bl_info = {
    "name": "Extrude Blup",
    "author": "Your Name",
    "version": (1, 0, 0),
    "blender": (3, 6, 0),
    "location": "Preferences > Add-ons > Extrude Blup",
    "description": "Plays sounds while you model: extrude, bevel, inset, live segment count and more",
    "category": "Mesh",
}

import functools
import os
import random

import aud
import bpy
import numpy as np
from bpy.app.handlers import persistent
from bpy.props import (
    BoolProperty, CollectionProperty, EnumProperty,
    FloatProperty, IntProperty, StringProperty,
)

ADDON_ID = __package__ or __name__
RATE = 44100
IDLE_INTERVAL = 0.05      # polling interval while nothing modal is running
ACTIVE_INTERVAL = 0.02    # polling interval while a modal operator (bevel, loop cut...) runs
FADE_OUT = 0.05           # seconds; avoids a click when a long file is trimmed

# (label, comma separated keywords, pitch, enabled by default)
DEFAULT_RULES = (
    ("Extrude",   "extrude",                         1.00, True),
    ("Bevel",     "bevel",                           1.15, True),
    ("Inset",     "inset",                           1.30, True),
    ("Loop Cut",  "loopcut",                         0.90, False),
    ("Delete",    "mesh_ot_delete,object_ot_delete", 0.70, False),
    ("Merge",     "merge",                           1.20, False),
    ("Duplicate", "duplicate",                       1.00, False),
)

# Modal operators watched live: (keyword in operator id, property to watch, preference flag)
LIVE_TARGETS = (
    ("bevel",   "segments",    "live_bevel"),
    ("loopcut", "number_cuts", "live_loopcut"),
)

_UNSET = object()
_device = None
_last_history_ptr = _UNSET
_file_cache = {}          # (path, max length) -> aud.Sound or None
_synth = {}               # built-in fallback sounds, created on first use
_live_state = {}          # operator id -> last seen value
_live_handle = None
_load_error = ""


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def _prefs():
    addon = bpy.context.preferences.addons.get(ADDON_ID)
    return addon.preferences if addon else None


def _matches(keywords, idname):
    return any(k and k in idname for k in (s.strip().lower() for s in keywords.split(",")))


def _find_prop(props, name):
    """Reads an operator property. Macro operators (loop cut) keep it in a sub operator."""
    try:
        rna = props.bl_rna.properties
        if name in rna:
            return getattr(props, name)
        for p in rna:
            if p.type == 'POINTER':
                sub = getattr(props, p.identifier, None)
                if sub is not None and name in sub.bl_rna.properties:
                    return getattr(sub, name)
    except (AttributeError, TypeError, ReferenceError):
        pass
    return None


# ---------------------------------------------------------------
# Sound creation and playback
# ---------------------------------------------------------------
def _bubble(f0, sweep, dur, decay):
    """One bubble: a sine wave with rising frequency and exponential decay."""
    t = np.arange(int(RATE * dur)) / RATE
    phase = 2 * np.pi * np.cumsum(f0 * (1.0 + sweep * t)) / RATE
    wave = np.sin(phase) + 0.15 * np.sin(2 * phase)
    return wave * np.exp(-t / decay) * np.minimum(1.0, t / 0.004)


def _to_sound(arr):
    arr = arr / max(1e-6, np.max(np.abs(arr))) * 0.8
    return aud.Sound.buffer(arr.astype(np.float32).reshape(-1, 1), RATE)


def _blup_blup(f0):
    out = np.zeros(int(RATE * 0.32))
    first = _bubble(f0, 7.0, 0.14, 0.035)
    second = _bubble(f0 * 1.3, 7.0, 0.14, 0.035) * 0.7
    offset = int(RATE * 0.10)
    out[:len(first)] += first
    out[offset:offset + len(second)] += second
    return _to_sound(out)


def _synth_sounds():
    if not _synth:
        _synth["blup"] = [_blup_blup(f) for f in (380.0, 440.0, 500.0)]
        _synth["blip"] = [_to_sound(_bubble(520.0, 7.0, 0.10, 0.025))]
    return _synth


def _load_file(path, max_len):
    """Loads, trims, fades and caches a sound file. Returns None on failure."""
    global _load_error
    key = (path, round(max_len, 3))
    if key in _file_cache:
        return _file_cache[key]

    snd = None
    real = bpy.path.abspath(path)
    if not os.path.isfile(real):
        _load_error = f"File not found: {real}"
    else:
        try:
            fade = min(FADE_OUT, max_len * 0.5)
            snd = (aud.Sound.file(real).limit(0, max_len)
                   .fadeout(max(0.0, max_len - fade), fade).cache())
        except Exception as e:  # aud raises generic errors for unsupported files
            _load_error = f"Could not load {real}: {e}"
    _file_cache[key] = snd
    return snd


def _sound_for(path, max_len, fallback_kind):
    snd = _load_file(path, max_len) if path else None
    return snd if snd is not None else random.choice(_synth_sounds()[fallback_kind])


def _play(snd, volume, pitch):
    handle = _device.play(snd)
    handle.volume = volume
    handle.pitch = pitch
    return handle


def _play_rule(prefs, rule):
    snd = _sound_for(rule.sound_path, prefs.max_length, "blup")
    var = prefs.pitch_variation
    _play(snd, prefs.master_volume * rule.volume, rule.pitch * random.uniform(1 - var, 1 + var))


def _play_live(prefs, value):
    """Pitch rises by `live_semitones` for every segment."""
    global _live_handle
    if _live_handle is not None:
        _live_handle.stop()    # cut the previous blip so fast scrolling stays clean
    snd = _sound_for(prefs.live_sound_path, prefs.live_length, "blip")
    pitch = 2.0 ** (prefs.live_semitones * (value - 1) / 12.0)
    _live_handle = _play(snd, prefs.master_volume * prefs.live_volume, max(0.25, min(3.0, pitch)))


# ---------------------------------------------------------------
# Watching
# ---------------------------------------------------------------
def _poll_live(prefs):
    """Plays a blip when a watched value changes. Returns True while a modal operator runs."""
    seen = {}
    modal_active = False
    for win in bpy.context.window_manager.windows:
        for op in getattr(win, "modal_operators", ()):
            modal_active = True
            idname = op.bl_idname.lower()
            for keyword, prop, flag in LIVE_TARGETS:
                if keyword in idname and getattr(prefs, flag):
                    val = _find_prop(op.properties, prop)
                    if isinstance(val, int):
                        seen[idname] = val

    for key, val in seen.items():
        old = _live_state.get(key)
        if old is not None and val != old:    # first sight only records the start value
            _play_live(prefs, val)

    _live_state.clear()
    _live_state.update(seen)
    return modal_active


def _poll_history(prefs):
    """Plays the first matching rule when a new operator lands in the operator history."""
    global _last_history_ptr
    ops = bpy.context.window_manager.operators
    ptr = ops[-1].as_pointer() if len(ops) else 0

    if _last_history_ptr is _UNSET:           # first tick: only remember the current state
        _last_history_ptr = ptr
        return
    if ptr == _last_history_ptr:
        return
    _last_history_ptr = ptr

    if not (ptr and prefs.enabled):
        return
    idname = ops[-1].bl_idname.lower()
    for rule in prefs.rules:
        if rule.enabled and _matches(rule.keyword, idname):
            _play_rule(prefs, rule)
            break


def _tick():
    interval = IDLE_INTERVAL
    prefs = _prefs()
    if prefs is None:
        return interval
    try:
        if not prefs.initialized:
            _fill_defaults(prefs)
        if prefs.enabled and prefs.live_enabled and _poll_live(prefs):
            interval = ACTIVE_INTERVAL
        _poll_history(prefs)
    except Exception:
        # A timer that raises is removed by Blender, so never let an error escape.
        pass
    return interval


@persistent
def _on_depsgraph_update(*_):
    """Second, event based trigger: the preview mesh changes whenever the segment count does."""
    prefs = _prefs()
    if prefs is not None and prefs.initialized and prefs.enabled and prefs.live_enabled:
        try:
            _poll_live(prefs)
        except Exception:
            pass


# ---------------------------------------------------------------
# Properties and preferences
# ---------------------------------------------------------------
def _fill_defaults(prefs):
    prefs.rules.clear()
    for label, keyword, pitch, enabled in DEFAULT_RULES:
        rule = prefs.rules.add()
        rule.label, rule.keyword, rule.pitch, rule.enabled = label, keyword, pitch, enabled
    prefs.active_index = 0
    prefs.initialized = True


class ExtrudeBlupRule(bpy.types.PropertyGroup):
    label: StringProperty(name="Name", default="New Rule")
    keyword: StringProperty(
        name="Keywords",
        description="Text to look for in the operator id. Separate several with commas "
                    "(example: bevel,inset). Not case sensitive",
    )
    sound_path: StringProperty(
        name="Sound File", subtype='FILE_PATH',
        description="mp3, wav or ogg. Leave empty to use the built-in blup",
    )
    volume: FloatProperty(name="Volume", default=1.0, min=0.0, max=2.0)
    pitch: FloatProperty(name="Pitch", default=1.0, min=0.25, max=3.0)
    enabled: BoolProperty(name="Enabled", default=True)


class ExtrudeBlupPrefs(bpy.types.AddonPreferences):
    bl_idname = ADDON_ID

    enabled: BoolProperty(name="Add-on Enabled", default=True)
    master_volume: FloatProperty(name="Master Volume", default=0.5, min=0.0, max=1.0)
    pitch_variation: FloatProperty(
        name="Pitch Variation", default=0.08, min=0.0, max=0.3,
        description="Randomly shifts the pitch a little each time so it never gets monotonous",
    )
    max_length: FloatProperty(
        name="Max Length (s)", default=1.0, min=0.1, max=5.0,
        description="Longer files are trimmed with a short fade out",
    )
    rules: CollectionProperty(type=ExtrudeBlupRule)
    active_index: IntProperty(default=0)
    initialized: BoolProperty(default=False, options={'HIDDEN'})

    live_enabled: BoolProperty(name="Live Segment Sound", default=True)
    live_bevel: BoolProperty(name="Bevel segments", default=True)
    live_loopcut: BoolProperty(name="Loop Cut number of cuts", default=True)
    live_sound_path: StringProperty(
        name="Sound File", subtype='FILE_PATH',
        description="A short mp3, wav or ogg. Leave empty to use the built-in blip",
    )
    live_length: FloatProperty(
        name="Max Length (s)", default=0.25, min=0.05, max=1.0,
        description="Keep it short so fast scrolling does not turn into mush",
    )
    live_volume: FloatProperty(name="Volume", default=1.0, min=0.0, max=2.0)
    live_semitones: FloatProperty(
        name="Semitones per Step", default=1.0, min=0.0, max=4.0,
        description="How much the pitch rises with every added segment (0 = constant pitch)",
    )

    def draw(self, context):
        layout = self.layout
        col = layout.column()
        for prop in ("enabled", "master_volume", "pitch_variation", "max_length"):
            col.prop(self, prop)

        box = layout.box()
        box.label(text="Live Segment Sound (while scrolling the count)", icon='MOD_BEVEL')
        box.prop(self, "live_enabled")
        sub = box.column()
        sub.enabled = self.live_enabled
        row = sub.row()
        row.prop(self, "live_bevel")
        row.prop(self, "live_loopcut")
        sub.prop(self, "live_sound_path")
        row = sub.row()
        row.prop(self, "live_volume")
        row.prop(self, "live_length")
        sub.prop(self, "live_semitones")
        sub.operator("extrude_blup.live_test", icon='PLAY_SOUND')

        layout.separator()
        layout.label(text="Finished operations (first matching rule plays):")
        row = layout.row()
        row.template_list("EXTRUDEBLUP_UL_rules", "", self, "rules", self, "active_index", rows=6)
        side = row.column(align=True)
        for icon, action in (('ADD', 'ADD'), ('REMOVE', 'REMOVE'), (None, None),
                             ('TRIA_UP', 'UP'), ('TRIA_DOWN', 'DOWN')):
            if action:
                side.operator("extrude_blup.rule_action", icon=icon, text="").action = action
            else:
                side.separator()

        if 0 <= self.active_index < len(self.rules):
            rule = self.rules[self.active_index]
            box = layout.box()
            box.prop(rule, "label")
            box.prop(rule, "keyword")
            box.prop(rule, "sound_path")
            row = box.row()
            row.prop(rule, "volume")
            row.prop(rule, "pitch")
            box.operator("extrude_blup.rule_test", icon='PLAY_SOUND')

        layout.operator("extrude_blup.defaults", icon='LOOP_BACK')


class EXTRUDEBLUP_UL_rules(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "enabled", text="")
        row.prop(item, "label", text="", emboss=False)
        row.label(text=item.keyword or "-")


# ---------------------------------------------------------------
# Operators
# ---------------------------------------------------------------
def _reset_sound_cache():
    global _load_error
    _file_cache.clear()
    _load_error = ""


def _report_load_error(op):
    global _load_error
    if _load_error:
        op.report({'WARNING'}, _load_error)
        _load_error = ""


class EXTRUDEBLUP_OT_rule_action(bpy.types.Operator):
    bl_idname = "extrude_blup.rule_action"
    bl_label = "Edit Rule List"
    bl_options = {'INTERNAL'}

    action: EnumProperty(items=(
        ('ADD', "Add", "Add a rule"),
        ('REMOVE', "Remove", "Remove the selected rule"),
        ('UP', "Move Up", "Move the selected rule up"),
        ('DOWN', "Move Down", "Move the selected rule down"),
    ))

    def execute(self, context):
        prefs = _prefs()
        i, n = prefs.active_index, len(prefs.rules)
        if self.action == 'ADD':
            prefs.rules.add()
            prefs.active_index = n
        elif self.action == 'REMOVE' and 0 <= i < n:
            prefs.rules.remove(i)
            prefs.active_index = max(0, min(i, n - 2))
        elif self.action in {'UP', 'DOWN'}:
            j = i - 1 if self.action == 'UP' else i + 1
            if 0 <= i < n and 0 <= j < n:
                prefs.rules.move(i, j)
                prefs.active_index = j
        return {'FINISHED'}


class EXTRUDEBLUP_OT_rule_test(bpy.types.Operator):
    bl_idname = "extrude_blup.rule_test"
    bl_label = "Test Sound"

    def execute(self, context):
        prefs = _prefs()
        if 0 <= prefs.active_index < len(prefs.rules):
            _reset_sound_cache()
            _play_rule(prefs, prefs.rules[prefs.active_index])
            _report_load_error(self)
        return {'FINISHED'}


def _live_test_step(value):
    prefs = _prefs()
    if prefs is not None:
        _play_live(prefs, value)


class EXTRUDEBLUP_OT_live_test(bpy.types.Operator):
    bl_idname = "extrude_blup.live_test"
    bl_label = "Test Segment Sound (1 to 6)"

    def execute(self, context):
        _reset_sound_cache()
        for i, value in enumerate(range(1, 7)):
            bpy.app.timers.register(functools.partial(_live_test_step, value),
                                    first_interval=0.01 + i * 0.15)
        _live_test_step(1)    # plays the first step now so a load error is known right away
        _report_load_error(self)
        return {'FINISHED'}


class EXTRUDEBLUP_OT_defaults(bpy.types.Operator):
    bl_idname = "extrude_blup.defaults"
    bl_label = "Restore Default Rules"

    def execute(self, context):
        _fill_defaults(_prefs())
        return {'FINISHED'}


# ---------------------------------------------------------------
# Registration
# ---------------------------------------------------------------
classes = (
    ExtrudeBlupRule,
    ExtrudeBlupPrefs,
    EXTRUDEBLUP_UL_rules,
    EXTRUDEBLUP_OT_rule_action,
    EXTRUDEBLUP_OT_rule_test,
    EXTRUDEBLUP_OT_live_test,
    EXTRUDEBLUP_OT_defaults,
)


def register():
    global _device
    for cls in classes:
        bpy.utils.register_class(cls)
    if bpy.app.background:    # no UI, no operators to listen to
        return
    _device = aud.Device()
    if _on_depsgraph_update not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph_update)
    if not bpy.app.timers.is_registered(_tick):
        bpy.app.timers.register(_tick, first_interval=IDLE_INTERVAL, persistent=True)


def unregister():
    global _device, _live_handle, _last_history_ptr
    if bpy.app.timers.is_registered(_tick):
        bpy.app.timers.unregister(_tick)
    if _on_depsgraph_update in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph_update)
    _reset_sound_cache()
    _synth.clear()
    _live_state.clear()
    _live_handle = None
    _last_history_ptr = _UNSET
    _device = None
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)