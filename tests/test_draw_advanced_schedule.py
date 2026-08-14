from auto_capcut.core.draw_models import (
    DrawAction,
    DrawActionType,
    DrawImagePlan,
    DrawMode,
    DrawStyle,
    NormalizedRect,
    SceneImage,
    SceneObject,
)
from auto_capcut.core.draw_renderer import Stroke, build_advanced_schedule, assign_strokes_to_objects


def _scene(order=("a", "b", "c")) -> SceneImage:
    objects = (
        SceneObject("a", "art", NormalizedRect(0.00, 0.00, 0.30, 1.00)),
        SceneObject("b", "warning", NormalizedRect(0.35, 0.00, 0.30, 1.00)),
        SceneObject("c", "art", NormalizedRect(0.70, 0.00, 0.30, 1.00)),
    )
    return SceneImage("001.png", (100, 100), objects, tuple(order))


def _plan(duration=2_000_000, params=None):
    return DrawImagePlan(
        1,
        "001.png",
        0,
        duration,
        DrawMode.ADVANCED,
        DrawStyle.V1,
        "manual",
        (DrawAction(DrawActionType.DRAW, 0, duration, params or {}),),
    )


def test_assignment_is_unique_and_uses_overlap_for_centroid_outside() -> None:
    scene = _scene()
    strokes = (
        Stroke(((0.05, 0.10), (0.20, 0.10))),
        Stroke(((0.30, 0.10), (0.45, 0.10))),
        Stroke(((0.80, 0.10), (0.90, 0.10))),
    )
    assigned, unmatched = assign_strokes_to_objects(strokes, scene)
    assert [len(assigned[key]) for key in ("a", "b", "c")] == [1, 1, 1]
    assert unmatched == ()
    assert sum(len(value) for value in assigned.values()) == len(strokes)


def test_explicit_order_overrides_scene_order_and_groups_do_not_interleave() -> None:
    scene = _scene(("a", "b", "c"))
    strokes = (
        Stroke(((0.05, 0.10), (0.20, 0.10))),
        Stroke(((0.80, 0.10), (0.90, 0.10))),
        Stroke(((0.40, 0.10), (0.55, 0.10))),
    )
    schedule = build_advanced_schedule(strokes, _plan(params={"order": "c,a,b", "pause_each": "0.10"}), scene)
    assert schedule.resolved_order[:3] == ("c", "a", "b")
    object_phases = [phase for phase in schedule.phases if phase.kind == "object"]
    assert [phase.object_id for phase in object_phases] == ["c", "a", "b"]
    assert sum(phase.duration_us for phase in schedule.phases) == schedule.sketch_end_us
    assert any(phase.kind == "travel" for phase in schedule.phases)


def test_unmatched_policy_is_a_separate_group() -> None:
    scene = _scene()
    strokes = (Stroke(((0.01, 0.01), (0.02, 0.01))), Stroke(((0.32, 0.45), (0.33, 0.45))))
    schedule = build_advanced_schedule(strokes, _plan(params={"unmatched": "last"}), scene)
    assert schedule.unmatched_count == 1
    assert schedule.groups[-1].object_id == "__unmatched__"
    assert len(schedule.groups[-1].strokes) == 1


def test_scene_order_is_used_when_order_is_absent() -> None:
    scene = _scene(("c", "a", "b"))
    strokes = (
        Stroke(((0.05, 0.10), (0.20, 0.10))),
        Stroke(((0.80, 0.10), (0.90, 0.10))),
        Stroke(((0.40, 0.10), (0.55, 0.10))),
    )
    schedule = build_advanced_schedule(strokes, _plan(), scene)
    assert schedule.resolved_order == ("c", "a", "b")
