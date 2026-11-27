"""
pytest tests/test_reverse_trace.py -v

Geometric tests for reverse ray tracing functionality.

These tests verify that rays propagate correctly through various combinations
of reflections and transmissions. Each test uses a 1x1 camera pixel to trace
a single ray and confirms whether the ray reaches the detector based on the
scene geometry.

Consistent with reverse ray traces, these tests validate geometry only.

Each test includes:
1. Geometry that should allow the ray to reach the detector (non-zero pixel)
2. Geometry that should prevent the ray from reaching the detector (zero pixel)

The pass-fail criterion is the trace log. Every scene is traced with
`log_level=logging.DEBUG` into a per-test log file, and that log must match,
verbatim, the known sequence of records pasted into the test that traces it -
the expectation sits beside the geometry that produced it. Every record, in
trace order (descend into a bounce, return from it, unwind), each with its
`TYPE:`, its `from:`/`target:` labels, its `action:`, its hop counter, bounce
depth and source count. Records carry no geometry floats, no timestamps and no
array content, so an expectation is byte-stable. Two things follow:

- A semantic error (a label that resolves to the wrong element, a context query
  that never fires, shaders fired at the wrong vertex, a path that dies
  silently) fails the test, not merely the pixel value.
- A developer reading a failure sees the whole actual trace against the whole
  known sequence and can tell which line diverges, and so whether the scene
  geometry or the tracer is at fault.

Names in the logs are the `node_id`s each test issues at construction - required and
validated by `validate_node_id` in `src/utilities/logconfig.py` - so a log reads like
the scene it traces: CAM_1, LAMP_1, SPHERE_1...

How to read a known sequence, top to bottom:
  ORIGIN       the sensor opens its legs (n: rays traced; one, in these tests)
  LEG          a ray walks from `from:` toward `target:` (START, then the
               reflection point of the element it last struck); `path:` is the
               breadcrumb of traversals that produced it
  CONTEXT      a visibility query succeeded: `target:` (a source) is visible
               from the vertex `from:`. No CONTEXT at a vertex means the
               source is in shadow from there
  SHADERS      a shader fired and contributed light; `src:` is the number of
               sources visible from that vertex, so `src: 0` is a shadow and
               `src: 1` on a reflection_model is a lit surface. `emission_model`
               is the SENSOR's own direct-view term - its cosine is taken
               against the sensor's pointing direction - so it fires only on the
               START leg; a bounce vertex gathers light through the surface's
               reflection model, never through an emission of its own
  INTERSECT    the element struck and the traversal that continues the path
  TERMINATION  a path that ends here (escaped to infinity, budget spent)

`hop:` is the global recursion counter, `b:` the bounce depth, `d:` the
transmission depth (0 in these tests - no transmissions). The tail of a
sequence is the unwind: each vertex, returning from its child leg, applies the
reflection model to the light that child gathered. So a known sequence carries
exactly ONE `emission_model` record (on the START leg), one `LEG` record per
vertex visited, and one `reflection_model` record per surface struck.

Viewing a trace in 3D (the ray debugger)

Every test carries a commented-out `scene.attach_ray_debugger(...)` call on the
line below the scene. Uncomment it to see that test's geometry and the rays it
traced in a vtk window (gold = elements, green = sensor, blue = the traced leg,
magenta = a leg that found a source, red arrow = the sensor's pointing
direction). Press `s` in the window to save a screenshot into `path`. The plot
fires at interpreter exit, so run ONE test at a time, and give it `-s` so the
window receives the event loop:

    pytest "tests/test_reverse_trace.py::TestSingleReflection::test_one_reflection_success" -s

Two conditions the call depends on:

- It must stay where it is - BEFORE the `scene +=` lines. `Scene.__iadd__` is
  what hands each element to the debugger, so attaching after them plots rays
  over an empty scene.
- It needs `vtk`, installed in the `luminous` conda env (python 3.10), not in
  base. Left commented out, no test imports vtk at all and the suite runs, and
  stays as fast, as it was.

`scene.attach_ray_debugger(display_3d_plot=False)` renders off-screen instead of
opening a window, which is what to use when only the file matters.
"""

import logging
import pytest
import numpy as np
from src.math.vector import Vector
from src.scene.scene import Scene
from src.element.element import SphereElement
from src.element.source import IsotropicSource
from src.element.detector import Camera
from src.utilities.logconfig import read_log_lines


def pixel_is_illuminated(camera: Camera) -> bool:
    """
    Check if a 1x1 camera pixel received any light.
    
    Args:
        camera: Camera with 1x1 resolution
        
    Returns:
        True if pixel has non-zero intensity, False otherwise
    """
    data = camera._reverse_trace_data

    if any(arr.size > 1 for arr in (data.x, data.y, data.z)):
        raise ValueError("Sensor data should be 1-dimensional. Ensure sensor has 1 pixel only.")

    return bool((data.x[0] > 0) or (data.y[0] > 0) or (data.z[0] > 0))


def traced_scene(tmp_path) -> Scene:
    """
    A reverse-tracing scene that records every step to a per-test log file.

    The log is the pass-fail criterion of these tests: `Scene.__init__` attaches
    a `DEBUG` file handler, so every hop, context query, shader emission and
    termination the tracer makes is written down for `assert_log_matches` below.

    Args:
        tmp_path: pytest's per-test temporary directory

    Returns:
        A Scene logging to `<tmp_path>/trace.log`
    """
    return Scene(reverse_trace=True, log_level=logging.DEBUG,
                 log_file=str(tmp_path / "trace.log"))


def assert_log_matches(scene: Scene, expected_log: str) -> None:
    """
    Assert the trace log is exactly the known sequence of records.

    Compares every record, in order, field for field - not a substring, not a
    sample of stages. On failure both traces are printed in full so the
    diverging line, and so which of the geometry and the tracer is at fault,
    can be read off directly.

    Args:
        scene: Scene whose `raytrace()` has already been called
        expected_log: the known sequence, one record per line
    """
    actual = "\n".join(read_log_lines(scene.log_file))
    expected = "\n".join(line.strip() for line in expected_log.strip().splitlines())
    assert actual == expected, (
        "the trace log does not match the known sequence\n"
        f"--- actual trace ({scene.log_file}) ---\n{actual}\n"
        f"--- known sequence ---\n{expected}"
    )


class TestSingleReflection:
    """Test geometry with one reflection."""

    def test_one_reflection_success(self, tmp_path):
        """Ray should reach detector via one reflection when geometry allows."""
        scene = traced_scene(tmp_path)
        # ray debugger - uncomment to view this trace (see module docstring; one test at a time, -s)
        # scene.attach_ray_debugger(path="./results", filename="one_reflection_success")

        source = IsotropicSource(
            node_id="LAMP_1",
            center=Vector(0, -1, 6),
            radius=0.3,
            color=Vector(1, 1, 1)
        )

        mirror_sphere = SphereElement(
            node_id="MIRROR_1",
            center=Vector(0, 0.3, 2),
            radius=0.5,
            color=Vector(0.9, 0.9, 0.9),
            user_params={'specular': 0.8, 'n_s': 50}
        )

        # 1x1 camera
        camera = Camera(
            node_id="CAM_1",
            width=1,
            height=1,
            position=Vector(0, 0, 4),
            pointing_direction=Vector(0, 0, -1),
            screen_width=0.01,
            screen_height=0.01
        )

        scene += source
        scene += mirror_sphere
        scene += camera
        scene.raytrace()

        assert pixel_is_illuminated(camera), \
            "Ray should reach detector via one reflection"

        expected_log = """
            ORIGIN       CAM_1             CAM_1             init_trace                      1
            LEG          CAM_1             START             reverse_leg         1    0   0               CAM_1
            INTERSECT    CAM_1             MIRROR_1          R_EXT               1    0   0  1
            LEG          MIRROR_1          SURF-REFLECT      reverse_leg         2    1   0               CAM_1->R_EXT(MIRROR_1)
            CONTEXT      MIRROR_1          LAMP_1            query_visibility    2    1   0  1
            TERMINATION  MIRROR_1          INFINITY          ray_escaped         2    1   0  1
            CONTEXT      MIRROR_1          LAMP_1            query_visibility    2    0   0  1
            SHADERS      CAM_1             MIRROR_1          reflection_model    2    0   0  1       1
            """
        assert_log_matches(scene, expected_log)

    def test_one_reflection_failure(self, tmp_path):
        """Ray should NOT reach detector when geometry prevents one reflection."""
        scene = traced_scene(tmp_path)
        # ray debugger - uncomment to view this trace (see module docstring; one test at a time, -s)
        # scene.attach_ray_debugger(path="./results", filename="one_reflection_failure")

        source = IsotropicSource(
            node_id="LAMP_1",
            center=Vector(0, 8, 6),
            radius=0.3,
            color=Vector(1, 1, 1)
        )

        mirror_sphere = SphereElement(
            node_id="MIRROR_1",
            center=Vector(0, 0.3, 2),
            radius=0.5,
            color=Vector(0.9, 0.9, 0.9),
            user_params={'specular': 0.8, 'n_s': 50}
        )

        # 1x1 camera
        camera = Camera(
            node_id="CAM_1",
            width=1,
            height=1,
            position=Vector(0, 0, 4),
            pointing_direction=Vector(0, 0, -1),
            screen_width=0.01,
            screen_height=0.01
        )

        scene += source
        scene += mirror_sphere
        scene += camera
        scene.raytrace()

        assert not pixel_is_illuminated(camera), \
            "Ray should NOT reach detector when source is blocked by sphere"

        expected_log = """
            ORIGIN       CAM_1             CAM_1             init_trace                      1
            LEG          CAM_1             START             reverse_leg         1    0   0               CAM_1
            INTERSECT    CAM_1             MIRROR_1          R_EXT               1    0   0  1
            LEG          MIRROR_1          SURF-REFLECT      reverse_leg         2    1   0               CAM_1->R_EXT(MIRROR_1)
            TERMINATION  MIRROR_1          INFINITY          ray_escaped         2    1   0  1
            """
        assert_log_matches(scene, expected_log)


class TestTwoReflections:
    """Test geometry with two reflections."""

    def test_two_reflections_success(self, tmp_path):
        """Ray should reach detector via two reflections when geometry allows."""
        scene = traced_scene(tmp_path)
        # ray debugger - uncomment to view this trace (see module docstring; one test at a time, -s)
        # scene.attach_ray_debugger(path="./results", filename="two_reflections_success")

        source = IsotropicSource(
            node_id="LAMP_1",
            center=Vector(-0.3, 0, 1),
            radius=0.3,
            color=Vector(1, 1, 1)
        )

        sphere1 = SphereElement(
            node_id="SPHERE_1",
            center=Vector(-0.3, 0, 2),
            radius=0.5,
            color=Vector(1, 0, 0),
            user_params={'specular': 0.8, 'n_s': 50}
        )

        sphere2 = SphereElement(
            node_id="SPHERE_2",
            center=Vector(2, 0, 3.5),
            radius=0.5,
            color=Vector(0, 1, 0),
            user_params={'specular': 0.8, 'n_s': 50}
        )

        # 1x1 camera
        camera = Camera(
            node_id="CAM_1",
            width=1,
            height=1,
            position=Vector(0, 0, 4),
            pointing_direction=Vector(0, 0, -1),
            screen_width=0.01,
            screen_height=0.01
        )

        scene += source
        scene += sphere1
        scene += sphere2
        scene += camera
        scene.raytrace()

        assert pixel_is_illuminated(camera), \
            "Ray should reach detector via two reflections"

        expected_log = """
            ORIGIN       CAM_1             CAM_1             init_trace                      1
            LEG          CAM_1             START             reverse_leg         1    0   0               CAM_1
            INTERSECT    CAM_1             SPHERE_1          R_EXT               1    0   0  1
            LEG          SPHERE_1          SURF-REFLECT      reverse_leg         2    1   0               CAM_1->R_EXT(SPHERE_1)
            INTERSECT    SPHERE_1          SPHERE_2          R_EXT               2    1   0  1
            LEG          SPHERE_2          SURF-REFLECT      reverse_leg         3    2   0               CAM_1->R_EXT(SPHERE_1)->R_EXT(SPHERE_2)
            CONTEXT      SPHERE_2          LAMP_1            query_visibility    3    2   0  1
            TERMINATION  SPHERE_2          INFINITY          ray_escaped         3    2   0  1
            CONTEXT      SPHERE_2          LAMP_1            query_visibility    3    1   0  1
            SHADERS      SPHERE_1          SPHERE_2          reflection_model    3    1   0  1       1
            SHADERS      CAM_1             SPHERE_1          reflection_model    3    0   0  1       1
            """
        assert_log_matches(scene, expected_log)

    def test_two_reflections_failure(self, tmp_path):
        """Ray should NOT reach detector when geometry prevents two reflections."""
        scene = traced_scene(tmp_path)
        # ray debugger - uncomment to view this trace (see module docstring; one test at a time, -s)
        scene.attach_ray_debugger(path="./results", filename="two_reflections_failure")

        source = IsotropicSource(
            node_id="LAMP_1",
            center=Vector(-1, 0, 1),
            radius=0.3,
            color=Vector(1, 1, 1)
        )

        sphere1 = SphereElement(
            node_id="SPHERE_1",
            center=Vector(-0.3, 0, 2),
            radius=0.5,
            color=Vector(1, 0, 0),
            user_params={'specular': 0.8, 'n_s': 50}
        )

        sphere2 = SphereElement(
            node_id="SPHERE_2",
            center=Vector(2, 0, 3.5),
            radius=0.5,
            color=Vector(0, 1, 0),
            user_params={'specular': 0.8, 'n_s': 50}
        )

        # 1x1 camera
        camera = Camera(
            node_id="CAM_1",
            width=1,
            height=1,
            position=Vector(0, 0, 4),
            pointing_direction=Vector(0, 0, -1),
            screen_width=0.01,
            screen_height=0.01
        )

        scene += source
        scene += sphere1
        scene += sphere2
        scene += camera
        scene.raytrace()

        assert not pixel_is_illuminated(camera), \
            "Ray should NOT reach detector when two-bounce geometry is wrong"

        expected_log = """
            ORIGIN       CAM_1             CAM_1             init_trace                      1
            LEG          CAM_1             START             reverse_leg         1    0   0               CAM_1
            INTERSECT    CAM_1             SPHERE_1          R_EXT               1    0   0  1
            LEG          SPHERE_1          SURF-REFLECT      reverse_leg         2    1   0               CAM_1->R_EXT(SPHERE_1)
            INTERSECT    SPHERE_1          SPHERE_2          R_EXT               2    1   0  1
            LEG          SPHERE_2          SURF-REFLECT      reverse_leg         3    2   0               CAM_1->R_EXT(SPHERE_1)->R_EXT(SPHERE_2)
            TERMINATION  SPHERE_2          INFINITY          ray_escaped         3    2   0  1
            """
        assert_log_matches(scene, expected_log)


class TestThreeReflections:
    """Test geometry with three reflections."""

    def test_three_reflections_success(self, tmp_path):
        """Ray should reach detector via three reflections when geometry allows."""
        scene = traced_scene(tmp_path)
        # ray debugger - uncomment to view this trace (see module docstring; one test at a time, -s)
        # scene.attach_ray_debugger(path="./results", filename="three_reflections_success")

        source = IsotropicSource(
            node_id="LAMP_1",
            center=Vector(-0.95, 0, 1),
            radius=0.3,
            color=Vector(1, 1, 1)
        )

        sphere1 = SphereElement(
            node_id="SPHERE_1",
            center=Vector(-0.3, 0, 2),
            radius=0.5,
            color=Vector(1, 0, 0),
            user_params={'specular': 0.8, 'n_s': 50}
        )

        sphere2 = SphereElement(
            node_id="SPHERE_2",
            center=Vector(2, 0, 3.5),
            radius=0.5,
            color=Vector(0, 1, 0),
            user_params={'specular': 0.8, 'n_s': 50}
        )

        sphere3 = SphereElement(
            node_id="SPHERE_3",
            center=Vector(2.5, 0.5, 1.2),
            radius=0.5,
            color=Vector(0, 1, 0),
            user_params={'specular': 0.8, 'n_s': 50}
        )

        # 1x1 camera
        camera = Camera(
            node_id="CAM_1",
            width=1,
            height=1,
            position=Vector(0, 0, 4),
            pointing_direction=Vector(0, 0, -1),
            screen_width=0.01,
            screen_height=0.01
        )

        scene += source
        scene += sphere1
        scene += sphere2
        scene += sphere3
        scene += camera
        scene.raytrace()

        assert pixel_is_illuminated(camera), \
            "Ray should reach detector via three reflections"

        # The log this trace must produce, verbatim. Three mirrors: the lamp is found at
        # the third bounce (CONTEXT at b: 3), answered on the unwind by src: 1 on the
        # record where SPHERE_2 shades SPHERE_3; the leg that found it then escapes
        # (TERMINATION), and the chain unwinds SPHERE_3 -> SPHERE_2 -> SPHERE_1 -> CAM_1,
        # each vertex shading the one it struck. Only SPHERE_3 can see the lamp; the
        # bounces below it are in shadow (src: 0), which is what makes this a three-bounce
        # path.
        expected_log = """
ORIGIN       CAM_1             CAM_1             init_trace                      1
LEG          CAM_1             START             reverse_leg         1    0   0               CAM_1
INTERSECT    CAM_1             SPHERE_1          R_EXT               1    0   0  1
LEG          SPHERE_1          SURF-REFLECT      reverse_leg         2    1   0               CAM_1->R_EXT(SPHERE_1)
INTERSECT    SPHERE_1          SPHERE_2          R_EXT               2    1   0  1
LEG          SPHERE_2          SURF-REFLECT      reverse_leg         3    2   0               CAM_1->R_EXT(SPHERE_1)->R_EXT(SPHERE_2)
INTERSECT    SPHERE_2          SPHERE_3          R_EXT               3    2   0  1
LEG          SPHERE_3          SURF-REFLECT      reverse_leg         4    3   0               CAM_1->R_EXT(SPHERE_1)->R_EXT(SPHERE_2)->R_EXT(SPHERE_3)
CONTEXT      SPHERE_3          LAMP_1            query_visibility    4    3   0  1
TERMINATION  SPHERE_3          INFINITY          ray_escaped         4    3   0  1
CONTEXT      SPHERE_3          LAMP_1            query_visibility    4    2   0  1
SHADERS      SPHERE_2          SPHERE_3          reflection_model    4    2   0  1       1
SHADERS      SPHERE_1          SPHERE_2          reflection_model    4    1   0  1       1
SHADERS      CAM_1             SPHERE_1          reflection_model    4    0   0  1       1
"""
        assert_log_matches(scene, expected_log)

    def test_three_plus_one_reflections_success(self, tmp_path):
        """Ray should reach detector via four reflections along two paths."""
        # The log answers the TODO this test used to leave open. Its sequence is
        # byte-for-byte the one logged by the three-bounce test above, and that
        # is the whole story: the recursion closes the moment a leg sees the lamp
        # (TERMINATION at b: 3), so the second, four-bounce path is never walked
        # and the lit pixel is not attributed to it. Supporting two paths needs a
        # tracer that keeps walking after a source is found and reports every
        # path it found; the records to read them off are already here - one LEG
        # per hop with a breadcrumb, and src: counts per vertex.
        scene = traced_scene(tmp_path)
        # ray debugger - uncomment to view this trace (see module docstring; one test at a time, -s)
        # scene.attach_ray_debugger(path="./results", filename="three_plus_one_reflections_success")

        source = IsotropicSource(
            node_id="LAMP_1",
            center=Vector(-0.75, 0, 1),
            radius=0.3,
            color=Vector(1, 1, 1)
        )

        sphere1 = SphereElement(
            node_id="SPHERE_1",
            center=Vector(-0.3, 0, 2),
            radius=0.5,
            color=Vector(1, 0, 0),
            user_params={'specular': 0.8, 'n_s': 50}
        )

        sphere2 = SphereElement(
            node_id="SPHERE_2",
            center=Vector(2, 0, 3.5),
            radius=0.5,
            color=Vector(0, 1, 0),
            user_params={'specular': 0.8, 'n_s': 50}
        )

        sphere3 = SphereElement(
            node_id="SPHERE_3",
            center=Vector(2.5, 0.5, 1.2),
            radius=0.5,
            color=Vector(0, 1, 0),
            user_params={'specular': 0.8, 'n_s': 50}
        )

        # 1x1 camera
        camera = Camera(
            node_id="CAM_1",
            width=1,
            height=1,
            position=Vector(0, 0, 4),
            pointing_direction=Vector(0, 0, -1),
            screen_width=0.01,
            screen_height=0.01
        )

        scene += source
        scene += sphere1
        scene += sphere2
        scene += sphere3
        scene += camera
        scene.raytrace()

        assert pixel_is_illuminated(camera), \
            "Ray should reach detector via three reflections"

        # The log this trace must produce, verbatim - and it is byte-for-byte the
        # sequence of the three-bounce test above. That identity is the finding: no
        # record here claims two paths, which is the gap this test used only to note
        # in a comment.
        expected_log = """
ORIGIN       CAM_1             CAM_1             init_trace                      1
LEG          CAM_1             START             reverse_leg         1    0   0               CAM_1
INTERSECT    CAM_1             SPHERE_1          R_EXT               1    0   0  1
LEG          SPHERE_1          SURF-REFLECT      reverse_leg         2    1   0               CAM_1->R_EXT(SPHERE_1)
INTERSECT    SPHERE_1          SPHERE_2          R_EXT               2    1   0  1
LEG          SPHERE_2          SURF-REFLECT      reverse_leg         3    2   0               CAM_1->R_EXT(SPHERE_1)->R_EXT(SPHERE_2)
INTERSECT    SPHERE_2          SPHERE_3          R_EXT               3    2   0  1
LEG          SPHERE_3          SURF-REFLECT      reverse_leg         4    3   0               CAM_1->R_EXT(SPHERE_1)->R_EXT(SPHERE_2)->R_EXT(SPHERE_3)
CONTEXT      SPHERE_3          LAMP_1            query_visibility    4    3   0  1
TERMINATION  SPHERE_3          INFINITY          ray_escaped         4    3   0  1
CONTEXT      SPHERE_3          LAMP_1            query_visibility    4    2   0  1
SHADERS      SPHERE_2          SPHERE_3          reflection_model    4    2   0  1       1
SHADERS      SPHERE_1          SPHERE_2          reflection_model    4    1   0  1       1
SHADERS      CAM_1             SPHERE_1          reflection_model    4    0   0  1       1
"""
        assert_log_matches(scene, expected_log)