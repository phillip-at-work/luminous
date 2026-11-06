import pytest
import numpy as np
import sys

from src.state_machine.state_machine import MockNode, TopologicalRayTracer

# print_debug_output("debug_output.py", lane_history)
def print_debug_output(filename, lane_history):
    with open(filename, "w", encoding="utf-8") as f:
        # 1. Inspect the depth: Check if the first element is a list
        is_nested = len(lane_history) > 0 and isinstance(lane_history[0], list)
        
        # 2. Helper function to format a single line item and its step comment
        def write_item(item, step_num, indent):
            if "TYPE: INTERSECT" in item:
                parts = {}
                for p in item.split('|'):
                    if ':' in p:
                        k, v = p.split(':', 1)
                        parts[k.strip()] = v.strip()
                target = parts.get('target', 'UNKNOWN')
                action = parts.get('action', '')
                
                if 'transmit_in' in action: desc = f"In through {target}"
                elif 'transmit_out' in action: desc = f"Out through {target}"
                elif 'reflect' in action: desc = f"Reflection off {target}"
                else: desc = f"Interaction with {target}"
                
                f.write(f"\n{indent}# Step {step_num}: {desc}\n")
                step_num += 1
                
            f.write(f"{indent}{repr(item)},\n")
            return step_num

        # 3. Generate output matching the exact structure
        f.write("expected_sequence = [\n")
        
        if is_nested:
            for sublist in lane_history:
                f.write("    [\n")
                step_counter = 0
                for item in sublist:
                    if isinstance(item, str):
                        step_counter = write_item(item, step_counter, "        ")
                f.write("    ],\n")
        else:
            step_counter = 0
            for item in lane_history:
                if isinstance(item, str):
                    step_counter = write_item(item, step_counter, "    ")
                    
        f.write("]\n")

#
#
#

# =========================================================================
# TEST 1: Pristine Direct View  (1 detector, 1 source, 0 elements, batch_size = 1 -> 1 lane)
# =========================================================================
def test_source_in_direct_view():
    """One ray (batch_size = 1) from DET_1 in a pristine environment: nothing blocks or
    intersects it, and SOURCE_A is in direct view at the START vertex."""
    detector = MockNode("DET_1")
    light_source = MockNode("SOURCE_A")
    tracer = TopologicalRayTracer(
        starts=[detector], elements=[], ends=[light_source], 
        bounce_count=0, max_transmission_depth=0
    )
    
    batch_size = 1
    # TIMELINE SHAPE CONTRACT (enforced at raytrace() entry): ROWS = ray lanes, one per batch_size;
    # COLUMNS = [ends... | elements...] in constructor order. Here: 1 lane x 1 column (SOURCE_A),
    # because elements = [] leaves no occluder columns to report.
    step_0_distances = np.array([
        [5.0],  # Lane 0 sees SOURCE_A directly at 5.0
    ])
    
    tracer.raytrace(batch_size=batch_size, MOCK_distances_timeline=[step_0_distances])
    lane_history = tracer.tracker.history

    # print_debug_output("debug_output_test_source_in_direct_view.py", lane_history)
    
    # This is the ONLY geometry in this file that satisfies the EMISSION RULE (SOURCE_A in direct
    # view at the START vertex), so it is the only test whose lane_history carries a CONTEXT /
    # SHADERS emission pair. One lane in, one lane of history out.
    expected_sequence = [
        [
            'TYPE: ORIGIN | path: DETECTOR_DET_1 | target: DET_1 | action: init_trace',
            'TYPE: CONTEXT | path: DETECTOR_DET_1 | target: SOURCE_A | action: query_emission_from_AIR',
            'TYPE: SHADERS | path: DETECTOR_DET_1 | target: SCENE | action: schedule_emission_shader_from_AIR',
        ],
    ]
    
    assert lane_history == expected_sequence  # emission pair present => credited at START vertex (x == 5.0)


# =========================================================================
# TEST PATH GEOMETRY:
# Detector ➔ R_EXT(M1) ➔ R_EXT(M2) ➔ R_EXT(M3) ➔ Source [End node reached; NO emission]
# =========================================================================
def test_detector_to_three_consecutive_reflections_success():
    """Tests a reflection chain that successfully ends on a light source.
    All mirrors should log intersections and schedule reflection shaders.

    EMISSION RULE: the source is NOT in direct view of the START node (the step_0 SOURCE_A
    column is inf), so NO emission shader is scheduled even though the leaf vertex three
    bounces out sees SOURCE_A at 4.0. Reaching the end node at the terminal point of a
    reflection chain proves PATH CONNECTIVITY only: it terminates the ray and validates the
    deferred reflection shader commits, and nothing is credited from the source (x == 0.0).
    """
    detector = MockNode("DET_1")
    mirror1 = MockNode("MIRROR_1")
    mirror2 = MockNode("MIRROR_2")
    mirror3 = MockNode("MIRROR_3")
    light_source = MockNode("SOURCE_A")
    
    tracer = TopologicalRayTracer(
        starts=[detector], elements=[mirror1, mirror2, mirror3], ends=[light_source], 
        bounce_count=3, max_transmission_depth=0
    )
    
    # Columns map to: [SOURCE_A,        MIRROR_1,       MIRROR_2,       MIRROR_3]
    step_0 = np.array([[float('inf'),   1.0,            float('inf'),   float('inf')]]) # Hits Mirror 1
    step_1 = np.array([[float('inf'),   float('inf'),   2.0,            float('inf')]]) # Hits Mirror 2
    step_2 = np.array([[float('inf'),   float('inf'),   float('inf'),   3.0        ]]) # Hits Mirror 3
    step_3 = np.array([[4.0,            float('inf'),   float('inf'),   float('inf')]]) # Finally hits Source A
    
    tracer.raytrace(
        batch_size=1, 
        MOCK_distances_timeline=[step_0, step_1, step_2, step_3]
    )
    lane_history = tracer.tracker.history
    # print_debug_output("debug_output_test_detector_to_three_consecutive_reflections_success.py", lane_history)
    
    # EMISSION RULE: the step_0 SOURCE_A column is inf, so no emission pair is logged. The
    # INTERSECT / SHADERS reflection commits below DO fire because the end node was reached
    # (path connectivity) at MOCK_step == 3, and they unwind leaf-first.
    expected_sequence = [
        [
            'TYPE: ORIGIN | path: DETECTOR_DET_1 | target: DET_1 | action: init_trace',

            # Step 0: Interaction with MIRROR_3
            'TYPE: INTERSECT | path: DETECTOR_DET_1➔R_EXT(MIRROR_1)➔R_EXT(MIRROR_2) | target: MIRROR_3 | action: external_surface_bounce',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(MIRROR_1)➔R_EXT(MIRROR_2) | target: MIRROR_3 | action: schedule_reflection_shader',

            # Step 1: Interaction with MIRROR_2
            'TYPE: INTERSECT | path: DETECTOR_DET_1➔R_EXT(MIRROR_1) | target: MIRROR_2 | action: external_surface_bounce',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(MIRROR_1) | target: MIRROR_2 | action: schedule_reflection_shader',

            # Step 2: Interaction with MIRROR_1
            'TYPE: INTERSECT | path: DETECTOR_DET_1 | target: MIRROR_1 | action: external_surface_bounce',
            'TYPE: SHADERS | path: DETECTOR_DET_1 | target: MIRROR_1 | action: schedule_reflection_shader',
        ],
    ]
    
    assert lane_history == expected_sequence  # no emission pair => no source radiance credited (x == 0.0)


# =========================================================================
# EMISSION CONTRACT (framework rule). Two geometries differing ONLY at the START vertex:
#
#   CASE A: SOURCE_A at 4.0 with MIRROR_1 at 1.0 in front -> occluded at START
#           ➔ R_EXT(M1) ➔ R_EXT(M2) ➔ R_EXT(M3) ➔ SOURCE_A      => NO emission shader
#   CASE B: SOURCE_A at 0.5 with MIRROR_1 at 1.0 behind it -> clear at START
#                                                                     => emission shader fires
# =========================================================================
def test_emission_shader_requires_start_end_direct_view():
    """The emission shader is scheduled IF AND ONLY IF the starts node and the ends node are in
    direct view of one another at the START vertex. Direct view at the terminal point of a
    reflection chain proves path connectivity only and must never schedule the emission shader."""
    # --- CASE A: SOURCE_A is in direct view only at the terminal reflection vertex ---
    detector_a = MockNode("DET_1")
    mirror1_a = MockNode("MIRROR_1")
    mirror2_a = MockNode("MIRROR_2")
    mirror3_a = MockNode("MIRROR_3")
    light_source_a = MockNode("SOURCE_A")

    tracer_a = TopologicalRayTracer(
        starts=[detector_a], elements=[mirror1_a, mirror2_a, mirror3_a], ends=[light_source_a], 
        bounce_count=3, max_transmission_depth=0
    )

    # Columns map to: [SOURCE_A,        MIRROR_1,       MIRROR_2,       MIRROR_3]
    step_0 = np.array([[4.0,            1.0,            float('inf'),   float('inf')]]) # 4.0 > 1.0 => SOURCE_A occluded at START
    step_1 = np.array([[float('inf'),  float('inf'),   2.0,            float('inf')]])  # hits MIRROR_2
    step_2 = np.array([[float('inf'),  float('inf'),   float('inf'),   3.0        ]])   # hits MIRROR_3
    step_3 = np.array([[4.0,           float('inf'),   float('inf'),   float('inf')]])  # hits SOURCE_A at the terminal vertex

    tracer_a.raytrace(
        batch_size=1, 
        MOCK_distances_timeline=[step_0, step_1, step_2, step_3]
    )
    lane_history = tracer_a.tracker.history

    # print_debug_output("debug_output_test_emission_shader_requires_start_end_direct_view_a.py", lane_history)

    expected_sequence = [
        [
            'TYPE: ORIGIN | path: DETECTOR_DET_1 | target: DET_1 | action: init_trace',

            # Step 0: Interaction with MIRROR_3
            'TYPE: INTERSECT | path: DETECTOR_DET_1➔R_EXT(MIRROR_1)➔R_EXT(MIRROR_2) | target: MIRROR_3 | action: external_surface_bounce',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(MIRROR_1)➔R_EXT(MIRROR_2) | target: MIRROR_3 | action: schedule_reflection_shader',

            # Step 1: Interaction with MIRROR_2
            'TYPE: INTERSECT | path: DETECTOR_DET_1➔R_EXT(MIRROR_1) | target: MIRROR_2 | action: external_surface_bounce',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(MIRROR_1) | target: MIRROR_2 | action: schedule_reflection_shader',

            # Step 2: Interaction with MIRROR_1
            'TYPE: INTERSECT | path: DETECTOR_DET_1 | target: MIRROR_1 | action: external_surface_bounce',
            'TYPE: SHADERS | path: DETECTOR_DET_1 | target: MIRROR_1 | action: schedule_reflection_shader',
        ],
    ]

    assert lane_history == expected_sequence  # no emission pair => occluded at START, nothing credited (x == 0.0)

    # --- CASE B: SOURCE_A is in direct view of the START vertex ---
    detector_b = MockNode("DET_1")
    mirror1_b = MockNode("MIRROR_1")
    mirror2_b = MockNode("MIRROR_2")
    mirror3_b = MockNode("MIRROR_3")
    light_source_b = MockNode("SOURCE_A")

    tracer_b = TopologicalRayTracer(
        starts=[detector_b], elements=[mirror1_b, mirror2_b, mirror3_b], ends=[light_source_b], 
        bounce_count=3, max_transmission_depth=0
    )

    # Columns map to: [SOURCE_A,        MIRROR_1,       MIRROR_2,       MIRROR_3]
    step_0 = np.array([[0.5,            1.0,            float('inf'),   float('inf')]])  # 0.5 <= 1.0 => SOURCE_A visible at START

    tracer_b.raytrace(batch_size=1, MOCK_distances_timeline=[step_0])
    lane_history = tracer_b.tracker.history

    # print_debug_output("debug_output_test_emission_shader_requires_start_end_direct_view_b.py", lane_history)

    expected_sequence = [
        [
            'TYPE: ORIGIN | path: DETECTOR_DET_1 | target: DET_1 | action: init_trace',
            'TYPE: CONTEXT | path: DETECTOR_DET_1 | target: SOURCE_A | action: query_emission_from_AIR',
            'TYPE: SHADERS | path: DETECTOR_DET_1 | target: SCENE | action: schedule_emission_shader_from_AIR',
        ],
    ]

    assert lane_history == expected_sequence  # emission pair present => emission credited (x == 5.0)


# =========================================================================
# TEST PATH GEOMETRY:
# Detector ➔ R_EXT(M1) ➔ R_EXT(M2) ➔ R_EXT(M3) ➔ ✖ [Infinity Miss]
# =========================================================================
def test_detector_to_three_consecutive_reflections_miss():
    """Tests a reflection chain that eventually escapes into empty space.
    No shaders or surface intersections should be computed or logged."""
    detector = MockNode("DET_1")
    mirror1 = MockNode("MIRROR_1")
    mirror2 = MockNode("MIRROR_2")
    mirror3 = MockNode("MIRROR_3")
    light_source = MockNode("SOURCE_A")
    
    tracer = TopologicalRayTracer(
        starts=[detector], elements=[mirror1, mirror2, mirror3], ends=[light_source], 
        bounce_count=3, max_transmission_depth=0
    )
    
    # Columns map to: [SOURCE_A,        MIRROR_1,       MIRROR_2,       MIRROR_3]
    step_0 = np.array([[float('inf'),   1.0,            float('inf'),   float('inf')]]) # Hits Mirror 1
    step_1 = np.array([[float('inf'),   float('inf'),   2.0,            float('inf')]]) # Hits Mirror 2
    step_2 = np.array([[float('inf'),   float('inf'),   float('inf'),   3.0        ]]) # Hits Mirror 3
    step_3 = np.array([[float('inf'),   float('inf'),   float('inf'),   float('inf')]]) # Escapes to Infinity!
    
    tracer.raytrace(
        batch_size=1, 
        MOCK_distances_timeline=[step_0, step_1, step_2, step_3]
    )
    lane_history = tracer.tracker.history

    # print_debug_output("debug_output_test_detector_to_three_consecutive_reflections_miss.py", lane_history)
    
    expected_sequence = [
        [
            'TYPE: ORIGIN | path: DETECTOR_DET_1 | target: DET_1 | action: init_trace',
            'TYPE: TERMINATION | path: DETECTOR_DET_1➔R_EXT(MIRROR_1)➔R_EXT(MIRROR_2)➔R_EXT(MIRROR_3) | target: INFINITY | action: ray_escaped_at_step_3',
        ],
    ]
    
    assert lane_history == expected_sequence  # ray escapes to INFINITY => nothing credited (x == 0.0)

# =========================================================================
# TEST PATH GEOMETRY TREE:
#
# Detector ➔ GLASS_BLOCK (Front Surface Split)
#          ├── R_EXT ➔ Source
#          └── T_IN  ➔ Source
# =========================================================================
def test_simultaneous_reflection_and_transmission_at_boundary():
    """
    Tests that an active ray striking a transparent element cleanly splits:
    - Fork 1: Reflects externally and successfully hits the light source.
    - Fork 2: Transmits into the element and hits the light source from inside.

    EMISSION RULE: both forks contact SOURCE_A at the terminal point of a reflection /
    transmission chain (MOCK_step > 0), never at the START vertex, so NO emission shader is
    scheduled. Both forks still commit their surface shaders because the end node was reached.
    """
    detector = MockNode("DET_1")
    glass_block = MockNode("GLASS_BLOCK", transparent=True, refractive_index=1.5)
    light_source = MockNode("SOURCE_A")
    
    # Allow 1 bounce and 1 transmission
    tracer = TopologicalRayTracer(
        starts=[detector], elements=[glass_block], ends=[light_source],
        bounce_count=1, max_transmission_depth=1
    )

    # Step 0: Ray strikes the glass surface (distance 1.0)
    # Columns map to:   [SOURCE_A,          GLASS_BLOCK]
    step_0 = np.array(  [[float('inf'),     1.0]])

    # We construct the mock timeline so that BOTH branches find a light source at Step 1.
    # Step 1: Both forks propagate out from Step 0.
    step_1 = np.array(  [[2.0,              float('inf')]])
    
    tracer.raytrace(batch_size=1, MOCK_distances_timeline=[step_0, step_1])
    lane_history = tracer.tracker.history
    # print_debug_output("debug_output_test_simultaneous_reflection_and_transmission_at_boundary.py", lane_history)
    
    expected_sequence = [
        [
            'TYPE: ORIGIN | path: DETECTOR_DET_1 | target: DET_1 | action: init_trace',

            # Step 0: Interaction with GLASS_BLOCK
            'TYPE: INTERSECT | path: DETECTOR_DET_1 | target: GLASS_BLOCK | action: external_surface_bounce',
            'TYPE: SHADERS | path: DETECTOR_DET_1 | target: GLASS_BLOCK | action: schedule_reflection_shader',

            # Step 1: In through GLASS_BLOCK
            'TYPE: INTERSECT | path: DETECTOR_DET_1 | target: GLASS_BLOCK | action: transmit_in_from_AIR',
            'TYPE: SHADERS | path: DETECTOR_DET_1 | target: GLASS_BLOCK | action: schedule_transmission_shader',
        ],
    ]

    assert lane_history == expected_sequence

# =========================================================================
# TEST PATH GEOMETRY TREE:
#
# Detector ➔ GLASS_BLOCK (Front Surface Split)
#          ├── R_EXT ➔ Source  [Evaluated at Step 1]
#          └── T_IN  ➔ GLASS_BLOCK (Back Surface Split)
#                     ├── R_INT ➔ Source  [Internal Bounce]
#                     └── T_OUT ➔ Source  [Exit to AIR]
# =========================================================================
def test_volumetric_slab_internal_and_external_splitting():
    """
    Simulates a ray interacting with a glass slab where it splits into T_IN 
    and R_EXT at the front surface, and the internal T_IN path hits the back 
    surface to cleanly evaluate internal reflections and exterior transmissions.

    EMISSION RULE: every source contact occurs at MOCK_step > 0, so the sequence contains
    INTERSECT / SHADERS commits only - no CONTEXT / SHADERS emission pairs.
    """
    detector = MockNode("DET_1")
    glass_block = MockNode("GLASS_BLOCK", transparent=True, refractive_index=1.5)
    light_source = MockNode("SOURCE_A")
    
    tracer = TopologicalRayTracer(
        starts=[detector], elements=[glass_block], ends=[light_source],
        bounce_count=2, max_transmission_depth=2
    )
    
    # Columns map to: [SOURCE_A, GLASS_BLOCK]
    # Step 0: Ray strikes the front glass block surface
    step_0 = np.array([[float('inf'), 1.0]])
    
    # Step 1: 
    # - R_EXT branch sees light source at 2.0 (no elements block it)
    # - T_IN branch encounters the back wall of glass block at 1.0. 
    #   Light source is at 2.0 (occluded because 2.0 > 1.0)
    step_1 = np.array([[2.0, 1.0]]) 
    
    # Step 2: Children branches look out to finalize connections to the light source
    step_2 = np.array([[3.0, float('inf')]])
    
    tracer.raytrace(batch_size=1, MOCK_distances_timeline=[step_0, step_1, step_2])
    lane_history = tracer.tracker.history

    # print_debug_output("debug_output_test_volumetric_slab_internal_and_external_splitting.py", lane_history)

    expected_sequence = [
        [
            'TYPE: ORIGIN | path: DETECTOR_DET_1 | target: DET_1 | action: init_trace',

            # Step 0: Interaction with GLASS_BLOCK
            'TYPE: INTERSECT | path: DETECTOR_DET_1➔R_EXT(GLASS_BLOCK) | target: GLASS_BLOCK | action: external_surface_bounce',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(GLASS_BLOCK) | target: GLASS_BLOCK | action: schedule_reflection_shader',

            # Step 1: In through GLASS_BLOCK
            'TYPE: INTERSECT | path: DETECTOR_DET_1➔R_EXT(GLASS_BLOCK) | target: GLASS_BLOCK | action: transmit_in_from_AIR',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(GLASS_BLOCK) | target: GLASS_BLOCK | action: schedule_transmission_shader',

            # Step 2: Interaction with GLASS_BLOCK
            'TYPE: INTERSECT | path: DETECTOR_DET_1 | target: GLASS_BLOCK | action: external_surface_bounce',
            'TYPE: SHADERS | path: DETECTOR_DET_1 | target: GLASS_BLOCK | action: schedule_reflection_shader',

            # Step 3: Interaction with GLASS_BLOCK
            'TYPE: INTERSECT | path: DETECTOR_DET_1➔T_IN(GLASS_BLOCK) | target: GLASS_BLOCK | action: internal_wall_bounce',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔T_IN(GLASS_BLOCK) | target: GLASS_BLOCK | action: schedule_reflection_shader',

            # Step 4: Out through GLASS_BLOCK
            'TYPE: INTERSECT | path: DETECTOR_DET_1➔T_IN(GLASS_BLOCK) | target: GLASS_BLOCK | action: transmit_out_to_AIR',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔T_IN(GLASS_BLOCK) | target: GLASS_BLOCK | action: schedule_transmission_shader',

            # Step 5: In through GLASS_BLOCK
            'TYPE: INTERSECT | path: DETECTOR_DET_1 | target: GLASS_BLOCK | action: transmit_in_from_AIR',
            'TYPE: SHADERS | path: DETECTOR_DET_1 | target: GLASS_BLOCK | action: schedule_transmission_shader',
        ],
    ]
    
    assert lane_history == expected_sequence


# =========================================================================
# TEST PATH GEOMETRY TREE:
#
# Detector ➔ GLASS_BLOCK (Front Surface Split)
#          ├── R_EXT ➔ ✖ [TERMINATION: bounce_budget_exhausted]
#          └── T_IN  ➔ Source [SUCCESS]
# =========================================================================
def test_splitting_where_one_fork_exhausts_budget():
    """
    Tests that if a ray splits at a transparent surface, but one fork
    exceeds its allowed budget (e.g. bounce count limit) while the other 
    succeeds, the successful fork's shaders are still scheduled normally 
    while the failed fork writes a TERMINATION log.

    EMISSION RULE: "[SUCCESS]" below means the end node was reached at MOCK_step > 0, which
    validates the transmission shader commit; it does NOT trigger the emission shader.
    """
    detector = MockNode("DET_1")
    glass_block = MockNode("GLASS_BLOCK", transparent=True)
    light_source = MockNode("SOURCE_A")
    
    # Set bounce_count to 0, but max_transmission_depth to 1.
    # This means reflection is immediately blocked by budget, but transmission can proceed!
    tracer = TopologicalRayTracer(
        starts=[detector], elements=[glass_block], ends=[light_source],
        bounce_count=0, max_transmission_depth=1
    )
    
    step_0 = np.array([[float('inf'), 1.0]]) # Hits glass block
    step_1 = np.array([[2.0, float('inf')]]) # Paths look for light source
    
    tracer.raytrace(batch_size=1, MOCK_distances_timeline=[step_0, step_1])
    lane_history = tracer.tracker.history
    # print_debug_output("debug_output_test_splitting_where_one_fork_exhausts_budget.py", lane_history)

    expected_sequence = [
        [
            'TYPE: ORIGIN | path: DETECTOR_DET_1 | target: DET_1 | action: init_trace',
            'TYPE: TERMINATION | path: DETECTOR_DET_1 | target: GLASS_BLOCK | action: bounce_budget_exhausted',

            # Step 0: In through GLASS_BLOCK
            'TYPE: INTERSECT | path: DETECTOR_DET_1 | target: GLASS_BLOCK | action: transmit_in_from_AIR',
            'TYPE: SHADERS | path: DETECTOR_DET_1 | target: GLASS_BLOCK | action: schedule_transmission_shader',
        ],
    ]

    assert lane_history == expected_sequence

# =========================================================================
# TEST 7 PATH GEOMETRY TREE:
#
# Detector ➔ GLASS_A (Front Face Split)
#          ├── R_EXT ➔ Source [Success]
#          └── T_IN  ➔ GLASS_A (Back Face Split)
#                     ├── R_INT ➔ Source [Success]
#                     └── T_OUT ➔ GLASS_B (Front Face Split)
#                                ├── R_EXT ➔ Source [Success]
#                                └── T_IN  ➔ GLASS_B (Back Face Split)
#                                           ├── R_INT ➔ Source [Success]
#                                           └── T_OUT ➔ ✖ [TERMINATION: transmit_budget_exhausted]
# =========================================================================
def test_double_slab_multi_transmission_exhaustion():
    """
    Tests a continuous multi-slab transmission run that forces budget exhaustion
    at exactly the 4th transmission boundary interface.

    EMISSION RULE: all 15 source contacts in this tree occur at MOCK_step > 0, so the
    expected sequence carries INTERSECT / SHADERS commits and one TERMINATION only -
    every CONTEXT / SHADERS emission pair is absent by design.
    """
    detector = MockNode("DET_1")
    glass_a = MockNode("GLASS_A", transparent=True)
    glass_b = MockNode("GLASS_B", transparent=True)
    light_source = MockNode("SOURCE_A")
    
    # Allow plenty of bounces, but hard-limit transmissions to exactly 3
    tracer = TopologicalRayTracer(
        starts=[detector], elements=[glass_a, glass_b], ends=[light_source],
        bounce_count=5, max_transmission_depth=3
    )
    
    # Columns map to: [SOURCE_A, GLASS_A, GLASS_B]
    step_0 = np.array([[float('inf'), 1.0,         float('inf')]]) # Hits Glass A front
    step_1 = np.array([[2.0,          1.0,         float('inf')]]) # Hits Glass A back
    step_2 = np.array([[2.0,          float('inf'), 1.0         ]]) # Hits Glass B front
    step_3 = np.array([[2.0,          float('inf'), 1.0         ]]) # Hits Glass B back
    step_4 = np.array([[2.0,          float('inf'), float('inf')]]) # T_OUT tries to exit B
    
    tracer.raytrace(
        batch_size=1, 
        MOCK_distances_timeline=[step_0, step_1, step_2, step_3, step_4]
    )
    lane_history = tracer.tracker.history

    # print_debug_output("debug_output_test_double_slab_multi_transmission_exhaustion.py", lane_history)
    
    expected_sequence = [
        [
            'TYPE: ORIGIN | path: DETECTOR_DET_1 | target: DET_1 | action: init_trace',

            # Step 0: Interaction with GLASS_B
            'TYPE: INTERSECT | path: DETECTOR_DET_1➔R_EXT(GLASS_A)➔R_EXT(GLASS_A)➔R_EXT(GLASS_B) | target: GLASS_B | action: external_surface_bounce',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(GLASS_A)➔R_EXT(GLASS_A)➔R_EXT(GLASS_B) | target: GLASS_B | action: schedule_reflection_shader',

            # Step 1: In through GLASS_B
            'TYPE: INTERSECT | path: DETECTOR_DET_1➔R_EXT(GLASS_A)➔R_EXT(GLASS_A)➔R_EXT(GLASS_B) | target: GLASS_B | action: transmit_in_from_AIR',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(GLASS_A)➔R_EXT(GLASS_A)➔R_EXT(GLASS_B) | target: GLASS_B | action: schedule_transmission_shader',

            # Step 2: Interaction with GLASS_B
            'TYPE: INTERSECT | path: DETECTOR_DET_1➔R_EXT(GLASS_A)➔R_EXT(GLASS_A) | target: GLASS_B | action: external_surface_bounce',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(GLASS_A)➔R_EXT(GLASS_A) | target: GLASS_B | action: schedule_reflection_shader',

            # Step 3: Interaction with GLASS_B
            'TYPE: INTERSECT | path: DETECTOR_DET_1➔R_EXT(GLASS_A)➔R_EXT(GLASS_A)➔T_IN(GLASS_B) | target: GLASS_B | action: internal_wall_bounce',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(GLASS_A)➔R_EXT(GLASS_A)➔T_IN(GLASS_B) | target: GLASS_B | action: schedule_reflection_shader',

            # Step 4: Out through GLASS_B
            'TYPE: INTERSECT | path: DETECTOR_DET_1➔R_EXT(GLASS_A)➔R_EXT(GLASS_A)➔T_IN(GLASS_B) | target: GLASS_B | action: transmit_out_to_AIR',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(GLASS_A)➔R_EXT(GLASS_A)➔T_IN(GLASS_B) | target: GLASS_B | action: schedule_transmission_shader',

            # Step 5: In through GLASS_B
            'TYPE: INTERSECT | path: DETECTOR_DET_1➔R_EXT(GLASS_A)➔R_EXT(GLASS_A) | target: GLASS_B | action: transmit_in_from_AIR',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(GLASS_A)➔R_EXT(GLASS_A) | target: GLASS_B | action: schedule_transmission_shader',

            # Step 6: Interaction with GLASS_A
            'TYPE: INTERSECT | path: DETECTOR_DET_1➔R_EXT(GLASS_A) | target: GLASS_A | action: external_surface_bounce',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(GLASS_A) | target: GLASS_A | action: schedule_reflection_shader',

            # Step 7: Interaction with GLASS_B
            'TYPE: INTERSECT | path: DETECTOR_DET_1➔R_EXT(GLASS_A)➔T_IN(GLASS_A)➔R_EXT(GLASS_B) | target: GLASS_B | action: external_surface_bounce',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(GLASS_A)➔T_IN(GLASS_A)➔R_EXT(GLASS_B) | target: GLASS_B | action: schedule_reflection_shader',

            # Step 8: In through GLASS_B
            'TYPE: INTERSECT | path: DETECTOR_DET_1➔R_EXT(GLASS_A)➔T_IN(GLASS_A)➔R_EXT(GLASS_B) | target: GLASS_B | action: transmit_in_from_GLASS_A',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(GLASS_A)➔T_IN(GLASS_A)➔R_EXT(GLASS_B) | target: GLASS_B | action: schedule_transmission_shader',

            # Step 9: Interaction with GLASS_B
            'TYPE: INTERSECT | path: DETECTOR_DET_1➔R_EXT(GLASS_A)➔T_IN(GLASS_A) | target: GLASS_B | action: external_surface_bounce',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(GLASS_A)➔T_IN(GLASS_A) | target: GLASS_B | action: schedule_reflection_shader',

            # Step 10: Interaction with GLASS_B
            'TYPE: INTERSECT | path: DETECTOR_DET_1➔R_EXT(GLASS_A)➔T_IN(GLASS_A)➔T_IN(GLASS_B) | target: GLASS_B | action: internal_wall_bounce',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(GLASS_A)➔T_IN(GLASS_A)➔T_IN(GLASS_B) | target: GLASS_B | action: schedule_reflection_shader',

            # Step 11: Out through GLASS_B
            'TYPE: INTERSECT | path: DETECTOR_DET_1➔R_EXT(GLASS_A)➔T_IN(GLASS_A)➔T_IN(GLASS_B) | target: GLASS_B | action: transmit_out_to_GLASS_A',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(GLASS_A)➔T_IN(GLASS_A)➔T_IN(GLASS_B) | target: GLASS_B | action: schedule_transmission_shader',

            # Step 12: In through GLASS_B
            'TYPE: INTERSECT | path: DETECTOR_DET_1➔R_EXT(GLASS_A)➔T_IN(GLASS_A) | target: GLASS_B | action: transmit_in_from_GLASS_A',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(GLASS_A)➔T_IN(GLASS_A) | target: GLASS_B | action: schedule_transmission_shader',

            # Step 13: In through GLASS_A
            'TYPE: INTERSECT | path: DETECTOR_DET_1➔R_EXT(GLASS_A) | target: GLASS_A | action: transmit_in_from_AIR',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(GLASS_A) | target: GLASS_A | action: schedule_transmission_shader',

            # Step 14: Interaction with GLASS_A
            'TYPE: INTERSECT | path: DETECTOR_DET_1 | target: GLASS_A | action: external_surface_bounce',
            'TYPE: SHADERS | path: DETECTOR_DET_1 | target: GLASS_A | action: schedule_reflection_shader',

            # Step 15: Interaction with GLASS_B
            'TYPE: INTERSECT | path: DETECTOR_DET_1➔T_IN(GLASS_A)➔R_INT(GLASS_A)➔R_EXT(GLASS_B) | target: GLASS_B | action: external_surface_bounce',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔T_IN(GLASS_A)➔R_INT(GLASS_A)➔R_EXT(GLASS_B) | target: GLASS_B | action: schedule_reflection_shader',

            # Step 16: In through GLASS_B
            'TYPE: INTERSECT | path: DETECTOR_DET_1➔T_IN(GLASS_A)➔R_INT(GLASS_A)➔R_EXT(GLASS_B) | target: GLASS_B | action: transmit_in_from_GLASS_A',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔T_IN(GLASS_A)➔R_INT(GLASS_A)➔R_EXT(GLASS_B) | target: GLASS_B | action: schedule_transmission_shader',

            # Step 17: Interaction with GLASS_B
            'TYPE: INTERSECT | path: DETECTOR_DET_1➔T_IN(GLASS_A)➔R_INT(GLASS_A) | target: GLASS_B | action: external_surface_bounce',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔T_IN(GLASS_A)➔R_INT(GLASS_A) | target: GLASS_B | action: schedule_reflection_shader',

            # Step 18: Interaction with GLASS_B
            'TYPE: INTERSECT | path: DETECTOR_DET_1➔T_IN(GLASS_A)➔R_INT(GLASS_A)➔T_IN(GLASS_B) | target: GLASS_B | action: internal_wall_bounce',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔T_IN(GLASS_A)➔R_INT(GLASS_A)➔T_IN(GLASS_B) | target: GLASS_B | action: schedule_reflection_shader',

            # Step 19: Out through GLASS_B
            'TYPE: INTERSECT | path: DETECTOR_DET_1➔T_IN(GLASS_A)➔R_INT(GLASS_A)➔T_IN(GLASS_B) | target: GLASS_B | action: transmit_out_to_GLASS_A',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔T_IN(GLASS_A)➔R_INT(GLASS_A)➔T_IN(GLASS_B) | target: GLASS_B | action: schedule_transmission_shader',

            # Step 20: In through GLASS_B
            'TYPE: INTERSECT | path: DETECTOR_DET_1➔T_IN(GLASS_A)➔R_INT(GLASS_A) | target: GLASS_B | action: transmit_in_from_GLASS_A',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔T_IN(GLASS_A)➔R_INT(GLASS_A) | target: GLASS_B | action: schedule_transmission_shader',

            # Step 21: Interaction with GLASS_A
            'TYPE: INTERSECT | path: DETECTOR_DET_1➔T_IN(GLASS_A) | target: GLASS_A | action: internal_wall_bounce',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔T_IN(GLASS_A) | target: GLASS_A | action: schedule_reflection_shader',

            # Step 22: Interaction with GLASS_B
            'TYPE: INTERSECT | path: DETECTOR_DET_1➔T_IN(GLASS_A)➔T_OUT(GLASS_A)➔R_EXT(GLASS_B) | target: GLASS_B | action: external_surface_bounce',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔T_IN(GLASS_A)➔T_OUT(GLASS_A)➔R_EXT(GLASS_B) | target: GLASS_B | action: schedule_reflection_shader',

            # Step 23: In through GLASS_B
            'TYPE: INTERSECT | path: DETECTOR_DET_1➔T_IN(GLASS_A)➔T_OUT(GLASS_A)➔R_EXT(GLASS_B) | target: GLASS_B | action: transmit_in_from_AIR',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔T_IN(GLASS_A)➔T_OUT(GLASS_A)➔R_EXT(GLASS_B) | target: GLASS_B | action: schedule_transmission_shader',

            # Step 24: Interaction with GLASS_B
            'TYPE: INTERSECT | path: DETECTOR_DET_1➔T_IN(GLASS_A)➔T_OUT(GLASS_A) | target: GLASS_B | action: external_surface_bounce',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔T_IN(GLASS_A)➔T_OUT(GLASS_A) | target: GLASS_B | action: schedule_reflection_shader',

            # Step 25: Interaction with GLASS_B
            'TYPE: INTERSECT | path: DETECTOR_DET_1➔T_IN(GLASS_A)➔T_OUT(GLASS_A)➔T_IN(GLASS_B) | target: GLASS_B | action: internal_wall_bounce',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔T_IN(GLASS_A)➔T_OUT(GLASS_A)➔T_IN(GLASS_B) | target: GLASS_B | action: schedule_reflection_shader',
            'TYPE: TERMINATION | path: DETECTOR_DET_1➔T_IN(GLASS_A)➔T_OUT(GLASS_A)➔T_IN(GLASS_B) | target: GLASS_B | action: transmit_budget_exhausted',

            # Step 26: In through GLASS_B
            'TYPE: INTERSECT | path: DETECTOR_DET_1➔T_IN(GLASS_A)➔T_OUT(GLASS_A) | target: GLASS_B | action: transmit_in_from_AIR',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔T_IN(GLASS_A)➔T_OUT(GLASS_A) | target: GLASS_B | action: schedule_transmission_shader',

            # Step 27: Out through GLASS_A
            'TYPE: INTERSECT | path: DETECTOR_DET_1➔T_IN(GLASS_A) | target: GLASS_A | action: transmit_out_to_AIR',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔T_IN(GLASS_A) | target: GLASS_A | action: schedule_transmission_shader',

            # Step 28: In through GLASS_A
            'TYPE: INTERSECT | path: DETECTOR_DET_1 | target: GLASS_A | action: transmit_in_from_AIR',
            'TYPE: SHADERS | path: DETECTOR_DET_1 | target: GLASS_A | action: schedule_transmission_shader',
        ],
    ]


    assert lane_history == expected_sequence


# =========================================================================
# TEST 8 PATH GEOMETRY TREE:
#
# Detector ➔ MIRROR_1
#          └── R_EXT ➔ MIRROR_2
#                     └── R_EXT ➔ MIRROR_1
#                                └── R_EXT ➔ MIRROR_2
#                                           └── R_EXT ➔ ✖ [TERMINATION: bounce_budget_exhausted]
# =========================================================================
def test_inter_element_mirror_cave_bounce_exhaustion():
    """
    Tests that a ray trapped bouncing infinitely between two mirrors is 
    cleanly truncated the exact moment it hits the maximum allowed bounce threshold.
    """
    detector = MockNode("DET_1")
    mirror_1 = MockNode("MIRROR_1")
    mirror_2 = MockNode("MIRROR_2")
    light_source = MockNode("SOURCE_A")
    
    # Clamp bounce count to 3
    tracer = TopologicalRayTracer(
        starts=[detector], elements=[mirror_1, mirror_2], ends=[light_source],
        bounce_count=3, max_transmission_depth=0
    )
    
    # Columns map to: [SOURCE_A, MIRROR_1, MIRROR_2]
    step_0 = np.array([[float('inf'), 1.0,         float('inf')]]) # Hits Mirror 1 (Bounce 1)
    step_1 = np.array([[float('inf'), float('inf'), 1.0         ]]) # Hits Mirror 2 (Bounce 2)
    step_2 = np.array([[float('inf'), 1.0,         float('inf')]]) # Hits Mirror 1 (Bounce 3)
    step_3 = np.array([[float('inf'), float('inf'), 1.0         ]]) # Hits Mirror 2 (Bounce 4 -> Exhausted!)
    
    tracer.raytrace(
        batch_size=1, 
        MOCK_distances_timeline=[step_0, step_1, step_2, step_3]
    )
    lane_history = tracer.tracker.history

    # print_debug_output("debug_output_test_inter_element_mirror_cave_bounce_exhaustion.py", lane_history)

    expected_sequence = [
        [
            'TYPE: ORIGIN | path: DETECTOR_DET_1 | target: DET_1 | action: init_trace',
            'TYPE: TERMINATION | path: DETECTOR_DET_1➔R_EXT(MIRROR_1)➔R_EXT(MIRROR_2)➔R_EXT(MIRROR_1) | target: MIRROR_2 | action: bounce_budget_exhausted',
        ],
    ]

    assert lane_history == expected_sequence

# =========================================================================
# TEST 9 PATH GEOMETRY TREE:
#
# Detector ➔ LIGHT_GUIDE (Front Face Split)
#          ├── R_EXT ➔ Source [Success]
#          └── T_IN  ➔ LIGHT_GUIDE (Internal Boundary Wall 1)
#                     ├── T_OUT ➔ Source [Success]
#                     └── R_INT ➔ LIGHT_GUIDE (Internal Boundary Wall 2)
#                                ├── T_OUT ➔ Source [Success]
#                                └── R_INT ➔ ✖ [TERMINATION: bounce_budget_exhausted]
# =========================================================================
def test_light_guide_internal_reflection_exhaustion():
    """
    Simulates a ray trapped inside a dense translucent medium block, testing 
    that consecutive internal reflections (R_INT) increment the bounce budget 
    and terminate cleanly when exhausted.
    """
    detector = MockNode("DET_1")
    light_guide = MockNode("LIGHT_GUIDE", transparent=True)
    light_source = MockNode("SOURCE_A")
    
    # Cap bounces at 2
    tracer = TopologicalRayTracer(
        starts=[detector], elements=[light_guide], ends=[light_source],
        bounce_count=2, max_transmission_depth=5
    )
    
    # Columns map to: [SOURCE_A, LIGHT_GUIDE]
    step_0 = np.array([[float('inf'), 1.0]]) # Hits front face (T_IN, bounce=0)
    step_1 = np.array([[5.0,          1.0]]) # Hits wall 1 (R_INT, bounce=1)
    step_2 = np.array([[5.0,          1.0]]) # Hits wall 2 (R_INT, bounce=2)
    step_3 = np.array([[5.0,          1.0]]) # Hits wall 3 (R_INT, bounce=3 -> Exhausted!)
    
    tracer.raytrace(
        batch_size=1, 
        MOCK_distances_timeline=[step_0, step_1, step_2, step_3]
    )
    lane_history = tracer.tracker.history

    # print_debug_output("debug_output_test_light_guide_internal_reflection_exhaustion.py", lane_history)
    
    expected_sequence = [
        [
            'TYPE: ORIGIN | path: DETECTOR_DET_1 | target: DET_1 | action: init_trace',
            'TYPE: TERMINATION | path: DETECTOR_DET_1➔R_EXT(LIGHT_GUIDE)➔R_EXT(LIGHT_GUIDE) | target: LIGHT_GUIDE | action: bounce_budget_exhausted',
            'TYPE: TERMINATION | path: DETECTOR_DET_1➔R_EXT(LIGHT_GUIDE)➔R_EXT(LIGHT_GUIDE)➔T_IN(LIGHT_GUIDE) | target: LIGHT_GUIDE | action: bounce_budget_exhausted',
            'TYPE: TERMINATION | path: DETECTOR_DET_1➔R_EXT(LIGHT_GUIDE)➔T_IN(LIGHT_GUIDE)➔R_INT(LIGHT_GUIDE) | target: LIGHT_GUIDE | action: bounce_budget_exhausted',
            'TYPE: TERMINATION | path: DETECTOR_DET_1➔T_IN(LIGHT_GUIDE)➔R_INT(LIGHT_GUIDE)➔R_INT(LIGHT_GUIDE) | target: LIGHT_GUIDE | action: bounce_budget_exhausted',
        ],
    ]

    assert lane_history == expected_sequence

# ====================================================================================
# BATCH SIMULATION SCENE GRAPH:
#
# DETECTOR_DET_1 (Camera 1) ➔ Processes Batch Lanes [0, 1]
# │  ├── Lane 0 ➔ Strikes GLASS_BLOCK front face, splits into:
# │  │     ├── R_EXT ➔ Strikes SOURCE_A [Success]
# │  │     └── T_IN  ➔ Penetrates volume, exits back face ➔ Strikes SOURCE_B [Success]
# │  └── Lane 1 ➔ Misses everything entirely ➔ Escapes to INFINITY [Dark Pixel]
# │
# DETECTOR_DET_2 (Camera 2) ➔ Processes Batch Lanes [0, 1]
#    ├── Lane 0 ➔ Points straight at MIRROR_1 ➔ R_EXT ➔ Strikes SOURCE_A [Success]
#    └── Lane 1 ➔ Points directly into empty space ➔ Escapes to INFINITY [Dark Pixel]
# ====================================================================================
def test_multi_detector_multi_source_vectorized_batch():
    """
    Validates a complex environment utilizing multiple detectors, multiple light sources, 
    and elements simultaneously across a parallel batch execution tracking framework.

    EMISSION RULE: no lane has a source in direct view at its START vertex (every step_0
    source column is inf), so neither detector's sequence contains CONTEXT / SHADERS
    emission pairs. "[Success]" in the tree above denotes end-node reachability only.
    """
    # 1. Define Entities
    det1 = MockNode("DET_1")
    det2 = MockNode("DET_2")
    
    glass = MockNode("GLASS", transparent=True)
    mirror = MockNode("MIRROR")
    
    src_a = MockNode("SOURCE_A")
    src_b = MockNode("SOURCE_B")
    
    # Instantiate tracer configuration
    # Layout order for distance columns: [SOURCE_A, SOURCE_B, GLASS, MIRROR]
    tracer = TopologicalRayTracer(
        starts=[det1, det2], 
        elements=[glass, mirror], 
        ends=[src_a, src_b],
        bounce_count=1, 
        max_transmission_depth=1
    )
    
    batch_size = 2
    
    # 2. Construct Mock Distance Timeline Matrices
    # Step 0 Matrix Layout: [SOURCE_A, SOURCE_B, GLASS, MIRROR]
    step_0 = np.array([
        [float('inf'), float('inf'), 1.0,         float('inf')], # Lane 0: Points at Glass front face
        [float('inf'), float('inf'), float('inf'), float('inf')]  # Lane 1: Escapes to infinity immediately
    ])
    
    # Step 1: Evaluating the resulting secondary branches spawned from Step 0
    # - The R_EXT branch from Lane 0 looks out and hits SOURCE_A at distance 2.0
    # - The T_IN branch from Lane 0 hits the back face of the GLASS block at distance 1.0
    step_1 = np.array([
        [2.0,          float('inf'), 1.0,         float('inf')], # Lane 0 processing
        [float('inf'), float('inf'), float('inf'), float('inf')]  # Lane 1 dead space
    ])
    
    # Step 2: Evaluating deep boundary exits
    # - The T_OUT branch exiting the back of the glass strikes SOURCE_B at distance 3.0
    # - Concurrently, Camera 2 (DET_2) uses this step index mapping to bounce off the MIRROR
    step_2 = np.array([
        [float('inf'), 3.0,          float('inf'), float('inf')], # Lane 0 hits Source B
        [2.0,          float('inf'), float('inf'), 1.0         ]  # Lane 1 hits Mirror, bounces to Source A
    ])
    
    # Compile the timelines
    timeline = [step_0, step_1, step_2]
    
    # 3. Execute Vectorized Frame Trace
    tracer.raytrace(batch_size=batch_size, MOCK_distances_timeline=timeline)

    lane_history = tracer.tracker.history
    # print_debug_output("debug_output_test_multi_detector_multi_source_vectorized_batch.py", lane_history)
    
    expected_sequence = [
        [
            'TYPE: ORIGIN | path: DETECTOR_DET_1 | target: DET_1 | action: init_trace',
            'TYPE: TERMINATION | path: DETECTOR_DET_1➔R_EXT(GLASS) | target: GLASS | action: bounce_budget_exhausted',

            # Step 0: In through GLASS
            'TYPE: INTERSECT | path: DETECTOR_DET_1➔R_EXT(GLASS) | target: GLASS | action: transmit_in_from_AIR',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(GLASS) | target: GLASS | action: schedule_transmission_shader',

            # Step 1: Interaction with GLASS
            'TYPE: INTERSECT | path: DETECTOR_DET_1 | target: GLASS | action: external_surface_bounce',
            'TYPE: SHADERS | path: DETECTOR_DET_1 | target: GLASS | action: schedule_reflection_shader',

            # Step 2: Interaction with GLASS
            'TYPE: INTERSECT | path: DETECTOR_DET_1➔T_IN(GLASS) | target: GLASS | action: internal_wall_bounce',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔T_IN(GLASS) | target: GLASS | action: schedule_reflection_shader',
            'TYPE: TERMINATION | path: DETECTOR_DET_1➔T_IN(GLASS) | target: GLASS | action: transmit_budget_exhausted',

            # Step 3: In through GLASS
            'TYPE: INTERSECT | path: DETECTOR_DET_1 | target: GLASS | action: transmit_in_from_AIR',
            'TYPE: SHADERS | path: DETECTOR_DET_1 | target: GLASS | action: schedule_transmission_shader',
            'TYPE: ORIGIN | path: DETECTOR_DET_2 | target: DET_2 | action: init_trace',
            'TYPE: TERMINATION | path: DETECTOR_DET_2➔R_EXT(GLASS) | target: GLASS | action: bounce_budget_exhausted',

            # Step 4: In through GLASS
            'TYPE: INTERSECT | path: DETECTOR_DET_2➔R_EXT(GLASS) | target: GLASS | action: transmit_in_from_AIR',
            'TYPE: SHADERS | path: DETECTOR_DET_2➔R_EXT(GLASS) | target: GLASS | action: schedule_transmission_shader',

            # Step 5: Interaction with GLASS
            'TYPE: INTERSECT | path: DETECTOR_DET_2 | target: GLASS | action: external_surface_bounce',
            'TYPE: SHADERS | path: DETECTOR_DET_2 | target: GLASS | action: schedule_reflection_shader',

            # Step 6: Interaction with GLASS
            'TYPE: INTERSECT | path: DETECTOR_DET_2➔T_IN(GLASS) | target: GLASS | action: internal_wall_bounce',
            'TYPE: SHADERS | path: DETECTOR_DET_2➔T_IN(GLASS) | target: GLASS | action: schedule_reflection_shader',
            'TYPE: TERMINATION | path: DETECTOR_DET_2➔T_IN(GLASS) | target: GLASS | action: transmit_budget_exhausted',

            # Step 7: In through GLASS
            'TYPE: INTERSECT | path: DETECTOR_DET_2 | target: GLASS | action: transmit_in_from_AIR',
            'TYPE: SHADERS | path: DETECTOR_DET_2 | target: GLASS | action: schedule_transmission_shader',
        ],
        [
            'TYPE: ORIGIN | path: DETECTOR_DET_1 | target: DET_1 | action: init_trace',
            'TYPE: TERMINATION | path: DETECTOR_DET_1 | target: INFINITY | action: ray_escaped_at_step_0',
            'TYPE: ORIGIN | path: DETECTOR_DET_2 | target: DET_2 | action: init_trace',
            'TYPE: TERMINATION | path: DETECTOR_DET_2 | target: INFINITY | action: ray_escaped_at_step_0',
        ],
    ]

    assert lane_history == expected_sequence

# =========================================================================
# TIMELINE SHAPE CONTRACT
# The lane count comes from `batch_size` (it sizes the tracker and every mask), NOT from the
# matrix. Rows are ray lanes; columns are [ends... | elements...] in constructor order. Without
# the guard in raytrace(), NumPy silently broadcasts a short matrix across lanes, so deleting a
# lane's row leaves `batch_size` lanes logging identical tokens and the test keeps passing.
# =========================================================================
def test_timeline_shape_contract_is_enforced_at_entry():
    """Every step matrix must be (batch_size rows) x (len(ends) + len(elements) columns)."""
    detector, light_source, glass = MockNode("DET_1"), MockNode("SOURCE_A"), MockNode("GLASS", transparent=True)

    def build(elements=()):
        return TopologicalRayTracer(starts=[detector], elements=list(elements), ends=[light_source],
                                    bounce_count=1, max_transmission_depth=1)

    # Too FEW rows: NumPy would stretch row 0 over both lanes (this is the silent case)
    with pytest.raises(ValueError, match=r"MOCK_distances_timeline\[0\]: 1 row\(s\) but batch_size=2"):
        build().raytrace(batch_size=2, MOCK_distances_timeline=[np.array([[5.0]])])

    # Too MANY rows: the broadcast mask would index past the 1-lane tracker
    with pytest.raises(ValueError, match=r"MOCK_distances_timeline\[0\]: 2 row\(s\) but batch_size=1"):
        build().raytrace(batch_size=1, MOCK_distances_timeline=[np.array([[5.0], [5.0]])])

    # Too FEW columns: 1 end + 1 element = 2 columns expected
    with pytest.raises(ValueError, match=r"1 column\(s\) but len\(ends\) \+ len\(elements\) = 2"):
        build(elements=(glass,)).raytrace(batch_size=1, MOCK_distances_timeline=[np.array([[5.0]])])

    # Too MANY columns: extra columns would masquerade as extra elements in the blockage min
    with pytest.raises(ValueError, match=r"3 column\(s\) but len\(ends\) \+ len\(elements\) = 2"):
        build(elements=(glass,)).raytrace(batch_size=1, MOCK_distances_timeline=[np.array([[5.0, 1.0, 9.0]])])

    # A 1-D matrix is not a (lanes x nodes) table at all
    with pytest.raises(ValueError, match=r"expected a 2-D matrix"):
        build().raytrace(batch_size=1, MOCK_distances_timeline=[np.array([5.0])])

    # The offending step is reported by index, and a valid step 0 does not mask a bad step 1
    with pytest.raises(ValueError, match=r"MOCK_distances_timeline\[1\]"):
        build(elements=(glass,)).raytrace(batch_size=1, MOCK_distances_timeline=[
            np.array([[5.0, 1.0]]),
            np.array([[5.0, 1.0], [5.0, 1.0]]),
        ])

    # The contract is satisfiable: 2 lanes x 2 columns validates and traces
    tracer = build(elements=(glass,))
    rays = tracer.raytrace(batch_size=2, MOCK_distances_timeline=[np.array([[5.0, 1.0], [5.0, 1.0]])])
    assert len(tracer.tracker.history) == 2
    assert len(rays.x) == 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))