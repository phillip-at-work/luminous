import pytest
import numpy as np
import sys

from src.state_machine.state_machine import MockNode, TopologicalRayTracer

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
# TEST 1: Source in Direct View of Detector
# =========================================================================
def test_source_in_direct_view():
    """Tests a pristine environment where nothing blocks or intersects the rays."""
    detector = MockNode("DET_1")
    light_source = MockNode("SOURCE_A")
    tracer = TopologicalRayTracer(
        starts=[detector], elements=[], ends=[light_source], 
        bounce_count=0, max_transmission_depth=0
    )
    
    batch_size = 2
    output_rays = tracer.raytrace(batch_size=batch_size, MOCK_distances_timeline=[])
    
    expected_sequence = [
        "TYPE: ORIGIN | path: DETECTOR_DET_1 | target: DET_1 | action: init_trace",
        "TYPE: CONTEXT | path: DETECTOR_DET_1 | target: SOURCE_A | action: query_emission_from_AIR",
        "TYPE: SHADERS | path: DETECTOR_DET_1 | target: SCENE | action: schedule_emission_shader_from_AIR"
    ]
    
    for ray_idx in range(batch_size):
        lane_history = tracer.tracker.history[ray_idx]
        assert lane_history == expected_sequence


# =========================================================================
# TEST 2: Detector ➔ Reflect ➔ Reflect ➔ Reflect ➔ Source
# =========================================================================
def test_detector_to_three_consecutive_reflections():
    """Tests a deep reflection chain hitting three distinct mirrors sequentially."""
    detector = MockNode("DET_1")
    mirror1 = MockNode("MIRROR_1")
    mirror2 = MockNode("MIRROR_2")
    mirror3 = MockNode("MIRROR_3")
    light_source = MockNode("SOURCE_A")
    tracer = TopologicalRayTracer(
        starts=[detector], elements=[mirror1, mirror2, mirror3], ends=[light_source], 
        bounce_count=3, max_transmission_depth=0
    )
    
    m1_timeline = np.array([[1.0, float('inf'), float('inf')]])
    m2_timeline = np.array([[float('inf'), 2.0, float('inf')]])
    m3_timeline = np.array([[float('inf'), float('inf'), 3.0]])
    
    output_rays = tracer.raytrace(batch_size=1, MOCK_distances_timeline=[m1_timeline, m2_timeline, m3_timeline])
    lane_history = tracer.tracker.history[0]
    
    expected_sequence = [
        "TYPE: ORIGIN | path: DETECTOR_DET_1 | target: DET_1 | action: init_trace",
        "TYPE: CONTEXT | path: DETECTOR_DET_1 | target: SOURCE_A | action: query_emission_from_AIR",
        "TYPE: SHADERS | path: DETECTOR_DET_1 | target: SCENE | action: schedule_emission_shader_from_AIR",
        "TYPE: INTERSECT | path: DETECTOR_DET_1 | target: MIRROR_1 | action: external_surface_bounce",
        "TYPE: CONTEXT | path: DETECTOR_DET_1➔R_EXT(MIRROR_1) | target: SOURCE_A | action: query_emission_from_AIR",
        "TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(MIRROR_1) | target: SCENE | action: schedule_emission_shader_from_AIR",
        "TYPE: INTERSECT | path: DETECTOR_DET_1➔R_EXT(MIRROR_1) | target: MIRROR_2 | action: external_surface_bounce",
        "TYPE: CONTEXT | path: DETECTOR_DET_1➔R_EXT(MIRROR_1)➔R_EXT(MIRROR_2) | target: SOURCE_A | action: query_emission_from_AIR",
        "TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(MIRROR_1)➔R_EXT(MIRROR_2) | target: SCENE | action: schedule_emission_shader_from_AIR",
        "TYPE: INTERSECT | path: DETECTOR_DET_1➔R_EXT(MIRROR_1)➔R_EXT(MIRROR_2) | target: MIRROR_3 | action: external_surface_bounce",
        "TYPE: CONTEXT | path: DETECTOR_DET_1➔R_EXT(MIRROR_1)➔R_EXT(MIRROR_2)➔R_EXT(MIRROR_3) | target: SOURCE_A | action: query_emission_from_AIR",
        "TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(MIRROR_1)➔R_EXT(MIRROR_2)➔R_EXT(MIRROR_3) | target: SCENE | action: schedule_emission_shader_from_AIR",
        "TYPE: TERMINATION | path: DETECTOR_DET_1➔R_EXT(MIRROR_1)➔R_EXT(MIRROR_2)➔R_EXT(MIRROR_3) | target: INFINITY | action: ray_escaped_at_step_3",
        "TYPE: CONTEXT | path: DETECTOR_DET_1➔R_EXT(MIRROR_1)➔R_EXT(MIRROR_2) | target: MIRROR_3 | action: query_direct_illum_from_SOURCE_A",
        "TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(MIRROR_1)➔R_EXT(MIRROR_2) | target: MIRROR_3 | action: schedule_reflection_shader",
        "TYPE: CONTEXT | path: DETECTOR_DET_1➔R_EXT(MIRROR_1) | target: MIRROR_2 | action: query_direct_illum_from_SOURCE_A",
        "TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(MIRROR_1) | target: MIRROR_2 | action: schedule_reflection_shader",
        "TYPE: CONTEXT | path: DETECTOR_DET_1 | target: MIRROR_1 | action: query_direct_illum_from_SOURCE_A",
        "TYPE: SHADERS | path: DETECTOR_DET_1 | target: MIRROR_1 | action: schedule_reflection_shader"
    ]
    
    assert lane_history == expected_sequence

# =========================================================================
# TEST 3: Detector ➔ Transmit In ➔ Reflect/Transmit Out ➔ Reflect ➔ Source
# =========================================================================
def test_complex_split_transmission_and_reflection_path():
    """Tests a complex splitting dielectric path verifying simultaneous sub-paths."""
    detector = MockNode("DET_1")
    glass_prism = MockNode("GLASS_PRISM", transparent=True)
    opaque_mirror = MockNode("OPAQUE_MIRROR")
    light_source = MockNode("SOURCE_A")
    tracer = TopologicalRayTracer(
        starts=[detector], elements=[glass_prism, opaque_mirror], ends=[light_source], 
        bounce_count=1, max_transmission_depth=2
    )
    
    prism_timeline = np.array([[1.0, 2.0, float('inf')]])
    mirror_timeline = np.array([[float('inf'), float('inf'), 3.0]])
    
    output_rays = tracer.raytrace(batch_size=1, MOCK_distances_timeline=[prism_timeline, mirror_timeline])
    lane_history = tracer.tracker.history[0]
    
    expected_sequence = [
        "TYPE: ORIGIN | path: DETECTOR_DET_1 | target: DET_1 | action: init_trace",
        "TYPE: CONTEXT | path: DETECTOR_DET_1 | target: SOURCE_A | action: query_emission_from_AIR",
        "TYPE: SHADERS | path: DETECTOR_DET_1 | target: SCENE | action: schedule_emission_shader_from_AIR",
        
        # --- Transmission Side Branch (Step 0) ---
        "TYPE: INTERSECT | path: DETECTOR_DET_1 | target: GLASS_PRISM | action: transmit_in_from_AIR",
        "TYPE: CONTEXT | path: DETECTOR_DET_1➔T_IN(GLASS_PRISM) | target: SOURCE_A | action: query_emission_from_GLASS_PRISM",
        "TYPE: SHADERS | path: DETECTOR_DET_1➔T_IN(GLASS_PRISM) | target: SCENE | action: schedule_emission_shader_from_GLASS_PRISM",
        
        # Step 1: Transmit Out
        "TYPE: INTERSECT | path: DETECTOR_DET_1➔T_IN(GLASS_PRISM) | target: GLASS_PRISM | action: transmit_out_to_AIR",
        "TYPE: CONTEXT | path: DETECTOR_DET_1➔T_IN(GLASS_PRISM)➔T_OUT(GLASS_PRISM) | target: SOURCE_A | action: query_emission_from_AIR",
        "TYPE: SHADERS | path: DETECTOR_DET_1➔T_IN(GLASS_PRISM)➔T_OUT(GLASS_PRISM) | target: SCENE | action: schedule_emission_shader_from_AIR",
        
        # Step 2: Hits Opaque Mirror
        "TYPE: INTERSECT | path: DETECTOR_DET_1➔T_IN(GLASS_PRISM)➔T_OUT(GLASS_PRISM) | target: OPAQUE_MIRROR | action: external_surface_bounce",
        "TYPE: CONTEXT | path: DETECTOR_DET_1➔T_IN(GLASS_PRISM)➔T_OUT(GLASS_PRISM)➔R_EXT(OPAQUE_MIRROR) | target: SOURCE_A | action: query_emission_from_AIR",
        "TYPE: SHADERS | path: DETECTOR_DET_1➔T_IN(GLASS_PRISM)➔T_OUT(GLASS_PRISM)➔R_EXT(OPAQUE_MIRROR) | target: SCENE | action: schedule_emission_shader_from_AIR",
        "TYPE: TERMINATION | path: DETECTOR_DET_1➔T_IN(GLASS_PRISM)➔T_OUT(GLASS_PRISM)➔R_EXT(OPAQUE_MIRROR) | target: INFINITY | action: ray_escaped_at_step_3",
        "TYPE: CONTEXT | path: DETECTOR_DET_1➔T_IN(GLASS_PRISM)➔T_OUT(GLASS_PRISM) | target: OPAQUE_MIRROR | action: query_direct_illum_from_SOURCE_A",
        "TYPE: SHADERS | path: DETECTOR_DET_1➔T_IN(GLASS_PRISM)➔T_OUT(GLASS_PRISM) | target: OPAQUE_MIRROR | action: schedule_reflection_shader",
        "TYPE: SHADERS | path: DETECTOR_DET_1➔T_IN(GLASS_PRISM) | target: GLASS_PRISM | action: schedule_transmission_shader",
        
        # Step 1: Internal Reflection Branch
        "TYPE: INTERSECT | path: DETECTOR_DET_1➔T_IN(GLASS_PRISM) | target: GLASS_PRISM | action: internal_wall_bounce",
        "TYPE: CONTEXT | path: DETECTOR_DET_1➔T_IN(GLASS_PRISM)➔R_INT(GLASS_PRISM) | target: SOURCE_A | action: query_emission_from_GLASS_PRISM",
        "TYPE: SHADERS | path: DETECTOR_DET_1➔T_IN(GLASS_PRISM)➔R_INT(GLASS_PRISM) | target: SCENE | action: schedule_emission_shader_from_GLASS_PRISM",
        "TYPE: TERMINATION | path: DETECTOR_DET_1➔T_IN(GLASS_PRISM)➔R_INT(GLASS_PRISM) | target: OPAQUE_MIRROR | action: bounce_budget_exhausted",
        "TYPE: CONTEXT | path: DETECTOR_DET_1➔T_IN(GLASS_PRISM) | target: GLASS_PRISM | action: query_direct_illum_from_SOURCE_A",
        "TYPE: SHADERS | path: DETECTOR_DET_1➔T_IN(GLASS_PRISM) | target: GLASS_PRISM | action: schedule_reflection_shader",
        "TYPE: SHADERS | path: DETECTOR_DET_1 | target: GLASS_PRISM | action: schedule_transmission_shader",
        
        # --- Reflection Side Branch (Step 0) ---
        "TYPE: INTERSECT | path: DETECTOR_DET_1 | target: GLASS_PRISM | action: external_surface_bounce",
        "TYPE: CONTEXT | path: DETECTOR_DET_1➔R_EXT(GLASS_PRISM) | target: SOURCE_A | action: query_emission_from_AIR",
        "TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(GLASS_PRISM) | target: SCENE | action: schedule_emission_shader_from_AIR",
        "TYPE: INTERSECT | path: DETECTOR_DET_1➔R_EXT(GLASS_PRISM) | target: GLASS_PRISM | action: transmit_in_from_AIR",
        "TYPE: CONTEXT | path: DETECTOR_DET_1➔R_EXT(GLASS_PRISM)➔T_IN(GLASS_PRISM) | target: SOURCE_A | action: query_emission_from_GLASS_PRISM",
        "TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(GLASS_PRISM)➔T_IN(GLASS_PRISM) | target: SCENE | action: schedule_emission_shader_from_GLASS_PRISM",
        "TYPE: TERMINATION | path: DETECTOR_DET_1➔R_EXT(GLASS_PRISM)➔T_IN(GLASS_PRISM) | target: OPAQUE_MIRROR | action: bounce_budget_exhausted",
        "TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(GLASS_PRISM) | target: GLASS_PRISM | action: schedule_transmission_shader",
        "TYPE: TERMINATION | path: DETECTOR_DET_1➔R_EXT(GLASS_PRISM) | target: GLASS_PRISM | action: bounce_budget_exhausted",
        "TYPE: CONTEXT | path: DETECTOR_DET_1 | target: GLASS_PRISM | action: query_direct_illum_from_SOURCE_A",
        "TYPE: SHADERS | path: DETECTOR_DET_1 | target: GLASS_PRISM | action: schedule_reflection_shader"
    ]
    assert lane_history == expected_sequence


# =========================================================================
# TEST 4: Reflection off Passive Element Illuminated by Source
# =========================================================================
def test_reflection_off_passive_element_illuminated_by_source():
    """Tests a strict physical arrangement with zero entity mixing and verified order."""
    detector = MockNode("DET_1")
    passive_mirror = MockNode("PASSIVE_MIRROR")  
    light_source = MockNode("LIGHT_SOURCE_A")                       
    tracer = TopologicalRayTracer(
        starts=[detector], elements=[passive_mirror], ends=[light_source], 
        bounce_count=1, max_transmission_depth=0
    )
    
    mirror_timeline = np.array([[3.0]])
    output_rays = tracer.raytrace(batch_size=1, MOCK_distances_timeline=[mirror_timeline])
    lane_history = tracer.tracker.history[0]
    
    expected_sequence = [
        "TYPE: ORIGIN | path: DETECTOR_DET_1 | target: DET_1 | action: init_trace",
        "TYPE: CONTEXT | path: DETECTOR_DET_1 | target: LIGHT_SOURCE_A | action: query_emission_from_AIR",
        "TYPE: SHADERS | path: DETECTOR_DET_1 | target: SCENE | action: schedule_emission_shader_from_AIR",
        "TYPE: INTERSECT | path: DETECTOR_DET_1 | target: PASSIVE_MIRROR | action: external_surface_bounce",
        "TYPE: CONTEXT | path: DETECTOR_DET_1➔R_EXT(PASSIVE_MIRROR) | target: LIGHT_SOURCE_A | action: query_emission_from_AIR",
        "TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(PASSIVE_MIRROR) | target: SCENE | action: schedule_emission_shader_from_AIR",
        "TYPE: TERMINATION | path: DETECTOR_DET_1➔R_EXT(PASSIVE_MIRROR) | target: INFINITY | action: ray_escaped_at_step_1",
        "TYPE: CONTEXT | path: DETECTOR_DET_1 | target: PASSIVE_MIRROR | action: query_direct_illum_from_LIGHT_SOURCE_A",
        "TYPE: SHADERS | path: DETECTOR_DET_1 | target: PASSIVE_MIRROR | action: schedule_reflection_shader"
    ]
    assert lane_history == expected_sequence

# =========================================================================
# TEST 5: Strict 1:1 Branch Allocation (No Double Reflections)
# =========================================================================
def test_simultaneous_splitting_at_dielectric_boundary():
    """
    Tests a ray-splitting scenario at a semi-transparent interface.
    Asserts the exact timeline structure to guarantee that the ray splits 
    EXACTLY once into transmission and EXACTLY once into reflection.
    """
    detector = MockNode("DET_1")
    coated_glass = MockNode("COATED_GLASS", transparent=True)
    light_source = MockNode("SOURCE_A")
    
    tracer = TopologicalRayTracer(
        starts=[detector], elements=[coated_glass], ends=[light_source],
        bounce_count=1, max_transmission_depth=1
    )
    
    glass_timeline = np.array([[1.0, float('inf')]])
    timeline_data = [glass_timeline]
    
    output_rays = tracer.raytrace(batch_size=1, MOCK_distances_timeline=timeline_data)
    lane_history = tracer.tracker.history[0]

    expected_sequence = [
        "TYPE: ORIGIN | path: DETECTOR_DET_1 | target: DET_1 | action: init_trace",
        "TYPE: CONTEXT | path: DETECTOR_DET_1 | target: SOURCE_A | action: query_emission_from_AIR",
        "TYPE: SHADERS | path: DETECTOR_DET_1 | target: SCENE | action: schedule_emission_shader_from_AIR",
        
        # --- The Single Isolated Transmission Entry ---
        "TYPE: INTERSECT | path: DETECTOR_DET_1 | target: COATED_GLASS | action: transmit_in_from_AIR",
        "TYPE: CONTEXT | path: DETECTOR_DET_1➔T_IN(COATED_GLASS) | target: SOURCE_A | action: query_emission_from_COATED_GLASS",
        "TYPE: SHADERS | path: DETECTOR_DET_1➔T_IN(COATED_GLASS) | target: SCENE | action: schedule_emission_shader_from_COATED_GLASS",
        "TYPE: TERMINATION | path: DETECTOR_DET_1➔T_IN(COATED_GLASS) | target: INFINITY | action: ray_escaped_at_step_1",
        "TYPE: SHADERS | path: DETECTOR_DET_1 | target: COATED_GLASS | action: schedule_transmission_shader",
        
        # --- The Single Isolated Reflection Entry ---
        "TYPE: INTERSECT | path: DETECTOR_DET_1 | target: COATED_GLASS | action: external_surface_bounce",
        "TYPE: CONTEXT | path: DETECTOR_DET_1➔R_EXT(COATED_GLASS) | target: SOURCE_A | action: query_emission_from_AIR",
        "TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(COATED_GLASS) | target: SCENE | action: schedule_emission_shader_from_AIR",
        "TYPE: TERMINATION | path: DETECTOR_DET_1➔R_EXT(COATED_GLASS) | target: INFINITY | action: ray_escaped_at_step_1",
        "TYPE: CONTEXT | path: DETECTOR_DET_1 | target: COATED_GLASS | action: query_direct_illum_from_SOURCE_A",
        "TYPE: SHADERS | path: DETECTOR_DET_1 | target: COATED_GLASS | action: schedule_reflection_shader"
    ]
    
    assert lane_history == expected_sequence

# =========================================================================
# TEST 6: Stacked Materials (Contiguous Pure Transmissions)
# =========================================================================
def test_contiguous_transmissions_through_lens_stack():
    """Tests a ray penetrating a series of concentric optical filters."""
    detector = MockNode("DET_1")
    lens1 = MockNode("LENS_1", transparent=True)
    lens2 = MockNode("LENS_2", transparent=True)
    light_source = MockNode("SOURCE_A")
    tracer = TopologicalRayTracer(
        starts=[detector], elements=[lens1, lens2], ends=[light_source], 
        bounce_count=0, max_transmission_depth=4
    )
    
    l1_timeline = np.array([[1.0, float('inf'), float('inf'), 4.0]])
    l2_timeline = np.array([[float('inf'), 2.0, 3.0, float('inf')]])
    timeline_data = [l1_timeline, l2_timeline]
    
    output_rays = tracer.raytrace(batch_size=1, MOCK_distances_timeline=timeline_data)
    lane_history = tracer.tracker.history[0]

    # print_debug_output("debug_output.py", lane_history)
    
    expected_sequence = [
        'TYPE: ORIGIN | path: DETECTOR_DET_1 | target: DET_1 | action: init_trace',
        'TYPE: CONTEXT | path: DETECTOR_DET_1 | target: SOURCE_A | action: query_emission_from_AIR',
        'TYPE: SHADERS | path: DETECTOR_DET_1 | target: SCENE | action: schedule_emission_shader_from_AIR',

        # Step 0: In through LENS_1
        'TYPE: INTERSECT | path: DETECTOR_DET_1 | target: LENS_1 | action: transmit_in_from_AIR',
        'TYPE: CONTEXT | path: DETECTOR_DET_1➔T_IN(LENS_1) | target: SOURCE_A | action: query_emission_from_LENS_1',
        'TYPE: SHADERS | path: DETECTOR_DET_1➔T_IN(LENS_1) | target: SCENE | action: schedule_emission_shader_from_LENS_1',

        # Step 1: In through LENS_2
        'TYPE: INTERSECT | path: DETECTOR_DET_1➔T_IN(LENS_1) | target: LENS_2 | action: transmit_in_from_LENS_1',
        'TYPE: CONTEXT | path: DETECTOR_DET_1➔T_IN(LENS_1)➔T_IN(LENS_2) | target: SOURCE_A | action: query_emission_from_LENS_2',
        'TYPE: SHADERS | path: DETECTOR_DET_1➔T_IN(LENS_1)➔T_IN(LENS_2) | target: SCENE | action: schedule_emission_shader_from_LENS_2',

        # Step 2: Out through LENS_2
        'TYPE: INTERSECT | path: DETECTOR_DET_1➔T_IN(LENS_1)➔T_IN(LENS_2) | target: LENS_2 | action: transmit_out_to_LENS_1',
        'TYPE: CONTEXT | path: DETECTOR_DET_1➔T_IN(LENS_1)➔T_IN(LENS_2)➔T_OUT(LENS_2) | target: SOURCE_A | action: query_emission_from_LENS_1',
        'TYPE: SHADERS | path: DETECTOR_DET_1➔T_IN(LENS_1)➔T_IN(LENS_2)➔T_OUT(LENS_2) | target: SCENE | action: schedule_emission_shader_from_LENS_1',

        # Step 3: Out through LENS_1
        'TYPE: INTERSECT | path: DETECTOR_DET_1➔T_IN(LENS_1)➔T_IN(LENS_2)➔T_OUT(LENS_2) | target: LENS_1 | action: transmit_out_to_AIR',
        'TYPE: CONTEXT | path: DETECTOR_DET_1➔T_IN(LENS_1)➔T_IN(LENS_2)➔T_OUT(LENS_2)➔T_OUT(LENS_1) | target: SOURCE_A | action: query_emission_from_AIR',
        'TYPE: SHADERS | path: DETECTOR_DET_1➔T_IN(LENS_1)➔T_IN(LENS_2)➔T_OUT(LENS_2)➔T_OUT(LENS_1) | target: SCENE | action: schedule_emission_shader_from_AIR',
        'TYPE: TERMINATION | path: DETECTOR_DET_1➔T_IN(LENS_1)➔T_IN(LENS_2)➔T_OUT(LENS_2)➔T_OUT(LENS_1) | target: INFINITY | action: ray_escaped_at_step_4',
        'TYPE: SHADERS | path: DETECTOR_DET_1➔T_IN(LENS_1)➔T_IN(LENS_2)➔T_OUT(LENS_2) | target: LENS_1 | action: schedule_transmission_shader',
        'TYPE: TERMINATION | path: DETECTOR_DET_1➔T_IN(LENS_1)➔T_IN(LENS_2)➔T_OUT(LENS_2) | target: LENS_1 | action: bounce_budget_exhausted',
        'TYPE: SHADERS | path: DETECTOR_DET_1➔T_IN(LENS_1)➔T_IN(LENS_2) | target: LENS_2 | action: schedule_transmission_shader',
        'TYPE: TERMINATION | path: DETECTOR_DET_1➔T_IN(LENS_1)➔T_IN(LENS_2) | target: LENS_2 | action: bounce_budget_exhausted',
        'TYPE: SHADERS | path: DETECTOR_DET_1➔T_IN(LENS_1) | target: LENS_2 | action: schedule_transmission_shader',
        'TYPE: TERMINATION | path: DETECTOR_DET_1➔T_IN(LENS_1) | target: LENS_2 | action: bounce_budget_exhausted',
        'TYPE: SHADERS | path: DETECTOR_DET_1 | target: LENS_1 | action: schedule_transmission_shader',
        'TYPE: TERMINATION | path: DETECTOR_DET_1 | target: LENS_1 | action: bounce_budget_exhausted',
    ]


    assert lane_history == expected_sequence


import numpy as np
import pytest

# =========================================================================
# TEST 7: Peculiar Sequence (Complete 1:1 Exhaustive Match Profile)
# =========================================================================
def test_peculiar_interleaved_path_behavior():
    """Tests an interleaved external/internal boundary trajectory loop with absolute structural certainty."""
    detector = MockNode("DET_1")
    element_x = MockNode("ELEMENT_X", transparent=True)
    light_source = MockNode("SOURCE_A")
    tracer = TopologicalRayTracer(
        starts=[detector], elements=[element_x], ends=[light_source], 
        bounce_count=2, max_transmission_depth=2
    )
    
    x_timeline = np.array([[1.0, 2.0, 3.0, 4.0]])
    timeline_data = [x_timeline]
    
    output_rays = tracer.raytrace(batch_size=1, MOCK_distances_timeline=timeline_data)
    lane_history = tracer.tracker.history

    # print_debug_output("debug_output.py", lane_history)
    
    expected_sequence = [
        [
            'TYPE: ORIGIN | path: DETECTOR_DET_1 | target: DET_1 | action: init_trace',
            'TYPE: CONTEXT | path: DETECTOR_DET_1 | target: SOURCE_A | action: query_emission_from_AIR',
            'TYPE: SHADERS | path: DETECTOR_DET_1 | target: SCENE | action: schedule_emission_shader_from_AIR',

            # Step 0: In through ELEMENT_X
            'TYPE: INTERSECT | path: DETECTOR_DET_1 | target: ELEMENT_X | action: transmit_in_from_AIR',
            'TYPE: CONTEXT | path: DETECTOR_DET_1➔T_IN(ELEMENT_X) | target: SOURCE_A | action: query_emission_from_ELEMENT_X',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔T_IN(ELEMENT_X) | target: SCENE | action: schedule_emission_shader_from_ELEMENT_X',

            # Step 1: Out through ELEMENT_X
            'TYPE: INTERSECT | path: DETECTOR_DET_1➔T_IN(ELEMENT_X) | target: ELEMENT_X | action: transmit_out_to_AIR',
            'TYPE: CONTEXT | path: DETECTOR_DET_1➔T_IN(ELEMENT_X)➔T_OUT(ELEMENT_X) | target: SOURCE_A | action: query_emission_from_AIR',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔T_IN(ELEMENT_X)➔T_OUT(ELEMENT_X) | target: SCENE | action: schedule_emission_shader_from_AIR',
            'TYPE: TERMINATION | path: DETECTOR_DET_1➔T_IN(ELEMENT_X)➔T_OUT(ELEMENT_X) | target: ELEMENT_X | action: transmit_budget_exhausted',

            # Step 2: Interaction with ELEMENT_X
            'TYPE: INTERSECT | path: DETECTOR_DET_1➔T_IN(ELEMENT_X)➔T_OUT(ELEMENT_X) | target: ELEMENT_X | action: external_surface_bounce',
            'TYPE: CONTEXT | path: DETECTOR_DET_1➔T_IN(ELEMENT_X)➔T_OUT(ELEMENT_X)➔R_EXT(ELEMENT_X) | target: SOURCE_A | action: query_emission_from_AIR',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔T_IN(ELEMENT_X)➔T_OUT(ELEMENT_X)➔R_EXT(ELEMENT_X) | target: SCENE | action: schedule_emission_shader_from_AIR',
            'TYPE: TERMINATION | path: DETECTOR_DET_1➔T_IN(ELEMENT_X)➔T_OUT(ELEMENT_X)➔R_EXT(ELEMENT_X) | target: ELEMENT_X | action: transmit_budget_exhausted',

            # Step 3: Interaction with ELEMENT_X
            'TYPE: INTERSECT | path: DETECTOR_DET_1➔T_IN(ELEMENT_X)➔T_OUT(ELEMENT_X)➔R_EXT(ELEMENT_X) | target: ELEMENT_X | action: external_surface_bounce',
            'TYPE: CONTEXT | path: DETECTOR_DET_1➔T_IN(ELEMENT_X)➔T_OUT(ELEMENT_X)➔R_EXT(ELEMENT_X)➔R_EXT(ELEMENT_X) | target: SOURCE_A | action: query_emission_from_AIR',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔T_IN(ELEMENT_X)➔T_OUT(ELEMENT_X)➔R_EXT(ELEMENT_X)➔R_EXT(ELEMENT_X) | target: SCENE | action: schedule_emission_shader_from_AIR',
            'TYPE: TERMINATION | path: DETECTOR_DET_1➔T_IN(ELEMENT_X)➔T_OUT(ELEMENT_X)➔R_EXT(ELEMENT_X)➔R_EXT(ELEMENT_X) | target: INFINITY | action: ray_escaped_at_step_4',
            'TYPE: CONTEXT | path: DETECTOR_DET_1➔T_IN(ELEMENT_X)➔T_OUT(ELEMENT_X)➔R_EXT(ELEMENT_X) | target: ELEMENT_X | action: query_direct_illum_from_SOURCE_A',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔T_IN(ELEMENT_X)➔T_OUT(ELEMENT_X)➔R_EXT(ELEMENT_X) | target: ELEMENT_X | action: schedule_reflection_shader',
            'TYPE: CONTEXT | path: DETECTOR_DET_1➔T_IN(ELEMENT_X)➔T_OUT(ELEMENT_X) | target: ELEMENT_X | action: query_direct_illum_from_SOURCE_A',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔T_IN(ELEMENT_X)➔T_OUT(ELEMENT_X) | target: ELEMENT_X | action: schedule_reflection_shader',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔T_IN(ELEMENT_X) | target: ELEMENT_X | action: schedule_transmission_shader',

            # Step 4: Interaction with ELEMENT_X
            'TYPE: INTERSECT | path: DETECTOR_DET_1➔T_IN(ELEMENT_X) | target: ELEMENT_X | action: internal_wall_bounce',
            'TYPE: CONTEXT | path: DETECTOR_DET_1➔T_IN(ELEMENT_X)➔R_INT(ELEMENT_X) | target: SOURCE_A | action: query_emission_from_ELEMENT_X',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔T_IN(ELEMENT_X)➔R_INT(ELEMENT_X) | target: SCENE | action: schedule_emission_shader_from_ELEMENT_X',

            # Step 5: Out through ELEMENT_X
            'TYPE: INTERSECT | path: DETECTOR_DET_1➔T_IN(ELEMENT_X)➔R_INT(ELEMENT_X) | target: ELEMENT_X | action: transmit_out_to_AIR',
            'TYPE: CONTEXT | path: DETECTOR_DET_1➔T_IN(ELEMENT_X)➔R_INT(ELEMENT_X)➔T_OUT(ELEMENT_X) | target: SOURCE_A | action: query_emission_from_AIR',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔T_IN(ELEMENT_X)➔R_INT(ELEMENT_X)➔T_OUT(ELEMENT_X) | target: SCENE | action: schedule_emission_shader_from_AIR',
            'TYPE: TERMINATION | path: DETECTOR_DET_1➔T_IN(ELEMENT_X)➔R_INT(ELEMENT_X)➔T_OUT(ELEMENT_X) | target: ELEMENT_X | action: transmit_budget_exhausted',

            # Step 6: Interaction with ELEMENT_X
            'TYPE: INTERSECT | path: DETECTOR_DET_1➔T_IN(ELEMENT_X)➔R_INT(ELEMENT_X)➔T_OUT(ELEMENT_X) | target: ELEMENT_X | action: external_surface_bounce',
            'TYPE: CONTEXT | path: DETECTOR_DET_1➔T_IN(ELEMENT_X)➔R_INT(ELEMENT_X)➔T_OUT(ELEMENT_X)➔R_EXT(ELEMENT_X) | target: SOURCE_A | action: query_emission_from_AIR',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔T_IN(ELEMENT_X)➔R_INT(ELEMENT_X)➔T_OUT(ELEMENT_X)➔R_EXT(ELEMENT_X) | target: SCENE | action: schedule_emission_shader_from_AIR',
            'TYPE: TERMINATION | path: DETECTOR_DET_1➔T_IN(ELEMENT_X)➔R_INT(ELEMENT_X)➔T_OUT(ELEMENT_X)➔R_EXT(ELEMENT_X) | target: INFINITY | action: ray_escaped_at_step_4',
            'TYPE: CONTEXT | path: DETECTOR_DET_1➔T_IN(ELEMENT_X)➔R_INT(ELEMENT_X)➔T_OUT(ELEMENT_X) | target: ELEMENT_X | action: query_direct_illum_from_SOURCE_A',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔T_IN(ELEMENT_X)➔R_INT(ELEMENT_X)➔T_OUT(ELEMENT_X) | target: ELEMENT_X | action: schedule_reflection_shader',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔T_IN(ELEMENT_X)➔R_INT(ELEMENT_X) | target: ELEMENT_X | action: schedule_transmission_shader',

            # Step 7: Interaction with ELEMENT_X
            'TYPE: INTERSECT | path: DETECTOR_DET_1➔T_IN(ELEMENT_X)➔R_INT(ELEMENT_X) | target: ELEMENT_X | action: internal_wall_bounce',
            'TYPE: CONTEXT | path: DETECTOR_DET_1➔T_IN(ELEMENT_X)➔R_INT(ELEMENT_X)➔R_INT(ELEMENT_X) | target: SOURCE_A | action: query_emission_from_ELEMENT_X',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔T_IN(ELEMENT_X)➔R_INT(ELEMENT_X)➔R_INT(ELEMENT_X) | target: SCENE | action: schedule_emission_shader_from_ELEMENT_X',

            # Step 8: Out through ELEMENT_X
            'TYPE: INTERSECT | path: DETECTOR_DET_1➔T_IN(ELEMENT_X)➔R_INT(ELEMENT_X)➔R_INT(ELEMENT_X) | target: ELEMENT_X | action: transmit_out_to_AIR',
            'TYPE: CONTEXT | path: DETECTOR_DET_1➔T_IN(ELEMENT_X)➔R_INT(ELEMENT_X)➔R_INT(ELEMENT_X)➔T_OUT(ELEMENT_X) | target: SOURCE_A | action: query_emission_from_AIR',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔T_IN(ELEMENT_X)➔R_INT(ELEMENT_X)➔R_INT(ELEMENT_X)➔T_OUT(ELEMENT_X) | target: SCENE | action: schedule_emission_shader_from_AIR',
            'TYPE: TERMINATION | path: DETECTOR_DET_1➔T_IN(ELEMENT_X)➔R_INT(ELEMENT_X)➔R_INT(ELEMENT_X)➔T_OUT(ELEMENT_X) | target: INFINITY | action: ray_escaped_at_step_4',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔T_IN(ELEMENT_X)➔R_INT(ELEMENT_X)➔R_INT(ELEMENT_X) | target: ELEMENT_X | action: schedule_transmission_shader',
            'TYPE: TERMINATION | path: DETECTOR_DET_1➔T_IN(ELEMENT_X)➔R_INT(ELEMENT_X)➔R_INT(ELEMENT_X) | target: ELEMENT_X | action: bounce_budget_exhausted',
            'TYPE: CONTEXT | path: DETECTOR_DET_1➔T_IN(ELEMENT_X)➔R_INT(ELEMENT_X) | target: ELEMENT_X | action: query_direct_illum_from_SOURCE_A',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔T_IN(ELEMENT_X)➔R_INT(ELEMENT_X) | target: ELEMENT_X | action: schedule_reflection_shader',
            'TYPE: CONTEXT | path: DETECTOR_DET_1➔T_IN(ELEMENT_X) | target: ELEMENT_X | action: query_direct_illum_from_SOURCE_A',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔T_IN(ELEMENT_X) | target: ELEMENT_X | action: schedule_reflection_shader',
            'TYPE: SHADERS | path: DETECTOR_DET_1 | target: ELEMENT_X | action: schedule_transmission_shader',

            # Step 9: Interaction with ELEMENT_X
            'TYPE: INTERSECT | path: DETECTOR_DET_1 | target: ELEMENT_X | action: external_surface_bounce',
            'TYPE: CONTEXT | path: DETECTOR_DET_1➔R_EXT(ELEMENT_X) | target: SOURCE_A | action: query_emission_from_AIR',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(ELEMENT_X) | target: SCENE | action: schedule_emission_shader_from_AIR',

            # Step 10: In through ELEMENT_X
            'TYPE: INTERSECT | path: DETECTOR_DET_1➔R_EXT(ELEMENT_X) | target: ELEMENT_X | action: transmit_in_from_AIR',
            'TYPE: CONTEXT | path: DETECTOR_DET_1➔R_EXT(ELEMENT_X)➔T_IN(ELEMENT_X) | target: SOURCE_A | action: query_emission_from_ELEMENT_X',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(ELEMENT_X)➔T_IN(ELEMENT_X) | target: SCENE | action: schedule_emission_shader_from_ELEMENT_X',

            # Step 11: Out through ELEMENT_X
            'TYPE: INTERSECT | path: DETECTOR_DET_1➔R_EXT(ELEMENT_X)➔T_IN(ELEMENT_X) | target: ELEMENT_X | action: transmit_out_to_AIR',
            'TYPE: CONTEXT | path: DETECTOR_DET_1➔R_EXT(ELEMENT_X)➔T_IN(ELEMENT_X)➔T_OUT(ELEMENT_X) | target: SOURCE_A | action: query_emission_from_AIR',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(ELEMENT_X)➔T_IN(ELEMENT_X)➔T_OUT(ELEMENT_X) | target: SCENE | action: schedule_emission_shader_from_AIR',
            'TYPE: TERMINATION | path: DETECTOR_DET_1➔R_EXT(ELEMENT_X)➔T_IN(ELEMENT_X)➔T_OUT(ELEMENT_X) | target: ELEMENT_X | action: transmit_budget_exhausted',

            # Step 12: Interaction with ELEMENT_X
            'TYPE: INTERSECT | path: DETECTOR_DET_1➔R_EXT(ELEMENT_X)➔T_IN(ELEMENT_X)➔T_OUT(ELEMENT_X) | target: ELEMENT_X | action: external_surface_bounce',
            'TYPE: CONTEXT | path: DETECTOR_DET_1➔R_EXT(ELEMENT_X)➔T_IN(ELEMENT_X)➔T_OUT(ELEMENT_X)➔R_EXT(ELEMENT_X) | target: SOURCE_A | action: query_emission_from_AIR',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(ELEMENT_X)➔T_IN(ELEMENT_X)➔T_OUT(ELEMENT_X)➔R_EXT(ELEMENT_X) | target: SCENE | action: schedule_emission_shader_from_AIR',
            'TYPE: TERMINATION | path: DETECTOR_DET_1➔R_EXT(ELEMENT_X)➔T_IN(ELEMENT_X)➔T_OUT(ELEMENT_X)➔R_EXT(ELEMENT_X) | target: INFINITY | action: ray_escaped_at_step_4',
            'TYPE: CONTEXT | path: DETECTOR_DET_1➔R_EXT(ELEMENT_X)➔T_IN(ELEMENT_X)➔T_OUT(ELEMENT_X) | target: ELEMENT_X | action: query_direct_illum_from_SOURCE_A',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(ELEMENT_X)➔T_IN(ELEMENT_X)➔T_OUT(ELEMENT_X) | target: ELEMENT_X | action: schedule_reflection_shader',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(ELEMENT_X)➔T_IN(ELEMENT_X) | target: ELEMENT_X | action: schedule_transmission_shader',

            # Step 13: Interaction with ELEMENT_X
            'TYPE: INTERSECT | path: DETECTOR_DET_1➔R_EXT(ELEMENT_X)➔T_IN(ELEMENT_X) | target: ELEMENT_X | action: internal_wall_bounce',
            'TYPE: CONTEXT | path: DETECTOR_DET_1➔R_EXT(ELEMENT_X)➔T_IN(ELEMENT_X)➔R_INT(ELEMENT_X) | target: SOURCE_A | action: query_emission_from_ELEMENT_X',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(ELEMENT_X)➔T_IN(ELEMENT_X)➔R_INT(ELEMENT_X) | target: SCENE | action: schedule_emission_shader_from_ELEMENT_X',

            # Step 14: Out through ELEMENT_X
            'TYPE: INTERSECT | path: DETECTOR_DET_1➔R_EXT(ELEMENT_X)➔T_IN(ELEMENT_X)➔R_INT(ELEMENT_X) | target: ELEMENT_X | action: transmit_out_to_AIR',
            'TYPE: CONTEXT | path: DETECTOR_DET_1➔R_EXT(ELEMENT_X)➔T_IN(ELEMENT_X)➔R_INT(ELEMENT_X)➔T_OUT(ELEMENT_X) | target: SOURCE_A | action: query_emission_from_AIR',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(ELEMENT_X)➔T_IN(ELEMENT_X)➔R_INT(ELEMENT_X)➔T_OUT(ELEMENT_X) | target: SCENE | action: schedule_emission_shader_from_AIR',
            'TYPE: TERMINATION | path: DETECTOR_DET_1➔R_EXT(ELEMENT_X)➔T_IN(ELEMENT_X)➔R_INT(ELEMENT_X)➔T_OUT(ELEMENT_X) | target: INFINITY | action: ray_escaped_at_step_4',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(ELEMENT_X)➔T_IN(ELEMENT_X)➔R_INT(ELEMENT_X) | target: ELEMENT_X | action: schedule_transmission_shader',
            'TYPE: TERMINATION | path: DETECTOR_DET_1➔R_EXT(ELEMENT_X)➔T_IN(ELEMENT_X)➔R_INT(ELEMENT_X) | target: ELEMENT_X | action: bounce_budget_exhausted',
            'TYPE: CONTEXT | path: DETECTOR_DET_1➔R_EXT(ELEMENT_X)➔T_IN(ELEMENT_X) | target: ELEMENT_X | action: query_direct_illum_from_SOURCE_A',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(ELEMENT_X)➔T_IN(ELEMENT_X) | target: ELEMENT_X | action: schedule_reflection_shader',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(ELEMENT_X) | target: ELEMENT_X | action: schedule_transmission_shader',

            # Step 15: Interaction with ELEMENT_X
            'TYPE: INTERSECT | path: DETECTOR_DET_1➔R_EXT(ELEMENT_X) | target: ELEMENT_X | action: external_surface_bounce',
            'TYPE: CONTEXT | path: DETECTOR_DET_1➔R_EXT(ELEMENT_X)➔R_EXT(ELEMENT_X) | target: SOURCE_A | action: query_emission_from_AIR',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(ELEMENT_X)➔R_EXT(ELEMENT_X) | target: SCENE | action: schedule_emission_shader_from_AIR',

            # Step 16: In through ELEMENT_X
            'TYPE: INTERSECT | path: DETECTOR_DET_1➔R_EXT(ELEMENT_X)➔R_EXT(ELEMENT_X) | target: ELEMENT_X | action: transmit_in_from_AIR',
            'TYPE: CONTEXT | path: DETECTOR_DET_1➔R_EXT(ELEMENT_X)➔R_EXT(ELEMENT_X)➔T_IN(ELEMENT_X) | target: SOURCE_A | action: query_emission_from_ELEMENT_X',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(ELEMENT_X)➔R_EXT(ELEMENT_X)➔T_IN(ELEMENT_X) | target: SCENE | action: schedule_emission_shader_from_ELEMENT_X',

            # Step 17: Out through ELEMENT_X
            'TYPE: INTERSECT | path: DETECTOR_DET_1➔R_EXT(ELEMENT_X)➔R_EXT(ELEMENT_X)➔T_IN(ELEMENT_X) | target: ELEMENT_X | action: transmit_out_to_AIR',
            'TYPE: CONTEXT | path: DETECTOR_DET_1➔R_EXT(ELEMENT_X)➔R_EXT(ELEMENT_X)➔T_IN(ELEMENT_X)➔T_OUT(ELEMENT_X) | target: SOURCE_A | action: query_emission_from_AIR',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(ELEMENT_X)➔R_EXT(ELEMENT_X)➔T_IN(ELEMENT_X)➔T_OUT(ELEMENT_X) | target: SCENE | action: schedule_emission_shader_from_AIR',
            'TYPE: TERMINATION | path: DETECTOR_DET_1➔R_EXT(ELEMENT_X)➔R_EXT(ELEMENT_X)➔T_IN(ELEMENT_X)➔T_OUT(ELEMENT_X) | target: INFINITY | action: ray_escaped_at_step_4',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(ELEMENT_X)➔R_EXT(ELEMENT_X)➔T_IN(ELEMENT_X) | target: ELEMENT_X | action: schedule_transmission_shader',
            'TYPE: TERMINATION | path: DETECTOR_DET_1➔R_EXT(ELEMENT_X)➔R_EXT(ELEMENT_X)➔T_IN(ELEMENT_X) | target: ELEMENT_X | action: bounce_budget_exhausted',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(ELEMENT_X)➔R_EXT(ELEMENT_X) | target: ELEMENT_X | action: schedule_transmission_shader',
            'TYPE: TERMINATION | path: DETECTOR_DET_1➔R_EXT(ELEMENT_X)➔R_EXT(ELEMENT_X) | target: ELEMENT_X | action: bounce_budget_exhausted',
            'TYPE: CONTEXT | path: DETECTOR_DET_1➔R_EXT(ELEMENT_X) | target: ELEMENT_X | action: query_direct_illum_from_SOURCE_A',
            'TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(ELEMENT_X) | target: ELEMENT_X | action: schedule_reflection_shader',
            'TYPE: CONTEXT | path: DETECTOR_DET_1 | target: ELEMENT_X | action: query_direct_illum_from_SOURCE_A',
            'TYPE: SHADERS | path: DETECTOR_DET_1 | target: ELEMENT_X | action: schedule_reflection_shader',
        ],
    ]

    assert lane_history == expected_sequence

# =========================================================================
# TEST 8: Strict Budget Termination Profile
# =========================================================================
def test_subsurface_infinite_reflection_budget_exhaustion():
    """
    Validates that a trapped ray terminates exactly when its budget depletes.
    Guarantees that no hanging recursive states or phantom evaluations execute 
    after the termination limits are declared.
    """
    detector = MockNode("DET_1")
    trapping_core = MockNode("TRAPPING_CORE", transparent=True)
    light_source = MockNode("SOURCE_A")
    
    tracer = TopologicalRayTracer(
        starts=[detector], elements=[trapping_core], ends=[light_source],
        bounce_count=2, max_transmission_depth=1
    )
    
    core_timeline = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    timeline_data = [core_timeline]
    
    output_rays = tracer.raytrace(batch_size=1, MOCK_distances_timeline=timeline_data)
    lane_history = tracer.tracker.history[0]

    # print_debug_output("debug_output.py", lane_history)
    
    expected_sequence = [
        'TYPE: ORIGIN | path: DETECTOR_DET_1 | target: DET_1 | action: init_trace',
        'TYPE: CONTEXT | path: DETECTOR_DET_1 | target: SOURCE_A | action: query_emission_from_AIR',
        'TYPE: SHADERS | path: DETECTOR_DET_1 | target: SCENE | action: schedule_emission_shader_from_AIR',

        # Step 0: In through TRAPPING_CORE
        'TYPE: INTERSECT | path: DETECTOR_DET_1 | target: TRAPPING_CORE | action: transmit_in_from_AIR',
        'TYPE: CONTEXT | path: DETECTOR_DET_1➔T_IN(TRAPPING_CORE) | target: SOURCE_A | action: query_emission_from_TRAPPING_CORE',
        'TYPE: SHADERS | path: DETECTOR_DET_1➔T_IN(TRAPPING_CORE) | target: SCENE | action: schedule_emission_shader_from_TRAPPING_CORE',
        'TYPE: TERMINATION | path: DETECTOR_DET_1➔T_IN(TRAPPING_CORE) | target: TRAPPING_CORE | action: transmit_budget_exhausted',

        # Step 1: Interaction with TRAPPING_CORE
        'TYPE: INTERSECT | path: DETECTOR_DET_1➔T_IN(TRAPPING_CORE) | target: TRAPPING_CORE | action: internal_wall_bounce',
        'TYPE: CONTEXT | path: DETECTOR_DET_1➔T_IN(TRAPPING_CORE)➔R_INT(TRAPPING_CORE) | target: SOURCE_A | action: query_emission_from_TRAPPING_CORE',
        'TYPE: SHADERS | path: DETECTOR_DET_1➔T_IN(TRAPPING_CORE)➔R_INT(TRAPPING_CORE) | target: SCENE | action: schedule_emission_shader_from_TRAPPING_CORE',
        'TYPE: TERMINATION | path: DETECTOR_DET_1➔T_IN(TRAPPING_CORE)➔R_INT(TRAPPING_CORE) | target: TRAPPING_CORE | action: transmit_budget_exhausted',

        # Step 2: Interaction with TRAPPING_CORE
        'TYPE: INTERSECT | path: DETECTOR_DET_1➔T_IN(TRAPPING_CORE)➔R_INT(TRAPPING_CORE) | target: TRAPPING_CORE | action: internal_wall_bounce',
        'TYPE: CONTEXT | path: DETECTOR_DET_1➔T_IN(TRAPPING_CORE)➔R_INT(TRAPPING_CORE)➔R_INT(TRAPPING_CORE) | target: SOURCE_A | action: query_emission_from_TRAPPING_CORE',
        'TYPE: SHADERS | path: DETECTOR_DET_1➔T_IN(TRAPPING_CORE)➔R_INT(TRAPPING_CORE)➔R_INT(TRAPPING_CORE) | target: SCENE | action: schedule_emission_shader_from_TRAPPING_CORE',
        'TYPE: TERMINATION | path: DETECTOR_DET_1➔T_IN(TRAPPING_CORE)➔R_INT(TRAPPING_CORE)➔R_INT(TRAPPING_CORE) | target: TRAPPING_CORE | action: transmit_budget_exhausted',
        'TYPE: TERMINATION | path: DETECTOR_DET_1➔T_IN(TRAPPING_CORE)➔R_INT(TRAPPING_CORE)➔R_INT(TRAPPING_CORE) | target: TRAPPING_CORE | action: bounce_budget_exhausted',
        'TYPE: CONTEXT | path: DETECTOR_DET_1➔T_IN(TRAPPING_CORE)➔R_INT(TRAPPING_CORE) | target: TRAPPING_CORE | action: query_direct_illum_from_SOURCE_A',
        'TYPE: SHADERS | path: DETECTOR_DET_1➔T_IN(TRAPPING_CORE)➔R_INT(TRAPPING_CORE) | target: TRAPPING_CORE | action: schedule_reflection_shader',
        'TYPE: CONTEXT | path: DETECTOR_DET_1➔T_IN(TRAPPING_CORE) | target: TRAPPING_CORE | action: query_direct_illum_from_SOURCE_A',
        'TYPE: SHADERS | path: DETECTOR_DET_1➔T_IN(TRAPPING_CORE) | target: TRAPPING_CORE | action: schedule_reflection_shader',
        'TYPE: SHADERS | path: DETECTOR_DET_1 | target: TRAPPING_CORE | action: schedule_transmission_shader',

        # Step 3: Interaction with TRAPPING_CORE
        'TYPE: INTERSECT | path: DETECTOR_DET_1 | target: TRAPPING_CORE | action: external_surface_bounce',
        'TYPE: CONTEXT | path: DETECTOR_DET_1➔R_EXT(TRAPPING_CORE) | target: SOURCE_A | action: query_emission_from_AIR',
        'TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(TRAPPING_CORE) | target: SCENE | action: schedule_emission_shader_from_AIR',

        # Step 4: In through TRAPPING_CORE
        'TYPE: INTERSECT | path: DETECTOR_DET_1➔R_EXT(TRAPPING_CORE) | target: TRAPPING_CORE | action: transmit_in_from_AIR',
        'TYPE: CONTEXT | path: DETECTOR_DET_1➔R_EXT(TRAPPING_CORE)➔T_IN(TRAPPING_CORE) | target: SOURCE_A | action: query_emission_from_TRAPPING_CORE',
        'TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(TRAPPING_CORE)➔T_IN(TRAPPING_CORE) | target: SCENE | action: schedule_emission_shader_from_TRAPPING_CORE',
        'TYPE: TERMINATION | path: DETECTOR_DET_1➔R_EXT(TRAPPING_CORE)➔T_IN(TRAPPING_CORE) | target: TRAPPING_CORE | action: transmit_budget_exhausted',

        # Step 5: Interaction with TRAPPING_CORE
        'TYPE: INTERSECT | path: DETECTOR_DET_1➔R_EXT(TRAPPING_CORE)➔T_IN(TRAPPING_CORE) | target: TRAPPING_CORE | action: internal_wall_bounce',
        'TYPE: CONTEXT | path: DETECTOR_DET_1➔R_EXT(TRAPPING_CORE)➔T_IN(TRAPPING_CORE)➔R_INT(TRAPPING_CORE) | target: SOURCE_A | action: query_emission_from_TRAPPING_CORE',
        'TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(TRAPPING_CORE)➔T_IN(TRAPPING_CORE)➔R_INT(TRAPPING_CORE) | target: SCENE | action: schedule_emission_shader_from_TRAPPING_CORE',
        'TYPE: TERMINATION | path: DETECTOR_DET_1➔R_EXT(TRAPPING_CORE)➔T_IN(TRAPPING_CORE)➔R_INT(TRAPPING_CORE) | target: TRAPPING_CORE | action: transmit_budget_exhausted',
        'TYPE: TERMINATION | path: DETECTOR_DET_1➔R_EXT(TRAPPING_CORE)➔T_IN(TRAPPING_CORE)➔R_INT(TRAPPING_CORE) | target: TRAPPING_CORE | action: bounce_budget_exhausted',
        'TYPE: CONTEXT | path: DETECTOR_DET_1➔R_EXT(TRAPPING_CORE)➔T_IN(TRAPPING_CORE) | target: TRAPPING_CORE | action: query_direct_illum_from_SOURCE_A',
        'TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(TRAPPING_CORE)➔T_IN(TRAPPING_CORE) | target: TRAPPING_CORE | action: schedule_reflection_shader',
        'TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(TRAPPING_CORE) | target: TRAPPING_CORE | action: schedule_transmission_shader',

        # Step 6: Interaction with TRAPPING_CORE
        'TYPE: INTERSECT | path: DETECTOR_DET_1➔R_EXT(TRAPPING_CORE) | target: TRAPPING_CORE | action: external_surface_bounce',
        'TYPE: CONTEXT | path: DETECTOR_DET_1➔R_EXT(TRAPPING_CORE)➔R_EXT(TRAPPING_CORE) | target: SOURCE_A | action: query_emission_from_AIR',
        'TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(TRAPPING_CORE)➔R_EXT(TRAPPING_CORE) | target: SCENE | action: schedule_emission_shader_from_AIR',

        # Step 7: In through TRAPPING_CORE
        'TYPE: INTERSECT | path: DETECTOR_DET_1➔R_EXT(TRAPPING_CORE)➔R_EXT(TRAPPING_CORE) | target: TRAPPING_CORE | action: transmit_in_from_AIR',
        'TYPE: CONTEXT | path: DETECTOR_DET_1➔R_EXT(TRAPPING_CORE)➔R_EXT(TRAPPING_CORE)➔T_IN(TRAPPING_CORE) | target: SOURCE_A | action: query_emission_from_TRAPPING_CORE',
        'TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(TRAPPING_CORE)➔R_EXT(TRAPPING_CORE)➔T_IN(TRAPPING_CORE) | target: SCENE | action: schedule_emission_shader_from_TRAPPING_CORE',
        'TYPE: TERMINATION | path: DETECTOR_DET_1➔R_EXT(TRAPPING_CORE)➔R_EXT(TRAPPING_CORE)➔T_IN(TRAPPING_CORE) | target: TRAPPING_CORE | action: transmit_budget_exhausted',
        'TYPE: TERMINATION | path: DETECTOR_DET_1➔R_EXT(TRAPPING_CORE)➔R_EXT(TRAPPING_CORE)➔T_IN(TRAPPING_CORE) | target: TRAPPING_CORE | action: bounce_budget_exhausted',
        'TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(TRAPPING_CORE)➔R_EXT(TRAPPING_CORE) | target: TRAPPING_CORE | action: schedule_transmission_shader',
        'TYPE: TERMINATION | path: DETECTOR_DET_1➔R_EXT(TRAPPING_CORE)➔R_EXT(TRAPPING_CORE) | target: TRAPPING_CORE | action: bounce_budget_exhausted',
        'TYPE: CONTEXT | path: DETECTOR_DET_1➔R_EXT(TRAPPING_CORE) | target: TRAPPING_CORE | action: query_direct_illum_from_SOURCE_A',
        'TYPE: SHADERS | path: DETECTOR_DET_1➔R_EXT(TRAPPING_CORE) | target: TRAPPING_CORE | action: schedule_reflection_shader',
        'TYPE: CONTEXT | path: DETECTOR_DET_1 | target: TRAPPING_CORE | action: query_direct_illum_from_SOURCE_A',
        'TYPE: SHADERS | path: DETECTOR_DET_1 | target: TRAPPING_CORE | action: schedule_reflection_shader',
    ]

    assert lane_history == expected_sequence

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))