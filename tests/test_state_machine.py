import pytest
import numpy as np
import sys

from src.state_machine.state_machine import MockNode, TopologicalRayTracer



def assert_sequence_order(lane_history: list[str], expected_sequence: list[str]):
    """Helper utility to enforce a strict chronological state machine trajectory."""
    last_found_index = -1
    for expected_state in expected_sequence:
        assert expected_state in lane_history, f"Missing expected state: '{expected_state}'"
        
        # Search exclusively forward from the position of the last found event
        current_index = lane_history.index(expected_state, last_found_index + 1)
        assert current_index > last_found_index, f"State '{expected_state}' occurred out of order!"
        last_found_index = current_index


# =========================================================================
# TEST 1: Source in Direct View of Detector
# =========================================================================
def test_source_in_direct_view():
    """Tests a pristine environment where nothing blocks or intersects the rays."""
    detector = MockNode("DET_1")
    light_source = MockNode("SOURCE_A")
    
    tracer = TopologicalRayTracer(
        starts=[detector],
        elements=[],  
        ends=[light_source],
        bounce_count=0,
        max_transmission_depth=0
    )
    
    batch_size = 2
    timeline_data = []  
    
    output_rays = tracer.raytrace(batch_size=batch_size, MOCK_distances_timeline=timeline_data)
    
    expected_sequence = [
        "START_AT_DETECTOR_DET_1",
        "CHECK_SOURCE_SOURCE_A_FROM_AIR",
        "APPLY_EMISSION_MODEL_FROM_AIR"
    ]
    
    for ray_idx in range(batch_size):
        lane_history = tracer.tracker.history[ray_idx]
        assert_sequence_order(lane_history, expected_sequence)
        assert len(lane_history) == 3


# =========================================================================
# TEST 2: Detector ➔ Reflect ➔ Reflect ➔ Reflect ➔ Source
# =========================================================================
def test_detector_to_three_consecutive_reflections():
    """Tests a deep reflection chain hitting three distinct mirrors sequentially."""
    detector = MockNode("DET_1")
    mirror1 = MockNode("MIRROR_1", transparent=False)
    mirror2 = MockNode("MIRROR_2", transparent=False)
    mirror3 = MockNode("MIRROR_3", transparent=False)
    light_source = MockNode("SOURCE_A")
    
    tracer = TopologicalRayTracer(
        starts=[detector],
        elements=[mirror1, mirror2, mirror3],
        ends=[light_source],
        bounce_count=3,
        max_transmission_depth=0
    )
    
    batch_size = 1
    m1_timeline = np.array([[1.0, float('inf'), float('inf')]])
    m2_timeline = np.array([[float('inf'), 2.0, float('inf')]])
    m3_timeline = np.array([[float('inf'), float('inf'), 3.0]])
    
    timeline_data = [m1_timeline, m2_timeline, m3_timeline]
    
    output_rays = tracer.raytrace(batch_size=batch_size, MOCK_distances_timeline=timeline_data)
    lane_history = tracer.tracker.history[0]
    
    expected_sequence = [
        "START_AT_DETECTOR_DET_1",
        "CHECK_SOURCE_SOURCE_A_FROM_AIR",
        "APPLY_EMISSION_MODEL_FROM_AIR",
        
        # Mirror 1 (Bounce 0)
        "EXTERNAL_SURFACE_BOUNCE_OFF_MIRROR_1",
        "CHECK_SOURCE_SOURCE_A_FROM_AIR",  # Recursive frame start
        "APPLY_EMISSION_MODEL_FROM_AIR",
        
        # Mirror 2 (Bounce 1)
        "EXTERNAL_SURFACE_BOUNCE_OFF_MIRROR_2",
        "CHECK_SOURCE_SOURCE_A_FROM_AIR",  # Recursive frame start
        "APPLY_EMISSION_MODEL_FROM_AIR",
        
        # Mirror 3 (Bounce 2)
        "EXTERNAL_SURFACE_BOUNCE_OFF_MIRROR_3",
        "CHECK_SOURCE_SOURCE_A_FROM_AIR",  # Recursive frame start
        "APPLY_EMISSION_MODEL_FROM_AIR",
        "DIRECT_ILLUM_AT_MIRROR_3_FOR_SOURCE_A",
        "APPLY_REFLECTION_MODEL_ON_MIRROR_3",
        
        # Unwinding checks back up the stack
        "DIRECT_ILLUM_AT_MIRROR_2_FOR_SOURCE_A",
        "APPLY_REFLECTION_MODEL_ON_MIRROR_2",
        "DIRECT_ILLUM_AT_MIRROR_1_FOR_SOURCE_A",
        "APPLY_REFLECTION_MODEL_ON_MIRROR_1"
    ]
    
    assert_sequence_order(lane_history, expected_sequence)


# =========================================================================
# TEST 3: Detector ➔ Transmit In ➔ Reflect/Transmit Out ➔ Reflect ➔ Source
# =========================================================================
def test_complex_split_transmission_and_reflection_path():
    """Tests a complex splitting dielectric path verifying simultaneous sub-paths."""
    detector = MockNode("DET_1")
    glass_prism = MockNode("GLASS_PRISM", transparent=True)
    opaque_mirror = MockNode("OPAQUE_MIRROR", transparent=False)
    light_source = MockNode("SOURCE_A")
    
    tracer = TopologicalRayTracer(
        starts=[detector],
        elements=[glass_prism, opaque_mirror],
        ends=[light_source],
        bounce_count=1,
        max_transmission_depth=2
    )
    
    batch_size = 1
    prism_timeline = np.array([[1.0, 2.0, float('inf')]])
    mirror_timeline = np.array([[float('inf'), float('inf'), 3.0]])
    
    timeline_data = [prism_timeline, mirror_timeline]
    
    output_rays = tracer.raytrace(batch_size=batch_size, MOCK_distances_timeline=timeline_data)
    lane_history = tracer.tracker.history[0]
    
    # Because of ray splitting, we trace chronological key checkpoints down branches
    expected_sequence = [
        "START_AT_DETECTOR_DET_1",
        "CHECK_SOURCE_SOURCE_A_FROM_AIR",
        "APPLY_EMISSION_MODEL_FROM_AIR",
        
        # Step 0: Hit prism outer boundary from air
        "TRANSMIT_INTO_GLASS_PRISM_FROM_AIR",
        
        # Entering internal recursive frame 
        "CHECK_SOURCE_SOURCE_A_FROM_GLASS_PRISM",
        "APPLY_EMISSION_MODEL_FROM_GLASS_PRISM",
        
        # Step 1: Subsurface splitting occurs at the inner boundary wall
        "TRANSMIT_OUT_OF_GLASS_PRISM_TO_AIR",
        
        # Sub-branch: Ray escapes out into air
        "CHECK_SOURCE_SOURCE_A_FROM_AIR",
        "APPLY_EMISSION_MODEL_FROM_AIR",
        
        # Step 2: Escaped ray strikes external opaque mirror
        "EXTERNAL_SURFACE_BOUNCE_OFF_OPAQUE_MIRROR",
        "DIRECT_ILLUM_AT_OPAQUE_MIRROR_FOR_SOURCE_A",
        "APPLY_REFLECTION_MODEL_ON_OPAQUE_MIRROR",
        
        # Return to prism boundary hooks
        "APPLY_TRANSMISSION_MODEL_ON_GLASS_PRISM",
        "INTERNAL_WALL_BOUNCE_INSIDE_GLASS_PRISM",
        "DIRECT_ILLUM_AT_GLASS_PRISM_FOR_SOURCE_A",
        "APPLY_REFLECTION_MODEL_ON_GLASS_PRISM"
    ]
    
    assert_sequence_order(lane_history, expected_sequence)


def test_reflection_off_passive_element_illuminated_by_source():
    """Tests a strict physical arrangement with zero entity mixing and verified order."""
    detector = MockNode("DET_1")
    passive_mirror = MockNode("PASSIVE_MIRROR", transparent=False)  
    light_source = MockNode("LIGHT_SOURCE_A")                       
    
    tracer = TopologicalRayTracer(
        starts=[detector],
        elements=[passive_mirror],  
        ends=[light_source],        
        bounce_count=1,
        max_transmission_depth=0
    )
    
    batch_size = 1
    mirror_timeline = np.array([[3.0]])
    timeline_data = [mirror_timeline]
    
    output_rays = tracer.raytrace(batch_size=batch_size, MOCK_distances_timeline=timeline_data)
    
    # Extract the flat list for ray index 0
    lane_history = tracer.tracker.history[0]
    
    expected_sequence = [
        # --- FRAME 0: START AT DETECTOR ---
        "START_AT_DETECTOR_DET_1",
        # Detector checks for direct, unblocked view of the light source from air
        "CHECK_SOURCE_LIGHT_SOURCE_A_FROM_AIR",
        # Detector applies emission rules for what it sees directly in the background
        "APPLY_EMISSION_MODEL_FROM_AIR",
        
        # --- FRAME 1: INTERSECTING GEOMETRY ---
        # The ray strikes the mirror and spawns a reflection branch
        "EXTERNAL_SURFACE_BOUNCE_OFF_PASSIVE_MIRROR",
        
        # Inside the reflection branch, a new recursive frame starts at the mirror surface!
        # It queries if the light source can directly illuminate this point on the mirror
        "CHECK_SOURCE_LIGHT_SOURCE_A_FROM_AIR",
        # It applies emission brightness hitting this sub-path location
        "APPLY_EMISSION_MODEL_FROM_AIR",
        
        # The reflection frame finishes its checks and unwinds, allowing the parent frame's 
        # reflection loop to couple the bounced energy together
        "DIRECT_ILLUM_AT_PASSIVE_MIRROR_FOR_LIGHT_SOURCE_A",
        "APPLY_REFLECTION_MODEL_ON_PASSIVE_MIRROR"
    ]

    
    # Enforce order: verify each state exists and comes after the previous one
    last_found_index = -1
    for expected_state in expected_sequence:
        assert expected_state in lane_history, f"Missing expected state: {expected_state}"
        
        # Find where it occurs, starting the search *after* our last found state
        current_index = lane_history.index(expected_state, last_found_index + 1)
        
        assert current_index > last_found_index, f"State '{expected_state}' occurred out of order!"
        last_found_index = current_index





if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))