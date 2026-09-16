import numpy as np
from typing import List, Tuple, Any, Optional
from functools import reduce
import numbers

from ..math.vector import Vector
from ..math.tools import extract
from ..element.detector import Detector
from ..element.element import Element
from ..element.source import Source
from ..utilities.ray_debugger import NullRayDebugger, ConcreteRayDebugger

from ..utilities.logconfig import setup_logging
import logging
logger = logging.getLogger(__name__)

class MockNode:
    def __init__(self, node_id: str, transparent: bool = False, refractive_index: float = 1.0):
        self.node_id = node_id
        self.transparent = transparent
        self.refractive_index = refractive_index


class VectorizedStateTracker:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.history: List[List[str]] = [[] for _ in range(batch_size)]

    def log_state(self, mask: np.ndarray, state_string: str):
        idx = np.where(mask)[0]
        for i in idx:
            self.history[i].append(state_string)


class TopologicalRayTracer:
    """State machine testing framework mirroring original vectorized physics loops."""
    def __init__(self, starts: List[MockNode], elements: List[MockNode], ends: List[MockNode], 
                 bounce_count: int = 2, max_transmission_depth: int = 2):
        self.starts = starts        
        self.elements = elements    
        self.ends = ends            
        self.bounce_count = bounce_count  # Instance parameter to modify per-test
        self.max_transmission_depth = max_transmission_depth  # Instance parameter to modify per-test
        self.tracker: Optional[VectorizedStateTracker] = None
        self.INFINITE = float('inf')

    def raytrace(self, batch_size: int, MOCK_distances_timeline: List[np.ndarray]) -> Vector:
        self.tracker = VectorizedStateTracker(batch_size)
        total_accumulated_rays = Vector(np.zeros(batch_size), np.zeros(batch_size), np.zeros(batch_size))
        
        all_active = np.ones(batch_size, dtype=np.bool_)

        # Vectorized budget tracking maps matching individual pixel trajectory frames
        initial_bounces = np.zeros(batch_size, dtype=np.int32)
        initial_transmissions = np.zeros(batch_size, dtype=np.int32)

        for detector in self.starts:
            self.tracker.log_state(all_active, f"START_AT_DETECTOR_{detector.node_id}")
            
            dummy_origin = Vector(np.zeros(batch_size), np.zeros(batch_size), np.zeros(batch_size))
            dummy_direction = Vector(np.ones(batch_size), np.ones(batch_size), np.ones(batch_size))

            returned_rays = self._reverse_recursive_path_trace(
                detector=detector,
                origin=dummy_origin,
                direction=dummy_direction,
                bounces=initial_bounces,
                transmission_depths=initial_transmissions,
                medium_stack=(),  
                MOCK_distances_timeline=MOCK_distances_timeline,
                MOCK_step=0,
                active_mask=all_active
            )
                
            total_accumulated_rays += returned_rays
            
        return total_accumulated_rays

    def _reverse_recursive_path_trace(
        self, 
        detector: MockNode, 
        origin: Vector, 
        direction: Vector, 
        bounces: np.ndarray, 
        transmission_depths: np.ndarray, 
        medium_stack: Tuple[MockNode, ...], 
        MOCK_distances_timeline: List[np.ndarray], 
        MOCK_step: int,
        active_mask: np.ndarray
    ) -> Vector:
        batch_size = active_mask.shape
        rays = Vector(np.zeros(batch_size), np.zeros(batch_size), np.zeros(batch_size))

        # Vectorized budget safety guard
        alive_mask = active_mask & (bounces <= self.bounce_count) & (transmission_depths <= self.max_transmission_depth)
        if not np.any(alive_mask):
            return rays

        current_medium = medium_stack[-1] if medium_stack else None
        medium_id = current_medium.node_id if current_medium else "AIR"

        # =========================================================================
        # 1. EMISSION MODEL HOOK (Direct Source Illumination Check & Application)
        # =========================================================================
        # At the start of this ray generation frame, check if the current origin points 
        # have an unblocked line of sight straight to any light source.
        #
        # For the START frame, this represents a source directly in view of the detector.
        # For subsequent frames, this collects ambient light hitting the ray location.
        #
        # >>> IN THE ORIGINAL FRAMEWORK:
        # >>> ray_data = detector._emission_model(detector.pointing_direction, emission_intersection_map)
        # >>> rays += ray_data.place(intersection_point_illuminated)
        # =========================================================================
        for source_node in self.ends:
            self.tracker.log_state(alive_mask, f"CHECK_SOURCE_{source_node.node_id}_FROM_{medium_id}")
            
        self.tracker.log_state(alive_mask, f"APPLY_EMISSION_MODEL_FROM_{medium_id}")

        # Parse timeline step matrices cleanly avoiding tuple coercion crashes
        distances: List[np.ndarray] = []
        for element_timeline in MOCK_distances_timeline:
            max_steps = element_timeline.shape[1] if element_timeline.ndim > 1 else element_timeline.shape[0]
            
            if MOCK_step < max_steps:
                distances.append(element_timeline[:, MOCK_step] if element_timeline.ndim > 1 else element_timeline)
            else:
                distances.append(np.full(batch_size, self.INFINITE, dtype=np.float64))

        if not distances:
            return rays
            
        minimum_distances: np.ndarray = reduce(np.minimum, distances)

        # --- 2. SCENE ELEMENT INTERACTION LOOP ---
        for element, distance in zip(self.elements, distances):
            hit: np.ndarray = alive_mask & (minimum_distances != self.INFINITE) & (distance == minimum_distances)

            if not np.any(hit):
                continue

            is_exiting_this_element = (current_medium is not None and current_medium.node_id == element.node_id)

            # =========================================================================
            # ROUTINE A: TRANSMISSION BRANCH (Ray Splitting Part 1)
            # =========================================================================
            if element.transparent:
                can_transmit = hit & (transmission_depths < self.max_transmission_depth)
                if np.any(can_transmit):
                    if is_exiting_this_element:
                        next_stack = medium_stack[:-1]
                        dest_id = next_stack[-1].node_id if next_stack else "AIR"
                        self.tracker.log_state(can_transmit, f"TRANSMIT_OUT_OF_{element.node_id}_TO_{dest_id}")
                    else:
                        next_stack = medium_stack + (element,)
                        self.tracker.log_state(can_transmit, f"TRANSMIT_INTO_{element.node_id}_FROM_{medium_id}")

                    next_transmissions = transmission_depths.copy()
                    next_transmissions[can_transmit] += 1

                    # A1. Recurse down the transmission axis to gather light from deeper layers
                    transmitted_ray_data = self._reverse_recursive_path_trace(
                        detector, origin, direction, bounces, next_transmissions,
                        next_stack, MOCK_distances_timeline, MOCK_step + 1, active_mask=can_transmit
                    )
                    rays += transmitted_ray_data.place(can_transmit)

                    # =====================================================================
                    # 2. TRANSMISSION MODEL HOOK (Internal Volumetric Boundary Phase)
                    # =====================================================================
                    # If evaluating a boundary crossing context (internal/subsurface), execute 
                    # your medium translation, attenuation, or internal volumetric scaling rules.
                    #
                    # >>> IN THE ORIGINAL FRAMEWORK:
                    # >>> ray_data = detector._transmission_model(element, start_point, intersection_point, hit)
                    # >>> rays += ray_data.place(hit)
                    # =====================================================================
                    self.tracker.log_state(can_transmit, f"APPLY_TRANSMISSION_MODEL_ON_{element.node_id}")

            # =========================================================================
            # ROUTINE B: REFLECTION BRANCH (Ray Splitting Part 2)
            # =========================================================================
            can_bounce = hit & (bounces < self.bounce_count)
            if np.any(can_bounce):
                if is_exiting_this_element:
                    self.tracker.log_state(can_bounce, f"INTERNAL_WALL_BOUNCE_INSIDE_{element.node_id}")
                else:
                    self.tracker.log_state(can_bounce, f"EXTERNAL_SURFACE_BOUNCE_OFF_{element.node_id}")

                next_bounces = bounces.copy()
                next_bounces[can_bounce] += 1

                # B1. Recurse down the reflection path first to collect indirect bounced energy
                reflected_ray_data = self._reverse_recursive_path_trace(
                    detector, origin, direction, next_bounces, transmission_depths,
                    medium_stack, MOCK_distances_timeline, MOCK_step + 1, active_mask=can_bounce
                )
                rays += reflected_ray_data.place(can_bounce)

                # B2. Check direct illumination shadows hitting this reflective surface location
                for source_node in self.ends:
                    self.tracker.log_state(can_bounce, f"DIRECT_ILLUM_AT_{element.node_id}_FOR_{source_node.node_id}")

                # =====================================================================
                # 3. REFLECTION MODEL HOOK (Specular/Diffuse Reflection Shader)
                # =====================================================================
                # Combine the bounced indirect path data with direct illumination source metrics
                # to calculate the surface color mapping value for this reflective instance.
                #
                # >>> IN THE ORIGINAL FRAMEWORK:
                # >>> direct_illum_data = detector._reflection_model(element, intersection_point, ..., reflection_intersection_map, hit)
                # >>> rays += direct_illum_data.place(hit)
                # =====================================================================
                self.tracker.log_state(can_bounce, f"APPLY_REFLECTION_MODEL_ON_{element.node_id}")
                
                mock_source_energy = Vector(np.ones(batch_size) * 5.0, np.ones(batch_size) * 5.0, np.ones(batch_size) * 5.0)
                rays += mock_source_energy.place(can_bounce)

        return rays
