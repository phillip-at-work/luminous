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

import numpy as np
from functools import reduce
from typing import List, Optional, Tuple

class MockNode:
    def __init__(self, node_id: str, transparent: bool = False, refractive_index: float = 1.0):
        self.node_id = node_id
        self.transparent = transparent
        self.refractive_index = refractive_index


class VectorizedStateTracker:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.history: List[List[str]] = [[] for _ in range(batch_size)]

    def log_state(self, mask: np.ndarray, log_type: str, path: str, target: str, action: str):
        """Option A implementation generating scannable structural state tokens."""
        state_string = f"TYPE: {log_type} | path: {path} | target: {target} | action: {action}"
        idx = np.where(mask)[0]
        for i in idx:
            self.history[i].append(state_string)


class TopologicalRayTracer:
    def __init__(self, starts: List[MockNode], elements: List[MockNode], ends: List[MockNode], 
                 bounce_count: int = 2, max_transmission_depth: int = 2):
        self.starts = starts        
        self.elements = elements    
        self.ends = ends            
        self.bounce_count = bounce_count  
        self.max_transmission_depth = max_transmission_depth  
        self.tracker: Optional[VectorizedStateTracker] = None
        self.INFINITE = float('inf')

    def raytrace(self, batch_size: int, MOCK_distances_timeline: List[np.ndarray]) -> Vector:
        self.tracker = VectorizedStateTracker(batch_size)
        total_accumulated_rays = Vector(np.zeros(batch_size), np.zeros(batch_size), np.zeros(batch_size))
        all_active = np.ones(batch_size, dtype=np.bool_)

        initial_bounces = np.zeros(batch_size, dtype=np.int32)
        initial_transmissions = np.zeros(batch_size, dtype=np.int32)

        for detector in self.starts:
            path_root = f"DETECTOR_{detector.node_id}"
            self.tracker.log_state(all_active, "ORIGIN", path_root, detector.node_id, "init_trace")
            
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
                active_mask=all_active,
                path_id=path_root
            )
            total_accumulated_rays += returned_rays
            
        return total_accumulated_rays

    def _reverse_recursive_path_trace(
        self, detector: MockNode, origin: Vector, direction: Vector, bounces: np.ndarray, 
        transmission_depths: np.ndarray, medium_stack: Tuple[MockNode, ...], 
        MOCK_distances_timeline: List[np.ndarray], MOCK_step: int, active_mask: np.ndarray, path_id: str
    ) -> Vector:
        batch_size = active_mask.shape[0]
        rays = Vector(np.zeros(batch_size), np.zeros(batch_size), np.zeros(batch_size))

        in_budget = (bounces <= self.bounce_count) & (transmission_depths <= self.max_transmission_depth)
        alive_mask = active_mask & in_budget
        dead_mask = active_mask & ~in_budget

        if np.any(dead_mask):
            self.tracker.log_state(dead_mask, "TERMINATION", path_id, "BUDGET", f"hard_budget_exceeded_step_{MOCK_step}")

        if not np.any(alive_mask):
            return rays

        current_medium = medium_stack[-1] if medium_stack else None
        medium_id = current_medium.node_id if current_medium else "AIR"

        for source_node in self.ends:
            self.tracker.log_state(alive_mask, "CONTEXT", path_id, source_node.node_id, f"query_emission_from_{medium_id}")

        # TODO should we log this state only if we `query_emission_from_` and it returns True?
        self.tracker.log_state(alive_mask, "SHADERS", path_id, "SCENE", f"schedule_emission_shader_from_{medium_id}")

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
        
        escaped_mask = alive_mask & (minimum_distances == self.INFINITE)
        if np.any(escaped_mask):
            self.tracker.log_state(escaped_mask, "TERMINATION", path_id, "INFINITY", f"ray_escaped_at_step_{MOCK_step}")

        for element, distance in zip(self.elements, distances):
            hit: np.ndarray = alive_mask & (minimum_distances != self.INFINITE) & (distance == minimum_distances)

            if not np.any(hit):
                continue

            is_exiting_this_element = (current_medium is not None and current_medium.node_id == element.node_id)

            # ROUTINE A: TRANSMISSION BRANCH
            if element.transparent:
                can_transmit = hit & (transmission_depths < self.max_transmission_depth)
                transmit_blocked = hit & (transmission_depths >= self.max_transmission_depth)
                
                if np.any(transmit_blocked):
                    self.tracker.log_state(transmit_blocked, "TERMINATION", path_id, element.node_id, "transmit_budget_exhausted")

                if np.any(can_transmit):
                    if is_exiting_this_element:
                        next_stack = medium_stack[:-1]
                        dest_id = next_stack[-1].node_id if next_stack else "AIR"
                        self.tracker.log_state(can_transmit, "INTERSECT", path_id, element.node_id, f"transmit_out_to_{dest_id}")
                        branch_path_id = f"{path_id}➔T_OUT({element.node_id})"
                    else:
                        next_stack = medium_stack + (element,)
                        self.tracker.log_state(can_transmit, "INTERSECT", path_id, element.node_id, f"transmit_in_from_{medium_id}")
                        branch_path_id = f"{path_id}➔T_IN({element.node_id})"

                    next_transmissions = transmission_depths.copy()
                    next_transmissions[can_transmit] += 1
                    
                    transmitted_ray_data = self._reverse_recursive_path_trace(
                        detector, origin, direction, bounces, next_transmissions,
                        next_stack, MOCK_distances_timeline, MOCK_step + 1, 
                        active_mask=can_transmit, path_id=branch_path_id
                    )
                    rays += transmitted_ray_data.place(can_transmit)
                    self.tracker.log_state(can_transmit, "SHADERS", path_id, element.node_id, "schedule_transmission_shader")

            # ROUTINE B: REFLECTION BRANCH
            can_bounce = hit & (bounces < self.bounce_count)
            bounce_blocked = hit & (bounces >= self.bounce_count)
            
            if np.any(bounce_blocked):
                self.tracker.log_state(bounce_blocked, "TERMINATION", path_id, element.node_id, "bounce_budget_exhausted")

            if np.any(can_bounce):
                if is_exiting_this_element:
                    self.tracker.log_state(can_bounce, "INTERSECT", path_id, element.node_id, "internal_wall_bounce")
                    branch_path_id = f"{path_id}➔R_INT({element.node_id})"
                else:
                    self.tracker.log_state(can_bounce, "INTERSECT", path_id, element.node_id, "external_surface_bounce")
                    branch_path_id = f"{path_id}➔R_EXT({element.node_id})"

                next_bounces = bounces.copy()
                next_bounces[can_bounce] += 1

                reflected_ray_data = self._reverse_recursive_path_trace(
                    detector, origin, direction, next_bounces, transmission_depths,
                    medium_stack, MOCK_distances_timeline, MOCK_step + 1, 
                    active_mask=can_bounce, path_id=branch_path_id
                )
                rays += reflected_ray_data.place(can_bounce)

                for source_node in self.ends:
                    self.tracker.log_state(can_bounce, "CONTEXT", path_id, element.node_id, f"query_direct_illum_from_{source_node.node_id}")

                self.tracker.log_state(can_bounce, "SHADERS", path_id, element.node_id, "schedule_reflection_shader")
                
                mock_source_energy = Vector(np.zeros(batch_size), np.zeros(batch_size), np.zeros(batch_size))
                mock_source_energy.x[can_bounce] = 5.0
                mock_source_energy.y[can_bounce] = 5.0
                mock_source_energy.z[can_bounce] = 5.0
                rays += mock_source_energy.place(can_bounce)

        return rays
