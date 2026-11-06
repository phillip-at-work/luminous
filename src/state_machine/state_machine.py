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
        self._validate_timeline_shape(batch_size, MOCK_distances_timeline)
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

            # UNPACK THE TUPLE HERE:
            returned_rays, _ = self._reverse_recursive_path_trace(
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
        self,
        detector: MockNode,
        origin: Vector,
        direction: Vector,
        bounces: np.ndarray,
        transmission_depths: np.ndarray,
        medium_stack: Tuple[MockNode, ...],
        MOCK_distances_timeline: List[np.ndarray],
        MOCK_step: int,
        active_mask: np.ndarray,
        path_id: str
    ) -> Tuple[Vector, np.ndarray]:
        """
        Executes a vectorized, reverse-propagated path trace step.
        Splits evaluation into an end-node reachability / START<->END direct-view emission pass
        (PART 1) and a deferred secondary reflection/transmission scattering pass (PART 2).
        Supports simultaneous splitting (reflection and transmission forks).

        FRAMEWORK RULE: the emission model is scheduled IF AND ONLY IF the `starts` node and the
        `ends` node are in direct view of one another - i.e. that unobstructed line of sight
        exists at the START vertex (MOCK_step == 0). An `ends` node that is in direct view only
        at the terminal point of a reflection and/or transmission chain proves path connectivity
        (it terminates the ray and validates upstream deferred shader commits) and must never
        trigger the emission model. `starts`/`ends` are direction-agnostic by design; shader
        semantics are the developer's responsibility, and this rule is the consistency contract.
        """
        batch_size = active_mask.shape
        rays = Vector(np.zeros(batch_size), np.zeros(batch_size), np.zeros(batch_size))
        hit_source_downstream = np.zeros(batch_size, dtype=np.bool_)

        # 1. Budget Filtration
        in_budget = (bounces <= self.bounce_count) & (transmission_depths <= self.max_transmission_depth)
        alive_mask = active_mask & in_budget
        dead_mask = active_mask & ~in_budget

        if np.any(dead_mask):
            self.tracker.log_state(dead_mask, "TERMINATION", path_id, "BUDGET", f"hard_budget_exceeded_step_{MOCK_step}")

        if not np.any(alive_mask) or MOCK_step >= len(MOCK_distances_timeline):
            return rays, hit_source_downstream

        current_medium = medium_stack[-1] if medium_stack else None
        medium_id = current_medium.node_id if current_medium else "AIR"

        # Pull full distance row layout for the current time step
        # Columns layout expected: [Sources..., Elements...]
        step_distances = MOCK_distances_timeline[MOCK_step]

        # Isolate columns dedicated strictly to optical geometric elements to find nearest blockages
        offset = len(self.ends)
        element_distances = step_distances[:, offset:]
        
        if element_distances.shape[1] > 0:
            closest_element_dist = np.min(element_distances, axis=1)
        else:
            closest_element_dist = np.full(batch_size, self.INFINITE, dtype=np.float64)

        # =========================================================================
        # PART 1: END-NODE REACHABILITY + START<->END DIRECT-VIEW EMISSION (WITH OCCLUSION)
        # =========================================================================
        # FRAMEWORK RULE - two distinct concepts are evaluated separately here:
        #
        #   end_reached   - the traced ray terminates on an `ends` node with no optical
        #                   element interposed (occlusion-checked). This proves PATH
        #                   CONNECTIVITY only: it terminates the ray at this vertex and, on
        #                   unwinding, validates the deferred reflection/transmission shader
        #                   commits upstream.
        #
        #   emission_call - the emission model is scheduled IF AND ONLY IF the `starts` node
        #                   and the `ends` node are in direct view of ONE ANOTHER, i.e. that
        #                   unobstructed line of sight exists at the START vertex itself.
        #                   An `ends` node that is in direct view only at the terminal point
        #                   of a reflection and/or transmission chain is NOT sufficient to
        #                   trigger the emission model. `starts`/`ends` are deliberately
        #                   direction-agnostic; it is the developer's shader that assigns the
        #                   semantics, and this rule is the single consistency contract.
        #
        # MOCK_step == 0 is the structural marker of the START vertex: `raytrace` enters with
        # MOCK_step=0 and every recursion passes MOCK_step+1, so MOCK_step > 0 means this
        # vertex was created by a reflection/transmission extension.
        for idx, source_node in enumerate(self.ends):
            source_distance = step_distances[:, idx]
            
            # PHYSICAL OCCLUSION CRITERIA:
            # The light source is only in direct view if it is closer than any optical element!
            end_reached = alive_mask & (source_distance < self.INFINITE) & (source_distance <= closest_element_dist)
            
            # START<->END DIRECT-VIEW RULE: emission is eligible only at the START vertex.
            emission_call = end_reached & (MOCK_step == 0)
            
            if np.any(emission_call):
                self.tracker.log_state(emission_call, "CONTEXT", path_id, source_node.node_id, f"query_emission_from_{medium_id}")
                self.tracker.log_state(emission_call, "SHADERS", path_id, "SCENE", f"schedule_emission_shader_from_{medium_id}")
                
                # Directly accumulate verified incoming light into the ray state
                mock_source_energy = Vector(np.zeros(batch_size), np.zeros(batch_size), np.zeros(batch_size))
                mock_source_energy.x[emission_call] = 5.0
                mock_source_energy.y[emission_call] = 5.0
                mock_source_energy.z[emission_call] = 5.0
                rays += mock_source_energy.place(emission_call)
            
            # Mark this path track slice as historically validated (path connectivity, not
            # emission). This is what gates the deferred shader commits during the unwinding.
            if np.any(end_reached):
                hit_source_downstream[end_reached] = True

        # =========================================================================
        # PART 2: DEFERRED SECONDARY SCATTER PASS
        # =========================================================================
        # Prevent rays that cleanly hit a source from also escaping to infinity or hitting elements
        still_alive = alive_mask & ~hit_source_downstream
        
        if element_distances.shape[1] == 0 or not np.any(still_alive):
            return rays, hit_source_downstream

        minimum_distances = closest_element_dist
        
        # Handle elements escaping to infinity (only for rays that didn't hit a source)
        escaped_mask = still_alive & (minimum_distances == self.INFINITE)
        if np.any(escaped_mask):
            self.tracker.log_state(escaped_mask, "TERMINATION", path_id, "INFINITY", f"ray_escaped_at_step_{MOCK_step}")

        for idx, element in enumerate(self.elements):
            distance = element_distances[:, idx]
            hit = still_alive & (minimum_distances != self.INFINITE) & (distance == minimum_distances)

            if not np.any(hit):
                continue

            is_exiting_this_element = (current_medium is not None and current_medium.node_id == element.node_id)

            # -----------------------------------------------------------------
            # BRANCH 1: THE REFLECTION FORK (Always occurs on geometric surfaces)
            # -----------------------------------------------------------------
            can_bounce = hit & (bounces < self.bounce_count)
            bounce_blocked = hit & (bounces >= self.bounce_count)
            
            if np.any(bounce_blocked):
                self.tracker.log_state(bounce_blocked, "TERMINATION", path_id, element.node_id, "bounce_budget_exhausted")

            if np.any(can_bounce):
                if is_exiting_this_element:
                    # Internal reflection: ray stays INSIDE the current volume medium
                    branch_path_id_reflect = f"{path_id}➔R_INT({element.node_id})"
                    log_msg_reflect = "internal_wall_bounce"
                    next_medium_stack_reflect = medium_stack 
                else:
                    # External reflection: ray bounces off outside, remains in parent medium context
                    branch_path_id_reflect = f"{path_id}➔R_EXT({element.node_id})"
                    log_msg_reflect = "external_surface_bounce"
                    next_medium_stack_reflect = medium_stack

                next_bounces = bounces.copy()
                next_bounces[can_bounce] += 1

                # Recursive descent down reflection trajectory fork
                reflected_ray_data, reflect_hit_source = self._reverse_recursive_path_trace(
                    detector, origin, direction, next_bounces, transmission_depths,
                    next_medium_stack_reflect, MOCK_distances_timeline, MOCK_step + 1, 
                    active_mask=can_bounce, path_id=branch_path_id_reflect
                )

                # DEFERRED SHADING VALIDATION: Verify if downstream paths ever connect to light
                valid_reflection_shader = can_bounce & reflect_hit_source
                if np.any(valid_reflection_shader):
                    self.tracker.log_state(valid_reflection_shader, "INTERSECT", path_id, element.node_id, log_msg_reflect)
                    self.tracker.log_state(valid_reflection_shader, "SHADERS", path_id, element.node_id, "schedule_reflection_shader")
                    
                    rays += reflected_ray_data.place(valid_reflection_shader)
                    hit_source_downstream[valid_reflection_shader] = True

            # -----------------------------------------------------------------
            # BRANCH 2: THE TRANSMISSION FORK (Only occurs if element is transparent)
            # -----------------------------------------------------------------
            if element.transparent:
                can_transmit = hit & (transmission_depths < self.max_transmission_depth)
                transmit_blocked = hit & (transmission_depths >= self.max_transmission_depth)
                
                if np.any(transmit_blocked):
                    self.tracker.log_state(transmit_blocked, "TERMINATION", path_id, element.node_id, "transmit_budget_exhausted")

                if np.any(can_transmit):
                    if is_exiting_this_element:
                        # Exiting volume: pop current element off the medium stack
                        next_medium_stack_transmit = medium_stack[:-1]
                        dest_id = next_medium_stack_transmit[-1].node_id if next_medium_stack_transmit else "AIR"
                        branch_path_id_transmit = f"{path_id}➔T_OUT({element.node_id})"
                        log_msg_transmit = f"transmit_out_to_{dest_id}"
                    else:
                        # Entering volume: push new element onto the medium stack
                        next_medium_stack_transmit = medium_stack + (element,)
                        branch_path_id_transmit = f"{path_id}➔T_IN({element.node_id})"
                        log_msg_transmit = f"transmit_in_from_{medium_id}"

                    next_transmissions = transmission_depths.copy()
                    next_transmissions[can_transmit] += 1
                    
                    # Recursive descent down transmission trajectory fork
                    transmitted_ray_data, trans_hit_source = self._reverse_recursive_path_trace(
                        detector, origin, direction, bounces, next_transmissions,
                        next_medium_stack_transmit, MOCK_distances_timeline, MOCK_step + 1, 
                        active_mask=can_transmit, path_id=branch_path_id_transmit
                    )
                    
                    # DEFERRED SHADING VALIDATION: Verify if downstream paths ever connect to light
                    valid_transmission_shader = can_transmit & trans_hit_source
                    if np.any(valid_transmission_shader):
                        self.tracker.log_state(valid_transmission_shader, "INTERSECT", path_id, element.node_id, log_msg_transmit)
                        self.tracker.log_state(valid_transmission_shader, "SHADERS", path_id, element.node_id, "schedule_transmission_shader")
                        
                        rays += transmitted_ray_data.place(valid_transmission_shader)
                        hit_source_downstream[valid_transmission_shader] = True

        return rays, hit_source_downstream

    def _validate_timeline_shape(self, batch_size: int, MOCK_distances_timeline: List[np.ndarray]) -> None:
        """
        Enforces the mock-timeline contract at entry (called first from `raytrace`): every step
        matrix is (batch_size ROWS = ray lanes) x (len(ends) + len(elements) COLUMNS =
        [ends... | elements...], in constructor order).

        This guard exists because the lane count is set by `batch_size` - it sizes the tracker and
        every mask - NOT by the matrix. A matrix with FEWER rows than `batch_size` is silently
        broadcast by NumPy across the lanes: deleting a lane's row then changes nothing observable,
        and a test keeps passing against a history that still has `batch_size` lanes. A matrix with
        MORE rows crashes later, far from the cause. Fail fast, at the source, naming the step.
        """
        expected_cols = len(self.ends) + len(self.elements)
        for step, matrix in enumerate(MOCK_distances_timeline):
            if matrix.ndim != 2:
                raise ValueError(
                    f"MOCK_distances_timeline[{step}]: expected a 2-D matrix "
                    f"(rows = ray lanes, cols = [ends | elements]); got ndim={matrix.ndim}"
                )
            rows, cols = matrix.shape
            if rows != batch_size:
                raise ValueError(
                    f"MOCK_distances_timeline[{step}]: {rows} row(s) but batch_size={batch_size}. "
                    f"Rows are ray lanes - one row per lane, per step. A short matrix is silently "
                    f"broadcast across lanes by NumPy; a long one indexes past the tracker."
                )
            if cols != expected_cols:
                raise ValueError(
                    f"MOCK_distances_timeline[{step}]: {cols} column(s) but "
                    f"len(ends) + len(elements) = {expected_cols}. Columns are "
                    f"[ends... | elements...] in constructor order."
                )
