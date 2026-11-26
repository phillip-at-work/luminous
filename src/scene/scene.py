from functools import reduce
import numpy as np
from numpy.typing import NDArray
import time
import atexit
import copy
from abc import ABC

from ..math.vector import Vector
from ..math.tools import extract
from ..element.detector import Detector
from ..element.element import Element
from ..element.source import Source
from ..utilities.ray_debugger import NullRayDebugger, ConcreteRayDebugger

from ..utilities.logconfig import (
    setup_logging,
    trace,
    extend_path,
    most_recent_origin,
    read_log_table,
    ORIGIN,
    LEG,
    CONTEXT,
    INTERSECT,
    SHADERS,
    TERMINATION,
    R_EXT,
    R_INT,
    T_IN,
    T_OUT,
    INFINITY_TARGET,
)
import logging
logger = logging.getLogger(__name__)

INFINITE = 1.0e39
BOUNCE_COUNT = 10
MAX_TRANSMISSION_DEPTH = 50

class Scene:

    def __init__(self, reverse_trace=True, index_of_refraction=1, log_level=logging.CRITICAL, log_file="luminous.log", standoff_distance=1e-5) -> None:
        setup_logging(level=log_level, log_file=log_file)
        self.log_file = log_file
        self.start_time = None
        self.refractive_index = index_of_refraction
        self.REVERSE_TRACE = reverse_trace
        self.elements = list()
        self.detectors = list()
        self.sources = list()
        self.elements_and_detectors = list()
        self.intersection_map = list()
        self.ray_debugger_reverse = NullRayDebugger()
        self.ray_debugger_forward = NullRayDebugger()
        self.counter = 0
        self.standoff_distance = standoff_distance

    def show_trace(self, types=None) -> str:
        """Print this scene's trace table to the terminal, and return the string."""
        table = read_log_table(self.log_file, types=types)
        print(table)
        return table

    def elaspsed_time(self):
        if self.start_time is None:
            raise ValueError('Timer has not been started. Call `elapsted_time` after calling `raytrace`.')
        return time.perf_counter() - self.start_time

    def attach_ray_debugger(self, path="./results", filename="debug_ray_trace", display_3d_plot=True):
        self.ray_debugger_reverse = ConcreteRayDebugger()
        self.ray_debugger_forward = ConcreteRayDebugger()

        def plot_ray_trace_wrap_user_args():
            self.ray_debugger_reverse.plot(path=path, filename=filename, display_3d_plot=display_3d_plot)
            self.ray_debugger_forward.plot(path=path, filename=filename, display_3d_plot=display_3d_plot)
        atexit.register(plot_ray_trace_wrap_user_args)

    def __iadd__(self, obj):

        self._check_identity(obj)

        if isinstance(obj, Element):
            self.elements.append(obj)
            if self.REVERSE_TRACE:
                self.ray_debugger_reverse.add_element(obj, color=(255,215,0)) # scene elements (gold)
            else:
                self.ray_debugger_forward.add_element(obj, color=(255,215,0)) # scene elements (gold)
        elif isinstance(obj, Source):
            self.sources.append(obj)
        elif isinstance(obj, Detector):
            self.detectors.append(obj)
        else:
            raise TypeError("Only Elements, Sources, and Detectors can be added to Scenes!")

        return self

    def _check_identity(self, obj) -> None:
        """Reject a node this Scene cannot name, or has already named twice."""
        if not hasattr(obj, 'node_id'):
            return

        node_id = obj.node_id
        if node_id is None:
            raise ValueError(
                f"{type(obj).__name__} has no `node_id`; a blank id erases the `from` and "
                "`target` cells of every record it appears in, so no ray can be traced back to it"
            )

        for existing in self.elements + self.sources + self.detectors:
            if existing.node_id == node_id:
                raise ValueError(
                    f"node_id {node_id!r} is already used by a {type(existing).__name__} "
                    "in this Scene; trace records name nodes by id, so ids must be unique"
                )

    def raytrace(self):
        self.start_time = time.perf_counter()

        self.elements_and_detectors = self.elements + self.detectors

        # reverse ray trace, from detector to source
        if self.REVERSE_TRACE:

            for source in self.sources:

                # ray debugger
                self.ray_debugger_reverse.add_element(source, color=(0,50,150))

            for detector in self.detectors:

                detector.pixels = detector.create_screen_coord()
                pixel_rays = detector._compute_initial_ray_directions(detector.pixels)

                # ray debugger
                detector_pointing_dir: Vector = detector.position + detector.pointing_direction
                self.ray_debugger_reverse.add_point(detector.position, color=(0, 255, 0))
                self.ray_debugger_reverse.add_vector(start_point=detector.position, end_point=detector_pointing_dir, color=(255, 0, 0))
                self.ray_debugger_reverse.add_point(detector.pixels, color=(255,0,0))

                detector_label = detector.node_id
                trace(logger, ORIGIN, detector_label, detector.node_id, "init_trace", n=pixel_rays.x.size)
                detector._reverse_trace_data = self._reverse_recursive_path_trace(detector=detector, origin=detector.pixels, direction=pixel_rays, bounce=0, transmission_depth=0, recursion_enum="START", path=detector_label)

        # forward ray trace, from source to detector
        if not self.REVERSE_TRACE:

            # TODO
            # intersection-finding in path trace should detect SceneElements and Detectors
            # making it unnecessary to have a nested loop
            # would instead iterate over sources only.
            # probably still iterate over detectors once just to plot ray debugging

            for source in self.sources:

                source.pixels = source.create_screen_coord()
                pixel_rays = source._compute_initial_ray_directions(source.pixels)

                # ray debugger
                self.ray_debugger_reverse.add_point(source.pixels, color=(255,0,0))
            
                for detector in self.detectors:

                    detector.compute_surface_definition()
                    detector.pixels = source.pixels

                    # ray debugger
                    source_pointing_dir: Vector = source.position + source.pointing_direction
                    self.ray_debugger_reverse.add_point(source.position, color=(0, 255, 100))
                    self.ray_debugger_reverse.add_vector(start_point=source.position, end_point=source_pointing_dir, color=(255, 100, 0))
                    self.ray_debugger_reverse.add_point(source.pixels, color=(255,100,0))
                    self.ray_debugger_reverse.add_element(detector, color=(0,50,150))

                    source_label = source.node_id
                    trace(logger, ORIGIN, source_label, detector.node_id, "init_trace", n=pixel_rays.x.size)
                    self._forward_recursive_path_trace(detector=detector, origin=source.pixels, direction=pixel_rays, bounce=0, recursion_enum="START", source=source, path=source_label)

    def _forward_recursive_path_trace(self, detector: Detector, origin: Vector, direction: Vector, bounce: int, recursion_enum: str, source: Source = None, path: str = ""):

            self.counter += 1
            from_label = most_recent_origin(path)
            trace(logger, LEG, from_label, recursion_enum, "forward_leg", hop=self.counter, b=bounce, path=path)

            #
            # detection
            #

            # if not self.REVERSE_TRACE:

            #     for detector in self.detectors:
                    
            #         # for reverse traces, check to see if ray `origin` is in direct view of a source.

            #         direction_to_detector: Vector = source.center - origin
            #         direction_to_source_unit: Vector = direction_to_detector.norm()
            #         intersections_blocking_source: list[NDArray[np.float64]] = [element.intersect(origin, direction_to_source_unit) for element in self.elements]
            #         minimum_distances_with_standoff: NDArray[np.float64] = reduce(np.minimum, intersections_blocking_source)
            #         distances_to_source = direction_to_detector.magnitude()
            #         intersection_point_illuminated: NDArray[np.bool_] = minimum_distances_with_standoff >= distances_to_source

            #         if recursion_enum == 'START':
            #             ray_intersects_detector_surface = detector.pointing_direction.dot(direction_to_detector) > 0
            #             intersection_point_illuminated = intersection_point_illuminated & ray_intersects_detector_surface

            #         if np.sum(intersection_point_illuminated) < 1:
            #             continue

            #         direction_to_source_minima: Vector = direction_to_detector.extract(intersection_point_illuminated)
            #         origin_point_illuminated: Vector = origin.extract(intersection_point_illuminated)
            #         intersection_to_source: Vector = origin_point_illuminated + direction_to_source_minima

            #         self.ray_debugger_reverse.add_vector(start_point=origin_point_illuminated, end_point=intersection_to_source, color=(255,0,255)) # to sources

            #         self.intersection_map.append({'source': source, 'direction_to_source_unit': direction_to_source_unit, 'intersection_point_illuminated': intersection_point_illuminated})

            #         # # store rays for forward tracing
            #         # r = (origin_point_illuminated - intersection_to_source).norm()
            #         # source._enqueue_rays(origin=intersection_to_source, direction=r, detector=detector)

            #
            # reflections and transmissions
            #

            # TODO _reflection_model operates on the detector.pixels data structure directly
            # the boolean array indicating detector intersections will simply mask this data structure
            # and then detector.view_data will do the final integration
            # but this intersection logic will probably also need to independently record the ray trajectories?

            # TODO or I guess I can just do [detector.intersect(origin, direction) for detector in self.detectors]
            # does this suggeset I don't need a nested for loop `for source... for detector`?
            # I think I can avoid the nested for loop if I can know what type of detector I'm pointed at, so I can call the correct detector
            # but I suppose the expense of calling ~1-20 detectors is less than the ugly code it would generate here?
            # also - how can I allow to detectors to occupy the same space or allow rays to pass through detectors to hit other detectors

            # TODO conduct this calculation with self.elements + self.detectors?
            # possible to distinguish between object type SceneElement versus Detector?
            # if so, process `hit` as `hit_element` and then `hit_detector`?


            distances: list[NDArray[np.float64]] = [item.intersect(origin, direction) for item in self.elements_and_detectors]
            minimum_distances: NDArray[np.float64] = reduce(np.minimum, distances)
            
            for item, distance in zip(self.elements_and_detectors, distances):

                hit: NDArray[np.bool_] = (minimum_distances != INFINITE) & (distance == minimum_distances)

                # hit_detector: NDArray[np.bool_] = (minimum_distances != INFINITE) & (distance == minimum_distances) & (item is Detector)
                # hit_element: NDArray[np.bool_] = (minimum_distances != INFINITE) & (distance == minimum_distances) & (item is SceneElement)

                # if np.any(hit_detector):
                    # new intersection logic for detector
                # if np.any(hit_element):
                    # existing logic

                if np.any(hit):

                    start_point: Vector = origin.extract(hit)
                    incident_ray: Vector = direction.extract(hit)
                    ray_travel_distance: NDArray[np.float64] = extract(hit, distance)

                    intersection_point: Vector = start_point + incident_ray * ray_travel_distance
                
                    #
                    # transmission from volume
                    #

                    if recursion_enum in ['TRANS-IN', 'SUBSURF-REFLECT'] and bounce < BOUNCE_COUNT:

                        surface_normal_at_intersection: Vector = element.compute_inward_normal(intersection_point)
                        intersection_point_with_standoff: Vector = intersection_point - surface_normal_at_intersection * self.standoff_distance
                        
                        transmitted_ray = self._transmitted_ray(
                                incident_ray,
                                surface_normal_at_intersection,
                                element.refractive_index,
                                self.refractive_index
                            )
                        
                        trace(logger, INTERSECT, from_label, item.node_id, T_OUT, hop=self.counter, b=bounce, mask=hit)

                        self._forward_recursive_path_trace(detector, intersection_point_with_standoff, transmitted_ray, bounce, recursion_enum="TRANS-OUT", source=source, path=extend_path(path, T_OUT, item.node_id))

                    elif recursion_enum in ['TRANS-IN', 'SUBSURF-REFLECT']:
                        # inside a volume, but the bounce budget is spent: this fork dies here
                        trace(logger, TERMINATION, from_label, item.node_id, "bounces_exhausted", hop=self.counter, b=bounce, mask=hit)

                    #
                    # transmission into volume
                    #

                    if (element.transparent and recursion_enum not in ['TRANS-IN', 'SUBSURF-REFLECT']) and bounce < BOUNCE_COUNT:
                            
                        surface_normal_at_intersection: Vector = element.compute_outward_normal(intersection_point)
                        intersection_point_with_standoff: Vector = intersection_point - surface_normal_at_intersection * self.standoff_distance
                        
                        transmitted_ray = self._transmitted_ray(
                                incident_ray,
                                surface_normal_at_intersection,
                                self.refractive_index,
                                element.refractive_index
                            )
                        
                        trace(logger, INTERSECT, from_label, item.node_id, T_IN, hop=self.counter, b=bounce, mask=hit)

                        self._forward_recursive_path_trace(detector, intersection_point_with_standoff, transmitted_ray, bounce, recursion_enum="TRANS-IN", source=source, path=extend_path(path, T_IN, item.node_id))

                    elif element.transparent and recursion_enum not in ['TRANS-IN', 'SUBSURF-REFLECT']:
                        trace(logger, TERMINATION, from_label, item.node_id, "bounces_exhausted", hop=self.counter, b=bounce, mask=hit)

                    #
                    # surface and subsurface reflection
                    #

                    if bounce < BOUNCE_COUNT:

                        if recursion_enum == 'SUBSURF-REFLECT' or recursion_enum == 'TRANS-IN':

                            if recursion_enum == 'SUBSURF-REFLECT':
                                color = (255,0,0) # subsurface reflected ray (red)
                            elif recursion_enum == 'TRANS-IN':
                                color = (0,255,255) # transmitted ray (cyan)

                            self.ray_debugger_reverse.add_vector(start_point=start_point, end_point=intersection_point, color=color)
                            surface_normal_at_intersection: Vector = element.compute_inward_normal(intersection_point)
                            detector._transmission_model(element, start_point, intersection_point, hit)
                            trace(logger, SHADERS, from_label, item.node_id, "transmission_model", hop=self.counter, b=bounce, mask=hit)

                        else:

                            self.ray_debugger_reverse.add_vector(start_point=start_point, end_point=intersection_point, color=(0,0,255)) # surface reflected ray (blue)
                            surface_normal_at_intersection: Vector = element.compute_outward_normal(intersection_point)

                        intersection_point_with_standoff: Vector = intersection_point + surface_normal_at_intersection * self.standoff_distance
                        direction_to_origin_unit: Vector = (detector.position - intersection_point).norm()

                        detector._reflection_model(element,
                                                    intersection_point,
                                                    surface_normal_at_intersection,
                                                    direction_to_origin_unit,
                                                    self.intersection_map,
                                                    hit)
                        trace(logger, SHADERS, from_label, item.node_id, "reflection_model", hop=self.counter, b=bounce, mask=hit, src=len(self.intersection_map))

                        reflected_ray = self._reflected_ray(incident_ray, surface_normal_at_intersection)

                        traversal = R_INT if recursion_enum in ["TRANS-IN", "SUBSURF-REFLECT"] else R_EXT
                        e = "SUBSURF-REFLECT" if recursion_enum in ["TRANS-IN", "SUBSURF-REFLECT"] else "SURF-REFLECT"
                        trace(logger, INTERSECT, from_label, item.node_id, traversal, hop=self.counter, b=bounce, mask=hit)

                        self._forward_recursive_path_trace(detector, intersection_point_with_standoff, reflected_ray, bounce + 1, recursion_enum=e, source=source, path=extend_path(path, traversal, item.node_id))

                        # NOTE not sure if the existing `transmission from volume` is sufficient to allow rays to transmit and reflect for enum `SUBSURF-REFLECT`

                    else:
                        trace(logger, TERMINATION, from_label, item.node_id, "bounces_exhausted", hop=self.counter, b=bounce, mask=hit)

    def _reverse_recursive_path_trace(self, detector: Detector, origin: Vector, direction: Vector, bounce: int, transmission_depth: int, recursion_enum: str, source: Source = None, path: str = ""):

        self.counter += 1
        from_label = most_recent_origin(path)
        trace(logger, LEG, from_label, recursion_enum, "reverse_leg", hop=self.counter, b=bounce, d=transmission_depth, path=path)

        rays = Vector(0, 0, 0)

        #
        # source illumination
        #

        for source in self.sources:

            # for reverse traces, check to see if ray `origin` is in direct view of a source.

            direction_to_source: Vector = source.center - origin
            direction_to_source_unit: Vector = direction_to_source.norm()
            intersections_blocking_source: list[NDArray[np.float64]] = [element.intersect(origin, direction_to_source_unit) for element in self.elements]
            minimum_distances_with_standoff: NDArray[np.float64] = reduce(np.minimum, intersections_blocking_source)
            distances_to_source = direction_to_source.magnitude()
            intersection_point_illuminated: NDArray[np.bool_] = minimum_distances_with_standoff >= distances_to_source

            if recursion_enum == 'START':
                ray_intersects_detector_surface = detector.pointing_direction.dot(direction_to_source) > 0
                intersection_point_illuminated = intersection_point_illuminated & ray_intersects_detector_surface

            if np.sum(intersection_point_illuminated) < 1:
                continue

            trace(logger, CONTEXT, from_label, source.node_id, "query_visibility", hop=self.counter, b=bounce, d=transmission_depth, mask=intersection_point_illuminated)

            direction_to_source_minima: Vector = direction_to_source.extract(intersection_point_illuminated)
            origin_point_illuminated: Vector = origin.extract(intersection_point_illuminated)
            intersection_to_source: Vector = origin_point_illuminated + direction_to_source_minima

            self.ray_debugger_reverse.add_vector(start_point=origin_point_illuminated, end_point=intersection_to_source, color=(255,0,255)) # to sources

            self.intersection_map.append({'source': source, 'direction_to_source_unit': direction_to_source_unit, 'intersection_point_illuminated': intersection_point_illuminated})

            # # store rays for forward tracing
            # r = (origin_point_illuminated - intersection_to_source).norm()
            # source._enqueue_rays(origin=intersection_to_source, direction=r, detector=detector)

        ray_data = detector._emission_model(detector.pointing_direction, self.intersection_map)
        trace(logger, SHADERS, from_label, detector.node_id, "emission_model", hop=self.counter, b=bounce, d=transmission_depth, src=len(self.intersection_map))
        rays += ray_data.place(intersection_point_illuminated)

        self.intersection_map.clear()

        #
        # reflections and transmissions
        #

        distances: list[NDArray[np.float64]] = [element.intersect(origin, direction) for element in self.elements]
        minimum_distances: NDArray[np.float64] = reduce(np.minimum, distances)

        # a leg with no geometry in front of it escapes to infinity and stops here
        escaped_rays: NDArray[np.bool_] = minimum_distances == INFINITE
        if np.any(escaped_rays):
            trace(logger, TERMINATION, from_label, INFINITY_TARGET, "ray_escaped", hop=self.counter, b=bounce, d=transmission_depth, mask=escaped_rays)
        
        for element, distance in zip(self.elements, distances):

            hit: NDArray[np.bool_] = (minimum_distances != INFINITE) & (distance == minimum_distances)

            if np.any(hit):

                start_point: Vector = origin.extract(hit)
                incident_ray: Vector = direction.extract(hit)
                ray_travel_distance: NDArray[np.float64] = extract(hit, distance)

                intersection_point: Vector = start_point + incident_ray * ray_travel_distance
            
                #
                # transmission from volume
                #

                if recursion_enum in ['TRANS-IN', 'SUBSURF-REFLECT'] and transmission_depth < MAX_TRANSMISSION_DEPTH:

                    surface_normal_at_intersection: Vector = element.compute_inward_normal(intersection_point)
                    intersection_point_with_standoff: Vector = intersection_point - surface_normal_at_intersection * self.standoff_distance
                    
                    transmitted_ray = self._transmitted_ray(
                            incident_ray,
                            surface_normal_at_intersection,
                            element.refractive_index,
                            self.refractive_index
                        )
                    
                    trace(logger, INTERSECT, from_label, element.node_id, T_OUT, hop=self.counter, b=bounce, d=transmission_depth, mask=hit)

                    ray_data = self._reverse_recursive_path_trace(detector, intersection_point_with_standoff, transmitted_ray, bounce, transmission_depth + 1, recursion_enum="TRANS-OUT", source=source, path=extend_path(path, T_OUT, element.node_id))
                    rays += ray_data.place(hit)

                elif recursion_enum in ['TRANS-IN', 'SUBSURF-REFLECT']:
                    # inside a volume, but the transmission budget is spent: this fork dies here
                    trace(logger, TERMINATION, from_label, element.node_id, "transmit_budget_exhausted", hop=self.counter, b=bounce, d=transmission_depth, mask=hit)

                #
                # transmission into volume
                #

                if (element.transparent and recursion_enum not in ['TRANS-IN', 'SUBSURF-REFLECT']) and transmission_depth < MAX_TRANSMISSION_DEPTH:
                        
                    surface_normal_at_intersection: Vector = element.compute_outward_normal(intersection_point)
                    intersection_point_with_standoff: Vector = intersection_point - surface_normal_at_intersection * self.standoff_distance
                    
                    transmitted_ray = self._transmitted_ray(
                            incident_ray,
                            surface_normal_at_intersection,
                            self.refractive_index,
                            element.refractive_index
                        )

                    trace(logger, INTERSECT, from_label, element.node_id, T_IN, hop=self.counter, b=bounce, d=transmission_depth, mask=hit)

                    ray_data = self._reverse_recursive_path_trace(detector, intersection_point_with_standoff, transmitted_ray, bounce, transmission_depth + 1, recursion_enum="TRANS-IN", source=source, path=extend_path(path, T_IN, element.node_id))
                    rays += ray_data.place(hit)

                elif element.transparent and recursion_enum not in ['TRANS-IN', 'SUBSURF-REFLECT']:
                    # transparent, but the transmission budget is spent: this fork dies here
                    trace(logger, TERMINATION, from_label, element.node_id, "transmit_budget_exhausted", hop=self.counter, b=bounce, d=transmission_depth, mask=hit)

                #
                # surface and subsurface reflection
                #

                if recursion_enum == 'SUBSURF-REFLECT' or recursion_enum == 'TRANS-IN':

                    if recursion_enum == 'SUBSURF-REFLECT':
                        color = (255,0,0) # subsurface reflected ray (red)
                    elif recursion_enum == 'TRANS-IN':
                        color = (0,255,255) # transmitted ray (cyan)

                    self.ray_debugger_reverse.add_vector(start_point=start_point, end_point=intersection_point, color=color)
                    surface_normal_at_intersection: Vector = element.compute_inward_normal(intersection_point)
                    ray_data = detector._transmission_model(element, start_point, intersection_point, hit)
                    trace(logger, SHADERS, from_label, element.node_id, "transmission_model", hop=self.counter, b=bounce, d=transmission_depth, mask=hit)
                    rays += ray_data.place(hit)

                else:

                    self.ray_debugger_reverse.add_vector(start_point=start_point, end_point=intersection_point, color=(0,0,255)) # surface reflected ray (blue)
                    surface_normal_at_intersection: Vector = element.compute_outward_normal(intersection_point)

                # Always handle reflections when within bounce limit
                if bounce < BOUNCE_COUNT:
                    
                    intersection_point_with_standoff: Vector = intersection_point + surface_normal_at_intersection * self.standoff_distance
                    direction_to_origin_unit: Vector = (detector.position - intersection_point).norm()
                    
                    e = "SUBSURF-REFLECT" if recursion_enum in ["TRANS-IN", "SUBSURF-REFLECT"] else "SURF-REFLECT"
                    
                    # First, recursively trace reflected ray to gather light from bounces
                    traversal = R_INT if recursion_enum in ["TRANS-IN", "SUBSURF-REFLECT"] else R_EXT
                    trace(logger, INTERSECT, from_label, element.node_id, traversal, hop=self.counter, b=bounce, d=transmission_depth, mask=hit)
                    reflected_ray = self._reflected_ray(incident_ray, surface_normal_at_intersection)
                    reflected_ray_data = self._reverse_recursive_path_trace(detector, intersection_point_with_standoff, reflected_ray, bounce + 1, transmission_depth, recursion_enum=e, source=source, path=extend_path(path, traversal, element.node_id))
                    
                    # Second, check for direct illumination from sources at this surface
                    for source in self.sources:

                        # only call reflection model if the point of reflection is illuminated. otherwise cast a shadow.

                        direction_to_source: Vector = source.center - intersection_point
                        direction_to_source_unit: Vector = direction_to_source.norm()
                        intersections_blocking_source: list[NDArray[np.float64]] = [element.intersect(intersection_point_with_standoff, direction_to_source_unit) for element in self.elements]
                        minimum_distances_with_standoff: NDArray[np.float64] = reduce(np.minimum, intersections_blocking_source)
                        distances_to_source = direction_to_source.magnitude()
                        intersection_point_illuminated: NDArray[np.bool_] = minimum_distances_with_standoff >= distances_to_source

                        if np.sum(intersection_point_illuminated) < 1:
                            continue

                        # the leg departs this surface on a visibility query, so `from:` is the
                        # element being shaded: this record is the shadow test's pass/fail marker
                        trace(logger, CONTEXT, element.node_id, source.node_id, "query_visibility", hop=self.counter, b=bounce, d=transmission_depth, mask=intersection_point_illuminated)

                        self.intersection_map.append({'source': source, 'direction_to_source_unit': direction_to_source_unit, 'intersection_point_illuminated': intersection_point_illuminated})

                    # Apply reflection model for direct illumination (if any sources visible)
                    direct_illum_data = detector._reflection_model(element,
                                                            intersection_point,
                                                            surface_normal_at_intersection,
                                                            direction_to_origin_unit,
                                                            self.intersection_map,
                                                            hit)
                    trace(logger, SHADERS, from_label, element.node_id, "reflection_model", hop=self.counter, b=bounce, d=transmission_depth, mask=hit, src=len(self.intersection_map))
                    
                    # Combine both reflected ray contribution and direct illumination
                    rays += reflected_ray_data.place(hit)
                    rays += direct_illum_data.place(hit)
                                
                    self.intersection_map.clear()

                    # NOTE not sure if the existing `transmission from volume` is sufficient to allow rays to transmit and reflect for enum `SUBSURF-REFLECT`

                else:
                    # the bounce budget is spent at this surface, so this leg ends here
                    trace(logger, TERMINATION, from_label, element.node_id, "bounces_exhausted", hop=self.counter, b=bounce, d=transmission_depth, mask=hit)

        return rays

    def _total_internal_reflection(self, incident_ray: Vector, surface_normal: Vector, n1: float, n2: float) -> bool:
        # Determine if incident angle >= critical angle
        # REF: [de Greve, 2006, Reflections and Refractions in Ray Tracing]
        if n2 >= n1:
            return np.zeros(incident_ray.x.shape, dtype=bool)
        
        cos_theta_i = -incident_ray.dot(surface_normal)
        theta_i = np.arccos(np.clip(cos_theta_i, -1.0, 1.0))
        critical_angle = np.arcsin(np.clip(n2 / n1, -1.0, 1.0))
        
        return theta_i > critical_angle

    def _reflected_ray(self, incident_ray: Vector, surface_normal_at_intersection: Vector) -> Vector:
        # Reflected ray unit vector
        # REF: [de Greve, 2006, Reflections and Refractions in Ray Tracing]
        return incident_ray - surface_normal_at_intersection * 2 * incident_ray.dot(surface_normal_at_intersection)
    
    def _transmitted_ray(self, incident_ray: Vector, surface_normal: Vector, n1: float, n2: float) -> Vector:
        # Refracted ray unit vector
        # REF: [de Greve, 2006, Reflections and Refractions in Ray Tracing]
        n = n1 / n2
        cos_theta_i = -incident_ray.dot(surface_normal)
        sin_theta_t_squared = (n ** 2) * (1 - cos_theta_i**2)
        term1 = n * incident_ray
        term2 = n * cos_theta_i - np.sqrt(1 - sin_theta_t_squared)
        term3 = term2 * surface_normal
        transmitted_ray = term3 + term1
        return transmitted_ray.norm()
    
    def _reflection_transmission_weights(self, incident_ray: Vector, surface_normal: Vector, n1: float, n2: float):
        # Ray weightings for reflected and refracted components assuming unpolarized source (fresnel equations)
        # REF: [de Greve, 2006, Reflections and Refractions in Ray Tracing]
        n = n1 / n2
        cos_theta_i = -incident_ray.dot(surface_normal)
        sin_theta_t_squared = (n ** 2) * (1 - cos_theta_i**2)
        cos_theta_t = np.sqrt(1 - sin_theta_t_squared)
        perpendicular_component = ((n1 * cos_theta_i - n2 * cos_theta_t) / (n1 * cos_theta_i + n2 * cos_theta_t)) ** 2
        parallel_component = ((n2 * cos_theta_i - n1 * cos_theta_t) / (n2 * cos_theta_i + n1 * cos_theta_t)) ** 2
        r = (perpendicular_component + parallel_component) / 2
        t = 1 - r
        return r.astype(np.float32), t.astype(np.float32)
    
    def _critical_angle(self, n1, n2):
        return np.arcsin(n2 / n1)
    
    