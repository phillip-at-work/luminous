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
from luminous.src.utilities.ray_debugger import NullRayDebugger, ConcreteRayDebugger

from luminous.src.utilities.logconfig import setup_logging
import logging
logger = logging.getLogger(__name__)

INFINITE = 1.0e39
BOUNCE_COUNT = 2

class Scene:

    def __init__(self, reverse_trace=True, index_of_refraction=1, log_level=logging.CRITICAL, log_file="luminous.log", standoff_distance=1e-5) -> None:
        setup_logging(name='luminous', level=log_level, log_file=log_file)
        self.start_time = None
        self.refractive_index = index_of_refraction
        self.reverse_trace = reverse_trace
        self.elements = list()
        self.detectors = list()
        self.sources = list()
        self.intersection_map = list()
        self.ray_debugger_reverse = NullRayDebugger()
        self.ray_debugger_forward = NullRayDebugger()
        self.counter = 0
        self.standoff_distance = standoff_distance

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
        if isinstance(obj, Element):
            self.elements.append(obj)
            if self.reverse_trace:
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

    def raytrace(self):
        self.start_time = time.perf_counter()

        # reverse ray trace, from detector to source
        if self.reverse_trace:

            for source in self.sources:
                
                # ray debugger
                source.surface = source.compute_surface_definition()
                # self.ray_debugger_reverse.add_source(source.surface, color=(0,50,150)) # TODO some issue with this

            for detector in self.detectors:

                detector.pixels = detector.create_screen_coord()
                detector.surface = detector.compute_surface_definition()
                pixel_rays = detector._compute_initial_ray_directions(detector.pixels)

                # ray debugger
                detector_pointing_dir: Vector = detector.position + detector.pointing_direction
                self.ray_debugger_reverse.add_point(detector.position, color=(0, 255, 0))
                self.ray_debugger_reverse.add_vector(start_point=detector.position, end_point=detector_pointing_dir, color=(255, 0, 0))
                self.ray_debugger_reverse.add_point(detector.pixels, color=(255,0,0))

                logger.debug(f"REVERSE TRACE")
                ray_within_volume = {e: False for e in self.elements}
                detector._reverse_trace_data = self._reverse_recursive_trace(detector=detector, origin=detector.pixels, direction=pixel_rays, bounce=0, recursion_enum="START", ray_within_volume=ray_within_volume)

        # forward ray trace, from source to detector
        if not self.reverse_trace:

            for source in self.sources:

                source.pixels = source.create_screen_coord()
                source.surface = source.compute_surface_definition()
                pixel_rays = source._compute_initial_ray_directions(source.pixels)

                # ray debugger
                self.ray_debugger_reverse.add_source(source.surface, color=(0,50,150))
            
                for detector in self.detectors:

                    # ray debugger
                    source_pointing_dir: Vector = source.position + source.pointing_direction
                    self.ray_debugger_reverse.add_point(source.position, color=(0, 255, 100))
                    self.ray_debugger_reverse.add_vector(start_point=source.position, end_point=source_pointing_dir, color=(255, 100, 0))
                    self.ray_debugger_reverse.add_point(source.pixels, color=(255,100,0))

                    logger.debug(f"FORWARD TRACE")
                    ray_within_volume = {e: False for e in self.elements}
                    detector._forward_trace_data = self._forward_recursive_trace(source=source, detector=detector, origin=source.ray_emission_origin[detector], direction=pixel_rays, bounce=0, recursion_enum="START", ray_within_volume=ray_within_volume)

    def _forward_recursive_trace(self, source: Source, detector: Detector, origin: Vector, direction: Vector, bounce: int, recursion_enum: str, ray_within_volume: dict):
        
        self.counter += 1
        logger.debug(f"counter={self.counter}. FORWARD. enum={recursion_enum}")

        rays = Vector(0, 0, 0)

        # #
        # # source illumination
        # #

        # for source in self.sources:

        #     direction_to_source: Vector = source.center - origin
        #     direction_to_source_unit: Vector = direction_to_source.norm()
        #     intersections_blocking_source: list[NDArray[np.float64]] = [element.intersect(origin, direction_to_source_unit) for element in self.elements]
        #     minimum_distances_with_standoff: NDArray[np.float64] = reduce(np.minimum, intersections_blocking_source)
        #     distances_to_source = direction_to_source.magnitude()
        #     intersection_point_illuminated: NDArray[np.bool_] = minimum_distances_with_standoff >= distances_to_source

        #     if recursion_enum == 'START':
        #         ray_intersects_detector_surface = detector.pointing_direction.dot(direction_to_source) > 0
        #         intersection_point_illuminated = intersection_point_illuminated & ray_intersects_detector_surface

        #     if np.sum(intersection_point_illuminated) < 1:
        #         continue

        #     direction_to_source_minima: Vector = direction_to_source.extract(intersection_point_illuminated)
        #     origin_point_illuminated: Vector = origin.extract(intersection_point_illuminated)
        #     intersection_to_source: Vector = origin_point_illuminated + direction_to_source_minima

        #     self.ray_debugger_reverse.add_vector(start_point=origin_point_illuminated, end_point=intersection_to_source, color=(255,0,255)) # to sources

        #     self.intersection_map.append({'source': source, 'direction_to_source_unit': direction_to_source_unit, 'intersection_point_illuminated': intersection_point_illuminated})

        #     ray_data = detector._emission_model(detector.pointing_direction, self.intersection_map)
        #     rays += ray_data.place(intersection_point_illuminated)
                        
        #     self.intersection_map.clear()

        #     # store rays for forward tracing
        #     r = (origin_point_illuminated - intersection_to_source).norm()
        #     source._enqueue_rays(origin=intersection_to_source, direction=r, detector=detector)
        #     # TODO add abstract method to "filter" rays based upon emission characteristics

        #
        # detection
        #

        intersection_point_with_standoff: Vector = origin + direction * 0.001

        for detector in self.detectors:
            pass

        #
        # reflections and transmissions
        #

        distances: list[NDArray[np.float64]] = [element.intersect(origin, direction) for element in self.elements]
        minimum_distances: NDArray[np.float64] = reduce(np.minimum, distances)
        
        for element, distance in zip(self.elements, distances):

            logger.debug(f"element-distance iteration. counter={self.counter}. ray submerged={ray_within_volume[element]}. current enum={recursion_enum}")

            hit: NDArray[np.bool_] = (minimum_distances != INFINITE) & (distance == minimum_distances)

            if np.any(hit):

                start_point: Vector = origin.extract(hit)
                incident_ray: Vector = direction.extract(hit)
                ray_travel_distance: NDArray[np.float64] = extract(hit, distance)

                intersection_point: Vector = start_point + incident_ray * ray_travel_distance
            
                #
                # transmission from volume
                #

                if ray_within_volume[element] and bounce < BOUNCE_COUNT:

                    surface_normal_at_intersection: Vector = element.compute_inward_normal(intersection_point)
                    intersection_point_with_standoff: Vector = intersection_point - surface_normal_at_intersection * self.standoff_distance
                    
                    transmitted_ray = self._transmitted_ray(
                            incident_ray,
                            surface_normal_at_intersection,
                            element.refractive_index,
                            self.refractive_index
                        )

                    ray_within_volume[element] = False
                    
                    logger.debug(f"TRANSMISSION-OUT. counter={self.counter}. bounce={bounce}. current enum={recursion_enum}")

                    ray_data = self._forward_recursive_trace(source, detector, intersection_point_with_standoff, transmitted_ray, bounce+1, recursion_enum="TRANSMISSION-OUT", ray_within_volume=ray_within_volume)
                    rays += ray_data.place(hit)

                #
                # transmission into volume
                #

                if (element.transparent and not ray_within_volume[element] and recursion_enum not in ['TRANSMISSION-IN', 'SUBSURFACE-REFLECTION']) and bounce < BOUNCE_COUNT:
                        
                    surface_normal_at_intersection: Vector = element.compute_outward_normal(intersection_point)
                    intersection_point_with_standoff: Vector = intersection_point - surface_normal_at_intersection * self.standoff_distance
                    
                    transmitted_ray = self._transmitted_ray(
                            incident_ray,
                            surface_normal_at_intersection,
                            self.refractive_index,
                            element.refractive_index
                        )

                    ray_within_volume[element] = True
                    
                    logger.debug(f"TRANSMISSION-IN. counter={self.counter}. bounce={bounce}. current enum={recursion_enum}")

                    ray_data = self._forward_recursive_trace(source, detector, intersection_point_with_standoff, transmitted_ray, bounce+1, recursion_enum="TRANSMISSION-IN", ray_within_volume=ray_within_volume)
                    rays += ray_data.place(hit)

                #
                # surface and subsurface reflection
                #

                if recursion_enum == 'SUBSURFACE-REFLECTION':

                    self.ray_debugger_reverse.add_vector(start_point=start_point, end_point=intersection_point, color=(255,0,0)) # subsurface reflected ray (red)

                    surface_normal_at_intersection: Vector = element.compute_inward_normal(intersection_point)
                    intersection_point_with_standoff: Vector = intersection_point + surface_normal_at_intersection * self.standoff_distance
                    ray_data = detector._transmission_model(element, start_point, intersection_point)
                    rays += ray_data.place(hit)

                elif recursion_enum == 'TRANSMISSION-IN':

                    self.ray_debugger_reverse.add_vector(start_point=start_point, end_point=intersection_point, color=(0,255,255)) # transmitted ray (cyan)

                    surface_normal_at_intersection: Vector = element.compute_inward_normal(intersection_point)
                    intersection_point_with_standoff: Vector = intersection_point + surface_normal_at_intersection * self.standoff_distance
                    ray_data = detector._transmission_model(element, start_point, intersection_point)
                    rays += ray_data.place(hit)

                else:

                    self.ray_debugger_reverse.add_vector(start_point=start_point, end_point=intersection_point, color=(0,0,255)) # surface reflected ray (blue)

                    surface_normal_at_intersection: Vector = element.compute_outward_normal(intersection_point)
                    intersection_point_with_standoff: Vector = intersection_point + surface_normal_at_intersection * self.standoff_distance

                direction_to_origin_unit: Vector = (detector.position - intersection_point).norm()

                # for source in self.sources:

                #     direction_to_source: Vector = source.center - intersection_point
                #     direction_to_source_unit: Vector = direction_to_source.norm()
                #     intersections_blocking_source: list[NDArray[np.float64]] = [element.intersect(intersection_point_with_standoff, direction_to_source_unit) for element in self.elements]
                #     minimum_distances_with_standoff: NDArray[np.float64] = reduce(np.minimum, intersections_blocking_source)
                #     distances_to_source = direction_to_source.magnitude()
                #     intersection_point_illuminated: NDArray[np.bool_] = minimum_distances_with_standoff >= distances_to_source

                #     if np.sum(intersection_point_illuminated) < 1:
                #         continue

                #     self.intersection_map.append({'source': source, 'direction_to_source_unit': direction_to_source_unit, 'intersection_point_illuminated': intersection_point_illuminated})

                ray_data = detector._reflection_model(element,
                                                        intersection_point,
                                                        surface_normal_at_intersection,
                                                        direction_to_origin_unit,
                                                        self.intersection_map)
                rays += ray_data.place(hit)
                            
                # self.intersection_map.clear()

                if bounce < BOUNCE_COUNT:
                    
                    e = "SUBSURFACE-REFLECTION" if recursion_enum in ["TRANSMISSION-IN", "SUBSURFACE-REFLECTION"] else "SURFACE-REFLECTION"
                    logger.debug(f"SURFACE-REFLECTION. counter={self.counter}. bounce={bounce}. current enum={recursion_enum}")
                    reflected_ray = self._reflected_ray(incident_ray, surface_normal_at_intersection)                
                    ray_data = self._reverse_recursive_trace(detector, intersection_point_with_standoff, reflected_ray, bounce + 1, recursion_enum=e, ray_within_volume=ray_within_volume)
                    rays += ray_data.place(hit)

                    # NOTE the existing `transmission from volume` is sufficient to allow rays to transmit and reflect for enum `SUBSURFACE-REFLECTION`


        logger.debug(f"RETURNING. counter={self.counter}. enum={recursion_enum}. current enum={recursion_enum}")
        return rays

    def _reverse_recursive_trace(self, detector: Detector, origin: Vector, direction: Vector, bounce: int, recursion_enum: str, ray_within_volume: dict):

        self.counter += 1
        logger.debug(f"counter={self.counter}. REVERSE. enum={recursion_enum}")

        rays = Vector(0, 0, 0)

        #
        # source illumination
        #

        for source in self.sources:

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

            direction_to_source_minima: Vector = direction_to_source.extract(intersection_point_illuminated)
            origin_point_illuminated: Vector = origin.extract(intersection_point_illuminated)
            intersection_to_source: Vector = origin_point_illuminated + direction_to_source_minima

            self.ray_debugger_reverse.add_vector(start_point=origin_point_illuminated, end_point=intersection_to_source, color=(255,0,255)) # to sources

            self.intersection_map.append({'source': source, 'direction_to_source_unit': direction_to_source_unit, 'intersection_point_illuminated': intersection_point_illuminated})

            ray_data = detector._emission_model(detector.pointing_direction, self.intersection_map)
            rays += ray_data.place(intersection_point_illuminated)
                        
            self.intersection_map.clear()

            # # store rays for forward tracing
            # r = (origin_point_illuminated - intersection_to_source).norm()
            # source._enqueue_rays(origin=intersection_to_source, direction=r, detector=detector)

        #
        # reflections and transmissions
        #

        distances: list[NDArray[np.float64]] = [element.intersect(origin, direction) for element in self.elements]
        minimum_distances: NDArray[np.float64] = reduce(np.minimum, distances)
        
        for element, distance in zip(self.elements, distances):

            logger.debug(f"element-distance iteration. counter={self.counter}. ray submerged={ray_within_volume[element]}. current enum={recursion_enum}")

            hit: NDArray[np.bool_] = (minimum_distances != INFINITE) & (distance == minimum_distances)

            if np.any(hit):

                start_point: Vector = origin.extract(hit)
                incident_ray: Vector = direction.extract(hit)
                ray_travel_distance: NDArray[np.float64] = extract(hit, distance)

                intersection_point: Vector = start_point + incident_ray * ray_travel_distance
            
                #
                # transmission from volume
                #

                if ray_within_volume[element] and bounce < BOUNCE_COUNT:

                    surface_normal_at_intersection: Vector = element.compute_inward_normal(intersection_point)
                    intersection_point_with_standoff: Vector = intersection_point - surface_normal_at_intersection * self.standoff_distance
                    
                    transmitted_ray = self._transmitted_ray(
                            incident_ray,
                            surface_normal_at_intersection,
                            element.refractive_index,
                            self.refractive_index
                        )

                    ray_within_volume[element] = False
                    
                    logger.debug(f"TRANSMISSION-OUT. counter={self.counter}. bounce={bounce}. current enum={recursion_enum}")

                    ray_data = self._reverse_recursive_trace(detector, intersection_point_with_standoff, transmitted_ray, bounce, recursion_enum="TRANSMISSION-OUT", ray_within_volume=ray_within_volume)
                    rays += ray_data.place(hit)

                #
                # transmission into volume
                #

                if (element.transparent and not ray_within_volume[element] and recursion_enum not in ['TRANSMISSION-IN', 'SUBSURFACE-REFLECTION']) and bounce < BOUNCE_COUNT:
                        
                    surface_normal_at_intersection: Vector = element.compute_outward_normal(intersection_point)
                    intersection_point_with_standoff: Vector = intersection_point - surface_normal_at_intersection * self.standoff_distance
                    
                    transmitted_ray = self._transmitted_ray(
                            incident_ray,
                            surface_normal_at_intersection,
                            self.refractive_index,
                            element.refractive_index
                        )

                    ray_within_volume[element] = True
                    
                    logger.debug(f"TRANSMISSION-IN. counter={self.counter}. bounce={bounce}. current enum={recursion_enum}")

                    ray_data = self._reverse_recursive_trace(detector, intersection_point_with_standoff, transmitted_ray, bounce, recursion_enum="TRANSMISSION-IN", ray_within_volume=ray_within_volume)
                    rays += ray_data.place(hit)

                #
                # surface and subsurface reflection
                #

                if recursion_enum == 'SUBSURFACE-REFLECTION':

                    self.ray_debugger_reverse.add_vector(start_point=start_point, end_point=intersection_point, color=(255,0,0)) # subsurface reflected ray (red)

                    surface_normal_at_intersection: Vector = element.compute_inward_normal(intersection_point)
                    intersection_point_with_standoff: Vector = intersection_point + surface_normal_at_intersection * self.standoff_distance
                    ray_data = detector._transmission_model(element, start_point, intersection_point)
                    rays += ray_data.place(hit)

                elif recursion_enum == 'TRANSMISSION-IN':

                    self.ray_debugger_reverse.add_vector(start_point=start_point, end_point=intersection_point, color=(0,255,255)) # transmitted ray (cyan)

                    surface_normal_at_intersection: Vector = element.compute_inward_normal(intersection_point)
                    intersection_point_with_standoff: Vector = intersection_point + surface_normal_at_intersection * self.standoff_distance
                    ray_data = detector._transmission_model(element, start_point, intersection_point)
                    rays += ray_data.place(hit)

                else:

                    self.ray_debugger_reverse.add_vector(start_point=start_point, end_point=intersection_point, color=(0,0,255)) # surface reflected ray (blue)

                    surface_normal_at_intersection: Vector = element.compute_outward_normal(intersection_point)
                    intersection_point_with_standoff: Vector = intersection_point + surface_normal_at_intersection * self.standoff_distance

                direction_to_origin_unit: Vector = (detector.position - intersection_point).norm()

                for source in self.sources:

                    direction_to_source: Vector = source.center - intersection_point
                    direction_to_source_unit: Vector = direction_to_source.norm()
                    intersections_blocking_source: list[NDArray[np.float64]] = [element.intersect(intersection_point_with_standoff, direction_to_source_unit) for element in self.elements]
                    minimum_distances_with_standoff: NDArray[np.float64] = reduce(np.minimum, intersections_blocking_source)
                    distances_to_source = direction_to_source.magnitude()
                    intersection_point_illuminated: NDArray[np.bool_] = minimum_distances_with_standoff >= distances_to_source

                    if np.sum(intersection_point_illuminated) < 1:
                        continue

                    self.intersection_map.append({'source': source, 'direction_to_source_unit': direction_to_source_unit, 'intersection_point_illuminated': intersection_point_illuminated})

                ray_data = detector._reflection_model(element,
                                                        intersection_point,
                                                        surface_normal_at_intersection,
                                                        direction_to_origin_unit,
                                                        self.intersection_map)
                rays += ray_data.place(hit)
                            
                self.intersection_map.clear()

                if bounce < BOUNCE_COUNT:
                    
                    e = "SUBSURFACE-REFLECTION" if recursion_enum in ["TRANSMISSION-IN", "SUBSURFACE-REFLECTION"] else "SURFACE-REFLECTION"
                    logger.debug(f"SURFACE-REFLECTION. counter={self.counter}. bounce={bounce}. current enum={recursion_enum}")
                    reflected_ray = self._reflected_ray(incident_ray, surface_normal_at_intersection)                
                    ray_data = self._reverse_recursive_trace(detector, intersection_point_with_standoff, reflected_ray, bounce + 1, recursion_enum=e, ray_within_volume=ray_within_volume)
                    rays += ray_data.place(hit)

                    # NOTE the existing `transmission from volume` is sufficient to allow rays to transmit and reflect for enum `SUBSURFACE-REFLECTION`


        logger.debug(f"RETURNING. counter={self.counter}. enum={recursion_enum}. current enum={recursion_enum}")
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
    
    