from functools import reduce
import numpy as np
from numpy.typing import NDArray
import time
import atexit

from ..math.vector import Vector
from ..math.tools import extract
from ..detector.detector import Detector
from ..element.element import Element
from ..source.source import Source
from luminous.src.utilities.ray_debugger import NullRayDebugger, ConcreteRayDebugger

from luminous.src.utilities.logconfig import setup_logging
import logging
logger = logging.getLogger('luminous.scene')

FARAWAY = 1.0e39


class Scene:

    def __init__(self, index_of_refraction=1, log_level=20, log_file="luminous.log") -> None:
        setup_logging(name='luminous', level=log_level, log_file=log_file)
        self.start_time = None
        self.refractive_index = index_of_refraction
        self.elements = list()
        self.detectors = list()
        self.sources = list()
        self.ray_debugger = NullRayDebugger()

    def elaspsed_time(self):
        if self.start_time is None:
            raise ValueError('Timer has not been started. Call `elapsted_time` after calling `raytrace`.')
        return time.perf_counter() - self.start_time

    def attach_ray_debugger(self, path="./results", filename="debug_ray_trace", display_3d_plot=True):
        self.ray_debugger = ConcreteRayDebugger()

        def plot_ray_trace_wrap_user_args():
            self.ray_debugger.plot(path=path, filename=filename, display_3d_plot=display_3d_plot)
        atexit.register(plot_ray_trace_wrap_user_args)

    def compute_ray_directions(self, detector: Detector, detector_screen: Vector):

        initial_ray_dir = (detector_screen - detector.position)

        ray_dir_x = initial_ray_dir.x
        ray_dir_y = initial_ray_dir.y
        ray_dir_z = initial_ray_dir.z + detector.pointing_direction.z

        ray_dir = Vector(ray_dir_x, ray_dir_y, ray_dir_z).norm()

        return ray_dir

    def create_screen_coord(self, w, h, detector_pointing_direction, detector_pos, vertical_screen_shift=0, horizontal_screen_shift=0, rotation=0) -> Vector:
        '''
        Create and return a Vector where each nth component of the vector represents one unique pixel in the detection screen;
        where the larger detector axis is normalized to +/- 1 and the smaller axis is scaled to aspect ratio.
        The detector screen surface is necessarily centered on detector_pos and normal to detector_pointing_direction.
        '''
        aspect_ratio = w / h
        screen = (  -1 + horizontal_screen_shift, 
                    1 / aspect_ratio + vertical_screen_shift, 
                    1 + horizontal_screen_shift, 
                    -1 / aspect_ratio + vertical_screen_shift)
        w_row = np.linspace(screen[0], screen[2], w)
        h_col = np.linspace(screen[1], screen[3], h)
        rows = np.tile(w_row, h)
        columns = np.repeat(h_col, w)
        d = np.zeros(len(rows))

        screen_coords = Vector(rows, columns, d)

        if rotation != 0:
            rotation_matrix = self._rotation_matrix(detector_pointing_direction, rotation)
            screen_coords = self._apply_rotation(screen_coords, rotation_matrix)

        screen_coords = screen_coords + detector_pos
        return screen_coords

    def _rotation_matrix(self, axis, theta):
        """
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
        """
        axis = axis.norm()
        a = np.cos(theta / 2.0)
        b, c, d = -axis.x * np.sin(theta / 2.0), -axis.y * np.sin(theta / 2.0), -axis.z * np.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

    def _apply_rotation(self, vector, rotation_matrix):
        """
        Apply the rotation matrix to the vector.
        """
        original_vectors = np.stack((vector.x, vector.y, vector.z), axis=-1)
        rotated_vectors = np.dot(original_vectors, rotation_matrix.T)
        return Vector(rotated_vectors[:, 0], rotated_vectors[:, 1], rotated_vectors[:, 2])

    def __iadd__(self, obj):
        if isinstance(obj, Element):
            self.elements.append(obj)
        elif isinstance(obj, Source):
            self.sources.append(obj)
        elif isinstance(obj, Detector):
            self.detectors.append(obj)
        else:
            raise TypeError("Only Elements, Sources, and Detectors can be added to Scenes!")

        return self
        
    def raytrace(self):
        self.start_time = time.perf_counter()

        # TODO future implementations will permit multiple sources. but currently does not. hard coded for 0th.
        self.source = self.sources[0]
        self.source_pos: Vector = self.source.position

        # TODO re-work as distinct processes or threads
        for detector in self.detectors:

            detector_pixels: Vector = self.create_screen_coord(detector.width, detector.height, detector.pointing_direction, detector.position)
            pixel_incident_rays: Vector = self.compute_ray_directions(detector, detector_pixels)

            # ray debugger
            detector_dir_translate: Vector = detector.position + detector.pointing_direction
            self.ray_debugger.add_point(detector.position, color=(0, 255, 0))
            self.ray_debugger.add_vector(start_point=detector.position, end_point=detector_dir_translate, color=(255, 0, 0))
            self.ray_debugger.add_point(detector_pixels, color=(255,0,0))

            detector.data = self._recursive_trace(detector=detector, origin=detector_pixels, direction=pixel_incident_rays, elements=self.elements, bounce=0)

    def _recursive_trace(self, detector: Detector, origin: Vector, direction: Vector, elements: list['Element'], bounce: int):

        ''' 
        distances between origin and element surface, along `direction` vector
        len(distances) = number of elements in scene
        len(distances)[n] = number of rays
        '''
        distances: list[NDArray[np.float64]] = [s.intersect(origin, direction) for s in elements]

        '''
        Find element-wise minimum for rays
        e.g., one ray may intersect two elements. if so, count nearest hit as the true hit.
        len(minimum_distances) == numer of rays, where each value is the smallest distance from each sub-list of `distances`
        '''
        minimum_distances: NDArray[np.float64] = reduce(np.minimum, distances)

        rays = Vector(0, 0, 0)
        
        for element, distance in zip(elements, distances):

            '''
            len(hit) == len(minimum_distances)
            for each boolean in this array:
            if `minimum_distances != FARAWAY`, and, for the `distance` in `distances`, if that `distance` is `minimum_distances[n]`, then this ray intersects this `element`
            '''
            hit: NDArray[np.bool_] = (minimum_distances != FARAWAY) & (distance == minimum_distances)

            if np.any(hit):

                ray_travel_distance: NDArray[np.float64] = extract(hit, distance)
                ray_start_position: Vector = origin.extract(hit)
                incident_ray: Vector = direction.extract(hit)

                intersection_point: Vector = ray_start_position + incident_ray * ray_travel_distance
                surface_normal_at_intersection: Vector = element.compute_intersection_normal(intersection_point)

                if element.transparent:
                    tir = self._total_internal_reflection(incident_ray, surface_normal_at_intersection, self.refractive_index, element.refractive_index)
                    reflection_weight = np.ones(tir.shape, dtype=np.float16)
                    transmission_weight = np.zeros(tir.shape, dtype=np.float16)

                    if not np.all(tir):
                        non_tir_indices = np.logical_not(tir)
                        reflection_weight[non_tir_indices], transmission_weight[non_tir_indices] = self._reflection_transmission_weights(
                            incident_ray.extract(non_tir_indices),
                            surface_normal_at_intersection.extract(non_tir_indices),
                            self.refractive_index,
                            element.refractive_index
                        )

                    valid_transmission_indices = transmission_weight > 0
                    if np.any(valid_transmission_indices):

                        # TODO I think somewhere in this block I need to account for the color of the element through which the ray passes
                        # ray_data = detector._capture_data(...)

                        # initial intersection of ray and element
                        element_intersection = intersection_point.extract(valid_transmission_indices)

                        # this ray propogates inside the element
                        incident_ray_into_volume = self._transmitted_ray(
                            incident_ray.extract(valid_transmission_indices),
                            surface_normal_at_intersection.extract(valid_transmission_indices),
                            self.refractive_index,
                            element.refractive_index
                        )

                        incident_ray_into_volume = element_intersection + incident_ray_into_volume

                        ray_travel_distance_transmission: NDArray[np.float64] = element.intersect(element_intersection, incident_ray_into_volume)

                        intersection_point_transmission: Vector = element_intersection + incident_ray_into_volume * ray_travel_distance_transmission
                        surface_normal_within_volume: Vector = element.compute_intersection_normal(intersection_point_transmission)

                        incident_ray_from_volume = self._transmitted_ray(
                            intersection_point_transmission,
                            surface_normal_within_volume,
                            element.refractive_index,
                            self.refractive_index
                        )
                        # TODO do something with `incident_ray_from_volume`, e.g., continue tracing rays to subsequent elements somehow
                        # possibly correct to simply recurse here? ray_data += self._recursive_trace(...)
                        # perhaps these rays can be added to other data structure tracing rays associated with not `element.transparent` case?

                    if np.all(reflection_weight == 0):
                        continue

                # TODO probably some graceful way to marry these two segments of code together
                # or is it necessary to add a reflection contingency for element.transparent?

                intersection_point_with_standoff: Vector = intersection_point + surface_normal_at_intersection * 0.0001

                direction_to_source: Vector = self.source_pos - intersection_point_with_standoff
                direction_to_source_unit: Vector = direction_to_source.norm()
                direction_to_origin_unit: Vector = (detector.position - intersection_point).norm()
                intersections_blocking_source: list[NDArray[np.float64]] = [s.intersect(intersection_point_with_standoff, direction_to_source_unit) for s in elements]
                minimum_distances_with_standoff: NDArray[np.float64] = reduce(np.minimum, intersections_blocking_source)
                distances_to_source = direction_to_source.magnitude()
                intersection_point_illuminated: NDArray[np.bool_] = minimum_distances_with_standoff >= distances_to_source
                
                # ray debugger
                self.ray_debugger.add_vector(start_point=ray_start_position, end_point=intersection_point_with_standoff, color=(0,0,255)) # to elements
                illuminated_intersections: Vector = intersection_point.extract(intersection_point_illuminated)
                direction_to_source_minima: Vector = direction_to_source.extract(intersection_point_illuminated)
                intersection_to_source: Vector = illuminated_intersections + direction_to_source_minima
                self.ray_debugger.add_vector(start_point=illuminated_intersections, end_point=intersection_to_source, color=(255,0,255)) # to source
                    
                # detect
                ray_data = detector._capture_data(surface_normal_at_intersection, direction_to_source_unit, element, intersection_point, intersection_point_illuminated)

                # reflect
                if bounce < 2:
                    reflected_ray = self._reflected_ray(incident_ray, surface_normal_at_intersection)
                    ray_data += self._recursive_trace(detector, intersection_point_with_standoff, reflected_ray, elements, bounce + 1) * element.specularity

                ray_data += detector._calculate_model(surface_normal_at_intersection, direction_to_source_unit, direction_to_origin_unit, intersection_point_illuminated)

                rays += ray_data.place(hit)

        return rays
    
    def _total_internal_reflection(self, incident_ray: Vector, surface_normal: Vector, n1: float, n2: float) -> bool:
        # Determine if incident angle >= critical angle
        # See Reflections and Refractions in Ray Tracing, 2006, Greve
        if n2 >= n1:
            return np.zeros(incident_ray.x.shape, dtype=bool)
        
        cos_theta_i = -incident_ray.dot(surface_normal)
        theta_i = np.arccos(np.clip(cos_theta_i, -1.0, 1.0))
        critical_angle = np.arcsin(np.clip(n2 / n1, -1.0, 1.0))
        
        return theta_i > critical_angle

    def _reflected_ray(self, incident_ray: Vector, surface_normal_at_intersection: Vector) -> Vector:
        # Reflected ray unit vector. See Reflections and Refractions in Ray Tracing, 2006, Greve
        return incident_ray - surface_normal_at_intersection * 2 * incident_ray.dot(surface_normal_at_intersection)
    
    def _transmitted_ray(self, incident_ray: Vector, surface_normal: Vector, n1: float, n2: float) -> Vector:
        # Refracted ray unit vector. See Reflections and Refractions in Ray Tracing, 2006, Greve
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
        # See Reflections and Refractions in Ray Tracing, 2006, Greve
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
    
    