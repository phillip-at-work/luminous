from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from ..math.vector import Vector
from .source import Source
from .shape import Shape, Square, Circle

import logging
from luminous.src.utilities.logconfig import setup_logging
logger = logging.getLogger(__name__)


class Detector(ABC):
    def __init__(self):
        '''
        Parameters:
            _data (Vector): Detector's accumulated data, called directly from within `Scene`
            position (Vector): Detector's absolute position in 3D space
            width (float): Detector width in pixels
            height (float): Detector height in pixels
            pointing_direction (Vector): vector defining `Detector` pointing direction
            screen_width (float, optional): The physical width (X-direction) of the detector screen in scene units. Default is 2.0.
            screen_height (float, optional): The physical height (Y-direction) of the detector screen. If None, calculated from screen_width and detector aspect ratio.
        '''

        # instance attributes directly assigned from Scene
        self._reverse_trace_data = None
        self._forward_trace_data = None
        self.pixels = None

    @abstractmethod
    def _reflection_model(  self, 
                            element: Shape,
                            intersection_point: Vector,
                            surface_normal_at_intersection: Vector, 
                            direction_to_origin_unit: Vector, 
                            intersection_map: list[dict],
                            intersection_mask: NDArray[np.bool_]) -> Vector | None:
        '''
        Tint or otherwise weight pixel data model at reflection event.

        Support for multiple sources MUST be implemented here!
        e.g., shaded pixel values are the sum or weighted sum of the contributions of sources which illuminate that pixel

        Parameters:
            element (Element): The element where the intersection occurs. Use this handle to access element-specific parameters needed for reflection model. Note that additional element parameters can be passed as key:value pairs using Element arg `user_params`.
            intersection_point (Vector): Point of intersection
            surface_normal_at_intersection (Vector): Normal vector with respect to element's surface at point of intersection
            direction_to_origin_unit (Vector): Source direction for ray(s). Note that because rays are traced in reverse, these rays originate from the Detector (not the Source) or the previous reflection
            intersection_map (list[dict]): A list of dicts representing how each source interacts with `intersection_point`. Each dict in the list contains the following keys: 'source' represents one Source object, 'direction_to_source_unit' is the Vector direction from `intersection_point` to Source, 'intersection_point_illuminated' is a numpy array of booleans indicating if corresponding rays are illuminated by that Source, else, are in a shadow
            reflection_weight (NDArray[np.number]): A numpy array of numbers where each value is [0, 1]. Indicates how much of the antecedent ray reflects (versus transmits).
        '''
        pass

    @abstractmethod
    def _transmission_model(self, 
                            element: Shape,
                            initial_intersection: Vector,
                            final_intersection: Vector,
                            intersection_mask: NDArray[np.bool_]) -> Vector | None:
        '''
        Tint or otherwise weight pixel data model at transmission event.

        # TODO document parameters
        '''
        pass

    @abstractmethod
    def _emission_model(    self, 
                            detection_area_normal: Vector, 
                            intersection_map: list[dict]) -> Vector | None:
        '''
        Tint or otherwise weight pixel data model when ray's origin position is in direct view of the source.
        This method need not be implemented for detectors used only for forward traces.

        # TODO document parameters
        '''
        pass

    @abstractmethod
    def view_data(self):
        '''
        Perform any final processing and return data to user

        Instance attribute `_reverse_trace_data` should be accessed, manipulated if necessary, and returned
        '''
        pass

    @abstractmethod
    def _compute_initial_ray_directions(self):
        '''
        Compute initial ray trajectories FROM a detector. Used in reverse traces only.
        '''
        pass

class PowerMeter(Detector, Circle):
    '''
    Ideal power meter for measurements in watts
    '''
    def __init__(self, center: Vector, radius: float, pointing_direction: Vector, pixel_count: int = None):
        Circle.__init__(self, center, radius, pointing_direction, pixel_count)
        Detector.__init__(self)

    def _reflection_model(self, element, intersection_point, surface_normal_at_intersection, direction_to_origin_unit, intersection_map, intersection_mask):
        # TODO apply reflection model directly to self.pixels
        pass
    
    def _transmission_model(self, element, initial_intersection, final_intersection, intersection_mask):
        raise NotImplementedError("Not yet implemented")
    
    def _emission_model(self, detection_area_normal, intersection_map):
        raise NotImplementedError("Not to be implemented. Power meter designed for use with forward path traces, which are source-driven.")

    def view_data(self, forward_trace_data=False):
        # TODO iterate through detected rays, integrate, return final value in watts
        pass

    def _compute_initial_ray_directions(self, pixels: Vector):
        raise NotImplementedError("Not yet implemented")

class Camera(Detector, Square):
    '''
    Simple simulated camera using Blinn-Phong shading for pixels.

    NOTE    elements must include `user_params={'specular': s, 'n_s': n}`
            where s is a float and n is a float
    '''
    def __init__(self, width: int, height: int, position: Vector, pointing_direction: Vector, screen_width: float, screen_height: float):
        Square.__init__(self, width, height, position, pointing_direction, screen_width, screen_height)
        Detector.__init__(self)

        # unilluminated pixels display this noise floor
        self.ambient_dark = Vector(0, 0, 0)

    def _reflection_model(self, element, intersection_point, surface_normal_at_intersection, direction_to_origin_unit, intersection_map, intersection_mask):

        s = Vector(0,0,0)

        for v in intersection_map:

            direction_to_source_unit = v['direction_to_source_unit']
            intersection_point_illuminated = v['intersection_point_illuminated']

            # lambertian diffuse shading
            # REF: [Akenine-Moller et al., 2018, Real Time Rendering, {Section: 5.2}]
            dot_product_clamped = np.maximum(surface_normal_at_intersection.dot(direction_to_source_unit), 0)
            lambertian_diffuse_shading = element.surface_color(intersection_point) * dot_product_clamped

            # blinn-phong spectral shading
            # REF: [de Vries, 2020, Learning OpenGL - Graphics Programming, {Section: 33.1}]
            phong = np.clip(surface_normal_at_intersection.dot((direction_to_source_unit + direction_to_origin_unit).norm()), 0, 1)
            phong_specular_shading = element.specular * np.power(phong, element.n_s)

            s += (lambertian_diffuse_shading + phong_specular_shading) * intersection_point_illuminated

        return self.ambient_dark + s
    
    def _transmission_model(self, element, initial_intersection, final_intersection, intersection_mask):

        # TODO this current implementation isn't great
        # next version should do two things: diminish the intensity of the ray passing through the medium and tint that ray based upon element color

        internal_travel_distance: Vector = (final_intersection - initial_intersection).magnitude() 
        beers_law_attenuation = element.surface_color(initial_intersection) * np.exp(-internal_travel_distance)
        return beers_law_attenuation
    
    def _emission_model(self, detection_area_normal: Vector, intersection_map):

        s = Vector(0,0,0)

        for v in intersection_map:

            source = v['source']
            direction_to_source_unit = v['direction_to_source_unit']
            intersection_point_illuminated = v['intersection_point_illuminated']

            dot_product_clamped = np.maximum(detection_area_normal.dot(direction_to_source_unit), 0)

            s += source.color * dot_product_clamped * intersection_point_illuminated

        return self.ambient_dark + s

    def view_data(self, forward_trace_data=False):
        '''
        Pixel values in `_data` represent colors
        '''

        d = self._forward_trace_data if forward_trace_data else self._reverse_trace_data            
        logger.debug(f"data.size.x: {d.x.size}. height: {self.height}. width: {self.width}")
        
        rgb = [
            Image.fromarray(
                (255 * np.clip(c, 0, 1).reshape((self.height, self.width))).astype(np.uint8),
                "L",
            )
            for c in d.components()
        ]

        return Image.merge("RGB", rgb)
    
    def _compute_initial_ray_directions(self, pixels: Vector):

        # TODO placeholder implementation. in the future, rays should propogate
        # through a pinhole, or through the extent of a pupil in a lens assembly
        # as marginal rays

        initial_ray_dir = (pixels - self.position)

        ray_dir_x = initial_ray_dir.x
        ray_dir_y = initial_ray_dir.y
        ray_dir_z = initial_ray_dir.z + self.pointing_direction.z

        ray_dir = Vector(ray_dir_x, ray_dir_y, ray_dir_z).norm()

        return ray_dir