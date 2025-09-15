from abc import ABC, abstractmethod
import numpy as np
from PIL import Image

from ..math.vector import Vector
from ..element.element import Element


class Detector(ABC):
    def __init__(self, width: int, height: int, position: Vector, pointing_direction: Vector):
        '''
        Parameters:
            position (Vector): Detector's absolute position in 3D space
            width (float): Detector width in pixels
            height (float): Detector height in pixels
            pointing_direction (Vector): vector defining `Detector` pointing direction
        '''
        self._data = None
        self.position = position
        self.width = width
        self.height = height
        self.pointing_direction = pointing_direction.norm()

    @abstractmethod
    def _reflection_model(  self, 
                            element: Element,
                            intersection_point: Vector,
                            surface_normal_at_intersection: Vector, 
                            direction_to_origin_unit: Vector, 
                            intersection_map: list[dict]) -> Vector:
        '''
        Recursively compile data for detector model

        Support for multiple sources MUST be implemented here!
        e.g., shaded pixel values are the sum or weighted sum of the contributions of sources which illuminate that pixel

        For a succinct description of reflection models, see 'Learning OpenGL - Graphics Programming' Chapter 6, 2020, de Vries

        Parameters:
            element (Element): The element where the intersection occurs. Use this handle to access element-specific parameters needed for reflection model. Note that additional element parameters can be passed as key:value pairs using Element arg `user_params`.
            intersection_point (Vector): Point of intersection
            surface_normal_at_intersection (Vector): Normal vector with respect to element's surface at point of intersection
            direction_to_origin_unit (Vector): Source direction for ray(s). Note that because rays are traced in reverse, these rays originate from the Detector (not the Source) or the previous reflection
            intersection_map (list[dict]): A list of dicts representing how each source interacts with `intersection_point`. Each dict in the list contains the following keys: 'source' represents one Source object, 'direction_to_source_unit' is the Vector direction from `intersection_point` to Source, 'intersection_point_illuminated' is a numpy array of booleans indicating if corresponding rays are illuminated by that Source, else, are in a shadow
        '''
        pass

    @abstractmethod
    def view_data(self):
        '''
        Perform any final processing and return data to user

        Instance attribute `_data` should be accessed, manipulated if necessary, and returned
        '''
        pass


class PowerMeter(Detector):
    def __init__(self, width: int, height: int, position: Vector, pointing_direction: Vector):
        super().__init__(width, height, position, pointing_direction)

    def _reflection_model(self):
        raise NotImplementedError(f"Method not yet implementated in {self.__class__}")

    def view_data(self):
        raise NotImplementedError(f"Method not yet implementated in {self.__class__}")


class Camera(Detector):
    '''
    Simple simulated camera using Phong shading for pixels.

    NOTE    elements must include `user_params={'specular': s, 'n_s': n}`
            where s is a float and n is a float
    '''
    def __init__(self, width: int, height: int, position: Vector, pointing_direction: Vector):
        super().__init__(width, height, position, pointing_direction)

        # unilluminated pixels display this noise floor
        self.ambient_dark = Vector(0, 0, 0)

    def _reflection_model(self, element, intersection_point, surface_normal_at_intersection, direction_to_origin_unit, intersection_map):

        s = Vector(0,0,0)

        for v in intersection_map:

            direction_to_source_unit = v['direction_to_source_unit']
            intersection_point_illuminated = v['intersection_point_illuminated']

            # lambertian diffuse shading. See Section 5.2, Real Time Rendering, Akenine-Moller et al., 2018
            dot_product_clamped = np.maximum(surface_normal_at_intersection.dot(direction_to_source_unit), 0)
            lambertian_diffuse_shading = element.surface_color(intersection_point) * dot_product_clamped

            # blinn-phong spectral shading. See Section 33.1, Learning OpenGL - Graphics Programming, de Vries, 2020
            phong = np.clip(surface_normal_at_intersection.dot((direction_to_source_unit + direction_to_origin_unit).norm()), 0, 1)
            phong_specular_shading = element.specular * np.power(phong, element.n_s)

            s += (lambertian_diffuse_shading + phong_specular_shading) * intersection_point_illuminated

        return self.ambient_dark + s

    def view_data(self):
        '''
        Pixel data is a color
        '''
        rgb = [
            Image.fromarray(
                (255 * np.clip(c, 0, 1).reshape((self.height, self.width))).astype(np.uint8),
                "L",
            )
            for c in self._data.components()
        ]

        return Image.merge("RGB", rgb)