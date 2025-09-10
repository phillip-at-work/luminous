from abc import ABC, abstractmethod
import numpy as np
from PIL import Image

from ..math.vector import Vector


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
    def _reflection_model(self) -> Vector:
        '''
        Recursively compile data for detector model

        Support for multiple sources MUST be integrated here
        e.g., shaded pixel values are the sum or weighted sum of the contributions of sources which illuminate that pixel

        For a succinct description of reflection models, see 'Learning OpenGL - Graphics Programming' Chapter 6, 2020, de Vries
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

    NOTE    that elements must implement `user_params={'diffuse': v, 'specular': s, 'n_s': n}`
            where v is a Vector, s is a float, and n is a float
    '''
    def __init__(self, width: int, height: int, position: Vector, pointing_direction: Vector):
        super().__init__(width, height, position, pointing_direction)

        # unilluminated pixels display this noise floor
        self.lambertian_model_dark_term = Vector(0, 0, 0)

    def _reflection_model(self, surface_normal_at_intersection, direction_to_source_unit, element, intersection_point, direction_to_origin_unit, intersection_point_illuminated):        
        # lambertian shader model
        # See Section 5.2, Real Time Rendering, Akenine-Moller et al., 2018
        dot_product_clamped = np.maximum(surface_normal_at_intersection.dot(direction_to_source_unit), 0)
        lambertian_shading = element.surface_color(intersection_point) * dot_product_clamped * intersection_point_illuminated
        lambertian_shader_model = self.lambertian_model_dark_term + lambertian_shading

        # phong highlights
        phong = np.clip(surface_normal_at_intersection.dot((direction_to_source_unit + direction_to_origin_unit).norm()), 0, 1)
        phong_highlights = element.diffuse * element.specular * np.power(phong, element.n_s) * intersection_point_illuminated
        return lambertian_shader_model + phong_highlights
    
        # TODO support for scenes including multiple sources must be implemented HERE in the shader model. see Akenine-Moller.

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