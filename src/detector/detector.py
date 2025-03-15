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
        self.position = position
        self.width = width
        self.height = height
        self.pointing_direction = pointing_direction.norm()

    @abstractmethod
    def _capture_data(self):
        '''
        Compile data for detector model
        '''
        pass

    @abstractmethod
    def _calculate_model(self):
        '''
        Perform calculations related to detector model
        '''
        pass

    @abstractmethod
    def view_data(self):
        '''
        Perform any final processing and return data to user
        '''
        pass


class PowerMeter(Detector):
    def __init__(self, width: int, height: int, position: Vector, pointing_direction: Vector):
        super().__init__(width, height, position, pointing_direction)

    def _capture_data(self):
        raise NotImplementedError(f"Method not yet implementated in {self.__class__}")
    
    def _calculate_model(self):
        raise NotImplementedError(f"Method not yet implementated in {self.__class__}")
    
    def view_data(self):
        raise NotImplementedError(f"Method not yet implementated in {self.__class__}")


class Camera(Detector):
    def __init__(self, width: int, height: int, position: Vector, pointing_direction: Vector):
        super().__init__(width, height, position, pointing_direction)
        self.data = Vector(0, 0, 0)

    # def initialize_ray_sum(self):
    #     self.rays = Vector(0, 0, 0)

    def _capture_data(self, surface_normal_at_intersection, direction_to_source_unit, element,intersection_point, intersection_point_illuminated):
        # Ambient
        color = Vector(0.05, 0.05, 0.05)
        # Lambert shading (diffuse)
        lv = np.maximum(surface_normal_at_intersection.dot(direction_to_source_unit), 0)
        dc = element.diffuse_color(intersection_point) * lv * intersection_point_illuminated
        return color + dc
    
    def _calculate_model(self, surface_normal_at_intersection, direction_to_source_unit, direction_to_origin_unit, intersection_point_illuminated):
        # Blinn-Phong shading (specular)
        phong = surface_normal_at_intersection.dot((direction_to_source_unit + direction_to_origin_unit).norm())
        return Vector(1, 1, 1) * np.power(np.clip(phong, 0, 1), 50) * intersection_point_illuminated

    def view_data(self):
        '''
        Pixel data is a color
        '''
        rgb = [
            Image.fromarray(
                (255 * np.clip(c, 0, 1).reshape((self.height, self.width))).astype(np.uint8),
                "L",
            )
            for c in self.data.components()
        ]

        return Image.merge("RGB", rgb)