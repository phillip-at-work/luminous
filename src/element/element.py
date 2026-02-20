from abc import ABC, abstractmethod
import numpy as np

from ..math.vector import Vector
from .shape import Sphere


class Element(ABC):

    '''An object in the scene that does not emit light'''

    def __init__(self, center: Vector, color: Vector, transparent: bool = False, refractive_index: float = 1.0, user_params: dict = None):
        '''
        Parameters:
            center (Vector): Element's center position in 3D space
            color (Vector): Element's color
            refractive_index (float): Refractive index of the element
            user_params (dict): Additional user-defined parameters
        '''
        self.center = center
        self.color = color
        self.transparent = transparent
        self.refractive_index = refractive_index
        self.user_params = user_params or dict()

        # merge `user_params` into instance
        for key, value in self.user_params.items():
            if hasattr(self, key):
                raise AttributeError(f"Attribute '{key}' already exists in the instance and cannot be overwritten.")
            setattr(self, key, value)

    @abstractmethod
    def surface_color(self, M: Vector) -> Vector:
        '''
        Calculate or otherwise return the surface color of the object at point of intersection.
        '''
        pass
    
class SphereElement(Element, Sphere):
    def __init__(self, center: Vector, radius: float, color: Vector, transparent: bool = False, refractive_index: float = 1.0, user_params=None):
        super().__init__(center, color, transparent, refractive_index, user_params)
        self.radius = radius

    def surface_color(self, M: Vector) -> Vector:
        return self.color

class CheckeredSphereElement(SphereElement):
    def __init__(self, center: Vector, radius: float, color: Vector, transparent: bool = False, refractive_index: float = 1.0, checker_color: Vector = Vector(1, 1, 1), user_params=None):
        super().__init__(center, radius, color, transparent, refractive_index, user_params)
        self.checker_color = checker_color

    def surface_color(self, M: Vector) -> Vector:
        checker = ((M.x * 2).astype(int) % 2) == ((M.z * 2).astype(int) % 2)
        return self.color * checker + self.checker_color * (1 - checker)    