from abc import ABC, abstractmethod
import numpy as np

from ..math.vector import Vector

FARAWAY = 1.0e39

#
# not for user instantiation
#

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

class Source(ABC):
    '''An object in the scene which does emit light'''
    pass

class SceneObject(ABC):

    @abstractmethod
    def intersect(self, origin: Vector, direction: Vector):
        '''
        Calculate intersection of a ray with the object.
        '''
        pass

    @abstractmethod
    def surface_color(self, M: Vector) -> Vector:
        '''
        Calculate or otherwise return the surface color of the object at point of intersection.
        '''
        pass

    @abstractmethod
    def compute_outward_normal(self, intersection_point: Vector) -> Vector:
        '''
        Compute normal vector, facing away from the object, associated with incident Vector's point of intersection.
        '''
        pass

    @abstractmethod
    def compute_inward_normal(self, intersection_point: Vector) -> Vector:
        '''
        Compute normal vector, facing into the object, associated with incident Vector's point of intersection.
        '''
        pass

class Sphere(SceneObject):
    def __init__(self, center: Vector, radius: float):
        self.radius = radius
        self.center = center

    def intersect(self, origin: Vector, direction: Vector):
        b = 2 * direction.dot(origin - self.center)
        c = (
            abs(self.center)
            + abs(origin)
            - 2 * self.center.dot(origin)
            - self.radius**2
        )
        discriminant = (b**2) - (4 * c)
        sq = np.sqrt(np.maximum(0, discriminant))
        h0 = (-b - sq) / 2
        h1 = (-b + sq) / 2
        h = np.where((h0 > 0) & (h0 < h1), h0, h1)
        pred = (discriminant > 0) & (h > 0)
        val = np.where(pred, h, FARAWAY)
        return val

    def surface_color(self, M: Vector) -> Vector:
        return self.color

    def compute_outward_normal(self, intersection_point: Vector) -> Vector:
        normal_at_intersection = (intersection_point - self.center).norm()
        return normal_at_intersection
    
    def compute_inward_normal(self, intersection_point: Vector) -> Vector:
        normal_at_intersection = (self.center - intersection_point).norm()
        return normal_at_intersection
    
#
# user elements for scenes
#
    
class SphereElement(Element, Sphere):
    def __init__(self, center: Vector, radius: float, color: Vector, transparent: bool = False, refractive_index: float = 1.0, user_params=None):
        super().__init__(center, color, transparent, refractive_index, user_params)
        self.radius = radius

    def intersect(self, origin: Vector, direction: Vector):
        return super().intersect(origin, direction)

    def surface_color(self, M: Vector) -> Vector:
        return super().surface_color(M)

    def compute_outward_normal(self, intersection_point: Vector) -> Vector:
        return super().compute_outward_normal(intersection_point)
    
    def compute_inward_normal(self, intersection_point: Vector) -> Vector:
        return super().compute_inward_normal(intersection_point)

class CheckeredSphereElement(SphereElement):
    def __init__(self, center: Vector, radius: float, color: Vector, transparent: bool = False, refractive_index: float = 1.0, checker_color: Vector = Vector(1, 1, 1), user_params=None):
        super().__init__(center, radius, color, transparent, refractive_index, user_params)
        self.checker_color = checker_color

    def surface_color(self, M: Vector) -> Vector:
        checker = ((M.x * 2).astype(int) % 2) == ((M.z * 2).astype(int) % 2)
        return self.color * checker + self.checker_color * (1 - checker)

#
# user sources for scenes
#

class IsotropicSource(Source, Sphere):
    def __init__(self, center: Vector, radius: float, color: Vector, pointing_direction: Vector):
        self.center = center
        self.radius = radius
        self.color = color
        self.pointing_direction = None # semantics for isotropic sources. no specific pointing direction.        