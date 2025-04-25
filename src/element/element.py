from abc import ABC, abstractmethod
import numpy as np

from ..math.vector import Vector

FARAWAY = 1.0e39

class Element(ABC):
    def __init__(self, center: Vector, color: Vector, reflectance: float = 0.5, transmittance: float = 0.0, refractive_index: float = 1.0):
        '''
        Parameters:
            center (Vector): Element's center position in 3D space
            color (Vector): Element's color
            reflectance (float): Reflectance coefficient of the element
            transmittance (float): Transmittance coefficient of the element
            refractive_index (float): Refractive index of the element
        '''
        self.center = center
        self.color = color
        self.reflectance = reflectance
        self.transmittance = transmittance
        self.refractive_index = refractive_index

    @abstractmethod
    def intersect(self, origin: Vector, direction: Vector):
        '''
        Calculate intersection of a ray with the element.
        '''
        pass

    @abstractmethod
    def diffuse_color(self, M: Vector) -> Vector:
        '''
        Calculate the diffuse color at a given point on the element.
        '''
        pass

    @abstractmethod
    def compute_intersection_normal(self, intersection_point: Vector) -> Vector:
        '''
        Compute normal vector associated with incident Vector's point of intersection.
        '''
        pass

class Sphere(Element):
    def __init__(self, center: Vector, radius: float, color: Vector, reflectance: float = 0.5, transmittance: float = 0.0, refractive_index: float = 1.0):
        super().__init__(center, color, reflectance, transmittance, refractive_index)
        self.radius = radius

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

    def diffuse_color(self, M: Vector) -> Vector:
        return self.color

    def compute_intersection_normal(self, intersection_point: Vector) -> Vector:
        normal_at_intersection = (intersection_point - self.center) * (1.0 / self.radius)
        return normal_at_intersection

class CheckeredSphere(Sphere):
    def diffuse_color(self, M: Vector) -> Vector:
        checker = ((M.x * 2).astype(int) % 2) == ((M.z * 2).astype(int) % 2)
        return self.color * checker