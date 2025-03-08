import numpy as np

from .tools import extract


class Vector():

    def __init__(self, x, y, z):
        (self.x, self.y, self.z) = (x, y, z)

    def __mul__(self, other):
        return Vector(self.x * other, self.y * other, self.z * other)

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def dot(self, other):
        return (self.x * other.x) + (self.y * other.y) + (self.z * other.z)
        
    def cross(self, other, normalize=True):
        cross_product = Vector(self.y * other.z - self.z * other.y,
                               self.z * other.x - self.x * other.z,
                               self.x * other.y - self.y * other.x)
        return cross_product.norm() if normalize else cross_product

    def __abs__(self):
        return self.dot(self)

    def norm(self):
        mag = self.magnitude()
        return self * (1.0 / np.where(mag == 0, 1, mag))

    def components(self):
        return (self.x, self.y, self.z)

    def extract(self, cond):
        return Vector(
            extract(cond, self.x), extract(cond, self.y), extract(cond, self.z)
        )

    def place(self, cond):
        r = Vector(np.zeros(cond.shape), np.zeros(cond.shape), np.zeros(cond.shape))
        np.place(r.x, cond, self.x)
        np.place(r.y, cond, self.y)
        np.place(r.z, cond, self.z)
        return r
    
    def magnitude(self):
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)