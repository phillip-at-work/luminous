import numpy as np
import numbers

from .tools import extract


class Vector():
    __array_priority__ = 1000

    def __init__(self, x, y, z):
        (self.x, self.y, self.z) = (x, y, z)

    def __mul__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x * other.x, self.y * other.y, self.z * other.z)
        elif isinstance(other, (np.ndarray, numbers.Number)):
            return Vector(self.x * other, self.y * other, self.z * other)
        else:
            raise TypeError("Vector multiplication not compatible with type: %s" % type(other))
        
    def __rmul__(self, other):
        if isinstance(other, (np.ndarray, numbers.Number)):
            return self.__mul__(other)
        else:
            raise TypeError("Vector multiplication not compatible with type: %s" % type(other))

    def __add__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x + other.x, self.y + other.y, self.z + other.z)
        elif isinstance(other, (np.ndarray, numbers.Number)):
            return Vector(self.x + other, self.y + other, self.z + other)
        else:
            raise TypeError("Vector addition not compatible with type: %s" % type(other))
        
    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x - other.x, self.y - other.y, self.z - other.z)
        elif isinstance(other, (np.ndarray, numbers.Number)):
            return Vector(self.x - other, self.y - other, self.z - other)
        else:
            raise TypeError("Vector subtraction not compatible with type: %s" % type(other))
    
    def __rsub__(self, other):
        return self.__sub__(other)

    def dot(self, other):
        return (self.x * other.x) + (self.y * other.y) + (self.z * other.z)
        
    def cross(self, other, normalize=True):
        cross_product = Vector(self.y * other.z - self.z * other.y,
                               self.z * other.x - self.x * other.z,
                               self.x * other.y - self.y * other.x)
        return cross_product.norm() if normalize else cross_product

    def __abs__(self):
        return self.dot(self)
    
    def __pow__(self, power, modulo=None):
        if isinstance(power, (np.ndarray, numbers.Number)):
            return Vector(self.x ** power, self.y ** power, self.z ** power)
        else:
            raise TypeError("Vector power not compatible with type: %s" % type(power))

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
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # required to allow ndarray * Vector to be commutative
        # see: https://numpy.org/doc/stable/user/basics.subclassing.html
        if method == '__call__':
            if ufunc == np.multiply:
                if len(inputs) == 2:
                    if isinstance(inputs[0], Vector) and isinstance(inputs[1], (np.ndarray, numbers.Number)):
                        return Vector(inputs[0].x * inputs[1], inputs[0].y * inputs[1], inputs[0].z * inputs[1])
                    elif isinstance(inputs[1], Vector) and isinstance(inputs[0], (np.ndarray, numbers.Number)):
                        return Vector(inputs[1].x * inputs[0], inputs[1].y * inputs[0], inputs[1].z * inputs[0])
            elif ufunc == np.add:
                if len(inputs) == 2:
                    if isinstance(inputs[0], Vector) and isinstance(inputs[1], (np.ndarray, numbers.Number)):
                        return Vector(inputs[0].x + inputs[1], inputs[0].y + inputs[1], inputs[0].z + inputs[1])
                    elif isinstance(inputs[1], Vector) and isinstance(inputs[0], (np.ndarray, numbers.Number)):
                        return Vector(inputs[1].x + inputs[0], inputs[1].y + inputs[0], inputs[1].z + inputs[0])
            elif ufunc == np.subtract:
                if len(inputs) == 2:
                    if isinstance(inputs[0], Vector) and isinstance(inputs[1], (np.ndarray, numbers.Number)):
                        return Vector(inputs[0].x - inputs[1], inputs[0].y - inputs[1], inputs[0].z - inputs[1])
                    elif isinstance(inputs[1], Vector) and isinstance(inputs[0], (np.ndarray, numbers.Number)):
                        return Vector(inputs[1].x - inputs[0], inputs[1].y - inputs[0], inputs[1].z - inputs[0])
            elif ufunc == np.power:
                if len(inputs) == 2:
                    if isinstance(inputs[0], Vector) and isinstance(inputs[1], (np.ndarray, numbers.Number)):
                        return Vector(inputs[0].x ** inputs[1], inputs[0].y ** inputs[1], inputs[0].z ** inputs[1])
                    elif isinstance(inputs[1], Vector) and isinstance(inputs[0], (np.ndarray, numbers.Number)):
                        return Vector(inputs[1].x ** inputs[0], inputs[1].y ** inputs[0], inputs[1].z ** inputs[0])
        return NotImplemented