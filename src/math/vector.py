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
        if isinstance(other, Vector):
            return Vector(other.x - self.x, other.y - self.y, other.z - self.z)
        elif isinstance(other, (np.ndarray, numbers.Number)):
            return Vector(other - self.x, other - self.y, other - self.z)
        else:
            raise TypeError("Subtraction not compatible with type: %s" % type(other))

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
        
    def _merge(self, other):

        # merge arrays. redundant entries persist.
        
        def to_array(value):
            if isinstance(value, (float, int)):
                return np.array([value])
            elif isinstance(value, np.ndarray):
                return value
            else:
                raise TypeError("Unsupported type for merging: %s" % type(value))

        if not isinstance(other, Vector):
            raise TypeError("Cannot merge with non-Vector object")

        x1, y1, z1 = map(to_array, [self.x, self.y, self.z])
        x2, y2, z2 = map(to_array, [other.x, other.y, other.z])

        merged_x = np.concatenate((x1, x2))
        merged_y = np.concatenate((y1, y2))
        merged_z = np.concatenate((z1, z2))

        return Vector(merged_x, merged_y, merged_z)

    def norm(self):
        mag = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        return Vector(self.x / mag, self.y / mag, self.z / mag)

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
            elif ufunc == np.exp:
                if len(inputs) == 1:
                    input_vector = inputs[0]
                    return Vector(np.exp(input_vector.x), np.exp(input_vector.y), np.exp(input_vector.z))

        return NotImplemented