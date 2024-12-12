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
        mag = np.sqrt(abs(self))
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
    
    def __str__(self):
        '''
        Print a vector as labelled columns.
        Method is generally slow and should be considered as a debug tool only.
        '''

        x_array = np.atleast_1d(self.x).astype(float)
        y_array = np.atleast_1d(self.y).astype(float)
        z_array = np.atleast_1d(self.z).astype(float)

        max_length = max(len(x_array), len(y_array), len(z_array))

        x_array = np.pad(x_array, (0, max_length - len(x_array)), constant_values=np.nan)
        y_array = np.pad(y_array, (0, max_length - len(y_array)), constant_values=np.nan)
        z_array = np.pad(z_array, (0, max_length - len(z_array)), constant_values=np.nan)

        max_len_x = max(len(f"{x:.1f}") for x in x_array if not np.isnan(x))
        max_len_y = max(len(f"{y:.1f}") for y in y_array if not np.isnan(y))
        max_len_z = max(len(f"{z:.1f}") for z in z_array if not np.isnan(z))

        max_len = max(max_len_x, max_len_y, max_len_z)

        header = f"{'x'.center(max_len)}    {'y'.center(max_len)}    {'z'.center(max_len)}\n"

        x_str = "\n".join(f"{x:>{max_len}.1f}" if not np.isnan(x) else ' ' * max_len for x in x_array)
        y_str = "\n".join(f"{y:>{max_len}.1f}" if not np.isnan(y) else ' ' * max_len for y in y_array)
        z_str = "\n".join(f"{z:>{max_len}.1f}" if not np.isnan(z) else ' ' * max_len for z in z_array)

        vector_str = "\n".join(f"{x_val}    {y_val}    {z_val}" for x_val, y_val, z_val in zip(x_str.splitlines(), y_str.splitlines(), z_str.splitlines()))

        return header + vector_str