from abc import ABC, abstractmethod

from ..math.vector import Vector

class Source(ABC):
    pass


class Laser(Source):
    pass


class Point(Source):
    def __init__(self, position: Vector, pointing_direction: Vector):
        self.position = position
        self.pointing_direction = pointing_direction.norm()


class Custom(Source):
    pass
