from abc import ABC, abstractmethod

from ..math.vector import Vector

import logging
logger = logging.getLogger('luminous.source')

class Source(ABC):
    pass


class Laser(Source):
    pass


class Isotropic(Source):
    def __init__(self, position: Vector, pointing_direction: Vector):
        self.position = position
        self.pointing_direction = pointing_direction.norm()


class Custom(Source):
    pass
