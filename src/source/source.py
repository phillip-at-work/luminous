from abc import ABC, abstractmethod

from ..math.vector import Vector

import logging
from luminous.src.utilities.logconfig import setup_logging
logger = logging.getLogger(__name__)

class Source(ABC):
    pass


class Laser(Source):
    pass


class Isotropic(Source):
    def __init__(self, position: Vector, color: Vector, pointing_direction: Vector):
        self.position = position
        self.color = color
        self.pointing_direction = None # semantics for isotropic sources. no specific pointing direction.


class Custom(Source):
    pass
