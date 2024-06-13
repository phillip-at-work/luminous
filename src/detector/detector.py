from abc import ABC, abstractmethod

from ..math.vector import Vector


class Detector(ABC):
    pass


class PowerMeter(Detector):
    pass


class Imager(Detector):
    def __init__(self, position: Vector, width: int, height: int):
        self.position = position
        self.width = width
        self.height = height
