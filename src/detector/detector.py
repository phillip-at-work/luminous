from abc import ABC, abstractmethod

from ..math.vector import Vector


class Detector(ABC):
    pass


class PowerMeter(Detector):
    pass


class Imager(Detector):
    def __init__(self, width: int, height: int, position: Vector, pointing_direction: Vector):
        '''
        Parameters:
            position (Vector): Detector's absolute position in 3D space
            width (float): Detector surface width
            height (float): Detector surface height
            pointing_direction (Vector): a noral vector defining Detector's pointing direction
        '''
        self.position = position
        self.width = width
        self.height = height
        self.pointing_direction = pointing_direction.norm()