from abc import ABC, abstractmethod

from ..math.vector import Vector


class Detector(ABC):
    @abstractmethod
    def capture_data(self):
        pass

    @abstractmethod
    def view_data(self):
        pass


class PowerMeter(Detector):
    def capture_data(self):
        raise NotImplementedError(f"Method not yet implementated in {self.__class__}")
    
    def view_data(self):
        raise NotImplementedError(f"Method not yet implementated in {self.__class__}")


class Imager(Detector):
    def __init__(self, width: int, height: int, position: Vector, pointing_direction: Vector):
        '''
        Parameters:
            position (Vector): Detector's absolute position in 3D space
            width (float): Detector width in pixels
            height (float): Detector height in pixels
            pointing_direction (Vector): vector defining `Detector` pointing direction
        '''
        self.position = position
        self.width = width
        self.height = height
        self.pointing_direction = pointing_direction.norm()

    def capture_data(self):
        pass

    def view_data(self):
        pass