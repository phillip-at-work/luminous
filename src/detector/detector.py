from abc import ABC, abstractmethod
import numpy as np
from PIL import Image

from ..math.vector import Vector


class Detector(ABC):
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

    @abstractmethod
    def _capture_data(self):
        '''
        Compile data and iteratively realize the model unique to that detector
        '''
        pass

    @abstractmethod
    def view_data(self):
        '''
        Perform any final processing and return data to user
        '''
        pass


class PowerMeter(Detector):
    def __init__(self, width: int, height: int, position: Vector, pointing_direction: Vector):
        super().__init__(width, height, position, pointing_direction)
        
    def _capture_data(self):
        raise NotImplementedError(f"Method not yet implementated in {self.__class__}")
    
    def view_data(self):
        raise NotImplementedError(f"Method not yet implementated in {self.__class__}")


class Imager(Detector):
    def __init__(self, width: int, height: int, position: Vector, pointing_direction: Vector):
        super().__init__(width, height, position, pointing_direction)

    def _capture_data(self):
        pass

    def view_data(self):
        '''
        Pixel data is a color
        '''
        pass