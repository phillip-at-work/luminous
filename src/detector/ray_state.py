from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from ..math.vector import Vector
from ..element.element import Element

class RayState(ABC):
    def __init__(self, width: int, height: int, ):
        '''
        Parameters:
            width (float): Detector width in pixels
            height (float): Detector height in pixels
        '''
        self.width = width
        self.height = height

        self._data = None

    @abstractmethod
    def _initial_state(self):
        '''
        Initial state of a ray before any reflections or refractions
        '''
        pass

    @abstractmethod
    def _reflection_model(  self, 
                            element: Element,
                            intersection_point: Vector,
                            surface_normal_at_intersection: Vector, 
                            direction_to_origin_unit: Vector, 
                            intersection_map: list[dict],
                            reflection_weights: NDArray[np.number]) -> Vector:
        '''
        Maintain status of ray through reflection event
        '''
        pass

    @abstractmethod
    def _transmission_into_volume_model(self, 
                                        element: Element,
                                        initial_intersection: Vector,
                                        final_intersection: Vector,
                                        transmission_weights: NDArray[np.number]):        
        '''
        Maintain status of ray as it transmits into volume
        '''
        pass

    @abstractmethod
    def _transmission_from_volume_model(self, 
                                        element: Element,
                                        initial_intersection: Vector,
                                        final_intersection: Vector,
                                        transmission_weights: NDArray[np.number]):        
        '''
        Maintain status of ray as it transmits from volume
        '''
        pass

class RayIsInsideElement(RayState):
    def __init__(self, width: int, height: int):
        super().__init__(width, height)
        self._initial_state()

    def _initial_state(self):

        # assumed that the detector is not inside an element
        self._data = np.zeros(self.width * self.height, dtype=bool)

    def _reflection_model(self, intersection_point):
        
        # if the ray is reflecting, it's not inside an element
        return np.zeros(intersection_point.x.shape, dtype=bool)
    
    def _transmission_into_volume_model(self, element, initial_intersection, final_intersection, transmission_weights):
        
        # rays necessarily refract into our out of elements
        return np.ones(initial_intersection.x.shape, dtype=bool)
    
    def _transmission_from_volume_model(self, element, initial_intersection, final_intersection, transmission_weights):
        
        pass

    def view_data(self):
        pass