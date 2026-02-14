from abc import ABC, abstractmethod
import numpy as np

from ..math.vector import Vector
from .shape import Circle, Sphere

class Source(ABC):

    '''An object in the scene which does emit light'''
    
    def __init__(self):

        # these represent rays which terminate at a source during a reverse ray trace
        # enqueued as though they were leaving the source. not used as of 2/11/26.
        self.ray_emission_direction = dict()
        self.ray_emission_origin = dict()

    def _enqueue_rays(self, origin: Vector, direction: Vector, detector):
        '''
        Enqueue rays from a reverse ray trace, which can be replayed in the forward direction.
        '''
        
        if self.ray_emission_direction.get(detector) is None:
            self.ray_emission_direction[detector] = direction
            self.ray_emission_origin[detector] = origin
        
        else:
            self.ray_emission_direction[detector]._merge(direction)
            self.ray_emission_origin[detector]._merge(origin)

class IsotropicSource(Source, Sphere):
    def __init__(self, center: Vector, radius: float, color: Vector, pointing_direction: Vector):
        Source.__init__(self)
        self.center = center
        self.radius = radius
        self.color = color
        self.pointing_direction = None # semantics for isotropic sources. no specific pointing direction.    