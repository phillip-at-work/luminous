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

    @abstractmethod
    def _compute_initial_ray_directions(self):
        '''
        Compute initial ray trajectories FROM a source. Used in forward traces only.
        '''
        pass

    def _enqueue_rays(self, origin: Vector, direction: Vector, detector):
        '''
        Enqueue rays from a reverse ray trace, which can be replayed in the forward direction.
        When replaying rays, use initial recursion call: _recursive_path_trace(... origin=source.ray_emission_origin[detector], ...)
        '''
        
        if self.ray_emission_direction.get(detector) is None:
            self.ray_emission_direction[detector] = direction
            self.ray_emission_origin[detector] = origin
        
        else:
            self.ray_emission_direction[detector]._merge(direction)
            self.ray_emission_origin[detector]._merge(origin)

class IsotropicSource(Source, Sphere):
    def __init__(self, center: Vector, radius: float, color: Vector):
        Sphere.__init__(self, center, radius)
        Source.__init__(self)
        self.color = color

    def _compute_initial_ray_directions(self, pixels: Vector):
        pass

class Laser(Source, Circle):
    def __init__(self, center: Vector, radius: float, pointing_direction: Vector, color: Vector, pixel_count: int):
        Circle.__init__(self, center, radius, pointing_direction, pixel_count)
        Source.__init__(self)
        self.color = color

    def _compute_initial_ray_directions(self, pixels: Vector):

        # TODO future implementations will allow for some extent of bream divergence

        initial_ray_dir = (pixels - self.center)

        ray_dir_x = initial_ray_dir.x
        ray_dir_y = initial_ray_dir.y
        ray_dir_z = initial_ray_dir.z + self.pointing_direction.z

        ray_dir = Vector(ray_dir_x, ray_dir_y, ray_dir_z)

        return ray_dir