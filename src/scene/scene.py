from functools import reduce
import numpy as np
from numpy.typing import NDArray
from abc import ABC
from PIL import Image
import time
import atexit
from typing import Tuple

from ..math.vector import Vector
from ..math.tools import extract
from ..detector.detector import Detector
from ..source.source import Source
from luminous.src.utilities.ray_debugger import NullRayDebugger, ConcreteRayDebugger

from luminous.src.utilities.logconfig import setup_logging
import logging
logger = logging.getLogger('luminous.scene')

FARAWAY = 1.0e39

##
## Scene
##


class Scene:

    def __init__(self, log_level=20, log_file="luminous.log") -> None:
        setup_logging(name='luminous', level=log_level, log_file=log_file)
        self.elements = list()
        self.ray_debugger = NullRayDebugger()
        self.compute_color_data = True # implement as boolean or null pattern

    def attach_ray_debugger(self, path="./results", filename="debug_ray_trace", display_3d_plot=True):
        self.ray_debugger = ConcreteRayDebugger()

        def plot_ray_trace_wrap_user_args():
            self.ray_debugger.plot(path=path, filename=filename, display_3d_plot=display_3d_plot)
        atexit.register(plot_ray_trace_wrap_user_args)

    def compute_ray_directions(self, detector_pointing_direction: Vector, detector_screen: Vector):

        initial_ray_dir = (detector_screen - self.detector_pos)

        ray_dir_x = initial_ray_dir.x
        ray_dir_y = initial_ray_dir.y
        ray_dir_z = initial_ray_dir.z + detector_pointing_direction.z

        ray_dir = Vector(ray_dir_x, ray_dir_y, ray_dir_z).norm()

        return ray_dir

    def create_screen_coord(self, w, h, detector_pointing_direction, detector_pos, vertical_screen_shift=0, horizontal_screen_shift=0, rotation=0) -> Vector:
        '''
        Create and return a Vector where each nth component of the vector represents one unique pixel in the detection screen;
        where the larger detector axis is normalized to +/- 1 and the smaller axis is scaled to aspect ratio.
        The detector screen surface is necessarily centered on detector_pos and normal to detector_pointing_direction.
        '''
        aspect_ratio = w / h
        screen = (  -1 + horizontal_screen_shift, 
                    1 / aspect_ratio + vertical_screen_shift, 
                    1 + horizontal_screen_shift, 
                    -1 / aspect_ratio + vertical_screen_shift)
        w_row = np.linspace(screen[0], screen[2], w)
        h_col = np.linspace(screen[1], screen[3], h)
        rows = np.tile(w_row, h)
        columns = np.repeat(h_col, w)
        d = np.zeros(len(rows))

        screen_coords = Vector(rows, columns, d)

        if rotation != 0:
            rotation_matrix = self._rotation_matrix(detector_pointing_direction, rotation)
            screen_coords = self._apply_rotation(screen_coords, rotation_matrix)

        screen_coords = screen_coords + detector_pos
        return screen_coords

    def _rotation_matrix(self, axis, theta):
        """
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
        """
        axis = axis.norm()
        a = np.cos(theta / 2.0)
        b, c, d = -axis.x * np.sin(theta / 2.0), -axis.y * np.sin(theta / 2.0), -axis.z * np.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

    def _apply_rotation(self, vector, rotation_matrix):
        """
        Apply the rotation matrix to the vector.
        """
        original_vectors = np.stack((vector.x, vector.y, vector.z), axis=-1)
        rotated_vectors = np.dot(original_vectors, rotation_matrix.T)
        return Vector(rotated_vectors[:, 0], rotated_vectors[:, 1], rotated_vectors[:, 2])

    def __iadd__(self, obj):
        if isinstance(obj, Element):
            self.elements.append(obj)
        elif isinstance(obj, Source):
            # TODO if self.source is not None, warn a user that they're overwriting their Source
            self.source = obj
        elif isinstance(obj, Detector):
            # TODO warn
            self.detector = obj
        else:
            raise TypeError("Only Elements, Sources, and Detectors can be added to Scenes!")

        return self
        
    def raytrace(self):
        self.start_time = time.perf_counter()

        self.detector_width = self.detector.width
        self.detector_height = self.detector.height
        self.detector_pos = self.detector.position
        detector_pointing_direction = self.detector.pointing_direction

        self.source_pos = self.source.position

        detector_pixels = self.create_screen_coord(self.detector.width, self.detector.height, detector_pointing_direction, self.detector_pos)
        self.pixel_incident_rays = self.compute_ray_directions(detector_pointing_direction, detector_pixels)

        self.ray_debugger.add_point(self.detector_pos, color=(0, 255, 0))
        detector_dir_translate = self.detector_pos + detector_pointing_direction
        self.ray_debugger.add_vector(start_point=self.detector_pos, end_point=detector_dir_translate, color=(255, 0, 0))
        self.ray_debugger.add_point(detector_pixels, color=(255,0,0))

        return self._recursive_trace(origin=detector_pixels, direction=self.pixel_incident_rays, elements=self.elements, bounce=0)

    def _recursive_trace(self, origin: Vector, direction: Vector, elements: list['Element'], bounce: int):

        ''' 
        distances between origin and element surface, along `direction` vector
        len(distances) = number of elements in scene
        len(distances)[n] = number of rays
        '''
        distances: list[NDArray[np.float64]] = [s.intersect(origin, direction) for s in elements]

        '''
        Find element-wise minimum for each detector pixel (ray)
        e.g., one ray may intersect two elements. if so, count nearest hit as the true hit.
        len(nearest) == numer of rays, where each value is the smallest distance from each sub-list of `distances`
        '''
        nearest: NDArray[np.float64] = reduce(np.minimum, distances)

        rays = Vector(0, 0, 0)
        for element, distance in zip(elements, distances):

            '''
            len(hit) == len(nearest)
            for each boolean in this array:
            if `nearest != FARAWAY`, and, for the `distance` in `distances`, if that `distance` is `nearest[n]`, then this ray intersects this `element`
            '''
            hit: NDArray[np.bool_] = (nearest != FARAWAY) & (distance == nearest)

            if np.any(hit):

                ray_travel_distance = extract(hit, distance)
                ray_start_position = origin.extract(hit)
                ray_pointing_direction = direction.extract(hit)

                intersection_point: Vector = ray_start_position + ray_pointing_direction * ray_travel_distance

                self.ray_debugger.add_vector(start_point=ray_start_position, end_point=intersection_point, color=(0,0,255))

                surface_normal_at_intersection: Vector = element.compute_intersection_geometry(intersection_point)

                # TODO at least some of this is unique to the color model
                to_source = (self.source_pos - intersection_point).norm()  # direction to light
                to_origin = (self.detector_pos - intersection_point).norm()  # direction to ray origin

                nudged = intersection_point + surface_normal_at_intersection * 0.0001  # M nudged to avoid itself

                # Shadow: find if the point is shadowed or not.
                # This amounts to finding out if M can see the light
                light_distances = [s.intersect(nudged, to_source) for s in elements]
                light_nearest = reduce(np.minimum, light_distances)
                seelight = light_distances[elements.index(element)] == light_nearest # TODO elements.index?

                if self.compute_color_data:
                    # Ambient
                    color = Vector(0.05, 0.05, 0.05) # TODO this should be a scene property or element property? what about for moving objects?

                    # Lambert shading (diffuse)
                    lv = np.maximum(surface_normal_at_intersection.dot(to_source), 0)
                    color += element.diffuse_color(intersection_point) * lv * seelight

                    # Reflection
                    if bounce < 2:
                        rayD = (ray_pointing_direction - surface_normal_at_intersection * 2 * ray_pointing_direction.dot(surface_normal_at_intersection)).norm()
                        color += self._recursive_trace(nudged, rayD, elements, bounce + 1) * element.reflectance

                    # Blinn-Phong shading (specular)
                    phong = surface_normal_at_intersection.dot((to_source + to_origin).norm())
                    color += Vector(1, 1, 1) * np.power(np.clip(phong, 0, 1), 50) * seelight

                rays += color.place(hit)

        return rays

    def resolve_rays(self, rays):
        rgb = [
            Image.fromarray(
                (255 * np.clip(c, 0, 1).reshape((self.detector_height, self.detector_width))).astype(np.uint8),
                "L",
            )
            for c in rays.components()
        ]

        return Image.merge("RGB", rgb)
    
    def elaspsed_time(self):
        return time.perf_counter() - self.start_time


##
## Elements
##


class Element(ABC):
    # TODO create and document abstract methods
    pass


class Cube(Element):
    pass


class PolygonVolume(Element):
    pass


class Sphere(Element):
    def __init__(self, center: Vector, radius: float, color: Vector, reflectance=0.5):
        self.center = center
        self.radius = radius
        self.color = color
        self.reflectance = reflectance

    def intersect(self, origin: Vector, direction: Vector):
        b = 2 * direction.dot(origin - self.center)
        c = (
            abs(self.center)
            + abs(origin)
            - 2 * self.center.dot(origin)
            - self.radius**2
        )
        discriminant = (b**2) - (4 * c)
        sq = np.sqrt(np.maximum(0, discriminant))
        h0 = (-b - sq) / 2
        h1 = (-b + sq) / 2
        h = np.where((h0 > 0) & (h0 < h1), h0, h1)
        pred = (discriminant > 0) & (h > 0)
        val = np.where(pred, h, FARAWAY)
        return val

    def diffuse_color(self, M):
        return self.color

    def compute_intersection_geometry(self, intersection_point: Vector) -> Vector:
        normal_at_intersection = (intersection_point - self.center) * (1.0 / self.radius)
        return normal_at_intersection

class CheckeredSphere(Sphere):
    def diffuse_color(self, M):
        checker = ((M.x * 2).astype(int) % 2) == ((M.z * 2).astype(int) % 2)
        return self.color * checker