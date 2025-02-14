from functools import reduce
import numpy as np
from numpy.typing import NDArray
from abc import ABC
from PIL import Image
import time
import atexit

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

    def compute_ray_directions(self, camera_normal: Vector, detector_screen):
        up = Vector(0, 1, 0) # TODO (0,0,1)?
        right = camera_normal.cross(up, normalize=False)
        initial_ray_dir = (detector_screen - self.detector_pos).norm()
        
        right_components = right.components()
        up_components = up.components()
        normal_components = camera_normal.components()
        
        original_vectors = np.stack((initial_ray_dir.x, initial_ray_dir.y, initial_ray_dir.z), axis=-1)
        rotated_vectors = np.dot(original_vectors, [right_components, up_components, normal_components])
        ray_dir_x, ray_dir_y, ray_dir_z = rotated_vectors.T.astype(np.float64)
        
        return Vector(ray_dir_x, ray_dir_y, ray_dir_z)
        
    def create_screen_coord(self, w, h, vertical_screen_shift=0, horizontal_screen_shift=0) -> Vector:
        '''
        Create and return a Vector where each nth component of the vector represents one unique pixel in the detection screen;
        where the larger detector axis is normalized to +/- 1 and the smaller axis is scaled to aspect ratio.
        The detector screen surface is necessarily centered on 0 (x-y plane) by default.
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
        return Vector(rows, columns, d)

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

        self.detector_width: float = self.detector.width
        self.detector_height: float = self.detector.height
        self.detector_pos: Vector = self.detector.position
        detector_pointing_direction: Vector = self.detector.pointing_direction

        self.source_pos: Vector = self.source.position
        # TODO source should also have pointing direction, like detector. for an isotropic source, use arbitrary default.
        source_pointing_direction: Vector = self.source.pointing_direction

        detector_screen: Vector = self.create_screen_coord(self.detector.width, self.detector.height)
        self.detector_pixels: Vector = self.compute_ray_directions(detector_pointing_direction, detector_screen)

        # debug ray plotting: detector
        self.ray_debugger.add_point(self.detector_pos, color=(0,255,0))
        detector_dir_translate = self.detector_pos + detector_pointing_direction
        self.ray_debugger.add_vector(start_point=self.detector_pos, end_point=detector_dir_translate, color=(255,0,0))

        return self._recursive_trace(origin=self.detector_pos, direction=self.detector_pixels, elements=self.elements, bounce=0)

    def _recursive_trace(self, origin: Vector, direction: Vector, elements: list['Element'], bounce: int):

        ''' 
        distances between origin and element surface, along direction vector
        length of outer list == number of elements in scene for which intersections exist
        length of inner np.ndarrays == number of detector pixels (rays)
        '''
        distances: list[np.ndarray] = [s.intersect(origin, direction) for s in elements]

        '''
        Find element-wise minimum for each detector pixel (ray)
        e.g., identify which element, if any, is hit first
        '''
        nearest: np.ndarray = reduce(np.minimum, distances)

        rays = Vector(0, 0, 0)
        for element, distance in zip(elements, distances):

            hit: NDArray[np.bool_] = (nearest != FARAWAY) & (distance == nearest)

            if np.any(hit):
                # TODO rename and identify how this block works
                dc = extract(hit, distance)
                Oc = origin.extract(hit)
                Dc = direction.extract(hit)
                M, N = element.new_ray_direction(Oc, Dc, dc)

                if self.compute_color_data:
                    to_source = (self.source_pos - M).norm()  # direction to light
                    to_origin = (self.detector_pos - M).norm()  # direction to ray origin
                    nudged = M + N * 0.0001  # M nudged to avoid itself

                    # Shadow: find if the point is shadowed or not.
                    # This amounts to finding out if M can see the light
                    light_distances = [s.intersect(nudged, to_source) for s in elements]
                    light_nearest = reduce(np.minimum, light_distances)
                    seelight = light_distances[elements.index(element)] == light_nearest # TODO elements.index?

                    # Ambient
                    color = Vector(0.05, 0.05, 0.05) # TODO this should be a scene property or element property? what about for moving objects?

                    # Lambert shading (diffuse)
                    lv = np.maximum(N.dot(to_source), 0)
                    color += element.diffuse_color(M) * lv * seelight

                    # Reflection
                    if bounce < 2:
                        rayD = (Dc - N * 2 * Dc.dot(N)).norm()
                        color += self._recursive_trace(nudged, rayD, elements, bounce + 1) * element.reflectance

                    # Blinn-Phong shading (specular)
                    phong = N.dot((to_source + to_origin).norm())
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

    def new_ray_direction(self, origin: Vector, D: Vector, d: Vector):
        M = origin + D * d  # intersection point
        N = (M - self.center) * (1.0 / self.radius)  # normal
        return M, N

class CheckeredSphere(Sphere):
    def diffuse_color(self, M):
        checker = ((M.x * 2).astype(int) % 2) == ((M.z * 2).astype(int) % 2)
        return self.color * checker