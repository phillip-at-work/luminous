from functools import reduce
import numpy as np
from abc import ABC

from ..math.vector import Vector
from ..math.tools import extract
from ..detector.detector import Detector
from ..source.source import Source
from PIL import Image
import time

FARAWAY = 1.0e39

##
## Scene
##


class Scene:

    def __init__(self, source: Source, detector: Detector):

        # TODO
        # add parameters to `init` associated with default objects
        # e.g., if a user doesn't pass in a `Medium`, make one with a default config

        self.detector_width = detector.width
        self.detector_height = detector.height

        self.elements = list()

        self.source_pos: Vector = source.position
        self.detector_pos: Vector = detector.position
        self.Q = self.vector_field(detector.width, detector.height)
        self.ray_dir: Vector = (self.Q - self.detector_pos).norm()

    def vector_field(self, w, h):
        r = float(w) / h
        S = (-1, 1 / r + 0.25, 1, -1 / r + 0.25)
        x_vals = np.linspace(S[0], S[2], w)
        y_vals = np.linspace(S[1], S[3], h)
        x = np.tile(x_vals, h)
        y = np.repeat(y_vals, w)
        return Vector(x, y, 0)

    def __iadd__(self, obj):
        if not isinstance(obj, Element):
            raise TypeError("Only Elements can be added to Scenes!")
        self.elements.append(obj)
        return self

    def raytrace(self, origin=None, direction=None, elements=None, bounce=0):

        if origin is None:
            origin = self.detector_pos
        if direction is None:
            direction = self.ray_dir
        if elements is None:
            elements = self.elements

        distances = [s.intersect(origin, direction) for s in elements]
        nearest = reduce(np.minimum, distances)
        rays = Vector(0, 0, 0)
        for s, d in zip(elements, distances):
            hit = (nearest != FARAWAY) & (d == nearest)
            if np.any(hit):
                dc = extract(hit, d)
                Oc = origin.extract(hit)
                Dc = direction.extract(hit)
                cc = s.light(
                    self.source_pos,
                    self.detector_pos,
                    Oc,
                    Dc,
                    dc,
                    elements,
                    self,
                    bounce,
                )
                rays += cc.place(hit)
        return rays

    def resolve_rays(self, rays):
        rgb = [
            Image.fromarray(
                (
                    255
                    * np.clip(c, 0, 1).reshape(
                        (self.detector_height, self.detector_width)
                    )
                ).astype(np.uint8),
                "L",
            )
            for c in rays.components()
        ]

        return Image.merge("RGB", rgb)


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
    def __init__(self, center: Vector, radius: float, diffuse: Vector, mirror=0.5):
        self.center = center
        self.radius = radius
        self.diffuse = diffuse
        self.mirror = mirror

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

    def diffusecolor(self, M):
        return self.diffuse

    def light(  # TODO rename
        self,
        source: Vector,
        detector: Vector,
        origin: Vector,
        D: Vector,
        d: Vector,
        elements: list[Element],
        scene: Scene,
        bounce: int,
    ):
        M = origin + D * d  # intersection point
        N = (M - self.center) * (1.0 / self.radius)  # normal
        to_source = (source - M).norm()  # direction to light
        to_origin = (detector - M).norm()  # direction to ray origin
        nudged = M + N * 0.0001  # M nudged to avoid itself

        # Shadow: find if the point is shadowed or not.
        # This amounts to finding out if M can see the light
        light_distances = [s.intersect(nudged, to_source) for s in elements]
        light_nearest = reduce(np.minimum, light_distances)
        seelight = light_distances[elements.index(self)] == light_nearest

        # Ambient
        color = Vector(0.05, 0.05, 0.05)

        # Lambert shading (diffuse)
        lv = np.maximum(N.dot(to_source), 0)
        color += self.diffusecolor(M) * lv * seelight

        # Reflection
        if bounce < 10:
            rayD = (D - N * 2 * D.dot(N)).norm()
            color += scene.raytrace(nudged, rayD, elements, bounce + 1) * self.mirror

        # Blinn-Phong shading (specular)
        phong = N.dot((to_source + to_origin).norm())
        color += Vector(1, 1, 1) * np.power(np.clip(phong, 0, 1), 50) * seelight
        return color


class CheckeredSphere(Sphere):
    def diffusecolor(self, M):
        checker = ((M.x * 2).astype(int) % 2) == ((M.z * 2).astype(int) % 2)
        return self.diffuse * checker
