import matplotlib.pyplot as plt
import numpy as np

from src.math.vector import Vector
from src.scene.scene import Scene
from src.scene.scene import Sphere, CheckeredSphere
from src.detector.detector import Imager
from src.source.source import Point


iterations = 100
t = 0

for i in range(iterations):

    # TODO source should also have pointing direction. for an isotropic source, use arbitrary default.
    source = Point(position=Vector(5, 5, -10), pointing_direction=Vector(0, 0, 1))
    detector = Imager(width=400, height=300, position=Vector(0, 0.35, -1), pointing_direction=Vector(0, 0, 1))
    scene = Scene(source=source, detector=detector)

    scene += Sphere(Vector(0.75, 0.1, 1), 0.6, Vector(0, 0, 1))
    scene += Sphere(Vector(-0.75, 0.1, 2.25), 0.6, Vector(0.5, 0.223, 0.5))
    scene += Sphere(Vector(-2.75, 0.1, 3.5), 0.6, Vector(1, 0.572, 0.184))
    scene += CheckeredSphere(Vector(0, -99999.5, 0), 99999, Vector(0.75, 0.75, 0.75), 0.25)

    rays = scene.raytrace()
    image = scene.resolve_rays(rays)
    t += scene.elaspsed_time()


print(f"runtime: {t/iterations}")
# 0.0522s

plt.imshow(image)
plt.show(block=True)