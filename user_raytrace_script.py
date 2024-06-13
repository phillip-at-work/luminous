import matplotlib.pyplot as plt

from src.math.vector import Vector
from src.scene.scene import Scene
from src.scene.scene import Sphere, CheckeredSphere
from src.detector.detector import Imager
from src.source.source import Point


source = Point(Vector(5, 5, -10))
detector = Imager(Vector(0, 0.35, -1), 400, 300)
scene = Scene(source, detector)

scene += Sphere(Vector(0.75, 0.1, 1), 0.6, Vector(0, 0, 1))
scene += Sphere(Vector(-0.75, 0.1, 2.25), 0.6, Vector(0.5, 0.223, 0.5))
scene += Sphere(Vector(-2.75, 0.1, 3.5), 0.6, Vector(1, 0.572, 0.184))
scene += CheckeredSphere(Vector(0, -99999.5, 0), 99999, Vector(0.75, 0.75, 0.75), 0.25)

rays = scene.raytrace()
image = scene.resolve_rays(rays)

plt.imshow(image)
plt.show(block=True)
