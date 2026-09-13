"""
Geometric tests for reverse ray tracing functionality.

These tests verify that rays propagate correctly through various combinations
of reflections and transmissions. Each test uses a 1x1 camera pixel to trace
a single ray and confirms whether the ray reaches the detector based on the
scene geometry.

Consistent with reverse ray traces, these tests validate geometry only.

Each test includes:
1. Geometry that should allow the ray to reach the detector (non-zero pixel)
2. Geometry that should prevent the ray from reaching the detector (zero pixel)
"""

import pytest
import numpy as np
from src.math.vector import Vector
from src.scene.scene import Scene
from src.element.element import SphereElement
from src.element.source import IsotropicSource
from src.element.detector import Camera


def pixel_is_illuminated(camera: Camera) -> bool:
    """
    Check if a 1x1 camera pixel received any light.
    
    Args:
        camera: Camera with 1x1 resolution
        
    Returns:
        True if pixel has non-zero intensity, False otherwise
    """
    data = camera._reverse_trace_data

    if any(arr.size > 1 for arr in (data.x, data.y, data.z)):
        raise ValueError("Sensor data should be 1-dimensional. Ensure sensor has 1 pixel only.")

    return bool((data.x[0] > 0) or (data.y[0] > 0) or (data.z[0] > 0))


class TestSingleReflection:
    """Test geometry with one reflection."""
    
    def test_one_reflection_success(self):
        """Ray should reach detector via one reflection when geometry allows."""
        scene = Scene(reverse_trace=True)
        # scene.attach_ray_debugger()

        source = IsotropicSource(
            center=Vector(0, -1, 2),
            radius=0.3,
            color=Vector(1, 1, 1)
        )

        mirror_sphere = SphereElement(
            center=Vector(0, 0.3, 2),
            radius=0.5,
            color=Vector(0.9, 0.9, 0.9),
            user_params={'specular': 0.8, 'n_s': 50}
        )
        
        # 1x1 camera
        camera = Camera(
            width=1,
            height=1,
            position=Vector(0, 0, 4),
            pointing_direction=Vector(0, 0, -1),
            screen_width=0.01,
            screen_height=0.01
        )
        
        scene += source
        scene += mirror_sphere
        scene += camera
        scene.raytrace()
        
        assert pixel_is_illuminated(camera), \
            "Ray should reach detector via one reflection"
    
    def test_one_reflection_failure(self):
        """Ray should NOT reach detector when geometry prevents one reflection."""
        scene = Scene(reverse_trace=True)
        # scene.attach_ray_debugger()

        source = IsotropicSource(
            center=Vector(0, 0.3, -1),
            radius=0.3,
            color=Vector(1, 1, 1)
        )

        mirror_sphere = SphereElement(
            center=Vector(0, 0.3, 2),
            radius=0.5,
            color=Vector(0.9, 0.9, 0.9),
            user_params={'specular': 0.8, 'n_s': 50}
        )
        
        # 1x1 camera
        camera = Camera(
            width=1,
            height=1,
            position=Vector(0, 0, 4),
            pointing_direction=Vector(0, 0, -1),
            screen_width=0.01,
            screen_height=0.01
        )
        
        scene += source
        scene += mirror_sphere
        scene += camera
        scene.raytrace()
        
        assert not pixel_is_illuminated(camera), \
            "Ray should NOT reach detector when source is blocked by sphere"

class TestTwoReflections:
    """Test geometry with two reflections."""
    
    def test_two_reflections_success(self):
        """Ray should reach detector via two reflections when geometry allows."""
        scene = Scene(reverse_trace=True)
        # scene.attach_ray_debugger()

        source = IsotropicSource(
            center=Vector(-0.3, 0, 1),
            radius=0.3,
            color=Vector(1, 1, 1)
        )

        sphere1 = SphereElement(
            center=Vector(-0.3, 0, 2),
            radius=0.5,
            color=Vector(1, 0, 0),
            user_params={'specular': 0.8, 'n_s': 50}
        )

        sphere2 = SphereElement(
            center=Vector(2, 0, 3.5),
            radius=0.5,
            color=Vector(0, 1, 0),
            user_params={'specular': 0.8, 'n_s': 50}
        )
        
        # 1x1 camera
        camera = Camera(
            width=1,
            height=1,
            position=Vector(0, 0, 4),
            pointing_direction=Vector(0, 0, -1),
            screen_width=0.01,
            screen_height=0.01
        )
        
        scene += source
        scene += sphere1
        scene += sphere2
        scene += camera
        scene.raytrace()
        
        assert pixel_is_illuminated(camera), \
            "Ray should reach detector via two reflections"
    
    def test_two_reflections_failure(self):
        """Ray should NOT reach detector when geometry prevents two reflections."""
        scene = Scene(reverse_trace=True)
        # scene.attach_ray_debugger()

        source = IsotropicSource(
            center=Vector(-1, 0, 1),
            radius=0.3,
            color=Vector(1, 1, 1)
        )

        sphere1 = SphereElement(
            center=Vector(-0.3, 0, 2),
            radius=0.5,
            color=Vector(1, 0, 0),
            user_params={'specular': 0.8, 'n_s': 50}
        )

        sphere2 = SphereElement(
            center=Vector(2, 0, 3.5),
            radius=0.5,
            color=Vector(0, 1, 0),
            user_params={'specular': 0.8, 'n_s': 50}
        )
        
        # 1x1 camera
        camera = Camera(
            width=1,
            height=1,
            position=Vector(0, 0, 4),
            pointing_direction=Vector(0, 0, -1),
            screen_width=0.01,
            screen_height=0.01
        )
        
        scene += source
        scene += sphere1
        scene += sphere2
        scene += camera
        scene.raytrace()
        
        assert not pixel_is_illuminated(camera), \
            "Ray should NOT reach detector when two-bounce geometry is wrong"


class TestThreeReflections:
    """Test geometry with three reflections."""
    
    def test_three_reflections_success(self):
        """Ray should reach detector via three reflections when geometry allows."""
        scene = Scene(reverse_trace=True)
        # scene.attach_ray_debugger()

        source = IsotropicSource(
            center=Vector(-0.95, 0, 1),
            radius=0.3,
            color=Vector(1, 1, 1)
        )

        sphere1 = SphereElement(
            center=Vector(-0.3, 0, 2),
            radius=0.5,
            color=Vector(1, 0, 0),
            user_params={'specular': 0.8, 'n_s': 50}
        )

        sphere2 = SphereElement(
            center=Vector(2, 0, 3.5),
            radius=0.5,
            color=Vector(0, 1, 0),
            user_params={'specular': 0.8, 'n_s': 50}
        )

        sphere3 = SphereElement(
            center=Vector(2.5, 0.5, 1.2),
            radius=0.5,
            color=Vector(0, 1, 0),
            user_params={'specular': 0.8, 'n_s': 50}
        )
        
        # 1x1 camera
        camera = Camera(
            width=1,
            height=1,
            position=Vector(0, 0, 4),
            pointing_direction=Vector(0, 0, -1),
            screen_width=0.01,
            screen_height=0.01
        )
        
        scene += source
        scene += sphere1
        scene += sphere2
        scene += sphere3
        scene += camera
        scene.raytrace()
        
        assert pixel_is_illuminated(camera), \
            "Ray should reach detector via three reflections"
    
    def test_three_plus_one_reflections_success(self):
        """Ray should reach detector via four reflections along two paths"""
        # TODO not yet supported in the framework!
        # not clear to me how to test this. currently, it returns a valid pixel
        # but this does not keep track of if the pixel originates from two paths!

        scene = Scene(reverse_trace=True)
        # scene.attach_ray_debugger()

        source = IsotropicSource(
            center=Vector(-0.75, 0, 1),
            radius=0.3,
            color=Vector(1, 1, 1)
        )

        sphere1 = SphereElement(
            center=Vector(-0.3, 0, 2),
            radius=0.5,
            color=Vector(1, 0, 0),
            user_params={'specular': 0.8, 'n_s': 50}
        )

        sphere2 = SphereElement(
            center=Vector(2, 0, 3.5),
            radius=0.5,
            color=Vector(0, 1, 0),
            user_params={'specular': 0.8, 'n_s': 50}
        )

        sphere3 = SphereElement(
            center=Vector(2.5, 0.5, 1.2),
            radius=0.5,
            color=Vector(0, 1, 0),
            user_params={'specular': 0.8, 'n_s': 50}
        )
        
        # 1x1 camera
        camera = Camera(
            width=1,
            height=1,
            position=Vector(0, 0, 4),
            pointing_direction=Vector(0, 0, -1),
            screen_width=0.01,
            screen_height=0.01
        )
        
        scene += source
        scene += sphere1
        scene += sphere2
        scene += sphere3
        scene += camera
        scene.raytrace()
        
        assert pixel_is_illuminated(camera), \
            "Ray should reach detector via three reflections"


class TestSingleTransmission:
    """Test geometry with one transmission."""
    
    def test_one_transmission_success(self):
        """Ray should reach detector via one transmission when geometry allows."""
        scene = Scene(reverse_trace=True)
        
        # Source directly behind transparent sphere
        source = IsotropicSource(
            center=Vector(0, 0, 0),
            radius=0.3,
            color=Vector(1, 1, 1)
        )
        
        # Transparent sphere between source and camera
        glass_sphere = SphereElement(
            center=Vector(0, 0, 2),
            radius=0.5,
            color=Vector(0.9, 0.9, 1.0),
            transparent=True,
            refractive_index=1.5,
            user_params={'specular': 0.1, 'n_s': 10}
        )
        
        # 1x1 camera looking through sphere at source
        camera = Camera(
            width=1,
            height=1,
            position=Vector(0, 0, 4),
            pointing_direction=Vector(0, 0, -1),
            screen_width=0.01,
            screen_height=0.01
        )
        
        scene += source
        scene += glass_sphere
        scene += camera
        scene.raytrace()
        
        assert pixel_is_illuminated(camera), \
            "Ray should reach detector via one transmission"
    
    def test_one_transmission_failure(self):
        """Ray should NOT reach detector when opaque object blocks transmission."""
        scene = Scene(reverse_trace=True)
        
        # Source behind opaque sphere
        source = IsotropicSource(
            center=Vector(0, 0, 0),
            radius=0.3,
            color=Vector(1, 1, 1)
        )
        
        # Opaque sphere blocks transmission
        opaque_sphere = SphereElement(
            center=Vector(0, 0, 2),
            radius=0.5,
            color=Vector(0.9, 0.9, 0.9),
            transparent=False,  # Opaque
            user_params={'specular': 0.1, 'n_s': 10}
        )
        
        camera = Camera(
            width=1,
            height=1,
            position=Vector(0, 0, 4),
            pointing_direction=Vector(0, 0, -1),
            screen_width=0.01,
            screen_height=0.01
        )
        
        scene += source
        scene += opaque_sphere
        scene += camera
        scene.raytrace()
        
        assert not pixel_is_illuminated(camera), \
            "Ray should NOT reach detector when opaque object blocks path"


class TestTwoTransmissions:
    """Test geometry with two transmissions."""
    
    def test_two_transmissions_success(self):
        """Ray should reach detector via two transmissions when geometry allows."""
        scene = Scene(reverse_trace=True)
        
        # Source at origin
        source = IsotropicSource(
            center=Vector(0, 0, 0),
            radius=0.3,
            color=Vector(1, 1, 1)
        )
        
        # First transparent sphere
        glass_sphere1 = SphereElement(
            center=Vector(0, 0, 1.5),
            radius=0.4,
            color=Vector(0.9, 0.9, 1.0),
            transparent=True,
            refractive_index=1.5,
            user_params={'specular': 0.1, 'n_s': 10}
        )
        
        # Second transparent sphere
        glass_sphere2 = SphereElement(
            center=Vector(0, 0, 2.5),
            radius=0.4,
            color=Vector(0.9, 1.0, 0.9),
            transparent=True,
            refractive_index=1.5,
            user_params={'specular': 0.1, 'n_s': 10}
        )
        
        # 1x1 camera
        camera = Camera(
            width=1,
            height=1,
            position=Vector(0, 0, 4),
            pointing_direction=Vector(0, 0, -1),
            screen_width=0.01,
            screen_height=0.01
        )
        
        scene += source
        scene += glass_sphere1
        scene += glass_sphere2
        scene += camera
        scene.raytrace()
        
        assert pixel_is_illuminated(camera), \
            "Ray should reach detector via two transmissions"
    
    def test_two_transmissions_failure(self):
        """Ray should NOT reach detector when opaque object blocks transmission path."""
        scene = Scene(reverse_trace=True)
        
        # Source at origin
        source = IsotropicSource(
            center=Vector(0, 0, 0),
            radius=0.3,
            color=Vector(1, 1, 1)
        )
        
        # First transparent sphere
        glass_sphere = SphereElement(
            center=Vector(0, 0, 1.5),
            radius=0.4,
            color=Vector(0.9, 0.9, 1.0),
            transparent=True,
            refractive_index=1.5,
            user_params={'specular': 0.1, 'n_s': 10}
        )
        
        # Opaque sphere blocks path
        opaque_sphere = SphereElement(
            center=Vector(0, 0, 2.5),
            radius=0.4,
            color=Vector(0.9, 0.9, 0.9),
            transparent=False,
            user_params={'specular': 0.1, 'n_s': 10}
        )
        
        camera = Camera(
            width=1,
            height=1,
            position=Vector(0, 0, 4),
            pointing_direction=Vector(0, 0, -1),
            screen_width=0.01,
            screen_height=0.01
        )
        
        scene += source
        scene += glass_sphere
        scene += opaque_sphere
        scene += camera
        scene.raytrace()
        
        assert not pixel_is_illuminated(camera), \
            "Ray should NOT reach detector when opaque object blocks transmission path"


class TestThreeTransmissions:
    """Test geometry with three transmissions."""
    
    def test_three_transmissions_success(self):
        """Ray should reach detector via three transmissions when geometry allows."""
        scene = Scene(reverse_trace=True)
        
        # Source at origin
        source = IsotropicSource(
            center=Vector(0, 0, 0),
            radius=0.3,
            color=Vector(1, 1, 1)
        )
        
        # Three transparent spheres in a line
        glass_sphere1 = SphereElement(
            center=Vector(0, 0, 1.2),
            radius=0.3,
            color=Vector(0.9, 0.9, 1.0),
            transparent=True,
            refractive_index=1.5,
            user_params={'specular': 0.1, 'n_s': 10}
        )
        
        glass_sphere2 = SphereElement(
            center=Vector(0, 0, 2.0),
            radius=0.3,
            color=Vector(0.9, 1.0, 0.9),
            transparent=True,
            refractive_index=1.5,
            user_params={'specular': 0.1, 'n_s': 10}
        )
        
        glass_sphere3 = SphereElement(
            center=Vector(0, 0, 2.8),
            radius=0.3,
            color=Vector(1.0, 0.9, 0.9),
            transparent=True,
            refractive_index=1.5,
            user_params={'specular': 0.1, 'n_s': 10}
        )
        
        # 1x1 camera
        camera = Camera(
            width=1,
            height=1,
            position=Vector(0, 0, 4),
            pointing_direction=Vector(0, 0, -1),
            screen_width=0.01,
            screen_height=0.01
        )
        
        scene += source
        scene += glass_sphere1
        scene += glass_sphere2
        scene += glass_sphere3
        scene += camera
        scene.raytrace()
        
        assert pixel_is_illuminated(camera), \
            "Ray should reach detector via three transmissions"
    
    def test_three_transmissions_failure(self):
        """Ray should NOT reach detector when path is blocked."""
        scene = Scene(reverse_trace=True)
        
        # Source at origin
        source = IsotropicSource(
            center=Vector(0, 0, 0),
            radius=0.3,
            color=Vector(1, 1, 1)
        )
        
        # Two transparent spheres
        glass_sphere1 = SphereElement(
            center=Vector(0, 0, 1.2),
            radius=0.3,
            color=Vector(0.9, 0.9, 1.0),
            transparent=True,
            refractive_index=1.5,
            user_params={'specular': 0.1, 'n_s': 10}
        )
        
        glass_sphere2 = SphereElement(
            center=Vector(0, 0, 2.0),
            radius=0.3,
            color=Vector(0.9, 1.0, 0.9),
            transparent=True,
            refractive_index=1.5,
            user_params={'specular': 0.1, 'n_s': 10}
        )
        
        # Opaque sphere blocks path
        opaque_sphere = SphereElement(
            center=Vector(0, 0, 2.8),
            radius=0.3,
            color=Vector(0.9, 0.9, 0.9),
            transparent=False,
            user_params={'specular': 0.1, 'n_s': 10}
        )
        
        camera = Camera(
            width=1,
            height=1,
            position=Vector(0, 0, 4),
            pointing_direction=Vector(0, 0, -1),
            screen_width=0.01,
            screen_height=0.01
        )
        
        scene += source
        scene += glass_sphere1
        scene += glass_sphere2
        scene += opaque_sphere
        scene += camera
        scene.raytrace()
        
        assert not pixel_is_illuminated(camera), \
            "Ray should NOT reach detector when opaque object blocks transmission path"


class TestReflectTransmitReflect:
    """Test geometry with reflect-transmit-reflect sequence."""
    
    def test_reflect_transmit_reflect_success(self):
        """Ray should reach detector via reflect-transmit-reflect when geometry allows."""
        scene = Scene(reverse_trace=True)
        
        # Source positioned for reflect-transmit-reflect path
        source = IsotropicSource(
            center=Vector(-1.8, 0, 2),
            radius=0.3,
            color=Vector(1, 1, 1)
        )
        
        # First reflective sphere (camera reflects off this, shifted left)
        mirror_sphere1 = SphereElement(
            center=Vector(-0.3, 0, 2),
            radius=0.5,
            color=Vector(0.9, 0.9, 0.9),
            user_params={'specular': 0.8, 'n_s': 50}
        )
        
        # Transparent sphere in middle (ray transmits through)
        glass_sphere = SphereElement(
            center=Vector(-0.9, 0, 1.2),
            radius=0.4,
            color=Vector(0.9, 0.9, 1.0),
            transparent=True,
            refractive_index=1.5,
            user_params={'specular': 0.1, 'n_s': 10}
        )
        
        # Second reflective sphere (ray reflects toward source)
        mirror_sphere2 = SphereElement(
            center=Vector(-1.3, 0, 1.8),
            radius=0.4,
            color=Vector(0.9, 0.9, 0.9),
            user_params={'specular': 0.8, 'n_s': 50}
        )
        
        # 1x1 camera
        camera = Camera(
            width=1,
            height=1,
            position=Vector(0, 0, 4),
            pointing_direction=Vector(0, 0, -1),
            screen_width=0.01,
            screen_height=0.01
        )
        
        scene += source
        scene += mirror_sphere1
        scene += glass_sphere
        scene += mirror_sphere2
        scene += camera
        scene.raytrace()
        
        assert pixel_is_illuminated(camera), \
            "Ray should reach detector via reflect-transmit-reflect sequence"
    
    def test_reflect_transmit_reflect_failure(self):
        """Ray should NOT reach detector when geometry prevents reflect-transmit-reflect."""
        scene = Scene(reverse_trace=True)
        
        # Source positioned for path
        source = IsotropicSource(
            center=Vector(-1.8, 0, 2),
            radius=0.3,
            color=Vector(1, 1, 1)
        )
        
        # First reflective sphere
        mirror_sphere1 = SphereElement(
            center=Vector(-0.3, 0, 2),
            radius=0.5,
            color=Vector(0.9, 0.9, 0.9),
            user_params={'specular': 0.8, 'n_s': 50}
        )
        
        # Transparent sphere
        glass_sphere = SphereElement(
            center=Vector(-0.9, 0, 1.2),
            radius=0.4,
            color=Vector(0.9, 0.9, 1.0),
            transparent=True,
            refractive_index=1.5,
            user_params={'specular': 0.1, 'n_s': 10}
        )
        
        # Second reflective sphere
        mirror_sphere2 = SphereElement(
            center=Vector(-1.3, 0, 1.8),
            radius=0.4,
            color=Vector(0.9, 0.9, 0.9),
            user_params={'specular': 0.8, 'n_s': 50}
        )
        
        # Opaque blocker between second mirror and source
        blocker = SphereElement(
            center=Vector(-1.55, 0, 1.9),
            radius=0.3,
            color=Vector(0.5, 0.5, 0.5),
            user_params={'specular': 0.0, 'n_s': 1}
        )
        
        camera = Camera(
            width=1,
            height=1,
            position=Vector(0, 0, 4),
            pointing_direction=Vector(0, 0, -1),
            screen_width=0.01,
            screen_height=0.01
        )
        
        scene += source
        scene += mirror_sphere1
        scene += glass_sphere
        scene += mirror_sphere2
        scene += blocker
        scene += camera
        scene.raytrace()
        
        assert not pixel_is_illuminated(camera), \
            "Ray should NOT reach detector when path is blocked"

class TestReflectTransmitReflectTransmit:
    """Test geometry with reflect-transmit-reflect-transmit sequence."""
    
    def test_reflect_transmit_reflect_transmit_success(self):
        """Ray should reach detector via reflect-transmit-reflect-transmit when geometry allows."""
        scene = Scene(reverse_trace=True)
        
        # Source positioned for complex path
        source = IsotropicSource(
            center=Vector(-2.2, 0, 0),
            radius=0.3,
            color=Vector(1, 1, 1)
        )
        
        # First reflective sphere (camera reflects off this, shifted left)
        mirror_sphere1 = SphereElement(
            center=Vector(-0.3, 0, 2),
            radius=0.5,
            color=Vector(0.9, 0.9, 0.9),
            user_params={'specular': 0.8, 'n_s': 50}
        )
        
        # First transparent sphere
        glass_sphere1 = SphereElement(
            center=Vector(-0.9, 0, 1.2),
            radius=0.3,
            color=Vector(0.9, 0.9, 1.0),
            transparent=True,
            refractive_index=1.5,
            user_params={'specular': 0.1, 'n_s': 10}
        )
        
        # Second reflective sphere
        mirror_sphere2 = SphereElement(
            center=Vector(-1.4, 0, 0.7),
            radius=0.4,
            color=Vector(0.9, 0.9, 0.9),
            user_params={'specular': 0.8, 'n_s': 50}
        )
        
        # Second transparent sphere
        glass_sphere2 = SphereElement(
            center=Vector(-1.9, 0, 0.2),
            radius=0.3,
            color=Vector(0.9, 1.0, 0.9),
            transparent=True,
            refractive_index=1.5,
            user_params={'specular': 0.1, 'n_s': 10}
        )
        
        # 1x1 camera
        camera = Camera(
            width=1,
            height=1,
            position=Vector(0, 0, 4),
            pointing_direction=Vector(0, 0, -1),
            screen_width=0.01,
            screen_height=0.01
        )
        
        scene += source
        scene += mirror_sphere1
        scene += glass_sphere1
        scene += mirror_sphere2
        scene += glass_sphere2
        scene += camera
        scene.raytrace()
        
        assert pixel_is_illuminated(camera), \
            "Ray should reach detector via reflect-transmit-reflect-transmit sequence"
    
    def test_reflect_transmit_reflect_transmit_failure(self):
        """Ray should NOT reach detector when geometry prevents the sequence."""
        scene = Scene(reverse_trace=True)
        
        # Source positioned incorrectly (opposite side)
        source = IsotropicSource(
            center=Vector(2.2, 0, 0),
            radius=0.3,
            color=Vector(1, 1, 1)
        )
        
        # First reflective sphere
        mirror_sphere1 = SphereElement(
            center=Vector(-0.3, 0, 2),
            radius=0.5,
            color=Vector(0.9, 0.9, 0.9),
            user_params={'specular': 0.8, 'n_s': 50}
        )
        
        # First transparent sphere
        glass_sphere1 = SphereElement(
            center=Vector(-0.9, 0, 1.2),
            radius=0.3,
            color=Vector(0.9, 0.9, 1.0),
            transparent=True,
            refractive_index=1.5,
            user_params={'specular': 0.1, 'n_s': 10}
        )
        
        # Second reflective sphere
        mirror_sphere2 = SphereElement(
            center=Vector(-1.4, 0, 0.7),
            radius=0.4,
            color=Vector(0.9, 0.9, 0.9),
            user_params={'specular': 0.8, 'n_s': 50}
        )
        
        # Second transparent sphere
        glass_sphere2 = SphereElement(
            center=Vector(-1.9, 0, 0.2),
            radius=0.3,
            color=Vector(0.9, 1.0, 0.9),
            transparent=True,
            refractive_index=1.5,
            user_params={'specular': 0.1, 'n_s': 10}
        )
        
        camera = Camera(
            width=1,
            height=1,
            position=Vector(0, 0, 4),
            pointing_direction=Vector(0, 0, -1),
            screen_width=0.01,
            screen_height=0.01
        )
        
        scene += source
        scene += mirror_sphere1
        scene += glass_sphere1
        scene += mirror_sphere2
        scene += glass_sphere2
        scene += camera
        scene.raytrace()
        
        assert not pixel_is_illuminated(camera), \
            "Ray should NOT reach detector when source is positioned incorrectly"

class TestReflectTransmitTransmitReflect:
    """Test geometry with reflect-transmit-transmit-reflect sequence."""
    
    def test_reflect_transmit_transmit_reflect_success(self):
        """Ray should reach detector via reflect-transmit-transmit-reflect when geometry allows."""
        scene = Scene(reverse_trace=True)
        
        # Source positioned for this path
        source = IsotropicSource(
            center=Vector(-2.1, 0, 0.5),
            radius=0.3,
            color=Vector(1, 1, 1)
        )
        
        # First reflective sphere (camera reflects off this, shifted left)
        mirror_sphere1 = SphereElement(
            center=Vector(-0.3, 0, 2),
            radius=0.5,
            color=Vector(0.9, 0.9, 0.9),
            user_params={'specular': 0.8, 'n_s': 50}
        )
        
        # First transparent sphere
        glass_sphere1 = SphereElement(
            center=Vector(-0.7, 0, 1.3),
            radius=0.3,
            color=Vector(0.9, 0.9, 1.0),
            transparent=True,
            refractive_index=1.5,
            user_params={'specular': 0.1, 'n_s': 10}
        )
        
        # Second transparent sphere
        glass_sphere2 = SphereElement(
            center=Vector(-1.2, 0, 0.9),
            radius=0.3,
            color=Vector(0.9, 1.0, 0.9),
            transparent=True,
            refractive_index=1.5,
            user_params={'specular': 0.1, 'n_s': 10}
        )
        
        # Second reflective sphere
        mirror_sphere2 = SphereElement(
            center=Vector(-1.7, 0, 0.5),
            radius=0.4,
            color=Vector(0.9, 0.9, 0.9),
            user_params={'specular': 0.8, 'n_s': 50}
        )
        
        # 1x1 camera
        camera = Camera(
            width=1,
            height=1,
            position=Vector(0, 0, 4),
            pointing_direction=Vector(0, 0, -1),
            screen_width=0.01,
            screen_height=0.01
        )
        
        scene += source
        scene += mirror_sphere1
        scene += glass_sphere1
        scene += glass_sphere2
        scene += mirror_sphere2
        scene += camera
        scene.raytrace()
        
        assert pixel_is_illuminated(camera), \
            "Ray should reach detector via reflect-transmit-transmit-reflect sequence"
    
    def test_reflect_transmit_transmit_reflect_failure(self):
        """Ray should NOT reach detector when geometry prevents the sequence."""
        scene = Scene(reverse_trace=True)
        
        # Source positioned incorrectly (opposite side)
        source = IsotropicSource(
            center=Vector(2.1, 0, 0.5),
            radius=0.3,
            color=Vector(1, 1, 1)
        )
        
        # First reflective sphere
        mirror_sphere1 = SphereElement(
            center=Vector(-0.3, 0, 2),
            radius=0.5,
            color=Vector(0.9, 0.9, 0.9),
            user_params={'specular': 0.8, 'n_s': 50}
        )
        
        # First transparent sphere
        glass_sphere1 = SphereElement(
            center=Vector(-0.7, 0, 1.3),
            radius=0.3,
            color=Vector(0.9, 0.9, 1.0),
            transparent=True,
            refractive_index=1.5,
            user_params={'specular': 0.1, 'n_s': 10}
        )
        
        # Second transparent sphere
        glass_sphere2 = SphereElement(
            center=Vector(-1.2, 0, 0.9),
            radius=0.3,
            color=Vector(0.9, 1.0, 0.9),
            transparent=True,
            refractive_index=1.5,
            user_params={'specular': 0.1, 'n_s': 10}
        )
        
        # Second reflective sphere
        mirror_sphere2 = SphereElement(
            center=Vector(-1.7, 0, 0.5),
            radius=0.4,
            color=Vector(0.9, 0.9, 0.9),
            user_params={'specular': 0.8, 'n_s': 50}
        )
        
        camera = Camera(
            width=1,
            height=1,
            position=Vector(0, 0, 4),
            pointing_direction=Vector(0, 0, -1),
            screen_width=0.01,
            screen_height=0.01
        )
        
        scene += source
        scene += mirror_sphere1
        scene += glass_sphere1
        scene += glass_sphere2
        scene += mirror_sphere2
        scene += camera
        scene.raytrace()
        
        assert not pixel_is_illuminated(camera), \
            "Ray should NOT reach detector when source is positioned incorrectly"

class TestTransmitReflectTransmit:
    """Test geometry with transmit-reflect-transmit sequence."""
    
    def test_transmit_reflect_transmit_success(self):
        """Ray should reach detector via transmit-reflect-transmit when geometry allows."""
        scene = Scene(reverse_trace=True)
        
        # Source positioned for this path
        source = IsotropicSource(
            center=Vector(0, 0, 0),
            radius=0.3,
            color=Vector(1, 1, 1)
        )
        
        # First transparent sphere (camera ray transmits through)
        glass_sphere1 = SphereElement(
            center=Vector(0, 0, 2.5),
            radius=0.4,
            color=Vector(0.9, 0.9, 1.0),
            transparent=True,
            refractive_index=1.5,
            user_params={'specular': 0.1, 'n_s': 10}
        )
        
        # Reflective sphere in middle
        mirror_sphere = SphereElement(
            center=Vector(0, 0.6, 1.5),
            radius=0.4,
            color=Vector(0.9, 0.9, 0.9),
            user_params={'specular': 0.8, 'n_s': 50}
        )
        
        # Second transparent sphere
        glass_sphere2 = SphereElement(
            center=Vector(0, 0, 0.8),
            radius=0.4,
            color=Vector(0.9, 1.0, 0.9),
            transparent=True,
            refractive_index=1.5,
            user_params={'specular': 0.1, 'n_s': 10}
        )
        
        # 1x1 camera
        camera = Camera(
            width=1,
            height=1,
            position=Vector(0, 0, 4),
            pointing_direction=Vector(0, 0, -1),
            screen_width=0.01,
            screen_height=0.01
        )
        
        scene += source
        scene += glass_sphere1
        scene += mirror_sphere
        scene += glass_sphere2
        scene += camera
        scene.raytrace()
        
        assert pixel_is_illuminated(camera), \
            "Ray should reach detector via transmit-reflect-transmit sequence"
    
    def test_transmit_reflect_transmit_failure(self):
        """Ray should NOT reach detector when geometry prevents the sequence."""
        scene = Scene(reverse_trace=True)
        
        # Source positioned for path (but will be blocked)
        source = IsotropicSource(
            center=Vector(0, 0, 0),
            radius=0.3,
            color=Vector(1, 1, 1)
        )
        
        # First transparent sphere
        glass_sphere1 = SphereElement(
            center=Vector(0, 0, 2.5),
            radius=0.4,
            color=Vector(0.9, 0.9, 1.0),
            transparent=True,
            refractive_index=1.5,
            user_params={'specular': 0.1, 'n_s': 10}
        )
        
        # Reflective sphere
        mirror_sphere = SphereElement(
            center=Vector(0, 0.6, 1.5),
            radius=0.4,
            color=Vector(0.9, 0.9, 0.9),
            user_params={'specular': 0.8, 'n_s': 50}
        )
        
        # Second transparent sphere
        glass_sphere2 = SphereElement(
            center=Vector(0, 0, 0.8),
            radius=0.4,
            color=Vector(0.9, 1.0, 0.9),
            transparent=True,
            refractive_index=1.5,
            user_params={'specular': 0.1, 'n_s': 10}
        )
        
        # Opaque blocker between glass_sphere2 and source
        blocker = SphereElement(
            center=Vector(0, 0, 0.4),
            radius=0.3,
            color=Vector(0.5, 0.5, 0.5),
            user_params={'specular': 0.0, 'n_s': 1}
        )
        
        camera = Camera(
            width=1,
            height=1,
            position=Vector(0, 0, 4),
            pointing_direction=Vector(0, 0, -1),
            screen_width=0.01,
            screen_height=0.01
        )
        
        scene += source
        scene += glass_sphere1
        scene += mirror_sphere
        scene += glass_sphere2
        scene += blocker
        scene += camera
        scene.raytrace()
        
        assert not pixel_is_illuminated(camera), \
            "Ray should NOT reach detector when path is blocked"

class TestTransmitReflectTransmitReflect:
    """Test geometry with transmit-reflect-transmit-reflect sequence."""
    
    def test_transmit_reflect_transmit_reflect_success(self):
        """Ray should reach detector via transmit-reflect-transmit-reflect when geometry allows."""
        scene = Scene(reverse_trace=True)
        
        # Source positioned for this path
        source = IsotropicSource(
            center=Vector(-1.5, 0, 0),
            radius=0.3,
            color=Vector(1, 1, 1)
        )
        
        # First transparent sphere (camera ray transmits through)
        glass_sphere1 = SphereElement(
            center=Vector(0, 0, 2.5),
            radius=0.4,
            color=Vector(0.9, 0.9, 1.0),
            transparent=True,
            refractive_index=1.5,
            user_params={'specular': 0.1, 'n_s': 10}
        )
        
        # First reflective sphere
        mirror_sphere1 = SphereElement(
            center=Vector(-0.3, 0, 1.7),
            radius=0.3,
            color=Vector(0.9, 0.9, 0.9),
            user_params={'specular': 0.8, 'n_s': 50}
        )
        
        # Second transparent sphere
        glass_sphere2 = SphereElement(
            center=Vector(-0.7, 0, 1.0),
            radius=0.3,
            color=Vector(0.9, 1.0, 0.9),
            transparent=True,
            refractive_index=1.5,
            user_params={'specular': 0.1, 'n_s': 10}
        )
        
        # Second reflective sphere
        mirror_sphere2 = SphereElement(
            center=Vector(-1.2, 0, 0.3),
            radius=0.3,
            color=Vector(0.9, 0.9, 0.9),
            user_params={'specular': 0.8, 'n_s': 50}
        )
        
        # 1x1 camera
        camera = Camera(
            width=1,
            height=1,
            position=Vector(0, 0, 4),
            pointing_direction=Vector(0, 0, -1),
            screen_width=0.01,
            screen_height=0.01
        )
        
        scene += source
        scene += glass_sphere1
        scene += mirror_sphere1
        scene += glass_sphere2
        scene += mirror_sphere2
        scene += camera
        scene.raytrace()
        
        assert pixel_is_illuminated(camera), \
            "Ray should reach detector via transmit-reflect-transmit-reflect sequence"
    
    def test_transmit_reflect_transmit_reflect_failure(self):
        """Ray should NOT reach detector when geometry prevents the sequence."""
        scene = Scene(reverse_trace=True)
        
        # Source positioned for path (but will be blocked)
        source = IsotropicSource(
            center=Vector(-1.5, 0, 0),
            radius=0.3,
            color=Vector(1, 1, 1)
        )
        
        # First transparent sphere
        glass_sphere1 = SphereElement(
            center=Vector(0, 0, 2.5),
            radius=0.4,
            color=Vector(0.9, 0.9, 1.0),
            transparent=True,
            refractive_index=1.5,
            user_params={'specular': 0.1, 'n_s': 10}
        )
        
        # First reflective sphere
        mirror_sphere1 = SphereElement(
            center=Vector(-0.3, 0, 1.7),
            radius=0.3,
            color=Vector(0.9, 0.9, 0.9),
            user_params={'specular': 0.8, 'n_s': 50}
        )
        
        # Second transparent sphere
        glass_sphere2 = SphereElement(
            center=Vector(-0.7, 0, 1.0),
            radius=0.3,
            color=Vector(0.9, 1.0, 0.9),
            transparent=True,
            refractive_index=1.5,
            user_params={'specular': 0.1, 'n_s': 10}
        )
        
        # Second reflective sphere
        mirror_sphere2 = SphereElement(
            center=Vector(-1.2, 0, 0.3),
            radius=0.3,
            color=Vector(0.9, 0.9, 0.9),
            user_params={'specular': 0.8, 'n_s': 50}
        )
        
        # Opaque blocker between second mirror and source
        blocker = SphereElement(
            center=Vector(-1.35, 0, 0.15),
            radius=0.2,
            color=Vector(0.5, 0.5, 0.5),
            user_params={'specular': 0.0, 'n_s': 1}
        )
        
        camera = Camera(
            width=1,
            height=1,
            position=Vector(0, 0, 4),
            pointing_direction=Vector(0, 0, -1),
            screen_width=0.01,
            screen_height=0.01
        )
        
        scene += source
        scene += glass_sphere1
        scene += mirror_sphere1
        scene += glass_sphere2
        scene += mirror_sphere2
        scene += blocker
        scene += camera
        scene.raytrace()
        
        assert not pixel_is_illuminated(camera), \
            "Ray should NOT reach detector when path is blocked"

class TestTransmitReflectReflectTransmit:
    """Test geometry with transmit-reflect-reflect-transmit sequence."""
    
    def test_transmit_reflect_reflect_transmit_success(self):
        """Ray should reach detector via transmit-reflect-reflect-transmit when geometry allows."""
        scene = Scene(reverse_trace=True)
        
        # Source positioned for this path
        source = IsotropicSource(
            center=Vector(0, 0, 0),
            radius=0.3,
            color=Vector(1, 1, 1)
        )
        
        # First transparent sphere (camera ray transmits through)
        glass_sphere1 = SphereElement(
            center=Vector(0, 0, 2.5),
            radius=0.4,
            color=Vector(0.9, 0.9, 1.0),
            transparent=True,
            refractive_index=1.5,
            user_params={'specular': 0.1, 'n_s': 10}
        )
        
        # First reflective sphere
        mirror_sphere1 = SphereElement(
            center=Vector(0, 0.5, 1.8),
            radius=0.3,
            color=Vector(0.9, 0.9, 0.9),
            user_params={'specular': 0.8, 'n_s': 50}
        )
        
        # Second reflective sphere
        mirror_sphere2 = SphereElement(
            center=Vector(0, 0.5, 1.2),
            radius=0.3,
            color=Vector(0.9, 0.9, 0.9),
            user_params={'specular': 0.8, 'n_s': 50}
        )
        
        # Second transparent sphere
        glass_sphere2 = SphereElement(
            center=Vector(0, 0, 0.6),
            radius=0.3,
            color=Vector(0.9, 1.0, 0.9),
            transparent=True,
            refractive_index=1.5,
            user_params={'specular': 0.1, 'n_s': 10}
        )
        
        # 1x1 camera
        camera = Camera(
            width=1,
            height=1,
            position=Vector(0, 0, 4),
            pointing_direction=Vector(0, 0, -1),
            screen_width=0.01,
            screen_height=0.01
        )
        
        scene += source
        scene += glass_sphere1
        scene += mirror_sphere1
        scene += mirror_sphere2
        scene += glass_sphere2
        scene += camera
        scene.raytrace()
        
        assert pixel_is_illuminated(camera), \
            "Ray should reach detector via transmit-reflect-reflect-transmit sequence"
    
    def test_transmit_reflect_reflect_transmit_failure(self):
        """Ray should NOT reach detector when geometry prevents the sequence."""
        scene = Scene(reverse_trace=True)
        
        # Source positioned for path (but will be blocked)
        source = IsotropicSource(
            center=Vector(0, 0, 0),
            radius=0.3,
            color=Vector(1, 1, 1)
        )
        
        # First transparent sphere
        glass_sphere1 = SphereElement(
            center=Vector(0, 0, 2.5),
            radius=0.4,
            color=Vector(0.9, 0.9, 1.0),
            transparent=True,
            refractive_index=1.5,
            user_params={'specular': 0.1, 'n_s': 10}
        )
        
        # First reflective sphere
        mirror_sphere1 = SphereElement(
            center=Vector(0, 0.5, 1.8),
            radius=0.3,
            color=Vector(0.9, 0.9, 0.9),
            user_params={'specular': 0.8, 'n_s': 50}
        )
        
        # Second reflective sphere
        mirror_sphere2 = SphereElement(
            center=Vector(0, 0.5, 1.2),
            radius=0.3,
            color=Vector(0.9, 0.9, 0.9),
            user_params={'specular': 0.8, 'n_s': 50}
        )
        
        # Second transparent sphere
        glass_sphere2 = SphereElement(
            center=Vector(0, 0, 0.6),
            radius=0.3,
            color=Vector(0.9, 1.0, 0.9),
            transparent=True,
            refractive_index=1.5,
            user_params={'specular': 0.1, 'n_s': 10}
        )
        
        # Opaque blocker between glass_sphere2 and source
        blocker = SphereElement(
            center=Vector(0, 0, 0.3),
            radius=0.2,
            color=Vector(0.5, 0.5, 0.5),
            user_params={'specular': 0.0, 'n_s': 1}
        )
        
        camera = Camera(
            width=1,
            height=1,
            position=Vector(0, 0, 4),
            pointing_direction=Vector(0, 0, -1),
            screen_width=0.01,
            screen_height=0.01
        )
        
        scene += source
        scene += glass_sphere1
        scene += mirror_sphere1
        scene += mirror_sphere2
        scene += glass_sphere2
        scene += blocker
        scene += camera
        scene.raytrace()
        
        assert not pixel_is_illuminated(camera), \
            "Ray should NOT reach detector when path is blocked"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
