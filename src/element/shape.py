from abc import ABC, abstractmethod
import numpy as np

from ..math.vector import Vector

FARAWAY = 1.0e39

class Shape(ABC):

    @abstractmethod
    def intersect(self, origin: Vector, direction: Vector):
        '''
        Calculate intersection of a ray with the object.
        '''
        pass

    @abstractmethod
    def compute_outward_normal(self, intersection_point: Vector) -> Vector:
        '''
        Compute normal vector, facing away from the object, associated with incident Vector's point of intersection.
        
        NOTE For 2d objects, return that object's pointing direction
        '''
        pass

    @abstractmethod
    def compute_inward_normal(self, intersection_point: Vector) -> Vector:
        '''
        Compute normal vector, facing into the object, associated with incident Vector's point of intersection.

        NOTE not implemented for 2d objects!
        '''
        pass

    @abstractmethod
    def compute_surface_definition(self):
        '''
        Computes geometry to describe a SceneObject's extension in space (2D or 3D).
        The surface definition of a Shape should be independent of the pixel definition.
        As these methods might not be called together.
        '''
        pass

    @abstractmethod
    def create_screen_coord(self):
        """
        Generates a set of 3D coordinates for each pixel on the detection screen.

        The detector screen is a rectangular plane in 3D space, centered at `detector_pos` and oriented perpendicular to `detector_pointing_direction`.
        The screen's physical size is specified by `screen_width` (and optionally `screen_height`); pixel centers are uniformly distributed across this surface.

        """
    @classmethod
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

    @classmethod
    def _apply_rotation(self, vector, rotation_matrix):
        """
        Apply the rotation matrix to the vector.
        """
        original_vectors = np.stack((vector.x, vector.y, vector.z), axis=-1)
        rotated_vectors = np.dot(original_vectors, rotation_matrix.T)
        return Vector(rotated_vectors[:, 0], rotated_vectors[:, 1], rotated_vectors[:, 2])


class Square(Shape):

    def __init__(self):
        self.top_left = None
        self.top_right = None
        self.bottom_left = None
        self.bottom_right = None

    def compute_surface_definition(self):
        '''
        Use this method to describe the square plane in 3d space.
        '''

        # normal vector, already normalized
        n = self.pointing_direction

        if abs(n.y) < 0.9:
            v_ref = Vector(0, 1, 0)
        else:
            v_ref = Vector(1, 0, 0)
        
        # right vector
        u = n.cross(v_ref)
        
        # up vector
        v = u.cross(n) 

        half_w = self.screen_width / 2.0
        half_h = self.screen_height / 2.0

        self.top_left = self.position - (u * half_w) + (v * half_h)
        self.top_right = self.position + (u * half_w) + (v * half_h)
        self.bottom_left = self.position - (u * half_w) - (v * half_h)
        self.bottom_right = self.position + (u * half_w) - (v * half_h)
        
    def create_screen_coord(
        self,
        vertical_screen_shift: float = 0.0,
        horizontal_screen_shift: float = 0.0,
        rotation: float = 0.0,
    ) -> Vector:

        '''
        Args:
        vertical_screen_shift (float, optional): Shifts the screen vertically (in the local Y direction) relative to `detector_pos`. Default is 0.0.
        horizontal_screen_shift (float, optional): Shifts the screen horizontally (in the local X direction) relative to `detector_pos`. Default is 0.0.
        rotation (float, optional): Rotates the screen about its normal (in radians). Default is 0.0.

        Returns:
        Vector: A Vector object containing the 3D coordinates for each pixel on the screen
        '''

        # superclass attributes
        w: int = self.width
        h: int = self.height
        detector_pointing_direction: Vector = self.pointing_direction
        detector_pos: Vector = self.position
        screen_width: float = self.screen_width
        screen_height: float = self.screen_height

        screen = (
            -screen_width / 2 + horizontal_screen_shift,
            screen_height / 2 + vertical_screen_shift,
            screen_width / 2 + horizontal_screen_shift,
            -screen_height / 2 + vertical_screen_shift
        )
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

    def intersect(self, ray_origin_point: Vector, ray_direction_from_origin: Vector):
        denom = super.pointing_direction.dot(ray_direction_from_origin)
        if denom > 0:
            # ray parallel to plane
            return FARAWAY

        t = super.pointing_direction.dot(self.top_left - ray_origin_point) / denom
        if t < 0:
            # intersection is behind the ray origin
            return FARAWAY

        P = ray_origin_point + ray_direction_from_origin * t

        corners = [self.top_left, self.top_right, self.bottom_right, self.bottom_left]
        inside = True
        for i in range(4):
            a = corners[i]
            b = corners[(i + 1) % 4]
            edge = b - a
            vp = P - a
            if super.pointing_direction.dot(edge.cross(vp)) < 0:
                inside = False
                break

        if inside:
            return P
        else:
            return FARAWAY

    def compute_outward_normal(self, intersection_point: Vector) -> Vector:
        return super.pointing_direction

    def compute_inward_normal(self, intersection_point: Vector) -> Vector:
        return NotImplementedError()
    

class Circle(Shape):
    def __init__(self, center: Vector, radius: float):
        self.radius = radius
        self.center = center

    def compute_surface_definition(self):
        # self.center, self.pointing_direction, and self.radius are sufficient to define the shape's plane
        pass

    def create_screen_coord(
        self,
        vertical_screen_shift: float = 0.0,
        horizontal_screen_shift: float = 0.0,
        rotation: float = 0.0,
    ) -> Vector:
        '''
        Generates roughly evenly distributed 3D coordinates on the circular plane.
        '''

        # normal vector, already normalized
        n = self.pointing_direction

        if abs(n.y) < 0.9:
            v_ref = Vector(0, 1, 0)
        else:
            v_ref = Vector(1, 0, 0)
        
        u = n.cross(v_ref)
        v = u.cross(n)

        # distribute points on a 2D disk using Fermat's spiral
        i = np.arange(self.pixel_count)
        golden_angle = np.pi * (3 - np.sqrt(5))
        
        # radius of each point (proportional to sqrt of index for even area distribution)
        # scaled by the circle's actual radius
        r = self.radius * np.sqrt(i / self.pixel_count)
        theta = i * golden_angle + rotation

        # convert polar (r, theta) to local 2D (x, y)
        local_x = r * np.cos(theta) + horizontal_screen_shift
        local_y = r * np.sin(theta) + vertical_screen_shift

        # map 2D plane coordinates to 3D space
        # point = center + (local_x * u_axis) + (local_y * v_axis)
        screen_coords = self.position + (u * local_x) + (v * local_y)

        return screen_coords

    def intersect(self, ray_origin_point: Vector, ray_direction_from_origin: Vector):
        denom = super.pointing_direction.dot(ray_direction_from_origin)
        if denom > 0:
            # ray parallel to plane
            return FARAWAY

        t = super.pointing_direction.dot(self.center - ray_origin_point) / denom
        if t < 0:
            # intersection is behind the ray origin
            return FARAWAY

        P = ray_origin_point + ray_direction_from_origin * t
        if (P - self.center).length() <= self.radius:
            return P
        else:
            return FARAWAY

    def compute_outward_normal(self, intersection_point: Vector) -> Vector:
        return super.pointing_direction
    
    def compute_inward_normal(self, intersection_point: Vector) -> Vector:
        raise NotImplementedError()


class Sphere(Shape):
    def __init__(self, center: Vector, radius: float):
        self.radius = radius
        self.center = center

    def compute_surface_definition(self):
        # self.center, and self.radius are sufficient to define the shape's volume
        pass

    def create_screen_coord(
        self,
        vertical_screen_shift: float = 0.0,
        horizontal_screen_shift: float = 0.0,
        rotation: float = 0.0,
    ) -> Vector:
        pass

    def intersect(self, ray_origin_point: Vector, ray_direction_from_origin: Vector):
        b = 2 * ray_direction_from_origin.dot(ray_origin_point - self.center)
        c = (
            abs(self.center)
            + abs(ray_origin_point)
            - 2 * self.center.dot(ray_origin_point)
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

    def compute_outward_normal(self, intersection_point: Vector) -> Vector:
        normal_at_intersection = (intersection_point - self.center).norm()
        return normal_at_intersection
    
    def compute_inward_normal(self, intersection_point: Vector) -> Vector:
        normal_at_intersection = (self.center - intersection_point).norm()
        return normal_at_intersection