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

    def compute_surface_definition(self, detector_coordinates: Vector, detector_pixel_width: int, detector_pixel_height: int):
        '''
        Use this method to find the square that captures a set of points in a plane.
        '''

        idx_top_left = 0
        idx_top_right = detector_pixel_width - 1
        idx_bottom_left = (detector_pixel_height - 1) * detector_pixel_width
        idx_bottom_right = detector_pixel_height * detector_pixel_width - 1

        self.top_left = Vector(
                np.array([detector_coordinates.x[idx_top_left]]),
                np.array([detector_coordinates.y[idx_top_left]]),
                np.array([detector_coordinates.z[idx_top_left]]) )
        self.top_right = Vector(
                np.array([detector_coordinates.x[idx_top_right]]),
                np.array([detector_coordinates.y[idx_top_right]]),
                np.array([detector_coordinates.z[idx_top_right]]) )
        self.bottom_left = Vector(
                np.array([detector_coordinates.x[idx_bottom_right]]),
                np.array([detector_coordinates.y[idx_bottom_right]]),
                np.array([detector_coordinates.z[idx_bottom_right]]))
        self.bottom_right = Vector(
                np.array([detector_coordinates.x[idx_bottom_left]]),
                np.array([detector_coordinates.y[idx_bottom_left]]),
                np.array([detector_coordinates.z[idx_bottom_left]]))
        
    def create_screen_coord(
        self,
        w: int,
        h: int,
        detector_pointing_direction: Vector,
        detector_pos: Vector,
        screen_width: float = 2.0,
        screen_height: float = None,
        vertical_screen_shift: float = 0.0,
        horizontal_screen_shift: float = 0.0,
        rotation: float = 0.0,
    ) -> Vector:

        '''
        Args:
        w (int): Number of pixels along the horizontal (width/X) direction of the screen.
        h (int): Number of pixels along the vertical (height/Y) direction of the screen.
        detector_pointing_direction (Vector): A unit vector specifying the normal direction the detector screen faces.
        detector_pos (Vector): The 3D position of the center of the detector screen.
        screen_width (float, optional): The physical width (X-direction) of the detector screen in scene units. Default is 2.0.
        screen_height (float, optional): The physical height (Y-direction) of the detector screen. If None, calculated from screen_width and detector aspect ratio.
        vertical_screen_shift (float, optional): Shifts the screen vertically (in the local Y direction) relative to `detector_pos`. Default is 0.0.
        horizontal_screen_shift (float, optional): Shifts the screen horizontally (in the local X direction) relative to `detector_pos`. Default is 0.0.
        rotation (float, optional): Rotates the screen about its normal (in radians). Default is 0.0.

        Returns:
        Vector: A Vector object containing the 3D coordinates for each pixel on the screen, suitable for ray tracing.
        '''
        
        aspect_ratio = w / h
        if screen_height is None:
            screen_height = screen_width / aspect_ratio

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
        pass

    def create_screen_coord(self):
        pass

    def intersect(self, ray_origin_point: Vector, ray_direction_from_origin: Vector):
        denom = self.normal.dot(ray_direction_from_origin)
        if denom > 0:
            # ray parallel to plane
            return FARAWAY

        t = self.normal.dot(self.center - ray_origin_point) / denom
        if t < 0:
            # intersection is behind the ray origin
            return FARAWAY

        P = ray_origin_point + ray_direction_from_origin * t
        if (P - self.center).length() <= self.radius:
            return P
        else:
            return FARAWAY

    def surface_color(self, M: Vector) -> Vector:
        return self.color

    def compute_outward_normal(self, intersection_point: Vector) -> Vector:
        raise NotImplementedError()
    
    def compute_inward_normal(self, intersection_point: Vector) -> Vector:
        raise NotImplementedError()


class Sphere(Shape):
    def __init__(self, center: Vector, radius: float):
        self.radius = radius
        self.center = center

    def compute_surface_definition(self):
        pass

    def create_screen_coord(self):
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