import os 
from luminous.src.math.vector import Vector
from luminous.src.element.element import Sphere, CheckeredSphere
import math
import numpy as np
import datetime
from abc import ABC, abstractmethod
import numbers
import inspect

class RayDebugger(ABC):
    @abstractmethod
    def add_vector(self, start_point, end_point, color):
        pass

    @abstractmethod
    def add_point(self, end_point, color):
        pass

    @abstractmethod
    def add_element(self, element, color=(0,0,0)):
        pass

    @abstractmethod
    def add_sphere(self, center, color, radius, opacity=0.3):
        pass

    @abstractmethod
    def plot(self):
        pass

class NullRayDebugger(RayDebugger):
    '''
    No-operation RayDebugger.
    '''

    def add_vector(self, end_point, start_point=(0,0,0), color=(0,0,0)):
        pass
    def add_point(self, end_point, color=(0,0,0)):
        pass
    def add_element(self, element, color=(0,0,0)):
        pass
    def add_sphere(self, center, color=(0,0,0), radius=0.1, opacity=0.3):
        pass
    def plot(self):
        pass

class ConcreteRayDebugger(RayDebugger):
    '''
    Actual RayDebugger compiles data and plots rays.
    '''
 
    def __init__(self, point_radius=0.01, shaft_radius=0.01, head_radius=0.015, head_length=0.05):
        import vtk
        self.vtk = vtk

        self.screenshot_counter = 0
        self.timestamp = datetime.datetime.now().strftime("%m%d%y%H%M%S")

        # vectors
        self.shaft_radius = shaft_radius
        self.head_radius = head_radius
        self.head_length = head_length
        self.vectors = vtk.vtkCellArray()
        self.vector_points = vtk.vtkPoints()
        self.vector_directions = vtk.vtkFloatArray()
        self.vector_directions.SetNumberOfComponents(3)
        self.vector_color_scalars = vtk.vtkUnsignedCharArray()
        self.vector_color_scalars.SetNumberOfComponents(3)
        self.vector_magnitudes = vtk.vtkFloatArray()
        self.vector_magnitudes.SetNumberOfComponents(1)

        # elements
        self.spheres = list()

        # points
        self.point_radius = point_radius
        self.points = vtk.vtkPoints()
        self.point_color_scalars = vtk.vtkFloatArray()
        
        self.lookup_table = vtk.vtkLookupTable()
        self.lookup_table.SetNumberOfTableValues(256*256*256)
        self.lookup_table.Build()
 
        for i in range(256*256*256):
            r = (i // (256*256)) % 256
            g = (i // 256) % 256
            b = i % 256
            self.lookup_table.SetTableValue(i, r/255.0, g/255.0, b/255.0, 1.0)

        self.luminous_dialog_title = "luminous ray debugger"
 
    def add_point(self, p, color=(0,0,0)):
 
        color_scalar = color[0] * 256 * 256 + color[1] * 256 + color[2]

        if isinstance(p, Vector):
            p = p.components()

        if all(isinstance(item, numbers.Number) for item in p):
            self.points.InsertNextPoint(p)
            self.point_color_scalars.InsertNextValue(color_scalar)
            return
        
        if all(isinstance(item, np.ndarray) for item in p):
            for x_end, y_end, z_end in zip(*p):
                end = (x_end, y_end, z_end)
                self.points.InsertNextPoint(end)
                self.point_color_scalars.InsertNextValue(color_scalar)
            return

        m = inspect.currentframe().f_code.co_name
        raise TypeError(f"Unknown data structure inside call: {self.__class__.__name__}.{m}")
    
    def add_sphere(self, p, color=(0,0,0), radius=0.1, opacity=0.5):

        if isinstance(p, Vector):
            p = p.components()

        # case 1: single point
        if all(isinstance(item, numbers.Number) for item in p):
            self.spheres.append((p, color, radius, opacity))
            return

        # case 2: numpy arrays
        if all(isinstance(item, np.ndarray) for item in p):
            for x, y, z in zip(*p):
                self.spheres.append(((x, y, z), color, radius, opacity))
            return

        raise TypeError(f"Unknown data structure in add_sphere")
 
    def add_vector(self, start_point, end_point, color=(0,0,0)):

        color_scalar = color[0] * 256 * 256 + color[1] * 256 + color[2]

        if isinstance(end_point, Vector):
            end_point = end_point.components()
        if isinstance(start_point, Vector):
            start_point = start_point.components()

        # case 1: both start_point and end_point are single points (floats)
        if all(isinstance(item, numbers.Number) for item in start_point) and all(isinstance(item, numbers.Number) for item in end_point):
            self._insert_vector(start_point, end_point, color_scalar)
            return

        # case 2: both start_point and end_point are numpy arrays
        if all(isinstance(item, np.ndarray) for item in start_point) and all(isinstance(item, np.ndarray) for item in end_point):
            for start_x, start_y, start_z, end_x, end_y, end_z in zip(start_point[0], start_point[1], start_point[2], end_point[0], end_point[1], end_point[2]):
                self._insert_vector((start_x, start_y, start_z), (end_x, end_y, end_z), color_scalar)
            return

        # case 3: start_point is a group of points (numpy arrays), and end_point is a single point (floats)
        if all(isinstance(item, np.ndarray) for item in start_point) and all(isinstance(item, numbers.Number) for item in end_point):
            for start_x, start_y, start_z in zip(start_point[0], start_point[1], start_point[2]):
                self._insert_vector((start_x, start_y, start_z), end_point, color_scalar)
            return

        # case 4: end_point is a group of points (numpy arrays), and start_point is a single point (floats)
        if all(isinstance(item, numbers.Number) for item in start_point) and all(isinstance(item, np.ndarray) for item in end_point):
            for end_x, end_y, end_z in zip(end_point[0], end_point[1], end_point[2]):
                self._insert_vector(start_point, (end_x, end_y, end_z), color_scalar)
            return

        m = inspect.currentframe().f_code.co_name
        raise TypeError(f"Unknown data structure inside call: {self.__class__.__name__}.{m}")
    
    def add_element(self, element, color=(0,0,0)):
        
        if isinstance(element, Sphere) or isinstance(element, CheckeredSphere):
            self.add_sphere(p=element.center, color=color, radius=element.radius)
            return
        
        m = inspect.currentframe().f_code.co_name
        raise TypeError(f"Unknown element inside call: {self.__class__.__name__}.{m}")

    def _insert_vector(self, start_point, end_point, color_scalar):

        vector = [e - s for e, s in zip(end_point, start_point)]
        vector_magnitude = math.sqrt(sum(v ** 2 for v in vector))

        normalized_vector = [v / vector_magnitude for v in vector]

        self.vector_points.InsertNextPoint(start_point)
        self.vector_directions.InsertNextTuple(normalized_vector)
        self.vector_color_scalars.InsertNextTuple3(color_scalar // (256*256), (color_scalar // 256) % 256, color_scalar % 256)
        self.vector_magnitudes.InsertNextValue(vector_magnitude)

    def save_screenshot_callback(self, interactor, event):
        key = interactor.GetKeySym()
        if key == 's':
            self.screenshot_counter += 1
            screenshot_filename = f"{self.filename}_{self.timestamp}_{self.screenshot_counter}.png"
            screenshot_full_path = os.path.join(self.path, screenshot_filename)
            print(f"Saving screenshot as: {screenshot_full_path}")

            window_to_image_filter = self.vtk.vtkWindowToImageFilter()
            window_to_image_filter.SetInput(interactor.GetRenderWindow())
            window_to_image_filter.Update()

            image_writer = self.vtk.vtkPNGWriter()
            image_writer.SetFileName(screenshot_full_path)
            image_writer.SetInputConnection(window_to_image_filter.GetOutputPort())
            image_writer.Write()

    def plot(self, path, filename, display_3d_plot):

        self.path = path
        self.filename = filename
        os.makedirs(self.path, exist_ok=True)

        renderer = self.vtk.vtkRenderer()
        render_window = self.vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
 
        # points
        if self.points.GetNumberOfPoints() > 0:
            point_source = self.vtk.vtkSphereSource()
            point_source.SetRadius(self.point_radius)
            point_source.SetPhiResolution(16)
            point_source.SetThetaResolution(16)
            point_source.Update()
    
            point_polydata = self.vtk.vtkPolyData()
            point_polydata.SetPoints(self.points)
            point_polydata.GetPointData().SetScalars(self.point_color_scalars)
    
            point_glyphs = self.vtk.vtkGlyph3D()
            point_glyphs.SetSourceConnection(point_source.GetOutputPort())
            point_glyphs.SetInputData(point_polydata)
            point_glyphs.ScalingOff()
            point_glyphs.Update()
    
            point_mapper = self.vtk.vtkPolyDataMapper()
            point_mapper.SetInputConnection(point_glyphs.GetOutputPort())
            point_mapper.SetLookupTable(self.lookup_table)
            point_mapper.SetScalarRange(0, 256*256*256 - 1)
    
            point_actor = self.vtk.vtkActor()
            point_actor.SetMapper(point_mapper)
 
            renderer.AddActor(point_actor)

        # spheres
        for center, color, radius, opacity in self.spheres:
            sphere_source = self.vtk.vtkSphereSource()
            sphere_source.SetCenter(center)
            sphere_source.SetRadius(radius)
            sphere_source.SetThetaResolution(24) 
            sphere_source.SetPhiResolution(24)

            sphere_mapper = self.vtk.vtkPolyDataMapper()
            sphere_mapper.SetInputConnection(sphere_source.GetOutputPort())

            sphere_actor = self.vtk.vtkActor()
            sphere_actor.SetMapper(sphere_mapper)
            # Actor expects 0-1.0 range for colors
            sphere_actor.GetProperty().SetColor(color[0]/255, color[1]/255, color[2]/255)
            sphere_actor.GetProperty().SetOpacity(opacity)
            
            renderer.AddActor(sphere_actor)
 
        # vectors
        if self.vector_points.GetNumberOfPoints() > 0:
            arrow_source = self.vtk.vtkArrowSource()
            arrow_source.SetShaftRadius(self.shaft_radius)
            arrow_source.SetTipRadius(self.head_radius)
            arrow_source.SetTipLength(self.head_length)
            arrow_source.Update()

            for i in range(self.vector_points.GetNumberOfPoints()):
                start_point = self.vector_points.GetPoint(i)
                vector_direction = self.vector_directions.GetTuple(i)
                vector_magnitude = self.vector_magnitudes.GetValue(i)
                color = self.vector_color_scalars.GetTuple3(i)

                transform = self.vtk.vtkTransform()
                transform.Translate(start_point)

                original_vector = [1, 0, 0]
                cross_product = np.cross(original_vector, vector_direction)
                dot_product = np.dot(original_vector, vector_direction)
                angle = np.arccos(dot_product) * 180 / np.pi
                transform.RotateWXYZ(angle, cross_product)
                transform.Scale(vector_magnitude, 1, 1)

                transform_filter = self.vtk.vtkTransformPolyDataFilter()
                transform_filter.SetTransform(transform)
                transform_filter.SetInputConnection(arrow_source.GetOutputPort())
                transform_filter.Update()

                vector_mapper = self.vtk.vtkPolyDataMapper()
                vector_mapper.SetInputConnection(transform_filter.GetOutputPort())

                vector_actor = self.vtk.vtkActor()
                vector_actor.SetMapper(vector_mapper)
                vector_actor.GetProperty().SetColor(color[0] / 255, color[1] / 255, color[2] / 255)

                renderer.AddActor(vector_actor)
        
        renderer.GetRenderWindow().Render()
 
        if display_3d_plot:

            all_points = self.vtk.vtkPoints()
            for i in range(self.points.GetNumberOfPoints()):
                all_points.InsertNextPoint(self.points.GetPoint(i))

            for i in range(self.vector_points.GetNumberOfPoints()):
                start_point = self.vector_points.GetPoint(i)
                vector_direction = self.vector_directions.GetTuple(i)
                vector_magnitude = self.vector_magnitudes.GetValue(i)
                end_point = [start_point[j] + vector_direction[j] * vector_magnitude for j in range(3)]
                all_points.InsertNextPoint(start_point)
                all_points.InsertNextPoint(end_point)

            for s in self.spheres:
                all_points.InsertNextPoint(s[0])

            polydata = self.vtk.vtkPolyData()
            polydata.SetPoints(all_points)

            bounds = polydata.GetBounds()

            cube_axes = self.vtk.vtkCubeAxesActor()
            cube_axes.SetBounds(bounds)
            cube_axes.SetCamera(renderer.GetActiveCamera())
            cube_axes.GetTitleTextProperty(0).SetColor(1, 1, 1)
            cube_axes.GetTitleTextProperty(1).SetColor(1, 1, 1)
            cube_axes.GetTitleTextProperty(2).SetColor(1, 1, 1)
            cube_axes.GetLabelTextProperty(0).SetColor(1, 1, 1)
            cube_axes.GetLabelTextProperty(1).SetColor(1, 1, 1)
            cube_axes.GetLabelTextProperty(2).SetColor(1, 1, 1)
            cube_axes.GetXAxesGridlinesProperty().SetColor(1, 1, 1)
            cube_axes.GetYAxesGridlinesProperty().SetColor(1, 1, 1)
            cube_axes.GetZAxesGridlinesProperty().SetColor(1, 1, 1)
            cube_axes.DrawXGridlinesOn()
            cube_axes.DrawYGridlinesOn()
            cube_axes.DrawZGridlinesOn()
            cube_axes.SetXTitle("X")
            cube_axes.SetYTitle("Y")
            cube_axes.SetZTitle("Z")
            cube_axes.XAxisMinorTickVisibilityOff()
            cube_axes.YAxisMinorTickVisibilityOff()
            cube_axes.ZAxisMinorTickVisibilityOff()
 
            renderer.AddActor(cube_axes)
            renderer.SetBackground(0,0,0)

            instructions = self.vtk.vtkTextActor()
            instructions.SetInput("Press 's' to save a screenshot")
            instructions.GetTextProperty().SetColor(1, 1, 1)
            instructions.GetTextProperty().SetFontSize(24)
            instructions.SetPosition(10, 10)
            renderer.AddActor(instructions)

            render_window_interactor = self.vtk.vtkRenderWindowInteractor()
            render_window_interactor.SetRenderWindow(render_window)

            render_window_interactor.AddObserver("KeyPressEvent", self.save_screenshot_callback)

            render_window_interactor.Initialize()
            render_window.SetWindowName(self.luminous_dialog_title)
            render_window.Render()
            render_window_interactor.Start()
            
        else:
            render_window.SetOffScreenRendering(1)
            render_window.Render()