import os 
from luminous.src.math.vector import Vector
import math
import numpy as np
import datetime

class RayDebugger:
    def add_vector(self, start_point, end_point, color):
        raise NotImplementedError("Only NullRayDebugger or ConcreteRayDebugger should be instantiated.")
    def add_point(self, end_point, color):
        raise NotImplementedError("Only NullRayDebugger or ConcreteRayDebugger should be instantiated.")

class NullRayDebugger(RayDebugger):
    '''
    No-operation RayDebugger.
    '''

    def add_vector(self, end_point, start_point=(0,0,0), color=(0,0,0)):
        pass
    def add_point(self, end_point, color=(0,0,0)):
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
 
    def add_point(self, end_point, color=(0,0,0)):
 
        color_scalar = color[0] * 256 * 256 + color[1] * 256 + color[2]
 
        if isinstance(end_point, Vector):
            end_point = end_point.components()
 
        try:
            for x_end, y_end, z_end in zip(*end_point):
                end = (x_end, y_end, z_end)
                self.points.InsertNextPoint(end)
                self.point_color_scalars.InsertNextValue(color_scalar)
 
        except TypeError:
            self.points.InsertNextPoint(end_point)
            self.point_color_scalars.InsertNextValue(color_scalar)
 
    def add_vector(self, start_point, end_point, color=(0,0,0)):

        if isinstance(end_point, Vector):
            end_point = end_point.components()
        if isinstance(start_point, Vector):
            start_point = start_point.components()
        
        vector = [e - s for e, s in zip(end_point, start_point)]
        vector_magnitude = math.sqrt(sum(v ** 2 for v in vector))

        normalized_vector = [v / vector_magnitude for v in vector]

        self.vector_points.InsertNextPoint(start_point)
        self.vector_directions.InsertNextTuple(normalized_vector)
        self.vector_color_scalars.InsertNextTuple3(color[0], color[1], color[2])
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

    def plot(self, path="./results", filename="debug_ray_trace", display_3d_plot=True):

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