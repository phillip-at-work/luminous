import os 
from luminous.src.math.vector import Vector

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

    def __init__(self, point_radius=0.01, shaft_radius=0.05, head_radius=0.1, head_length=0.2):
        import vtk
        self.vtk = vtk
        self.point_radius = point_radius
        # TODO shaft_radius, head_radius, and head_length intended for vectors
        self.shaft_radius = shaft_radius
        self.head_radius = head_radius
        self.head_length = head_length
        self.points = vtk.vtkPoints()
        self.scalars = vtk.vtkFloatArray()
        self.lookup_table = vtk.vtkLookupTable()
        self.luminous_dialog_title = "luminous ray debugger"

        self.lookup_table.SetNumberOfTableValues(256*256*256)
        self.lookup_table.Build()

        for i in range(256*256*256):
            r = (i // (256*256)) % 256
            g = (i // 256) % 256
            b = i % 256
            self.lookup_table.SetTableValue(i, r/255.0, g/255.0, b/255.0, 1.0)

    def add_point(self, end_point, color=(0,0,0)):

        color_scalar = color[0] * 256 * 256 + color[1] * 256 + color[2]

        if isinstance(end_point, Vector):
            end_point = end_point.components()

        try:
            for x_end, y_end, z_end in zip(*end_point):
                end = (x_end, y_end, z_end)
                self.points.InsertNextPoint(end)
                self.scalars.InsertNextValue(color_scalar)

        except TypeError:
            self.points.InsertNextPoint(end_point)
            self.scalars.InsertNextValue(color_scalar)

    def add_vector(self, end_point, start_point=(0,0,0), color=(0,0,0)):

        color_scalar = color[0] * 256 * 256 + color[1] * 256 + color[2]

        if isinstance(end_point, Vector):
            end_point = end_point.components()
        if isinstance(start_point, Vector):
            start_point = start_point.components()

        try:
            for x_end, y_end, z_end, x_start, y_start, z_start in zip(*end_point, *start_point):
                # TODO append multiple vectors to appropriate data structure
                pass
                
        except TypeError:
            # TODO append one vector to appropriate data structure
            pass

    def plot(self, path="./results", filename="debug_ray_trace", display_3d_plot=False):
        os.makedirs(path, exist_ok=True)
        full_path = os.path.join(path, filename)

        renderer = self.vtk.vtkRenderer()
        render_window = self.vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)

        sphere_source = self.vtk.vtkSphereSource()
        sphere_source.SetRadius(self.point_radius)
        sphere_source.SetPhiResolution(16)
        sphere_source.SetThetaResolution(16)
        sphere_source.Update()

        point_polydata = self.vtk.vtkPolyData()
        point_polydata.SetPoints(self.points)
        point_polydata.GetPointData().SetScalars(self.scalars)

        glyphs = self.vtk.vtkGlyph3D()
        glyphs.SetSourceConnection(sphere_source.GetOutputPort())
        glyphs.SetInputData(point_polydata)
        glyphs.ScalingOff()
        glyphs.Update()

        glyph_mapper = self.vtk.vtkPolyDataMapper()
        glyph_mapper.SetInputConnection(glyphs.GetOutputPort())
        glyph_mapper.SetLookupTable(self.lookup_table)
        glyph_mapper.SetScalarRange(0, 256*256*256 - 1)

        glyph_actor = self.vtk.vtkActor()
        glyph_actor.SetMapper(glyph_mapper)

        renderer.AddActor(glyph_actor)

        renderer.GetRenderWindow().Render()

        if display_3d_plot:

            all_points = self.vtk.vtkPoints()
            for i in range(self.points.GetNumberOfPoints()):
                all_points.InsertNextPoint(self.points.GetPoint(i))

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

            render_window_interactor = self.vtk.vtkRenderWindowInteractor()
            render_window_interactor.SetRenderWindow(render_window)

            render_window_interactor.Initialize()
            render_window.SetWindowName(self.luminous_dialog_title)

            render_window.Render()
            render_window_interactor.Start()
        else:
            render_window.SetOffScreenRendering(1)
            render_window.Render()

        color_array = self.vtk.vtkUnsignedCharArray()
        color_array.SetNumberOfComponents(3)
        color_array.SetName("Colors")
        for i in range(self.points.GetNumberOfPoints()):
            scalar_value = self.scalars.GetValue(i)
            r = (scalar_value // (256*256)) % 256
            g = (scalar_value // 256) % 256
            b = scalar_value % 256
            color_array.InsertNextTuple3(r, g, b)

        point_polydata.GetPointData().AddArray(color_array)

        ply_writer = self.vtk.vtkPLYWriter()
        ply_writer.SetFileName(full_path + ".ply")
        ply_writer.SetInputData(point_polydata)
        ply_writer.SetArrayName("Colors")
        ply_writer.SetColorModeToDefault()
        ply_writer.Write()