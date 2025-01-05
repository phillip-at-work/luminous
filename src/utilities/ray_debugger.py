import os 

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

class ConcreteRayDebugger:
    '''
    Actual RayDebugger compiles data and plots rays.
    Prints .obj, .mtl to specified directory.
    '''

    def __init__(self, shaft_radius=0.05, head_radius=0.1, head_length=0.2, point_radius=None):
        import vtk
        self.vtk = vtk
        self.shaft_radius = shaft_radius
        self.head_radius = head_radius
        self.head_length = head_length
        self.point_radius = point_radius if point_radius is not None else head_radius
        self.renderer = vtk.vtkRenderer()
        self.vectors = list()
        self.points = list()
        self.luminous_dialog_title = "luminous ray debugger"

    def add_vector(self, end_point, start_point=(0,0,0), color=(0,0,0)):
        self.vectors.append((start_point, end_point, color))

    def add_point(self, end_point, color=(0,0,0)):
        self.points.append((end_point, color))

    def create_arrow(self, start_point, end_point, color):
        vector = [end_point[i] - start_point[i] for i in range(3)]
        length = self.vtk.vtkMath.Norm(vector)
        self.vtk.vtkMath.Normalize(vector)

        arrow_source = self.vtk.vtkArrowSource()
        arrow_source.SetShaftRadius(self.shaft_radius)
        arrow_source.SetTipRadius(self.head_radius)
        arrow_source.SetTipLength(self.head_length / (self.head_length + length))

        transform = self.vtk.vtkTransform()
        transform.Translate(start_point)

        cross_vec = [0.0, 0.0, 0.0]
        self.vtk.vtkMath.Cross([1, 0, 0], vector, cross_vec)
        angle = self.vtk.vtkMath.DegreesFromRadians(self.vtk.vtkMath.AngleBetweenVectors([1, 0, 0], vector))
        transform.RotateWXYZ(angle, cross_vec)

        transform.Scale(length, 1, 1)

        transform_filter = self.vtk.vtkTransformPolyDataFilter()
        transform_filter.SetTransform(transform)
        transform_filter.SetInputConnection(arrow_source.GetOutputPort())
        transform_filter.Update()

        mapper = self.vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(transform_filter.GetOutputPort())

        actor = self.vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)

        return actor

    def create_sphere(self, center, radius, color):
        sphere_source = self.vtk.vtkSphereSource()
        sphere_source.SetCenter(center)
        sphere_source.SetRadius(radius)
        sphere_source.SetPhiResolution(16)
        sphere_source.SetThetaResolution(16)
        sphere_source.Update()

        mapper = self.vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere_source.GetOutputPort())

        actor = self.vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)

        return actor

    def plot(self, path="./results", filename="debug_ray_trace", display_3d_plot=False):
        os.makedirs(path, exist_ok=True)
        full_path = os.path.join(path, filename)

        render_window = self.vtk.vtkRenderWindow()
        render_window.AddRenderer(self.renderer)

        assembly = self.vtk.vtkAssembly()
        for start_point, end_point, color in self.vectors:
            arrow_actor = self.create_arrow(start_point, end_point, color)
            assembly.AddPart(arrow_actor)

        for center, color in self.points:
            sphere_actor = self.create_sphere(center, self.point_radius, color)
            assembly.AddPart(sphere_actor)

        self.renderer.AddActor(assembly)

        # export obj file with associated mtl
        obj_exporter = self.vtk.vtkOBJExporter()
        obj_exporter.SetFilePrefix(full_path)
        obj_exporter.SetRenderWindow(render_window)
        obj_exporter.Write()

        if display_3d_plot:
            for start_point, end_point, color in self.vectors:
                arrow_actor = self.create_arrow(start_point, end_point, color)
                self.renderer.AddActor(arrow_actor)

            for center, color in self.points:
                sphere_actor = self.create_sphere(center, self.point_radius, color)
                self.renderer.AddActor(sphere_actor)

            all_points = self.vtk.vtkPoints()
            for start_point, end_point, _ in self.vectors:
                all_points.InsertNextPoint(start_point)
                all_points.InsertNextPoint(end_point)
            for center, _ in self.points:
                all_points.InsertNextPoint(center)

            polydata = self.vtk.vtkPolyData()
            polydata.SetPoints(all_points)

            bounds = polydata.GetBounds()

            cube_axes = self.vtk.vtkCubeAxesActor()
            cube_axes.SetBounds(bounds)
            cube_axes.SetCamera(self.renderer.GetActiveCamera())
            cube_axes.GetTitleTextProperty(0).SetColor(0, 0, 0)
            cube_axes.GetTitleTextProperty(1).SetColor(0, 0, 0)
            cube_axes.GetTitleTextProperty(2).SetColor(0, 0, 0)
            cube_axes.GetLabelTextProperty(0).SetColor(0, 0, 0)
            cube_axes.GetLabelTextProperty(1).SetColor(0, 0, 0)
            cube_axes.GetLabelTextProperty(2).SetColor(0, 0, 0)
            cube_axes.GetXAxesGridlinesProperty().SetColor(0, 0, 0)
            cube_axes.GetYAxesGridlinesProperty().SetColor(0, 0, 0)
            cube_axes.GetZAxesGridlinesProperty().SetColor(0, 0, 0)
            cube_axes.DrawXGridlinesOn()
            cube_axes.DrawYGridlinesOn()
            cube_axes.DrawZGridlinesOn()
            cube_axes.SetXTitle("X")
            cube_axes.SetYTitle("Y")
            cube_axes.SetZTitle("Z")
            cube_axes.XAxisMinorTickVisibilityOff()
            cube_axes.YAxisMinorTickVisibilityOff()
            cube_axes.ZAxisMinorTickVisibilityOff()

            self.renderer.AddActor(cube_axes)
            self.renderer.SetBackground(1, 1, 1)

            render_window = self.vtk.vtkRenderWindow()
            render_window.AddRenderer(self.renderer)

            render_window_interactor = self.vtk.vtkRenderWindowInteractor()
            render_window_interactor.SetRenderWindow(render_window)

            render_window_interactor.Initialize()
            render_window.SetWindowName(self.luminous_dialog_title)

            render_window.Render()
            render_window_interactor.Start()

        else:

            render_window.SetOffScreenRendering(1)
            render_window.Render()