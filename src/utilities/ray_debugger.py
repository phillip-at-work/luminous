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

    def add_vector(self, start_point, end_point, color):
        pass
    def add_point(self, end_point, color):
        pass
    def plot(self):
        pass

class ConcreteRayDebugger(RayDebugger):
    '''
    Actual RayDebugger compiles data and plots rays.
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

    def add_vector(self, start_point, end_point, color):
        self.vectors.append((start_point, end_point, color))

    def add_point(self, end_point, color):
        self.points.append((end_point, color))

    def create_arrow(self, start_point, end_point, color):
        
        # vector direction and length
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
        angle = self.vtk.vtkMath.DegreesFromRadians(vtk.vtkMath.AngleBetweenVectors([1, 0, 0], vector))
        transform.RotateWXYZ(angle, cross_vec)

        transform.Scale(length, 1, 1)

        mapper = self.vtk.vtkPolyDataMapper()
        actor = self.vtk.vtkActor()
        transform_filter = self.vtk.vtkTransformPolyDataFilter()
        transform_filter.SetTransform(transform)
        transform_filter.SetInputConnection(arrow_source.GetOutputPort())
        transform_filter.Update()

        mapper.SetInputConnection(transform_filter.GetOutputPort())
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)

        return actor

    def create_sphere(self, center, radius, color):

        sphere_source = self.vtk.vtkSphereSource()
        sphere_source.SetCenter(center)
        sphere_source.SetRadius(radius)
        sphere_source.SetPhiResolution(16)
        sphere_source.SetThetaResolution(16)

        mapper = self.vtk.vtkPolyDataMapper()
        actor = self.vtk.vtkActor()
        mapper.SetInputConnection(sphere_source.GetOutputPort())
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)

        return actor
    
    def plot(self, path="./results", filename="debug_ray_trace", display_3d_plot=False):
        # Ensure the directory exists
        os.makedirs(path, exist_ok=True)
        full_path = os.path.join(path, filename + ".ply")

        # Create an append filter to combine all geometry
        append_filter = self.vtk.vtkAppendPolyData()

        for start_point, end_point, color in self.vectors:
            arrow_actor = self.create_arrow(start_point, end_point, color)
            append_filter.AddInputData(arrow_actor.GetMapper().GetInput())

        for center, color in self.points:
            sphere_actor = self.create_sphere(center, self.point_radius, color)
            append_filter.AddInputData(sphere_actor.GetMapper().GetInput())

        append_filter.Update()

        # ply file write
        ply_writer = self.vtk.vtkPLYWriter()
        ply_writer.SetFileName(full_path)
        ply_writer.SetInputConnection(append_filter.GetOutputPort())
        ply_writer.SetColorModeToDefault()  # This mode attempts to save color information if present
        ply_writer.Write()

        render_window = self.vtk.vtkRenderWindow()
        render_window.AddRenderer(self.renderer)

        # display dialog with 3d volume
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

            render_window_interactor = self.vtk.vtkRenderWindowInteractor()
            render_window_interactor.SetRenderWindow(render_window)
            render_window.Render()
            render_window_interactor.Initialize()
            render_window_interactor.Start()

        else:
            # perform headless render
            render_window.SetOffScreenRendering(1)
            render_window.Render()