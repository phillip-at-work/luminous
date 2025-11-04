from __future__ import annotations
from typing import TYPE_CHECKING, Type
from ..math.vector import Vector
import importlib

if TYPE_CHECKING:
    from .detector import Detector

class RenderTarget:
    _instances: dict[int, RenderTarget] = {}

    def __new__(cls, detector: Detector, render_targets: list[Type[Detector]] = None) -> RenderTarget:
        detector_id = id(detector)
        if detector_id not in cls._instances:
            instance = super(RenderTarget, cls).__new__(cls)
            cls._instances[detector_id] = instance
            instance._initialize(detector, render_targets)
        return cls._instances[detector_id]

    def _initialize(self, detector: Detector, render_targets: list[Type[Detector]]):
        self.detector = detector
        self.render_targets = render_targets or []
        self._data = {}
        self._processed_detectors = set()
        self._enqueue_detectors(self.render_targets)

    def _enqueue_detectors(self, detectors: list[Type[Detector]]):
        for target_cls in detectors:
            if target_cls not in self._processed_detectors:

                module_name = target_cls.__module__
                class_name = target_cls.__name__
                module = importlib.import_module(module_name)
                target_cls = getattr(module, class_name)
                target_instance = target_cls(self.detector.width, self.detector.height, self.detector.position, self.detector.pointing_direction)
                
                self._data[target_cls] = target_instance
                self._processed_detectors.add(target_cls)

                # recursively enqueue any render_targets of the newly added detector
                self._enqueue_detectors(target_instance.render_target.render_targets)

    def _enqueue_implicit_detector(self, detector_cls: Type[Detector]):
        if detector_cls not in self._processed_detectors:
            self.render_targets.insert(0, detector_cls)
            self._enqueue_detectors([detector_cls])
        else:
            # Move the detector to the front of the render_targets list if it already exists
            if detector_cls in self.render_targets:
                self.render_targets.remove(detector_cls)
            self.render_targets.insert(0, detector_cls)

    def _reflection_model(self, element, intersection_point: Vector, surface_normal_at_intersection: Vector, direction_to_origin_unit: Vector, intersection_map: list[dict]) -> Vector:
        # Call _reflection_model method for each render_target
        for target_cls in self.render_targets:
            target_instance = self._data[target_cls]
            target_instance._reflection_model(element, intersection_point, surface_normal_at_intersection, direction_to_origin_unit, intersection_map)
        
        # Call _reflection_model method for the main detector
        return self.detector._reflection_model(element, intersection_point, surface_normal_at_intersection, direction_to_origin_unit, intersection_map)

    def _transmission_model(self, element, initial_intersection, final_intersection: float) -> Vector:
        # Call _transmission_model method for each render_target
        for target_cls in self.render_targets:
            target_instance = self._data[target_cls]
            target_instance._transmission_model(element, initial_intersection, final_intersection)
        
        # Call _transmission_model method for the main detector
        return self.detector._transmission_model(element, initial_intersection, final_intersection)

    def get_data(self, target_cls: Type[Detector]):
        # use this method to recover data from implicit detectors
        # e.g., from my_other_detector._reflection_model()
        # call `other_detector_data = self.render_target.get_data(OtherDetector)`
        if target_cls in self._data:
            return self._data[target_cls]._data
        else:
            raise ValueError(
                "Data for detector type '%s' is not available. "
                "Ensure that the detector type is enqueued correctly." % target_cls.__name__
            )