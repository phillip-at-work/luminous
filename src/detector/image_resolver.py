from ..math.vector import Vector
from ..element.element import Element
from ..source.source import Source
from ..detector.detector import Detector
from collections import deque

from matplotlib import pyplot as plt
import math

import logging
from luminous.src.utilities.logconfig import setup_logging
logger = logging.getLogger(__name__)

ENUMS = ['REFLECTION', 'TRANSMISSION', 'SOURCED', 'PLACE']

class ImageResolver:
    def __init__(self):
        self.ray_record = dict()
        # self.detectors = set()
        self.intermediate_results = dict()
        # self.results = dict()
        # self.pixel_integrations = deque()
        # self.pixel_dimensions = deque()
        # self.clear_queues = False

    # def _map_source(self, origin: Vector, source: Source, element: Element, detector: Detector):
    #     '''
    #     origin: point of ray origin
    #     source: Source object, which also defines final point of intersection
    #     element: Element where reflection occurred
    #     detector: detector associated with this ray
    #     '''

    #     self.detectors.add(detector)

    #     n = source.pointing_direction if source.pointing_direction is not None else (source.position - origin).norm()
    #     r = RayRecord(origin, source.position, n, element, 'SOURCED')

    #     if self.ray_record.get(source) is None:

    #         if self.ray_record.get(detector) is None:
    #             self.ray_record[detector] = [r]
    #             self.ray_record[source] = [(detector, [r])]

    #         else:
    #             antecedent = self.ray_record[detector]
    #             antecedent.append(r)
    #             self.ray_record[source] = [(detector, antecedent)]

    #     else:

    #         if self.ray_record.get(detector) is None:
    #             self.ray_record[detector] = [r]
    #             self.ray_record[source].append((detector, [r]))

    #         else:
    #             antecedent = self.ray_record[detector]
    #             antecedent.append(r)
    #             self.ray_record[source].append((detector, antecedent))

    def _map_reflection(self, element, intersection_point, surface_normal_at_intersection, direction_to_origin_unit, intersection_map, detector):

        r = ReflectionRecord(element, intersection_point, surface_normal_at_intersection, direction_to_origin_unit, intersection_map)
        self._append(detector, r)

    def _map_transmission(self, element, intersection_point, full_transmitted_ray_within_volume, detector):

        r = TransmissionRecord(element, intersection_point, full_transmitted_ray_within_volume)
        self._append(detector, r)

    def _map_pixels(self, hit, detector):

        r = IntegrationRecord(hit)
        self._append(detector, r)

    def _append(self, detector, record):
        '''
        Append RayRecord to list, index into dict.
        '''

        if self.ray_record.get(detector) is None:
            self.ray_record[detector] = [record]
            return

        self.ray_record[detector].append(record)

    def _reverse_resolve(self, detector):

        self.intermediate_results[detector] = list()

        for r in self.ray_record[detector]:

            if r.enum == 'REFLECTION':

                e = r.element
                i = r.intersection_point
                n = r.surface_normal_at_intersection
                o = r.direction_to_origin_unit
                m = r.intersection_map
            
                self.intermediate_results[detector].append(detector._reflection_model(e, i, n, o, m))

                logger.debug(f"REFLECTION ray record. {detector._reflection_model(e, i, n, o, m).x.shape}")

            elif r.enum == 'TRANSMISSION':

                e = r.element
                i = r.intersection_point
                r = r.full_transmitted_ray_within_volume

                self.intermediate_results[detector].append(detector._transmission_model(e, i, r))

                logger.debug(f"TRANSMISSION ray record. {detector._transmission_model(e, i, r).x.shape}")

            # elif r.enum == 'INTEGRATE':

            #     print(f"INTEGRATION ray record")

            #     i = len(self.intermediate_results[detector]) - 1
            #     v = self.intermediate_results[detector][i]

            #     while i >= 1:
            #         print(f"      while loop")
            #         v += self.intermediate_results[detector][i-1]
            #         i -= 1
                
            #     self.intermediate_results[detector].clear()
            #     self.intermediate_results[detector].append(v.place(r.hit))

            #     print(f"final integrated x.shape: {self.intermediate_results[detector][0].x.shape}")

            elif r.enum == 'INTEGRATE':

                logger.debug(f"INTEGRATION ray record")

                i = 0 #len(self.intermediate_results[detector]) - 1
                v = self.intermediate_results[detector][i]

                logger.debug(f"initial integrated x.shape: {v.x.shape}")

                while i <= len(self.intermediate_results[detector]) - 2: #>= 1:
                    logger.debug(f"... while loop. adding x.shape: {self.intermediate_results[detector][i+1].x.shape}")
                    v += self.intermediate_results[detector][i+1]
                    i += 1
                
                self.intermediate_results[detector].clear()
                self.intermediate_results[detector].append(v.place(r.hit))

                logger.debug(f"final integrated x.shape: {self.intermediate_results[detector][0].x.shape}")




    def _resolve(self):
        pass

    # def _resolve(self):
    #     '''
    #     Iterate over Source keys, applying Detector transmission and reflection model.
    #     Final image integrates the contributions of all reflections and refractions, for every source.
    #     '''
    #     # print(f"ray record keys: {self.ray_record.keys()}")
    #     sources = [s for s in self.ray_record.keys() if isinstance(s, Source)]
    #     self.results = {key: Vector(0, 0, 0) for key in self.detectors}
    #     # results = dict()
    #     print(f"RESOLVER. resolving ##########")

    #     for s in sources:

    #         rays = self.ray_record[s] # a list of tuples
    #         for r in rays:

    #             if len(r) > 2:
    #                 raise IndexError("Ray record wrong size.")

    #             detector = r[0]     # detector
    #             ray_events = r[1]   # reflection and transmission events

    #             for event in reversed(ray_events):

    #                 o = event.origin
    #                 i = event.intersection_point
    #                 n = event.normal_at_intersection
    #                 e = event.element

    #                 print(f"RESOLVER. ### event enum: {event.enum}. ###")

    #                 if event.enum == 'REFLECTION':

    #                     if self.clear_queues:
    #                         pass
    #                         # clear queue content into self.results[detector]

    #                     v = detector._reflection_model_test(o, i, n, e)
    #                     self.pixel_integrations.append(v)

    #                     # if results.get(detector) is None:
    #                     #     results[detector] = v #.place(self.ray_element_hits)
    #                     #     print(f"RESOLVER. reflect... = v. reflect. {results[detector].x.size}")
    #                     # else:
    #                     #     results[detector] += v #.place(self.ray_element_hits)
    #                     #     print(f"RESOLVER. reflect... += v. reflect. {results[detector].x.size}")

                        

    #                 elif event.enum == 'TRANSMISSION':

    #                     if self.clear_queues:
    #                         pass
    #                         # clear queue content into self.results[detector]

    #                     v = detector._transmission_model_test(o, i, n, e)
    #                     self.pixel_integrations.append(v)

    #                     # if results.get(detector) is None:
    #                     #     results[detector] = v #.place(self.ray_element_hits)
    #                     #     print(f"RESOLVER. transmit... = v. transmit. {results[detector].x.size}")
    #                     # else:
    #                     #     results[detector] += v #.place(self.ray_element_hits)
    #                     #     print(f"RESOLVER. transmit... += v. transmit. {results[detector].x.size}")

    #                 elif event.enum == 'PLACE':
    #                     pass

    #                     # TODO need to get craftier with how to apply `place`

    #                     # print(f"resolver place...")

    #                     # if detector not in results.keys():
    #                     #     print(f"place. key not found.")
    #                     #     continue



    #                     # hit = 0
    #                     # if len(self.place_queue) > 0:
    #                     #     hit = self.place_queue.popleft()
                            
    #                     #     print(f"RESOLVER. popping place value for use. {hit.shape}")

    #                     #     if self.results.get(detector) is None:

    #                     #         print(f"place. = case. hit.shape={hit.shape}. data.x.size={results[detector].x.size}")
    #                     #         self.results[detector] = results[detector].place(hit)
                                
    #                     #     else:

    #                     #         print(f"place. += case. hit.shape={hit.shape}. data.x.size={results[detector].x.size}")
    #                     #         self.results[detector] += results[detector].place(hit)
                                

    #                     #     print(f"RESOLVER. appending place value after initial use. shape={event.hit.shape}")
    #                     #     self.place_queue.append(event.hit)

    #                     #     # if results[detector] is not None:
    #                     #     #     del results[detector]

    #                     # elif len(self.place_queue) == 0:
    #                     #     print(f"RESOLVER. appending place value only. shape={event.hit.shape}")
    #                     #     self.place_queue.append(event.hit)
                            



    #                 elif event.enum == 'SOURCED':

    #                     # TODO implement source solution ray model here

    #                     pass

    #     for detector in self.results.keys():
    #         print(f"_data.x size from image_resolver: {self.results[detector].x.size}")
    #         detector._data = self.results[detector]

# class RayRecord:
#     def __init__(self, origin, intersection_point, normal_at_intersection, element, enum, hit=None):
#         if enum not in ENUMS:
#             raise ValueError('enums went rogue. panic.')
#         self.origin = origin
#         self.intersection_point = intersection_point
#         self.normal_at_intersection = normal_at_intersection
#         self.element = element
#         self.enum = enum
#         self.hit = hit

class ReflectionRecord:
    def __init__(self, element, intersection_point, surface_normal_at_intersection, direction_to_origin_unit, intersection_map):
        self.element = element
        self.intersection_point = intersection_point
        self.surface_normal_at_intersection = surface_normal_at_intersection
        self.direction_to_origin_unit = direction_to_origin_unit
        self.intersection_map = intersection_map
        self.enum = 'REFLECTION'

class TransmissionRecord:
    def __init__(self, element, intersection_point, full_transmitted_ray_within_volume):
        self.element = element
        self.intersection_point = intersection_point
        self.full_transmitted_ray_within_volume = full_transmitted_ray_within_volume
        self.enum = 'TRANSMISSION'

class IntegrationRecord:
    def __init__(self, hit):
        self.hit = hit
        self.enum = "INTEGRATE"