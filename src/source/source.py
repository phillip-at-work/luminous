from abc import ABC, abstractmethod


class Source(ABC):
    pass


class Laser(Source):
    pass


class Point(Source):
    def __init__(self, position):
        self.position = position


class Custom(Source):
    pass
