from shapely.geometry import Polygon
from htrflow.utils.geometry import Bbox
from src.file_tools import read_json_file, write_json_file
from typing import Dict

class Line():
    def __init__(self, bbox: Bbox = None, polygon: Polygon = None, text: str = None):
        self.bbox = bbox
        self.polygon = polygon
        self.text = text

    def __getitem__(self, key):
        return getattr(self, key)

    @classmethod
    def from_dict(cls, data: Dict):
        return Line(bbox=Bbox(*data["bbox"]), polygon=Polygon(data["polygon"]), text=data["text"])

    @property
    def dict(self):
        return dict(
            bbox=[*self.bbox],
            polygon=[(x, y) for x, y in self.polygon.boundary.coords], 
            text=self.text
        )


class Region():
    def __init__(self, bbox: Bbox = None, polygon: Polygon = None, text: str = None, lines: list[Line] = None):
        self.bbox = bbox
        self.polygon = polygon
        self.lines = lines

        if text is None and lines is not None:
            self.text = " ".join([line.text for line in lines])
        elif text is None and lines is None:
            self.text = ""
        elif text is not None:
            self.text = text
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    @classmethod
    def from_dict(cls, data: Dict):
        lines = [Line.from_dict(line) for line in data["lines"]]
        return Region(bbox=Bbox(*data["bbox"]), polygon=Polygon(data["polygon"]), lines=lines)

    @property
    def dict(self):
        return dict(
            bbox=[*self.bbox],
            polygon=[(x, y) for x, y in self.polygon.boundary.coords], 
            lines=[line.dict for line in self.lines],
            text=self.text
        )


class Page():
    def __init__(self, regions: list[Region] = None, lines: list[Line] = None, text: str = None, path: str = None):
        self.regions = regions
        self.lines = lines

        if text is None and lines is not None:
            self.text = " ".join([line.text for line in lines])
        elif text is None and lines is None:
            self.text = ""
        elif text is not None:
            self.text = text

        self.path = path

    def __getitem__(self, key):
        return getattr(self, key)
    
    @classmethod
    def from_json(cls, json_path):
        data = read_json_file(json_path)
        regions = [Region.from_dict(region) for region in data["regions"]]
        lines = [Line.from_dict(line) for line in data["lines"]]
        path = data["path"]
        return Page(regions=regions, lines=lines, path=path)

    def to_json(self, json_path):
        write_json_file(self.dict, json_path)

    @property
    def dict(self):
        return dict(
            regions=[region.dict for region in self.regions],
            lines=[line.dict for line in self.lines],
            text=self.text,
            path=self.path,
        )


class ODOutput():
    def __init__(self, bboxes: list[Bbox], polygons: list[Polygon]):
        assert len(bboxes) == len(polygons), "Number of bboxes and polygons must be equal"

        self.bboxes = bboxes
        self.polygons = polygons
        self._objects_count = len(bboxes)

    def __iter__(self):
        for idx in range(len(self._objects_count)):
            yield self.__getitem__(idx)

    def __len__(self):
        return self._objects_count
    
    def _get_one(self, idx):
        return dict(bbox=self.bboxes[idx], polygon=self.polygons[idx])
    
    def __add__(self, other):
        return ODOutput(bboxes=self.bboxes + other.bboxes, polygons=self.polygons + other.polygons)
        
    def __getitem__(self, idx: int | slice):
        if isinstance(idx, int):
            return self._get_one(idx)
        else:
            return [self._get_one(i) for i in range(idx.start or 0, idx.stop or -1, idx.step or 1) if i < len(self.bboxes)]


# From Riksarkivet with slight modifications
class Ratio:
    """
    An unormalized `fraction.Fraction`

    This class makes it easy to compute the total error rate (and other
    fraction-based metrics) of several documents.

    Example: Let A and B be two documents with WER(A) = 1/5 and
    WER(B) = 1/100. In total, there are 2 errors and 105 words, so
    WER(A+B) = 2/105.

    Addition of two `Ratio` instances supports this:
    >>> Ratio(1, 5) + Ratio(1, 100)
    Ratio(2, 105)
    """

    def __init__(self, a, b):
        self.a = int(a)
        self.b = int(b)

    def __add__(self, other):
        if other == 0:
            # sum(a, b) performs the addition 0 + a + b internally,
            # which means that Ratio must support addition with 0 in
            # order to work with sum().
            return self
        return Ratio(self.a + other.a, self.b + other.b)

    __radd__ = __add__  # redirects int + Ratio to __add__

    def __float__(self):
        return 0.0 if self.b == 0 else float(self.a / self.b)

    def __gt__(self, other):
        return float(self) > float(other)

    def __lt__(self, other):
        return float(self) < float(other)

    def __eq__(self, other):
        return float(self) == float(other)

    def __str__(self):
        return f"{self.a}/{self.b}"