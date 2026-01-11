import cv2 as cv
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .extraction import CandidateExtraction
from .location import LFinderDetector, DashedBorderDetector, DataMatrixValidator
from .location.l_finder_detector import LPattern, LineSegment
from .location.dashed_border_detector import DataMatrixLocation
from .geometry import BorderFitter, PreciseLocation


@dataclass
class DetectionResult:
    candidate_box: Tuple[int, int, int, int]
    precise_location: Optional[PreciseLocation]
    l_patterns: List[LPattern]
    is_valid: bool
    score: float


class DataMatrixPipeline:

    def __init__(self,
                 canny_t1: int = 50,
                 canny_t2: int = 150,
                 min_area: float = 400.0,
                 min_perimeter: float = 80.0,
                 padding: int = 10):

        self.extractor = CandidateExtraction(
            canny_t1=canny_t1,
            canny_t2=canny_t2,
            min_area=min_area,
            min_perimeter=min_perimeter,
            padding=padding
        )

        self.l_finder = LFinderDetector()
        self.validator = DataMatrixValidator()
        self.border_fitter = BorderFitter()

    def process_frame(self, frame: np.ndarray) -> List[DetectionResult]:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        candidates = self.extractor.get_candidates(frame)
        results = []

        for (x, y, w, h) in candidates:
            region = gray[y:y+h, x:x+w]

            segments = self.l_finder.detect_lines(region)
            l_patterns = self.l_finder.find_l_patterns(segments)

            precise_location = None
            is_valid = False
            score = 0.0

            if len(l_patterns) > 0:
                l_pattern = l_patterns[0]
                
                validation = self.validator.validate(region, l_pattern)
                
                if validation.is_valid:
                    precise_location = self.border_fitter.fit(region, l_pattern)
                    is_valid = True
                    score = validation.score

            results.append(DetectionResult(
                candidate_box=(x, y, w, h),
                precise_location=precise_location,
                l_patterns=l_patterns,
                is_valid=is_valid,
                score=score
            ))

        valid_results = [r for r in results if r.is_valid]
        valid_results.sort(key=lambda r: r.score, reverse=True)
        
        return valid_results

    def draw_results(self, frame: np.ndarray, results: List[DetectionResult],
                     debug_view: bool = False) -> np.ndarray:
        output = frame.copy()

        for result in results:
            x, y, w, h = result.candidate_box

            if result.precise_location and result.is_valid:
                vertices = result.precise_location.get_ordered_vertices()
                abs_vertices = [(v[0] + x, v[1] + y) for v in vertices]
                
                pts = np.array(abs_vertices, dtype=np.int32)
                cv.polylines(output, [pts], True, (0, 255, 0), 2)

            elif debug_view:
                cv.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 1)

        return output
