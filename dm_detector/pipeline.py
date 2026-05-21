import cv2 as cv
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from dm_detector.extraction.candidate_extraction import CandidateExtraction
from dm_detector.location.l_finder_detector import LFinderDetector, LPattern
from dm_detector.location.validator import DataMatrixValidator
from dm_detector.location.dashed_border_detector import DashedBorderDetector
from dm_detector.geometry.border_fitter import BorderFitter, PreciseLocation

@dataclass
class DetectionResult:
    candidate_box: Tuple[int, int, int, int]
    precise_location: Optional[PreciseLocation]
    l_patterns: List[LPattern]
    is_valid: bool
    score: float

    def get_rectified_image(self, full_frame: np.ndarray, output_size: int = 400) -> Optional[np.ndarray]:
        if not self.precise_location or not self.l_patterns:
            return None

        vertices = np.array(self.precise_location.vertices, dtype=np.float32)

        cx, cy, _, _ = self.candidate_box
        l_corner_global = np.array(self.l_patterns[0].corner) + [cx, cy]
        corner_idx = int(np.argmin(np.linalg.norm(vertices - l_corner_global, axis=1)))

        v_adj1 = vertices[(corner_idx + 1) % 4]
        v_diag = vertices[(corner_idx + 2) % 4]
        v_adj2 = vertices[(corner_idx + 3) % 4]

        vec1 = v_adj1 - vertices[corner_idx]
        vec2 = v_adj2 - vertices[corner_idx]

        cross_prod = vec1[0] * vec2[1] - vec1[1] * vec2[0]

        if cross_prod < 0:
            horiz_v = v_adj1
            vert_v = v_adj2
        else:
            horiz_v = v_adj2
            vert_v = v_adj1

        ordered_src = np.array([
            vertices[corner_idx],
            horiz_v,
            v_diag,
            vert_v
        ], dtype=np.float32)

        dst_pts = np.array([
            [0, output_size - 1],
            [output_size - 1, output_size - 1],
            [output_size - 1, 0],
            [0, 0]
        ], dtype=np.float32)

        M = cv.getPerspectiveTransform(ordered_src, dst_pts)
        return cv.warpPerspective(full_frame, M, (output_size, output_size))

class DataMatrixPipeline:

    def __init__(self,
                 canny_t1: int = 50,
                 canny_t2: int = 150,
                 min_area: float = 400.0,
                 min_perimeter: float = 80.0,
                 padding: int = 25):

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
        self.dashed_detector = DashedBorderDetector()

    @staticmethod
    def parent_visited(visited: list, current: tuple):
        for v in visited:
            if current[0] >= v[0] and current[1] >= v[1] and current[2] <= v[2] and current[3] <= v[3]:
                return True
        return False

    def process_frame(self, frame: np.ndarray) -> List[DetectionResult]:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        candidates = self.extractor.get_candidates(frame)
        results = []

        visited_candidates = []

        candidates.sort(reverse=True, key=lambda c: c[2] * c[3])

        for (x, y, w, h) in candidates:
            region = np.ascontiguousarray(gray[y:y + h, x:x + w])

            if self.parent_visited(visited_candidates, (x, y, x + w, y + h)):
                continue
            visited_candidates.append((x, y, x + w, y + h))

            segments = self.l_finder.detect_lines(region)
            l_patterns = self.l_finder.find_l_patterns(region, segments)

            for idx, l_pattern in enumerate(l_patterns):
                validation = self.validator.validate(region, l_pattern)

                if not validation.is_valid:
                    continue

                dashed_result, edges = self.dashed_detector.detect(region, l_pattern)

                if dashed_result is None:
                    continue

                precise_location = self.border_fitter.fit(region, edges, l_pattern, rough_location=dashed_result)

                if precise_location is None:
                    bx, by, bw, bh = dashed_result.bounding_box
                    vertices = [
                        (float(bx), float(by)),
                        (float(bx + bw), float(by)),
                        (float(bx + bw), float(by + bh)),
                        (float(bx), float(by + bh))
                    ]
                    center = (float(bx + bw / 2), float(by + bh / 2))

                    precise_location = PreciseLocation(
                        vertices=vertices,
                        center=center,
                        angle=0.0,
                        size=(float(bw), float(bh))
                    )

                global_vertices = [(vx + x, vy + y) for vx, vy in precise_location.vertices]
                precise_location.vertices = global_vertices
                precise_location.center = (precise_location.center[0] + x, precise_location.center[1] + y)

                results.append(DetectionResult(
                    candidate_box=(x, y, w, h),
                    precise_location=precise_location,
                    l_patterns=[l_pattern],
                    is_valid=True,
                    score=validation.score
                ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    @staticmethod
    def draw_results(frame: np.ndarray, results: List[DetectionResult],
                     debug_view: bool = False) -> np.ndarray:
        output = frame.copy()

        for result in results:
            if result.precise_location and result.is_valid:
                vertices = result.precise_location.get_ordered_vertices()
                pts = np.array(vertices, dtype=np.int32)
                cv.polylines(output, [pts], True, (0, 255, 0), 2)

                if debug_view:
                    cx, cy = int(result.precise_location.center[0]), int(result.precise_location.center[1])
                    cv.circle(output, (cx, cy), 10, (255, 0, 0), -1)

        return output