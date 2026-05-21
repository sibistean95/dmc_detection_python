import cv2 as cv
import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass

from .l_finder_detector import LPattern

@dataclass
class DataMatrixLocation:
    l_pattern: LPattern
    upper_border: Tuple[int, int, int, int]
    right_border: Tuple[int, int, int, int]
    bounding_box: Tuple[int, int, int, int]
    quads: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]
    upper_outer_coords: List[Tuple[int, int]]
    right_outer_coords: List[Tuple[int, int]]

class DashedBorderDetector:

    def __init__(self, tau: int = 5, edge_threshold: int = 50, min_transitions: int = 9):
        self.tau = tau
        self.edge_threshold = edge_threshold
        self.min_transitions = min_transitions

    def get_detection_regions(self, l_pattern: LPattern,
                              img_shape: Tuple[int, int]) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int],
    Tuple[np.ndarray, np.ndarray], np.ndarray, Tuple[float, float], np.ndarray, Tuple[float, float], np.ndarray]:
        corner = np.array(l_pattern.corner, dtype=float)
        v_a = np.array(l_pattern.vertex1, dtype=float)
        v_b = np.array(l_pattern.vertex2, dtype=float)

        arm_a = v_a - corner
        arm_b = v_b - corner

        if abs(arm_a[0]) > abs(arm_a[1]):
            horiz_vertex, horiz_arm = v_a, arm_a
            vert_vertex, vert_arm = v_b, arm_b
        else:
            horiz_vertex, horiz_arm = v_b, arm_b
            vert_vertex, vert_arm = v_a, arm_a

        v_diag = horiz_vertex + vert_vertex - corner

        horiz_len = float(np.linalg.norm(horiz_arm))
        vert_len = float(np.linalg.norm(vert_arm))
        tau = self.tau
        img_h, img_w = img_shape
        depth_frac = 0.3

        def strip_aabb(p_start: np.ndarray, p_end: np.ndarray,
                       inward_unit: np.ndarray, inward_depth: float) -> Tuple[int, int, int, int]:
            offset = inward_unit * inward_depth
            pts = np.stack([p_start, p_end, p_start + offset, p_end + offset])
            x_min = int(np.floor(pts[:, 0].min())) - tau
            y_min = int(np.floor(pts[:, 1].min())) - tau
            x_max = int(np.ceil(pts[:, 0].max())) + tau
            y_max = int(np.ceil(pts[:, 1].max())) + tau
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(img_w, x_max)
            y_max = min(img_h, y_max)
            return x_min, y_min, max(1, x_max - x_min), max(1, y_max - y_min)

        inward_upper = -vert_arm / max(vert_len, 1e-6)
        upper_region = strip_aabb(vert_vertex, v_diag, inward_upper, depth_frac * vert_len)

        inward_right = -horiz_arm / max(horiz_len, 1e-6)
        right_region = strip_aabb(horiz_vertex, v_diag, inward_right, depth_frac * horiz_len)

        return (upper_region, right_region, (inward_right, inward_upper), vert_vertex,
                (horiz_len, depth_frac * vert_len), horiz_vertex, (horiz_len * depth_frac, vert_len), v_diag)

    @staticmethod
    def scan_edge_along_arms_direction(
            edge_img: np.ndarray,
            sample_img: np.ndarray,
            u_hat: np.ndarray,
            v_hat: np.ndarray,
            origin: np.ndarray,
            u_len: int,
            v_len: int
    ):
        best_v = 0
        best_transitions = 0
        for v in range(v_len):
            signal = np.array([
                int(edge_img[
                        max(0, min(int(round((origin + u * u_hat + v * v_hat)[1])), edge_img.shape[0] - 1)),
                        max(0, min(int(round((origin + u * u_hat + v * v_hat)[0])), edge_img.shape[1] - 1))
                    ])
                for u in range(u_len)
            ])
            binary = (signal > 0).astype(np.int8)
            transitions = int(np.sum(np.abs(np.diff(binary))))
            if transitions > best_transitions:
                best_transitions = transitions
                best_v = v

        most_edges_row = best_v

        edge_row = np.array([])
        sampled_coords = []
        for u in range(u_len):
            coords = origin + u * u_hat + most_edges_row * v_hat

            row = int(round(coords[1]))
            col = int(round(coords[0]))

            row = max(0, min(row, edge_img.shape[0] - 1))
            col = max(0, min(col, edge_img.shape[1] - 1))
            edge_row = np.append(edge_row, sample_img[row, col])
            sampled_coords.append((col, row))

        return most_edges_row, edge_row, sampled_coords

    @staticmethod
    def find_outer_border_line(
            edge_img: np.ndarray,
            u_hat: np.ndarray,
            v_hat: np.ndarray,
            origin: np.ndarray,
            u_len: int,
            inward_depth: int,
            margin: int = 10,
            threshold_frac: float = 0.4
    ) -> List[Tuple[int, int]]:
        outer_origin = origin - margin * v_hat
        total_v = margin + inward_depth

        transitions_per_v = []
        for v in range(total_v):
            signal = np.array([
                int(edge_img[
                    max(0, min(int(round((outer_origin + u * u_hat + v * v_hat)[1])), edge_img.shape[0] - 1)),
                    max(0, min(int(round((outer_origin + u * u_hat + v * v_hat)[0])), edge_img.shape[1] - 1))
                ])
                for u in range(u_len)
            ])
            binary = (signal > 0).astype(np.int8)
            transitions = int(np.sum(np.abs(np.diff(binary))))
            transitions_per_v.append(transitions)

        max_trans = max(transitions_per_v) if transitions_per_v else 0
        threshold = max_trans * threshold_frac

        outer_v = 0
        for v, t in enumerate(transitions_per_v):
            if t >= threshold:
                outer_v = v
                break

        sampled_coords = []
        for u in range(u_len):
            coords = outer_origin + u * u_hat + outer_v * v_hat
            row = max(0, min(int(round(coords[1])), edge_img.shape[0] - 1))
            col = max(0, min(int(round(coords[0])), edge_img.shape[1] - 1))
            sampled_coords.append((col, row))

        return sampled_coords

    @staticmethod
    def count_transitions(border: np.ndarray) -> int:
        if len(border) < 2:
            return 0
        b_min, b_max = float(np.min(border)), float(np.max(border))
        if b_max - b_min < 20.0:
            return 0
        threshold = (b_min + b_max) / 2.0
        binary_border = (border > threshold).astype(np.int8)
        transitions = np.sum(np.abs(np.diff(binary_border)))
        return int(transitions)

    @staticmethod
    def _auto_canny(image: np.ndarray, sigma: float = 0.33) -> np.ndarray:
        v = np.median(image)
        lower = max(0, int((1.0 - sigma) * v))
        upper = min(255, int((1.0 + sigma) * v))
        if lower < 50: lower = 50
        if upper < 100: upper = 100
        return cv.Canny(image, lower, upper)

    def detect(self, gray_img: np.ndarray, l_pattern: LPattern) -> Tuple[Optional[DataMatrixLocation], np.ndarray]:
        edges = self._auto_canny(gray_img)

        img_h, img_w = gray_img.shape[:2]

        (upper_region, right_region, (inward_right_unit, inward_upper_unit), vert_vertex, (border_len_upper, upper_inward),
         horiz_vertex, (right_inward, border_len_right), v_diag) = self.get_detection_regions(l_pattern, (img_h, img_w))

        upper_dashed_row, extracted_arr_upper, upper_coords = self.scan_edge_along_arms_direction(edges, gray_img, inward_right_unit * -1,
                                                                                    inward_upper_unit, vert_vertex,
                                                                                    int(border_len_upper), int(upper_inward))
        right_dashed_col, extracted_arr_right, right_coords = self.scan_edge_along_arms_direction(edges, gray_img, inward_upper_unit * -1,
                                                                                    inward_right_unit, horiz_vertex,
                                                                                    int(border_len_right), int(right_inward))

        t_upper = self.count_transitions(extracted_arr_upper)
        t_right = self.count_transitions(extracted_arr_right)

        if t_upper < self.min_transitions or t_right < self.min_transitions:
            return None, edges

        upper_outer_coords = self.find_outer_border_line(
            edges, inward_right_unit * -1, inward_upper_unit, vert_vertex,
            int(border_len_upper), int(upper_inward))
        right_outer_coords = self.find_outer_border_line(
            edges, inward_upper_unit * -1, inward_right_unit, horiz_vertex,
            int(border_len_right), int(right_inward))

        corners = np.array([l_pattern.corner, vert_vertex, horiz_vertex, v_diag])
        min_x = int(np.floor(corners[:, 0].min()))
        min_y = int(np.floor(corners[:, 1].min()))
        max_x = int(np.ceil(corners[:, 0].max()))
        max_y = int(np.ceil(corners[:, 1].max()))
        bounding_box = (min_x, min_y, max_x - min_x, max_y - min_y)

        quad_corners = np.array([l_pattern.corner, horiz_vertex, v_diag, vert_vertex])
        quads = (
            (int(quad_corners[0][0]), int(quad_corners[0][1])),
            (int(quad_corners[1][0]), int(quad_corners[1][1])),
            (int(quad_corners[2][0]), int(quad_corners[2][1])),
            (int(quad_corners[3][0]), int(quad_corners[3][1]))
        )

        return DataMatrixLocation(
            l_pattern=l_pattern,
            upper_border=upper_region,
            right_border=right_region,
            bounding_box=bounding_box,
            quads=quads,
            upper_outer_coords=upper_outer_coords,
            right_outer_coords=right_outer_coords
        ), edges