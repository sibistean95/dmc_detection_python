import cv2 as cv
import numpy as np
from typing import List, Tuple

class CandidateExtraction:
    def __init__(self,
                 canny_t1: int = 100,
                 canny_t2: int = 200,
                 min_area: float = 300.0,
                 min_perimeter: float = 80.0,
                 padding: int = 10):

        self.canny_t1 = canny_t1
        self.canny_t2 = canny_t2
        self.min_area = min_area
        self.min_perimeter = min_perimeter
        self.padding = padding

    def edge_detection(self, image_gray: np.ndarray) -> np.ndarray:
        return cv.Canny(image_gray, self.canny_t1, self.canny_t2)

    @staticmethod
    def morphological_processing(edges: np.ndarray) -> np.ndarray:
        kernel_row = cv.getStructuringElement(cv.MORPH_RECT, (4, 1))
        kernel_column = cv.getStructuringElement(cv.MORPH_RECT, (1, 4))

        dilate_row = cv.dilate(edges, kernel_row, iterations=2)
        dilate_column = cv.dilate(edges, kernel_column, iterations=2)

        candidate_region = cv.bitwise_and(dilate_row, dilate_column)

        kernel_open = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        processed = cv.morphologyEx(candidate_region, cv.MORPH_OPEN, kernel_open)

        return processed

    def contour_analysis(self, binary_map: np.ndarray, shape: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
        contours, _ = cv.findContours(binary_map, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        candidate_boxes = []
        img_h, img_w = shape

        for contour in contours:
            perimeter = cv.arcLength(contour, True)
            area = cv.contourArea(contour)

            if perimeter > self.min_perimeter and area > self.min_area:
                x, y, w, h = cv.boundingRect(contour)

                x_new = max(0, x - self.padding)
                y_new = max(0, y - self.padding)
                w_new = min(img_w - x_new, w + 2 * self.padding)
                h_new = min(img_h - y_new, h + 2 * self.padding)

                candidate_boxes.append((x_new, y_new, w_new, h_new))

        return candidate_boxes

    def get_candidates(self, frame: np.ndarray) -> list:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        edges = self.edge_detection(gray)
        preprocess = self.morphological_processing(edges)
        candidates = self.contour_analysis(preprocess, gray.shape)

        return candidates