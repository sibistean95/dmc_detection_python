import cv2 as cv
import numpy as np
from typing import List, Tuple


class DataMatrixLocator:
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.lsd = cv.createLineSegmentDetector(0)

    def preprocess_lsd(self, image: np.ndarray) -> np.ndarray:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        lines, width, precision, nfa = self.lsd.detect(gray)
        processed = np.zeros_like(gray)

        if lines is not None:
            lines = lines.reshape(-1, 4)
            for x1, y1, x2, y2 in lines:
                length = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                if length > 20:
                    cv.line(processed, (int(x1), int(y1)), (int(x2), int(y2)), 255, 2)

        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        processed = cv.dilate(processed, kernel, iterations=1)

        return processed

    @staticmethod
    def preprocess_image(image: np.ndarray) -> np.ndarray:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(gray, (7, 7), 0)

        edges = cv.Canny(blurred, 30, 100)

        kernel_row = cv.getStructuringElement(cv.MORPH_RECT, (7, 1))
        kernel_column = cv.getStructuringElement(cv.MORPH_RECT, (1, 7))

        closed_row = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel_row)
        closed_column = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel_column)

        processed = cv.bitwise_or(closed_row, closed_column)

        kernel_open = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        processed = cv.morphologyEx(processed, cv.MORPH_OPEN, kernel_open)

        return processed

    @staticmethod
    def get_binary_image(image: np.ndarray) -> np.ndarray:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        binary = cv.adaptiveThreshold(
            gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv.THRESH_BINARY, 21, 4
        )
        return binary

    @staticmethod
    def filter_candidates(contours: List[np.ndarray]) -> List[np.ndarray]:
        potential_codes = []
        for cnt in contours:
            perimeter = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.03 * perimeter, True)

            if len(approx) == 4:
                (x, y, w, h) = cv.boundingRect(approx)
                aspect_ratio = w / float(h)

                if 0.8 <= aspect_ratio <= 1.2:
                    potential_codes.append(approx)
        return potential_codes

    @staticmethod
    def validate_l_pattern(binary_image: np.ndarray, contour: np.ndarray) -> bool:
        x, y, w, h = cv.boundingRect(contour)
        if w < 20 or h < 20:
            return False

        mask = np.zeros_like(binary_image)
        cv.drawContours(mask, [contour], -1, 255, -1)

        mean_val = cv.mean(binary_image, mask=mask)[0]

        if mean_val > 230 or mean_val < 20:
            return False

        pts = [pt[0] for pt in contour]
        edge_types = []

        for i in range(4):
            p1 = pts[i]
            p2 = pts[(i + 1) % 4]

            vec = p2 - p1
            length = np.linalg.norm(vec)
            if length == 0:
                edge_types.append(0)
                continue

            steps = int(min(max(length, 10), 100))
            pixel_values = []

            for j in range(steps):
                alpha = j / steps
                lx = int(p1[0] + vec[0] * alpha)
                ly = int(p1[1] + vec[1] * alpha)

                if 0 <= lx < binary_image.shape[1] and 0 <= ly < binary_image.shape[0]:
                    pixel_values.append(binary_image[ly, lx])

            if not pixel_values:
                edge_types.append(0)
                continue

            transitions = 0
            black_count = 0
            for k in range(len(pixel_values)):
                if pixel_values[k] == 0:
                    black_count += 1
                if k > 0 and pixel_values[k] != pixel_values[k - 1]:
                    transitions += 1

            black_ratio = black_count / len(pixel_values)

            if black_ratio > 0.60 and transitions <= 3:
                edge_types.append(1)
            elif transitions >= 4:
                edge_types.append(-1)
            else:
                edge_types.append(0)

        solid_count = edge_types.count(1)
        dashed_count = edge_types.count(-1)

        if solid_count != 2:
            return False

        if dashed_count < 1:
            return False

        solids_indices = [i for i, x in enumerate(edge_types) if x == 1]
        idx1, idx2 = solids_indices[0], solids_indices[1]

        is_appropriate = (abs(idx1 - idx2) == 1) or (abs(idx1 - idx2) == 3)

        return is_appropriate

    def detect_in_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
        binary_img = self.get_binary_image(frame)

        methods = [
            ("LSD", self.preprocess_lsd(frame)),
            ("Row & Column Kernel", self.preprocess_image(frame))
        ]

        found = False

        for name, processed_img in methods:
            if found: break

            contours, _ = cv.findContours(processed_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            candidates = self.filter_candidates(list(contours))

            for cand in candidates:
                if self.validate_l_pattern(binary_img, cand):
                    cv.drawContours(frame, [cand], -1, (0, 255, 0), 3)

                    for pt in cand:
                        cv.circle(frame, tuple(pt[0]), 5, (0, 0, 255), -1)

                    x, y, w, h = cv.boundingRect(cand)
                    cv.putText(frame, f"DM Code ({name})", (x, y - 10),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    found = True
                    break

            if self.debug_mode:
                cv.imshow(f"Debug: {name}", processed_img)

        return frame, found


def run_webcam():
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Cannot access webcam.")
        return

    detector = DataMatrixLocator(debug_mode=True)
    print("Press 'q' to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result_frame, is_detected = detector.detect_in_frame(frame)
        cv.imshow("Data Matrix Detector", result_frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    run_webcam()