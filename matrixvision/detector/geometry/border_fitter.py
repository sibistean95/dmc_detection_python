import cv2 as cv
import numpy as np
from typing import List, Tuple, Optional

from matrixvision.data import PreciseLocation, LPattern, DataMatrixLocation
from matrixvision.debug import DebugSink, NullSink
from .polarity import PolarityChecker
from matrixvision.viz import show_fitted_lines, show_sampled_points


class BorderFitter:
    def __init__(self, debug: DebugSink = NullSink()):
        self.debug = debug
        self.polarity_checker = PolarityChecker(debug=self.debug)

    def fit(self, gray_img: np.ndarray, edge_img: np.ndarray, l_pattern: LPattern,
            rough_location: Optional[DataMatrixLocation] = None, gaussian_size: int = 3, dilate_size: int = 5,
            blob_min_area: int = 0, win_in: int = 20, win_out: int = 20, max_pts_outside: int = 3,
            inlier_threshold: float = 1.2) -> Optional[Tuple[PreciseLocation, bool]]:
        if rough_location is None:
            return self._fit_from_l_pattern(l_pattern), False

        return self._fit_clean_linefit(gray_img, rough_location, dilate=dilate_size, gaussian_size=gaussian_size,
                                           blob_min_area=blob_min_area, win_in=win_in, win_out=win_out,
                                           max_outside=max_pts_outside, inlier_threshold=inlier_threshold)

    @staticmethod
    def _build_precise_location(vertices_list: List[Tuple[float, float]]) -> PreciseLocation:
        pts = np.array(vertices_list)
        ctr = np.mean(pts, axis=0)
        side = pts[1] - pts[0]
        angle = float(np.degrees(np.arctan2(side[1], side[0])))
        w = float(np.linalg.norm(pts[1] - pts[0]))
        h = float(np.linalg.norm(pts[3] - pts[0]))

        return PreciseLocation(
            vertices=vertices_list,
            center=(float(ctr[0]), float(ctr[1])),
            angle=angle,
            size=(w, h)
        )

    def _fit_clean_linefit(self, gray_img: np.ndarray,
                           rough_location: DataMatrixLocation,
                           dilate: int = 5,
                           ransac_iterations: int = 1500,
                           blob_min_area: int = 0,
                           gaussian_size: int = 5,
                           win_in: int = 20,
                           win_out: int = 20,
                           max_outside: int = 10,
                           inlier_threshold: float = 1.2) -> Optional[Tuple[PreciseLocation, bool]]:
        """Quad-guided noise removal, then fit one *outer-tangent* line per side
        and intersect.

        The rough quad is used only as a *guide*: everything outside it is
        surrounding texture/noise and is masked away, leaving the DMC isolated.
        Each side is then fit to the cleaned boundary with an outer-tangent
        RANSAC (see ``_ransac_line_outer``) that prefers the outermost supported
        line rather than the one with the most inliers — so the dense first
        interior column/row can't win over the sparse, dashed outer border and
        clip it. No connected-component speck filter is used, because the timing
        dashes are themselves small components; the outer-tangent fit tolerates
        the few surviving specks instead.
        """
        quad = np.array([np.array(q, dtype=float) for q in rough_location.quads])
        center = quad.mean(axis=0)

        # 1. Binarize dark modules -> white. The threshold is Otsu computed over
        #    the DMC *interior* (the rough quad) only, not the whole ROI: on a
        #    noisy gray surface a global Otsu is biased by the large background
        #    and floods the image white, which then makes the dilated mask (not
        #    the DMC) the apparent border and pushes the fitted corners outward.
        #    Estimating the threshold from the DMC's own bimodal pixels adapts
        #    per image with no hardcoded value.
        blurred = cv.GaussianBlur(gray_img, (gaussian_size, gaussian_size), 0)
        # blurred = cv.medianBlur(gray_img, 5)
        # if inverted_mask:
        #     quad_mask = np.ones(blurred.shape[:2], dtype=np.uint8) * 255
        #     cv.fillConvexPoly(quad_mask, quad.astype(np.int32), 0)
        #     interior = blurred[quad_mask == 0]
        # else:
        quad_mask = np.zeros(blurred.shape[:2], dtype=np.uint8)
        cv.fillConvexPoly(quad_mask, quad.astype(np.int32), 255)
        interior = blurred[quad_mask > 0]
        thr, _ = cv.threshold(interior.reshape(-1, 1), 0, 255,
                              cv.THRESH_BINARY + cv.THRESH_OTSU)
        _, bw = cv.threshold(blurred, thr, 255, cv.THRESH_BINARY_INV)

        inverted = self.polarity_checker.has_inverted_polarity(gray_img, rough_location.l_pattern)

        if inverted:
            bw = cv.bitwise_not(bw)

        # inverted = False
        # white = np.where(bw == 255)
        # self.debug.log(f"[border-fitter] white pixels in binary image: {white}")
        # if white[0].shape[0] > 0.7 * bw.shape[0] * bw.shape[1]:
        #     inverted = True
        #     bw = cv2.bitwise_not(bw)

        kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 1))
        clean = cv.morphologyEx(bw, cv.MORPH_OPEN, kernel)

        num_labels, labels, stats, _ = cv.connectedComponentsWithStats(clean, connectivity=8)

        min_area = blob_min_area
        filtered = np.zeros_like(clean)

        for i in range(1, num_labels):
            if stats[i, cv.CC_STAT_AREA] >= min_area:
                filtered[labels == i] = 255

        need_rs = False

        if quad_mask.shape[0] < 700 or quad_mask.shape[1] < 700:
            need_rs = True

        self.debug.show("quad mask", cv.resize(quad_mask, dsize=None, fx=2.0, fy=2.0,
                                               interpolation=cv.INTER_AREA) if need_rs else quad_mask)
        self.debug.show("bw", cv.resize(bw, dsize=None, fx=2.0, fy=2.0, interpolation=cv.INTER_AREA) if need_rs else bw)
        self.debug.show("clean",
                        cv.resize(clean, dsize=None, fx=2.0, fy=2.0, interpolation=cv.INTER_AREA) if need_rs else clean)
        self.debug.show("filtered", cv.resize(filtered, dsize=None, fx=2.0, fy=2.0,
                                              interpolation=cv.INTER_AREA) if need_rs else filtered)

        # 2. mask away everything outside the quad
        mask = cv.dilate(quad_mask, cv.getStructuringElement(cv.MORPH_ELLIPSE, (dilate, dilate)))
        cleaned = cv.bitwise_and(filtered, mask)
        cleaned = cv.morphologyEx(cleaned, cv.MORPH_OPEN, np.ones((2, 2), np.uint8))

        self.debug.show("cleaned", cv.resize(cleaned, dsize=None, fx=2.0, fy=2.0,
                                             interpolation=cv.INTER_AREA) if need_rs else cleaned)
        self.debug.pause()

        # 3. fit an outer-tangent line per side
        lines = []
        all_pts = []
        for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
            p_start, p_end = quad[a], quad[b]
            direction = p_end - p_start
            length = float(np.linalg.norm(direction))
            if length < 2.0:
                return None
            u_hat = direction / length
            perp = np.array([-u_hat[1], u_hat[0]])
            if np.dot(perp, (p_start + p_end) / 2 - center) < 0:
                perp = -perp  # outward normal

            pts = self._scan_boundary_inward(cleaned, p_start, p_end, perp, win_in=win_in, win_out=win_out)
            if len(pts) < 2:
                return None
            show_sampled_points(cleaned, pts, color=(255, 0, 0), need_resize=need_rs, debug=self.debug)
            line = self._ransac_line_outer(
                pts, perp, direction, max_iterations=ransac_iterations, max_outside=max_outside,
                inlier_threshold=inlier_threshold
            )

            self.debug.log(f"[border-fitter] Line: {line}")

            if line is None:
                return None
            lines.append(line)
            all_pts.append(pts)

        # 4. adjacent-side intersections -> corners (quad order: corner,horiz,vdiag,vert)
        corner = self._intersect_lines(lines[3], lines[0])
        horiz = self._intersect_lines(lines[0], lines[1])
        v_diag = self._intersect_lines(lines[1], lines[2])
        vert = self._intersect_lines(lines[2], lines[3])

        self.debug.log(f"[border-fitter] Corners: {corner}, {horiz}, {v_diag}, {vert}")

        if any(v is None for v in (corner, horiz, v_diag, vert)):
            return None

        show_fitted_lines(cleaned, lines, all_pts, [corner, horiz, v_diag, vert], need_resize=need_rs,
                          debug=self.debug)

        return self._build_precise_location(
            [(float(p[0]), float(p[1])) for p in (corner, horiz, v_diag, vert)]
        ), inverted

    @staticmethod
    def _scan_boundary_inward(cleaned: np.ndarray, p_start: np.ndarray, p_end: np.ndarray,
                              perp: np.ndarray, win_out: float = 8.0, win_in: float = 8.0,
                              step: float = 0.5) -> np.ndarray:
        """Sample along a side and, at each sample, scan perpendicular from
        outside inward (``perp`` points outward) for the first two consecutive
        foreground pixels — the DMC's outer boundary on a noise-free image.
        Two-in-a-row rejects a lone surviving speck; gaps find nothing and are
        skipped. At dashed-border gap positions this returns the interior module
        edge; the outer-tangent fit downstream keeps those points *inside* the
        line rather than on it."""
        direction = p_end - p_start
        length = float(np.linalg.norm(direction))
        if length < 2.0:
            return np.empty((0, 2))
        u_hat = direction / length

        h, w = cleaned.shape[:2]

        def is_fg(q: np.ndarray) -> bool:
            ix, iy = int(round(q[0])), int(round(q[1]))
            return 0 <= iy < h and 0 <= ix < w and cleaned[iy, ix] > 0

        pts = []
        n = max(4, int(length))
        for i in range(n):
            base = p_start + (i + 0.5) * length / n * u_hat
            s = win_out
            while s >= -win_in:
                if is_fg(base + perp * s) and is_fg(base + perp * (s - step)):
                    pts.append(base + perp * s)
                    break
                s -= step

        return np.array(pts) if pts else np.empty((0, 2))

    @staticmethod
    def _ransac_line_outer(points: np.ndarray, perp_out: np.ndarray, expected_dir: np.ndarray,
                           max_iterations: int = 1500, inlier_threshold: float = 1.2,
                           max_outside: int = 2,
                           angle_tol_deg: float = 22.0) -> Optional[Tuple[np.ndarray, float]]:
        """Outer-tangent RANSAC.

        Among candidate lines whose direction is within ``angle_tol_deg`` of
        ``expected_dir``, keep only those with at most ``max_outside`` foreground
        points lying strictly *outward* of the line (i.e. the line is an outer
        boundary, not an interior one), and pick the best-supported such line.
        This makes the sparse dashed outer edge — whose tips define the true
        boundary — win over a denser interior line that would otherwise clip it.
        """
        n = len(points)
        if n < 2:
            return None
        perp_out = perp_out / (np.linalg.norm(perp_out) + 1e-12)
        expected_angle = float(np.arctan2(expected_dir[1], expected_dir[0]))
        tol = np.radians(angle_tol_deg)

        def angle_diff(a: float, b: float) -> float:
            d = abs(a - b) % np.pi
            return min(d, np.pi - d)

        best_inliers = None
        best_key = (-1, -1e18)  # (inlier_count, signed offset c) — more inliers, then more outward
        rng = np.random.default_rng(0)  # deterministic: same warp every run
        for _ in range(max_iterations):
            idx = rng.choice(n, 2, replace=False)
            p1, p2 = points[idx[0]], points[idx[1]]
            d = p2 - p1
            d_len = np.linalg.norm(d)
            if d_len < 1e-6:
                continue
            if angle_diff(float(np.arctan2(d[1], d[0])), expected_angle) > tol:
                continue
            normal = np.array([-d[1], d[0]]) / d_len
            if np.dot(normal, perp_out) < 0:
                normal = -normal  # orient normal outward
            c = -np.dot(normal, p1)
            signed = points @ normal + c  # > 0 means outward of the line
            inliers = np.abs(signed) < inlier_threshold
            outside = int(np.sum(signed > inlier_threshold))
            if outside > max_outside:
                continue
            key = (int(np.sum(inliers)), float(c))
            if key > best_key:
                best_key = key
                best_inliers = inliers

        if best_inliers is None or int(np.sum(best_inliers)) < 2:
            return None

        inlier_pts = points[best_inliers].astype(np.float32)
        vx, vy, x0, y0 = cv.fitLine(inlier_pts, cv.DIST_L2, 0, 0.01, 0.01).flatten()
        normal = np.array([-vy, vx])
        if np.dot(normal, perp_out) < 0:
            normal = -normal
        c = -(normal[0] * x0 + normal[1] * y0)
        return (normal, c)

    @staticmethod
    def _intersect_lines(line1: Tuple[np.ndarray, float],
                         line2: Tuple[np.ndarray, float]) -> Optional[np.ndarray]:
        n1, c1 = line1
        n2, c2 = line2

        A = np.array([n1, n2])
        b = np.array([-c1, -c2])

        det = np.linalg.det(A)
        if abs(det) < 1e-6:
            return None

        return np.linalg.solve(A, b)

    @staticmethod
    def _fit_from_l_pattern(l_pattern: LPattern) -> PreciseLocation:
        corner = np.array(l_pattern.corner)
        v1 = np.array(l_pattern.vertex1)
        v2 = np.array(l_pattern.vertex2)

        dir_h = v1 - corner
        dir_v = v2 - corner

        p1 = corner
        p2 = corner + dir_h
        p3 = corner + dir_h + dir_v
        p4 = corner + dir_v

        vertices = [
            (float(p1[0]), float(p1[1])),
            (float(p2[0]), float(p2[1])),
            (float(p3[0]), float(p3[1])),
            (float(p4[0]), float(p4[1]))
        ]

        center = np.mean([p1, p2, p3, p4], axis=0)

        return PreciseLocation(
            vertices=vertices,
            center=(float(center[0]), float(center[1])),
            angle=0,
            size=(float(np.linalg.norm(dir_h)), float(np.linalg.norm(dir_v)))
        )
