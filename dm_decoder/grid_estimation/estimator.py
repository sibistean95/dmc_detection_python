import numpy as np

class GridEstimator:
    def __init__(self, min_size: int = 10, margin: int = 10):
        self.min_size = min_size
        self.margin = margin
        self.valid_sizes = [10, 12, 14, 16, 18, 20, 22, 24, 26, 32, 36, 40,
                            44, 48, 52, 64, 72, 80, 88, 96, 104, 120, 132, 144]

    def snap_to_valid_size(self, estimated_n: int) -> int:
        return min(self.valid_sizes, key=lambda x: abs(x - estimated_n))

    @staticmethod
    def _count_transitions(line: np.ndarray, min_amplitude: int = 30) -> int:
        lo, hi = int(np.min(line)), int(np.max(line))

        if hi - lo < min_amplitude:
            return 0

        thr = (lo + hi) / 2.0
        binary = (line > thr).astype(np.int8)
        return int(np.sum(np.abs(np.diff(binary))))

    def estimate_grid(self, warp_gray: np.ndarray) -> tuple[int, int, float, float]:
        h, w = warp_gray.shape

        top_trans = [self._count_transitions(warp_gray[y, self.margin:w - self.margin]) for y in [4, 8, 12]]
        nx_est = int(np.median(top_trans)) + 2

        right_trans = [self._count_transitions(warp_gray[self.margin:h - self.margin, x]) for x in
                       [w - 5, w - 9, w - 13]]
        ny_est = int(np.median(right_trans)) + 2

        nx_est = max(self.min_size, nx_est)
        ny_est = max(self.min_size, ny_est)

        nx_snapped = self.snap_to_valid_size(nx_est)
        ny_snapped = self.snap_to_valid_size(ny_est)

        final_pitch_x = w / nx_snapped
        final_pitch_y = h / ny_snapped

        return nx_snapped, ny_snapped, final_pitch_x, final_pitch_y