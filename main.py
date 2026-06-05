import cv2 as cv
from dm_detector.pipeline import DataMatrixPipeline
from dm_decoder.grid_estimation.estimator import GridEstimator
from dm_decoder.sampling.sampler import ModuleSampler
from dm_decoder.mapping.utah_mapping import UtahMapper
from dm_decoder.decoding.decoder import DataMatrixDecoder
from utils import contrast_power_law


def main():
    image_path = "./test_images/dmc_test_3.jpeg"
    frame = cv.imread(image_path)

    if frame is None:
        print(f"Error: could not load image from {image_path}")
        return

    detector = DataMatrixPipeline()
    results = detector.process_frame(frame)

    output_frame = detector.draw_results(frame, results, debug_view=True)
    cv.imshow("1. Detection", output_frame)

    if results and results[0].is_valid:
        warped_bgr = results[0].get_rectified_image(frame, output_size=400)

        if warped_bgr is not None:
            warp_gray = cv.cvtColor(warped_bgr, cv.COLOR_BGR2GRAY)
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cv.imshow("2. Rectified image (warped preview)", clahe.apply(warp_gray))
            filtered = cv.medianBlur(warp_gray, 5)
            filtered = contrast_power_law(filtered, gamma=1.1)
            cv.imshow("3. Rectified and power law contrast", filtered)

            print("\n--- Geometric Grid Estimation ---")

            estimator = GridEstimator(margin=1)
            h, w = filtered.shape
            grid = estimator.estimate_grid(filtered)

            if grid is not None:
                col_centres, row_centres = grid
                nx, ny = len(col_centres), len(row_centres)
                print(f"[timing] estimated matrix size: {nx} cols x {ny} rows")

                bits = estimator.sample_matrix(warp_gray, col_centres, row_centres)
                print(bits)
                data_matrix = bits[1:-1, 1:-1]  # strip the 1-module border

                grid_vis = cv.cvtColor(warp_gray, cv.COLOR_GRAY2BGR)
                estimator.draw_module_grid(grid_vis, col_centres, row_centres)
                grid_vis = estimator.draw_module_numbers(grid_vis, col_centres, row_centres)
                cv.imshow("3. Final Grid", grid_vis)

            else:
                pitch, score = estimator.estimate_pitch(filtered)
                if pitch is not None:
                    print(f"[autocorr] pitch={pitch:.2f} px score={score:.2f}")
                    w_eff = w - 2 * estimator.margin
                    h_eff = h - 2 * estimator.margin
                    nx = int(round(w_eff / pitch))
                    ny = int(round(h_eff / pitch))
                    print(f"[autocorr] estimated matrix size: {nx} cols x {ny} rows")

                    sampler = ModuleSampler()
                    data_matrix = sampler.get_matrix_data(filtered, w / nx, h / ny, ny, nx)

                    grid_vis = cv.cvtColor(filtered, cv.COLOR_GRAY2BGR)
                    sampler.draw_grid(grid_vis, w / nx, h / ny)
                    cv.imshow("3. Final Grid", grid_vis)

                else:
                    print("could not estimate grid via new methods, trying fallback...")
                    nx_snapped, ny_snapped, final_pitch_x, final_pitch_y = estimator.estimate_grid(filtered)
                    print(f"[fallback] estimated matrix size: {nx_snapped} cols x {ny_snapped} rows")

                    sampler = ModuleSampler()
                    data_matrix = sampler.get_matrix_data(filtered, final_pitch_x, final_pitch_y, ny_snapped,
                                                          nx_snapped)

                    grid_vis = cv.cvtColor(filtered, cv.COLOR_GRAY2BGR)
                    sampler.draw_grid(grid_vis, final_pitch_x, final_pitch_y)
                    cv.imshow("3. Final Grid", grid_vis)

            if data_matrix is not None:
                print(f"\nData matrix size (without borders): {data_matrix.shape[1]}x{data_matrix.shape[0]}")

                print("\n--- UTAH MAPPING TEST ---")
                mapper = UtahMapper()
                codewords = mapper.map_to_codewords(data_matrix)

                print(f"Extracted {len(codewords)} total codewords (bytes)")
                print(f"Raw data bytes: {codewords}")

                print("\n--- REED SOLOMON ERROR CORRECTION & DECODING ---")
                decoder = DataMatrixDecoder()
                final_text = decoder.decode(codewords)

                print(f"\n[ FINAL TEXT EXTRACTION FROM DATA MATRIX CODE ] -> {final_text}")
                cv.waitKey(0)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()