import cv2 as cv
from dm_detector.pipeline import DataMatrixPipeline
from dm_decoder.grid_estimation.estimator import GridEstimator
from dm_decoder.sampling.sampler import ModuleSampler
from dm_decoder.mapping.utah_mapping import UtahMapper
from dm_decoder.decoding.decoder import DataMatrixDecoder

def main():
    image_path = "test_images/dmc_test_4.png"
    frame = cv.imread(image_path)

    if frame is None:
        print(f"Error: could not load image from {image_path}")
        return

    detector = DataMatrixPipeline()
    results = detector.process_frame(frame)

    output_frame = detector.draw_results(frame, results)
    cv.imshow("1. Detection", output_frame)

    if results and results[0].is_valid:
        warped_bgr = results[0].get_rectified_image(frame)

        if warped_bgr is not None:
            warp_gray = cv.cvtColor(warped_bgr, cv.COLOR_BGR2GRAY)
            warp_gray = cv.normalize(warp_gray, None, 0, 255, cv.NORM_MINMAX)
            warp_gray = cv.GaussianBlur(warp_gray, (3, 3), 0)

            cv.imshow("2. Rectified image (warped)", warp_gray)

            print("\n--- Geometric Grid Estimation ---")

            estimator = GridEstimator()
            nx_snapped, ny_snapped, final_pitch_x, final_pitch_y = estimator.estimate_grid(warp_gray)

            print(f"Snapped grid size: {nx_snapped} cols x {ny_snapped} rows")
            print(f"Module pixel size: {final_pitch_x:.2f} px")

            sampler = ModuleSampler()

            roi_color = cv.cvtColor(warp_gray, cv.COLOR_GRAY2BGR)
            sampler.draw_grid(roi_color, horizontal_pitch=final_pitch_x, vertical_pitch=final_pitch_y)
            cv.imshow("3. Grid visualization", roi_color)

            data_matrix = sampler.get_matrix_data(
                image=warp_gray,
                horizontal_pitch=final_pitch_x,
                vertical_pitch=final_pitch_y,
                rows=ny_snapped,
                cols=nx_snapped
            )

            print(f"\nData matrix size (without borders): {data_matrix.shape[1]}x{data_matrix.shape[0]}")
            print("Binary data region preview (0=white, 1=black):\n")

            for row in data_matrix:
                row_str = "".join(["1 " if val == 1 else "0 " for val in row])
                print(row_str)

            print("\n--- UTAH MAPPING TEST ---")
            mapper = UtahMapper()
            codewords = mapper.map_to_codewords(data_matrix)

            print(f"Extracted {len(codewords)} total codewords (bytes)")
            print(f"Raw data bytes: {codewords}")

            print("\n--- REED SOLOMON ERROR CORRECTION TEST ---")
            decoder = DataMatrixDecoder()
            final_text = decoder.decode(codewords)

            print(f"\n[ Final text extraction from data matrix code ] -> {final_text}")

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()