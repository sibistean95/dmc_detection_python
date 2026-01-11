import cv2 as cv
from dm_detector import DataMatrixPipeline


def main():
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Cannot access webcam")
        return

    pipeline = DataMatrixPipeline()

    print("Press 'q' to exit, 'd' to toggle debug view")
    debug_view = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = pipeline.process_frame(frame)
        output = pipeline.draw_results(frame, results, debug_view)

        cv.imshow("Data Matrix Detector", output)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            debug_view = not debug_view
            print(f"Debug view: {'ON' if debug_view else 'OFF'}")

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()