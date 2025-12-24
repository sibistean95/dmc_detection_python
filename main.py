import cv2 as cv
from dm_detector.extraction import CandidateExtraction

def main():
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Cannot access webcam")
        return

    extractor = CandidateExtraction(
        canny_t1 = 100,
        canny_t2 = 200,
        min_area = 300,
        min_perimeter = 80,
        padding = 10
    )

    print("Press 'q' to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        candidates = extractor.get_candidates(frame)

        for (x, y, w, h) in candidates:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)

            cv.putText(frame, "Candidate", (x, y - 5),
                       cv.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 255), 1)

        cv.imshow("Data Matrix Detector", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()