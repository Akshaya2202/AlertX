from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

def detect_video(video_path):

    cap = cv2.VideoCapture(
        video_path
    )

    detections = 0
    frames = 0

    confidence_sum = 0
    confidence_count = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        frames += 1

        results = model(
            frame
        )

        for r in results:

            detections += len(
                r.boxes
            )

            for box in r.boxes:

                confidence_sum += float(
                    box.conf
                )

                confidence_count += 1

    cap.release()

    average = detections / max(
        frames,
        1
    )

    if average < 2:

        alert = "Low Suspicious"

    elif average < 5:

        alert = "Medium Suspicious"

    else:

        alert = "High Suspicious"

    if confidence_count > 0:

        avg_confidence = round(

            (confidence_sum / confidence_count) * 100,

            2

        )

    else:

        avg_confidence = 0

    return alert, avg_confidence