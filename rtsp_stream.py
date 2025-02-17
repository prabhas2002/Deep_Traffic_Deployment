import cv2
import torch
from ultralytics import YOLO
import argparse

def main(args):
    model = YOLO(args.trained_weights)
    model.to(device=args.device)
    rtsp_url = args.rtsp_url

    cap = cv2.VideoCapture(rtsp_url)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = result.names[int(box.cls[0])]
                confidence = box.conf[0].item()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}: {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("YOLO RTSP", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLO RTSP Stream')
    parser.add_argument('--trained_weights', type=str, required=True, help='Path to trained weights file')
    parser.add_argument('--rtsp_url', type=str, required=True, help='RTSP stream URL')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run YOLO model on (e.g., "cpu", "cuda:0")')

    args = parser.parse_args()
    main(args)