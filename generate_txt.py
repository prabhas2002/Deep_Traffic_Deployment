from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import argparse
from datetime import datetime
import os


def tracking(args):
    model = YOLO(args.trained_weights)
    model.to(device=args.device)
    rtsp_url = args.rtsp_url
    cap = cv2.VideoCapture(rtsp_url)
    rtsp_ip = rtsp_url.split('@')[-1].split('/')[0].replace(':', '_')
    track_history = defaultdict(lambda: [])

    current_date = datetime.now().strftime("%Y-%m-%d")
    output_dir = f"./Results/{rtsp_ip}/{current_date}"
    os.makedirs(output_dir, exist_ok=True)
    if args.detailed:
        output_txt_file = f"{output_dir}/{current_date}_detailed.txt"
    else:
        output_txt_file = f"{output_dir}/{current_date}.txt"

    with open(output_txt_file, 'a') as file:
        frame_number = 1
        while cap.isOpened():
            success, frame = cap.read()

            if success:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                results = model.track(frame, tracker=args.type_tracker, persist=True)
                boxes = results[0].boxes.xywh.cpu()
                confidences = results[0].boxes.conf.cpu().tolist() if results[0].boxes.conf is not None else []
                track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []
                class_ids = results[0].boxes.cls.int().cpu().tolist() if results[0].boxes.cls is not None else []
                if not track_ids:
                    frame_number += 1
                    continue

                for box, track_id, class_id, confidence in zip(boxes, track_ids, class_ids, confidences):
                    x, y, w, h = box
                    x_center = x
                    y_center = y
                    box_width = w
                    box_height = h
                    track = track_history[track_id]
                    track.append((float(x), float(y)))
                    if len(track) > 30:
                        track.pop(0)

                    object_id = track_id
                    bb_left = x_center - box_width / 2
                    bb_top = y_center - box_height / 2
                    x, y, z = -1, -1, -1 
                    vehicle_type = model.names[class_id]

                    if args.detailed or len(track) == 1:  # Only append if detailed or first occurrence of object_id
                        file.write(f"{frame_number},{object_id},{vehicle_type},{bb_left},{bb_top},{box_width},{box_height},{confidence},{x},{y},{z},{current_time}\n")
                        file.flush()

                frame_number += 1

            else:
                break

    cap.release()
    cv2.destroyAllWindows()
    

def main():
    parser = argparse.ArgumentParser(description='Tracking using YOLO model')
    parser.add_argument('--trained_weights', type=str, required=True, help='Path to trained weights file')
    parser.add_argument('--rtsp_url', type=str, required=True, help='RTSP stream URL')
    parser.add_argument('--type_tracker', type=str, default='bytetrack.yaml', help='File to access type of tracker')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run YOLO model on (e.g., "cpu", "cuda:0")')
    parser.add_argument('--detailed', action='store_true', help='If set, create a detailed file with all data')

    args = parser.parse_args()
    tracking(args)


if __name__ == '__main__':
    main()