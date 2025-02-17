import argparse
import pandas as pd
from datetime import datetime
import os

def query_vehicle_count(date, start_time, end_time, confidence_threshold, file_path, detailed):
    valid_vehicle_types = ["caravan", "truck", "autorickshaw", "motorcycle", "car", "bicycle", "bus"]
    
    try:
        start_datetime = datetime.strptime(f"{date} {start_time}", "%Y-%m-%d %H:%M:%S")
        end_datetime = datetime.strptime(f"{date} {end_time}", "%Y-%m-%d %H:%M:%S")
        
        if start_datetime > end_datetime:
            print("Error: Start time must be less than or equal to end time.")
            return
        
        rtsp_ip = file_path.split('/')[-1]
        
        if detailed:
            data_file = f"{file_path}/{date}_detailed.txt"
        else:
            data_file = f"{file_path}/{date}.txt"
        
        if not os.path.exists(data_file):
            print(f"No data file found for the given date in {file_path}.")
            return
        
        columns = ["frame_number", "object_id", "vehicle_type", "bb_left", "bb_top", "box_width", "box_height", "confidence", "x", "y", "z", "timestamp"]
        df = pd.read_csv(data_file, names=columns)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        filtered_df = df[(df["timestamp"] >= start_datetime) & 
                         (df["timestamp"] <= end_datetime) & 
                         (df["confidence"] >= confidence_threshold) &
                         (df["vehicle_type"].isin(valid_vehicle_types))]
        vehicle_count = filtered_df["object_id"].nunique()
        print(f"Number of unique vehicles detected: {vehicle_count}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


def main():
    parser = argparse.ArgumentParser(description='Query number of vehicles based on timestamp and confidence score')
    parser.add_argument('--date', type=str, required=True, help='Date of the tracking file (YYYY-MM-DD)')
    parser.add_argument('--start_time', type=str, required=True, help='Start time (HH:MM:SS)')
    parser.add_argument('--end_time', type=str, required=True, help='End time (HH:MM:SS)')
    parser.add_argument('--confidence_threshold', type=float, default=0.3, help='Minimum confidence score')
    parser.add_argument('--file_path', type=str, default='./results', help='Path to results directory')
    parser.add_argument('--detailed', action='store_true', help='If set, query detailed files')

    args = parser.parse_args()
    query_vehicle_count(args.date, args.start_time, args.end_time, args.confidence_threshold, args.file_path, args.detailed)


if __name__ == '__main__':
    main()