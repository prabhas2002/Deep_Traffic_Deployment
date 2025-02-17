


# Deep Traffic Deployment

This project deploys a fine-tuned YOLO model with ByteTrack for real-time tracking of vehicles on multiple RTSP streams. The system is optimized for low latency, minimal memory usage, and efficient querying of tracking data.

## File Structure

```
.
├── rtsp_stream.py        # Visualizes tracking on RTSP streams
├── generate_txt.py       # Generates a TXT file in MOT format from tracking data
├── query.py              # Queries the generated TXT file for traffic density
├── requirements.txt      # List of required Python packages
└── README.md             # Project documentation
```

## Usage

### 1. Visualize Tracking on RTSP Stream

Run the following command to visualize the tracking on an RTSP stream:

```bash
python rtsp_stream.py --trained_weights <path to trained weights> --rtsp_url <RTSP stream URL> --device <device>
```

Example:

```bash
python rtsp_stream.py --trained_weights /path_to_pwd/yolo.pt --rtsp_url rtsp://admin:admin@172.16.37.11:554/1/h264minor --device cuda:0
```

### 2. Generate the TXT File in MOT Format Based on Tracking of RTSP Stream

To generate the TXT file in MOT format from tracking data:

```bash
python generate_txt.py --trained_weights <path to trained weights> --rtsp_url <RTSP stream URL> --device <device>
```

Example:

```bash
python generate_txt.py --trained_weights /path_to_pwd/yolo.pt --rtsp_url rtsp://admin:admin@172.16.37.11:554/1/h264minor --device cuda:0
```

To generate a detailed TXT file that may contain duplicate IDs and more tracking information (which could lead to high disk space consumption), use the `--detailed` option:

```bash
python generate_txt.py --trained_weights <path to trained weights> --rtsp_url <RTSP stream URL> --device <device> --detailed
```

**Note:** We recommend not using the `--detailed` option, as it may lead to excessive disk space consumption. Without `--detailed`, the generated file will be more efficient while maintaining accuracy.

### 3. Query the Generated TXT File

The `query.py` script allows you to query the generated MOT format file to calculate traffic density between two time stamps. Below is the command syntax:

```bash
python query.py --date <date> --start_time <start_time> --end_time <end_time> --file_path <file_path> --confidence_threshold <confidence_threshold>
```

**Arguments:**

- `--date <date>`: The date on which the tracking data was generated. The date should be in `YYYY-MM-DD` format.
- `--start_time <start_time>`: The starting time for the query in `HH:MM:SS` format. This is the time from which the traffic density will be calculated.
- `--end_time <end_time>`: The ending time for the query in `HH:MM:SS` format. This is the time up to which the traffic density will be calculated.
- `--file_path <file_path>`: The path to the directory where the MOT file is stored. This file contains the tracking data for the specified stream.
- `--confidence_threshold <confidence_threshold>`: The threshold value (from 0 to 1) for object detection confidence. Only detections with a confidence score greater than or equal to this threshold will be considered for the traffic density calculation.

Example command:

```bash
python query.py --date 2025-02-17 --start_time 11:13:54 --end_time 11:15:52 --file_path /path_to_pwd/Results/172.16.37.11_554/2025-02-17 --confidence_threshold 0.3
```

**Note:** Adjusting the `--confidence_threshold` helps control the quality of detected objects included in the traffic density calculation. Higher values may reduce false positives, while lower values may increase the chances of including noisy detections. The default value is 0.3.

### 4. Using the Model on Multiple RTSP Streams

To use the model on multiple RTSP streams, change the RTSP stream URL in the arguments and change the device accordingly based on the virtual space available in the system.
