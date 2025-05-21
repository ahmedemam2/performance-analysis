from flask import Flask, render_template, request, send_file, jsonify
import os
from werkzeug.utils import secure_filename
import supervision as sv
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time
from supervision.detection.utils import box_iou_batch
import uuid

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
REPORT_FOLDER = 'reports'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['REPORT_FOLDER'] = REPORT_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_video(input_path, output_path, report_path):
    model = YOLO('yolov8x.pt')
    tracker = sv.ByteTrack(track_activation_threshold=0.3, lost_track_buffer=60, minimum_matching_threshold=0.9)
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    track_history = defaultdict(lambda: {
        "active": False,
        "last_box": None,
        "last_seen_frame": -1,
        "missing_since_frame": -1,
        "reappearances": [],
        "id_switches": 0,
        "false_reids": 0,
        "premature_deletions": 0
    })
    start_time = time.time()
    total_frames = 0
    total_id_switches = 0
    total_false_reids = 0
    video_info = sv.VideoInfo.from_video_path(input_path)
    fps = video_info.fps

    def is_similar(box1, box2, threshold=0.7):
        box1 = np.array(box1).reshape(1, 4)
        box2 = np.array(box2).reshape(1, 4)
        iou = box_iou_batch(box1, box2)[0][0]
        return iou > threshold

    def callback(frame: np.ndarray, index: int) -> np.ndarray:
        nonlocal total_frames, total_id_switches, total_false_reids
        total_frames += 1
        results = model(frame, classes=[0], conf=0.6, iou=0.7)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)
        current_time = index / fps
        current_ids = set(detections.tracker_id) if detections.tracker_id is not None else set()
        
        if detections.tracker_id is not None:
            for i, bbox in enumerate(detections.xyxy):
                tracker_id = detections.tracker_id[i]
                for history_id, history in track_history.items():
                    if history["last_box"] is not None:
                        if is_similar(bbox, history["last_box"]):
                            if history["active"] and tracker_id != history_id:
                                track_history[history_id]["id_switches"] += 1
                                total_id_switches += 1
                            elif not history["active"]:
                                track_history[history_id]["false_reids"] += 1
                                total_false_reids += 1
        
        if detections.tracker_id is not None:
            for i, bbox in enumerate(detections.xyxy):
                tracker_id = detections.tracker_id[i]
                if tracker_id not in track_history or track_history[tracker_id]["last_box"] is None:
                    track_history[tracker_id]["last_box"] = bbox
                
                if not track_history[tracker_id]["active"] and track_history[tracker_id]["last_seen_frame"] != -1:
                    missing_duration = current_time - (track_history[tracker_id].get("missing_since_time", 0))
                    track_history[tracker_id]["reappearances"].append((current_time, missing_duration))
                
                track_history[tracker_id].update({
                    "active": True,
                    "last_box": bbox,
                    "last_seen_frame": index,
                    "missing_since_frame": -1
                })
        
        for tracker_id in list(track_history.keys()):
            if tracker_id not in current_ids and track_history[tracker_id]["active"]:
                frames_missing = index - track_history[tracker_id]["last_seen_frame"]
                if frames_missing < 10:  
                    track_history[tracker_id]["premature_deletions"] += 1
                
                track_history[tracker_id].update({
                    "active": False,
                    "missing_since_frame": index,
                    "missing_since_time": current_time
                })
        
        labels = []
        if detections.tracker_id is not None:
            labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
        
        annotated_frame = box_annotator.annotate(frame.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
        
        return annotated_frame

    try:
        sv.process_video(
            source_path=input_path,
            target_path=output_path,
            callback=callback
        )
    except Exception as e:
        print(f"Error processing video: {e}")
        return False

    elapsed_time = time.time() - start_time
    with open(report_path, 'w', encoding="utf-8") as f:
        f.write("Summary\n")
        f.write(f"Total frames processed: {total_frames}\n")
        f.write(f"Average FPS: {total_frames/elapsed_time:.1f}\n")
        f.write(f"Unique objects detected: {len(track_history)}\n")
        f.write(f"Total ID switches: {total_id_switches}\n")
        f.write(f"Total false re-IDs: {total_false_reids}\n")
        
        f.write("\nVideo Information:\n")
        f.write(f"- Source: {os.path.basename(input_path)}\n")
        f.write(f"- Resolution: {video_info.width}x{video_info.height}\n")
        f.write(f"- Duration: {video_info.total_frames/fps:.2f} seconds\n")
        f.write(f"- Original FPS: {fps}\n")
        
        f.write("\nPerformance Metrics:\n")
        f.write(f"- Processing time: {elapsed_time:.2f} seconds\n")
        f.write(f"- Processing speed: {video_info.total_frames/elapsed_time:.1f}x real-time\n")
        
        f.write("\nObject Details:\n")
        for tracker_id, data in sorted(track_history.items(), key=lambda x: x[1]['last_seen_frame']):
            if data['last_seen_frame'] != -1:
                f.write(f"\nObject #{tracker_id}:\n")
                f.write(f"- Active: {'Yes' if data['active'] else 'No'}\n")
                f.write(f"- First seen at frame: {data.get('first_seen_frame', 'N/A')}\n")
                f.write(f"- Last seen at frame: {data['last_seen_frame']}\n")
                f.write(f"- ID switches: {data['id_switches']}\n")
                f.write(f"- False re-IDs: {data['false_reids']}\n")
                f.write(f"- Reappearances: {len(data['reappearances'])}\n")
                if 'reappearances' in data and data['reappearances']:
                    avg_reappearance = sum(dur for (_, dur) in data['reappearances'])/len(data['reappearances'])
                    f.write(f"- Average reappearance time: {avg_reappearance:.2f} seconds\n")

    return True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Generate unique ID for this processing job
        processing_id = str(uuid.uuid4())
        
        # Save original file
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{processing_id}_{filename}")
        file.save(input_path)
        
        # Set output paths
        output_filename = f"processed_{filename}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        report_path = os.path.join(app.config['REPORT_FOLDER'], f"report_{processing_id}.txt")
        
        # Process video in background (in production, use Celery or similar)
        success = process_video(input_path, output_path, report_path)
        
        if success:
            return jsonify({
                'processing_id': processing_id,
                'original_filename': filename,
                'processed_video': output_filename,
                'report_file': f"report_{processing_id}.txt"
            })
        else:
            return jsonify({'error': 'Video processing failed'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/download/<filename>')
def download_file(filename):
    if filename.startswith('processed_'):
        directory = app.config['OUTPUT_FOLDER']
    elif filename.startswith('report_'):
        directory = app.config['REPORT_FOLDER']
    else:
        return jsonify({'error': 'File not found'}), 404
    
    file_path = os.path.join(directory, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)