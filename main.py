import cv2
import numpy as np
from ultralytics import YOLO
import threading
from flask import Flask, render_template, jsonify
import time

# --- FLASK APP SETUP ---
app = Flask(__name__)

# --- TIMING CONFIGURATION ---
BUFFER_IN = 10   # Seconds to wait before ADDING a person (Occupation)
BUFFER_OUT = 30  # Seconds to wait before REMOVING a person (Vacancy)

# Global variable for the frontend
occupancy_data = {
    "area_1": {"count": 0, "capacity": 4},
    "area_2": {"count": 0, "capacity": 4},
    "area_3": {"count": 0, "capacity": 4},
    "area_4": {"count": 0, "capacity": 4},
    "area_5": {"count": 0, "capacity": 8},
}

# State tracking
area_states = {
    "area_1": {"candidate_count": 0, "last_change_time": 0, "confirmed_count": 0},
    "area_2": {"candidate_count": 0, "last_change_time": 0, "confirmed_count": 0},
    "area_3": {"candidate_count": 0, "last_change_time": 0, "confirmed_count": 0},
    "area_4": {"candidate_count": 0, "last_change_time": 0, "confirmed_count": 0},
    "area_5": {"candidate_count": 0, "last_change_time": 0, "confirmed_count": 0},
}

# --- YOLO & DETECTION LOGIC ---
def run_detection():
    global occupancy_data, area_states
    
    # Load model
    model = YOLO("yolo11m.pt")
    names = model.names
    
    # Video source
    cap = cv2.VideoCapture('trim.mp4') 

    # Define Areas
    area_1 = [(76, 337), (215,347), (159, 421), (0,400)]
    area_2 = [(215,347), (433,352), (403,426), (159, 421)]
    area_3 = [(0,438), (382,463), (371,500), (0,500)]
    area_4 = [(677,346), (690, 380), (1020,360), (1020, 325)]
    area_5 = [(690, 380), (1020,360), (1020, 463), (744,479)]

    areas = [area_1, area_2, area_3, area_4, area_5]
    area_keys = ["area_1", "area_2", "area_3", "area_4", "area_5"]

    count_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        count_frame += 1
        if count_frame % 2 != 0:
            continue

        frame = cv2.resize(frame, (1020, 500))
        results = model.track(frame, persist=True, verbose=False)

        # 1. Get Instantaneous (Raw) Counts
        current_raw_counts = {k: 0 for k in area_keys}

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            
            for box, class_id in zip(boxes, class_ids):
                c = names[class_id]
                if 'person' in c:
                    x1, y1, x2, y2 = box
                    cx = int(x1+x2)//2
                    cy = int(y1+y2)//2

                    for i, area in enumerate(areas):
                        result = cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False)
                        if result >= 0:
                            current_raw_counts[area_keys[i]] += 1
                            cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)
                            cv2.rectangle(frame, (x1,y1), (x2,y2), (0, 255, 0), 1)
        
        for area in areas:
            cv2.polylines(frame, [np.array(area, np.int32)], True, (255, 255, 255), 1)

        # 2. Apply Dual-Timer Logic
        current_time = time.time()
        
        for key in area_keys:
            raw_count = current_raw_counts[key]
            state = area_states[key]

            # Step A: Check if the raw input is different from our "candidate"
            # If the camera sees something new, reset the timer.
            if raw_count != state["candidate_count"]:
                state["candidate_count"] = raw_count
                state["last_change_time"] = current_time
            
            # Step B: If the raw input is stable (equal to candidate), check the timer
            else:
                time_stable = current_time - state["last_change_time"]
                
                # Scenario 1: Count is INCREASING (People Arriving) -> Wait 10s
                if state["candidate_count"] > state["confirmed_count"]:
                    if time_stable >= BUFFER_IN:
                        state["confirmed_count"] = state["candidate_count"]

                # Scenario 2: Count is DECREASING (People Leaving) -> Wait 30s
                elif state["candidate_count"] < state["confirmed_count"]:
                    if time_stable >= BUFFER_OUT:
                        state["confirmed_count"] = state["candidate_count"]

            # Update global data (with capacity clamping from previous fix)
            occupancy_data[key]["count"] = min(state["confirmed_count"], occupancy_data[key]["capacity"])

            cv2.imshow("RGB", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
# --- FLASK ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/status')
def status():
    return jsonify(occupancy_data)

if __name__ == '__main__':
    t = threading.Thread(target=run_detection)
    t.daemon = True
    t.start()
    app.run(host='0.0.0.0', port=5000, debug=False)
        