import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import io

app = FastAPI()

# Load models once
trailer_model = YOLO("trailer_best.pt")
logo_model = YOLO("logo_best.pt")

# --- ADD YOUR CLASSIFICATION LOGIC HERE ---
def classify_tape_pattern(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 80, 40]); upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 80, 40]); upper_red2 = np.array([180, 255, 255])
    mask = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2))
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    temp_boxes = [cv2.boundingRect(c) for c in contours if cv2.boundingRect(c)[2] > cv2.boundingRect(c)[3] * 1.5]
    
    if len(temp_boxes) < 2: return 12 # Default fallback
    
    temp_boxes = sorted(temp_boxes, key=lambda b: b[0])
    ratio = temp_boxes[0][2] / (temp_boxes[1][0] - (temp_boxes[0][0] + temp_boxes[0][2]))
    return 12 if ratio < 1.3 else 18

@app.post("/analyze")
async def analyze_images(
    macro_file: UploadFile = File(...), 
    micro_file: UploadFile = File(...)
):
    # 1. Convert both uploads to OpenCV
    macro_data = await macro_file.read()
    macro_img = cv2.imdecode(np.frombuffer(macro_data, np.uint8), cv2.IMREAD_COLOR)
    
    micro_data = await micro_file.read()
    micro_img = cv2.imdecode(np.frombuffer(micro_data, np.uint8), cv2.IMREAD_COLOR)

    # 2. Get the Scale (Micro Image)
    tape_constant = classify_tape_pattern(micro_img)

    # 3. Run Flattening & Counting (Macro Image)
    # (Here you insert your full flatten_trailer_side and measure_flattened_length logic)
    # For now, let's assume the math results:
    final_length = 28.0 
    
    # 4. Final Data Package
    return {
        "tape_pattern": f"{tape_constant} inch",
        "length": final_length,
        "status": "Success"
    }