import cv2
import numpy as np
import base64
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware  # Added for browser permission
from ultralytics import YOLO
import io

app = FastAPI()

# --- CORS PERMISSION BLOCK ---
# This allows your local website to talk to your Railway server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. LOAD MODELS
# Ensure these filenames match what you uploaded to your GitHub repo
trailer_model = YOLO("trailer_best.pt")
logo_model = YOLO("logo_best.pt")

# --- HELPER FUNCTIONS ---

def encode_img(img):
    """Converts OpenCV image to Base64 string for the website"""
    if img is None: return ""
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

# --- CORE AI LOGIC (FROM YOUR COLAB) ---

def get_tape_pattern_analysis(micro_img):
    """CELL 3: Detects 12/18 inch pattern and returns (value, viz_string)"""
    hsv = cv2.cvtColor(micro_img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 80, 40]); upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 80, 40]); upper_red2 = np.array([180, 255, 255])
    mask = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2))
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    temp_boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > h * 1.5: temp_boxes.append((x, y, w, h))
    
    if len(temp_boxes) < 2: return 12, encode_img(micro_img)
    
    temp_boxes = sorted(temp_boxes, key=lambda b: b[0])
    ratio = temp_boxes[0][2] / (temp_boxes[1][0] - (temp_boxes[0][0] + temp_boxes[0][2]))
    pattern = 12 if ratio < 1.3 else 18
    
    # Draw logic for viz
    debug_micro = micro_img.copy()
    for (x, y, w, h) in temp_boxes[:2]:
        cv2.rectangle(debug_micro, (x, y), (x+w, y+h), (0, 255, 0), 3)
    
    return pattern, encode_img(debug_micro)

def process_trailer_vision(macro_img, pattern_inches):
    """CELL 2 & 4: Flattening + Statistical Strip Counting"""
    results = trailer_model(macro_img)[0]
    if len(results.boxes) == 0: return None, 0, ""
    
    # 1. Perspective Warp
    box = results.boxes[0].xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = map(int, box)
    src_pts = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    width, height = (x2 - x1), (y2 - y1)
    dst_pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    flat_canvas = cv2.warpPerspective(macro_img, matrix, (width, height))

    # 2. Strip Counting with Statistical Filter
    hsv = cv2.cvtColor(flat_canvas, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255])),
                         cv2.inRange(hsv, np.array([160, 50, 50]), np.array([180, 255, 255])))
    mask[0:int(flat_canvas.shape[0]*0.80), :] = 0 
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    temp_boxes = [cv2.boundingRect(c) for c in contours if cv2.boundingRect(c)[2] > cv2.boundingRect(c)[3]]
    
    if not temp_boxes: return flat_canvas, 0, encode_img(flat_canvas)
    
    median_w = np.median([b[2] for b in temp_boxes])
    valid_boxes = [b for b in temp_boxes if (median_w * 0.5) < b[2] < (median_w * 1.5)]
    
    # Draw strip boxes for viz
    strip_viz_img = flat_canvas.copy()
    for (bx, by, bw, bh) in valid_boxes:
        cv2.rectangle(strip_viz_img, (bx, by), (bx+bw, by+bh), (0, 255, 0), 2)
    
    final_length = (len(valid_boxes) * pattern_inches) / 12
    return flat_canvas, final_length, encode_img(strip_viz_img)

def get_brand_and_ink(flat_canvas, length_ft):
    """CELL 7 & 8: Logo Detection + Otsu Binarization"""
    results = logo_model(flat_canvas, conf=0.5)[0]
    if len(results.boxes) == 0: return "UNKNOWN", 0, "", encode_img(flat_canvas)
    
    box = results.boxes[0]
    brand = logo_model.names[int(box.cls[0].item())].upper()
    coords = list(map(int, box.xyxy[0].cpu().numpy()))
    
    # Otsu Logic
    roi = flat_canvas[coords[1]:coords[3], coords[0]:coords[2]]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if cv2.countNonZero(binary) > (binary.size / 2): binary = cv2.bitwise_not(binary)
    
    # Create Green Pixel Overlay
    ink_overlay = roi.copy()
    ink_overlay[binary > 0] = [0, 255, 0]
    
    # Master Manifest View
    manifest_img = flat_canvas.copy()
    cv2.rectangle(manifest_img, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 255), 5)
    cv2.putText(manifest_img, f"BRAND: {brand}", (coords[0], coords[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5)

    feet_per_px = length_ft / flat_canvas.shape[1]
    ink_sqft = cv2.countNonZero(binary) * (feet_per_px ** 2)
    
    return brand, ink_sqft, encode_img(ink_overlay), encode_img(manifest_img)

# --- THE API ENDPOINT ---

@app.post("/analyze")
async def analyze_truck(macro_file: UploadFile = File(...), micro_file: UploadFile = File(...)):
    # Read files
    macro_img = cv2.imdecode(np.frombuffer(await macro_file.read(), np.uint8), cv2.IMREAD_COLOR)
    micro_img = cv2.imdecode(np.frombuffer(await micro_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # 1. Micro Analysis
    pattern, micro_viz = get_tape_pattern_analysis(micro_img)
    
    # 2. Macro Geometry
    flat_canvas, length, strip_viz = process_trailer_vision(macro_img, pattern)
    if flat_canvas is None: return {"error": "Trailer not detected"}
    
    # 3. Logo & Ink
    brand, ink_sqft, ink_viz, manifest_viz = get_brand_and_ink(flat_canvas, length)
    
    # 4. Final Math
    height = (flat_canvas.shape[0] * (length / flat_canvas.shape[1]))
    total_area = length * height

    return {
        "results": {
            "fleet_owner": brand,
            "tape_calibration": f"{pattern} inch",
            "length_ft": round(length, 2),
            "height_ft": round(height, 2),
            "total_area_sqft": round(total_area, 2),
            "ink_yield_sqft": round(ink_sqft, 2)
        },
        "visuals": {
            "micro_pattern": micro_viz,      # The tape detection
            "flattened_strips": strip_viz,   # Flat trailer + green strip boxes
            "pixel_scan": ink_viz,           # The green Otsu overlay
            "final_manifest": manifest_viz   # Master view with Logo box
        }
    }
