import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import io

app = FastAPI()

# 1. LOAD MODELS ONCE (Saves memory and stay fast)
# Ensure these filenames match exactly what is in your folder!
trailer_model = YOLO("trailer_best.pt")
logo_model = YOLO("logo_best.pt")

# --- ENGINE ROOM: YOUR CORE LOGIC ---

def get_tape_pattern(micro_img):
    """CELL 3: Logic to distinguish 12-inch vs 18-inch tape"""
    hsv = cv2.cvtColor(micro_img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 80, 40]); upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 80, 40]); upper_red2 = np.array([180, 255, 255])
    mask = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2))
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    temp_boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > h * 1.5: temp_boxes.append((x, y, w, h))
    
    if len(temp_boxes) < 2: return 12 # Safe fallback
    
    temp_boxes = sorted(temp_boxes, key=lambda b: b[0])
    ratio = temp_boxes[0][2] / (temp_boxes[1][0] - (temp_boxes[0][0] + temp_boxes[0][2]))
    return 12 if ratio < 1.3 else 18

def process_trailer_geometry(macro_img, tape_pattern):
    """CELL 2 & 4: YOLO Flattening + Statistical Outlier Counting"""
    # YOLO Flattening
    results = trailer_model(macro_img)[0]
    if len(results.boxes) == 0: return None, 0
    
    box = results.boxes[0].xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = map(int, box)
    src_pts = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    width, height = (x2 - x1), (y2 - y1)
    dst_pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    flat_canvas = cv2.warpPerspective(macro_img, matrix, (width, height))

    # Strip Counting with Statistical Filter
    hsv = cv2.cvtColor(flat_canvas, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255])),
                         cv2.inRange(hsv, np.array([160, 50, 50]), np.array([180, 255, 255])))
    mask[0:int(flat_canvas.shape[0]*0.80), :] = 0 # ROI bottom 20%
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    temp_boxes = [cv2.boundingRect(c) for c in contours if cv2.boundingRect(c)[2] > cv2.boundingRect(c)[3]]
    
    if not temp_boxes: return flat_canvas, 0
    
    # Statistical Median Filter
    median_w = np.median([b[2] for b in temp_boxes])
    valid_count = len([b for b in temp_boxes if (median_w * 0.5) < b[2] < (median_w * 1.5)])
    
    final_length = (valid_count * tape_pattern) / 12
    return flat_canvas, final_length

def get_ink_area(flat_canvas, length_ft):
    """CELL 7 & 8: Brand Detection + Otsu's Binarization"""
    results = logo_model(flat_canvas, conf=0.5)[0]
    if len(results.boxes) == 0: return "UNKNOWN", 0, 0
    
    box = results.boxes[0]
    brand = logo_model.names[int(box.cls[0].item())].upper()
    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
    
    # Otsu's Masking
    roi = flat_canvas[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    if cv2.countNonZero(binary) > (binary.size / 2): binary = cv2.bitwise_not(binary)
    
    feet_per_px = length_ft / flat_canvas.shape[1]
    ink_sqft = cv2.countNonZero(binary) * (feet_per_px ** 2)
    
    return brand, ink_sqft, feet_per_px

# --- THE WAITER: API ENDPOINT ---

@app.post("/analyze")
async def analyze_fleet(macro_file: UploadFile = File(...), micro_file: UploadFile = File(...)):
    # Read files
    macro_bytes = await macro_file.read()
    micro_bytes = await micro_file.read()
    
    macro_img = cv2.imdecode(np.frombuffer(macro_bytes, np.uint8), cv2.IMREAD_COLOR)
    micro_img = cv2.imdecode(np.frombuffer(micro_bytes, np.uint8), cv2.IMREAD_COLOR)

    # 1. Calibrate Scale
    pattern = get_tape_pattern(micro_img)
    
    # 2. Process Geometry
    flat_canvas, length = process_trailer_geometry(macro_img, pattern)
    if flat_canvas is None: return {"error": "Trailer not detected in Macro image"}
    
    # 3. Process Ink & Branding
    brand, ink_sqft, ft_per_px = get_ink_area(flat_canvas, length)
    
    # 4. Final Math
    height = flat_canvas.shape[0] * ft_per_px
    total_area = length * height
    m_type = "Single Unit (Full Wrap)" if brand in ["AMAZON", "COCACOLA"] else "Set of Items (Discrete Decals)"

    return {
        "fleet_owner": brand,
        "tape_calibration": f"{pattern} inch",
        "trailer_dimensions": {"length_ft": round(length, 2), "height_ft": round(height, 2)},
        "total_area_sqft": round(total_area, 2),
        "manufacturing_type": m_type,
        "ink_yield_sqft": round(ink_sqft, 2),
        "required_graphic_area": round(total_area - ink_sqft, 2) if "Wrap" in m_type else round(ink_sqft, 2)
    }