import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import io

app = FastAPI()

# 1. LOAD MODELS (Load them once when server starts to stay fast)
trailer_model = YOLO("trailer_best.pt")
logo_model = YOLO("logo_best.pt")

# --- YOUR CORE LOGIC (Simplified for Web) ---

def process_pipeline(image):
    # STEP 1: FLATTEN (From your Cell 2)
    results = trailer_model(image)[0]
    if len(results.boxes) == 0:
        return {"error": "No trailer detected"}
    
    box = results.boxes[0].xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = map(int, box)
    src_pts = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    width, height = (x2 - x1), (y2 - y1)
    dst_pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    flat_canvas = cv2.warpPerspective(image, matrix, (width, height))

    # STEP 2: MEASURE (Using your DOT tape logic from Cell 4/6)
    # For the web demo, we will assume the 12-inch pattern for now 
    # to keep the demo snappy, or you can add your micro-image logic later!
    # Let's assume a standard length for the demo trigger:
    final_length = 28.0  # Placeholder: In production, you'd run your tape count here
    
    pixel_h, pixel_w = flat_canvas.shape[:2]
    feet_per_pixel = final_length / pixel_w
    final_height = pixel_h * feet_per_pixel
    final_area = final_length * final_height

    # STEP 3: LOGO & BRAND (From your Cell 7)
    logo_results = logo_model(flat_canvas, conf=0.5)[0]
    brand = "UNKNOWN"
    true_ink_sqft = 0

    if len(logo_results.boxes) > 0:
        box = logo_results.boxes[0]
        brand = logo_model.names[int(box.cls[0].item())].upper()
        coords = list(map(int, box.xyxy[0].cpu().numpy()))
        
        # STEP 4: TRUE INK (From your Cell 8)
        # (Otsu math goes here)
        x1_l, y1_l, x2_l, y2_l = coords
        roi = flat_canvas[y1_l:y2_l, x1_l:x2_l]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        if cv2.countNonZero(binary) > (binary.size / 2):
            binary = cv2.bitwise_not(binary)
        
        ink_pixels = cv2.countNonZero(binary)
        true_ink_sqft = ink_pixels * (feet_per_pixel ** 2)

    # STEP 5: BUSINESS LOGIC
    m_type = "Single Unit (Full Wrap)" if brand in ["AMAZON", "COCACOLA"] else "Set of Items (Discrete Decals)"

    return {
        "brand": brand,
        "length": round(final_length, 2),
        "height": round(final_height, 2),
        "total_area": round(final_area, 2),
        "manufacturing_type": m_type,
        "ink_area": round(true_ink_sqft, 2)
    }

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    # Convert upload to OpenCV format
    data = await file.read()
    nparr = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Run the pipeline
    report = process_pipeline(image)
    return report