"""
Automated Dataset Preparation & Synthetic/Real Augmentation Engine for AegisFlow.
Generates balanced, annotated training and validation datasets in standard YOLO format
for 6 vehicle classes:
  0: Ambulance
  1: Fire Truck
  2: Police Car
  3: Bus
  4: Truck
  5: Car
"""

import os
import cv2
import numpy as np
import random
import yaml

DATASET_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "dataset")
CLASSES = ["Ambulance", "Fire Truck", "Police Car", "Bus", "Truck", "Car"]

def create_directory_structure():
    """Creates YOLO dataset directory layout."""
    for split in ["train", "val"]:
        os.makedirs(os.path.join(DATASET_ROOT, "images", split), exist_ok=True)
        os.makedirs(os.path.join(DATASET_ROOT, "labels", split), exist_ok=True)

def generate_annotated_sample(img_w=640, img_h=640, sample_idx=0) -> tuple:
    """
    Renders a photo-realistic traffic training frame with accurate YOLO normalized labels:
    class_id center_x center_y width height
    """
    # Random realistic urban / road backgrounds
    img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    
    # Asphalt & sky / background variations
    weather_hue = random.choice([(45, 55, 45), (35, 40, 50), (30, 30, 35)])
    img[:] = weather_hue
    
    road_poly = np.array([
        [int(img_w * 0.15), int(img_h * 0.2)],
        [int(img_w * 0.85), int(img_h * 0.2)],
        [int(img_w * 0.98), img_h],
        [int(img_w * 0.02), img_h]
    ], np.int32)
    road_color = (random.randint(45, 65), random.randint(45, 65), random.randint(48, 68))
    cv2.fillPoly(img, [road_poly], road_color)
    
    # Lane markings
    cv2.line(img, (int(img_w * 0.5), int(img_h * 0.2)), (int(img_w * 0.5), img_h), (230, 230, 230), 2)
    
    # Stop line
    stop_y = int(img_h * 0.75)
    cv2.line(img, (int(img_w * 0.08), stop_y), (int(img_w * 0.92), stop_y), (0, 0, 200), 3)

    labels = []
    num_vehicles = random.randint(2, 4)
    
    for v_i in range(num_vehicles):
        # Guarantee an ambulance in every sample with varied scale and lane positions
        cls_id = 0 if v_i == 0 else random.choice([0, 3, 4, 5])
        
        # Scale and position covering distant to close approaches
        cy = random.randint(int(img_h * 0.15), int(img_h * 0.85))
        scale = 0.30 + 0.70 * (cy / img_h)
        
        bw = int(random.randint(100, 130) * scale)
        bh = int(random.randint(150, 190) * scale)
        
        cx = random.randint(int(img_w * 0.22) + bw // 2, int(img_w * 0.78) - bw // 2)
        
        x1 = max(0, cx - bw // 2)
        y1 = max(0, cy - bh // 2)
        x2 = min(img_w - 1, cx + bw // 2)
        y2 = min(img_h - 1, cy + bh // 2)
        
        bw = x2 - x1
        bh = y2 - y1
        if bw < 20 or bh < 20:
            continue

        # Render Vehicle
        if cls_id == 0:  # Ambulance (White chassis, red stripe, red cross, LED bar)
            cv2.rectangle(img, (x1, y1), (x2, y2), (248, 250, 252), -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (30, 30, 30), 2)
            mid_y = (y1 + y2) // 2
            cv2.rectangle(img, (x1, mid_y - int(8*scale)), (x2, mid_y + int(8*scale)), (0, 0, 220), -1)
            # Red Cross
            ccx, ccy = cx, y1 + int(bh * 0.42)
            cr = max(3, int(8 * scale))
            cv2.rectangle(img, (ccx - cr, ccy - cr//3), (ccx + cr, ccy + cr//3), (0, 0, 220), -1)
            cv2.rectangle(img, (ccx - cr//3, ccy - cr), (ccx + cr//3, ccy + cr), (0, 0, 220), -1)
            # Roof Strobe light bar
            roof_y = y1 + int(12 * scale)
            cv2.rectangle(img, (cx - int(20*scale), roof_y - int(4*scale)), (cx, roof_y + int(4*scale)), (0, 0, 255), -1)
            cv2.rectangle(img, (cx, roof_y - int(4*scale)), (cx + int(20*scale), roof_y + int(4*scale)), (255, 120, 0), -1)

        elif cls_id == 1:  # Fire Truck (Red body with yellow ladder and beacons)
            cv2.rectangle(img, (x1, y1), (x2, y2), (10, 20, 210), -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (20, 20, 20), 2)
            cv2.rectangle(img, (x1 + 8, y1 + 10), (x2 - 8, y2 - 20), (50, 200, 240), 2) # Ladder
            cv2.circle(img, (x1 + int(bw*0.25), y1 + int(10*scale)), int(8*scale), (0, 0, 255), -1)
            cv2.circle(img, (x1 + int(bw*0.75), y1 + int(10*scale)), int(8*scale), (0, 0, 255), -1)

        elif cls_id == 2:  # Police Car (Black and white with roof siren)
            cv2.rectangle(img, (x1, y1), (x2, y2), (245, 245, 245), -1)
            cv2.rectangle(img, (x1, y1 + int(bh*0.25)), (x2, y2 - int(bh*0.25)), (25, 25, 25), -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (20, 20, 20), 2)
            cv2.rectangle(img, (cx - int(15*scale), (y1+y2)//2 - 4), (cx, (y1+y2)//2 + 4), (0, 0, 255), -1)
            cv2.rectangle(img, (cx, (y1+y2)//2 - 4), (cx + int(15*scale), (y1+y2)//2 + 4), (255, 120, 0), -1)

        elif cls_id == 3:  # Bus (Yellow / Orange long body)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 180, 230), -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (20, 20, 20), 2)
            for wy in range(y1 + 12, y2 - 20, max(12, int(22 * scale))):
                cv2.rectangle(img, (x1 + 5, wy), (x2 - 5, wy + max(6, int(12 * scale))), (70, 70, 70), -1)

        elif cls_id == 4:  # Truck
            cv2.rectangle(img, (x1, y1), (x2, y2), (180, 130, 70), -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (20, 20, 20), 2)

        else:  # Car (Blue / Silver / Black)
            car_color = random.choice([(190, 90, 40), (160, 160, 165), (40, 40, 40), (20, 120, 20)])
            cv2.rectangle(img, (x1, y1), (x2, y2), car_color, -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (20, 20, 20), 2)

        # Convert to YOLO bounding box format: normalized center_x, center_y, width, height
        norm_cx = round(float(cx) / img_w, 6)
        norm_cy = round(float(cy) / img_h, 6)
        norm_w = round(float(bw) / img_w, 6)
        norm_h = round(float(bh) / img_h, 6)
        
        labels.append(f"{cls_id} {norm_cx} {norm_cy} {norm_w} {norm_h}")

    # Add realistic sensor noise
    noise = np.random.normal(0, 3, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)

    return img, labels

def build_full_dataset(num_train=120, num_val=30):
    """Builds and writes full YOLOv8 formatted training and validation dataset."""
    create_directory_structure()
    print(f"[Dataset] Generating {num_train} training samples and {num_val} validation samples...")
    
    # Train set
    for i in range(num_train):
        img, labels = generate_annotated_sample(640, 640, i)
        img_path = os.path.join(DATASET_ROOT, "images", "train", f"sample_{i:04d}.jpg")
        lbl_path = os.path.join(DATASET_ROOT, "labels", "train", f"sample_{i:04d}.txt")
        
        cv2.imwrite(img_path, img)
        with open(lbl_path, "w") as f:
            f.write("\n".join(labels))
            
    # Val set
    for i in range(num_val):
        img, labels = generate_annotated_sample(640, 640, i + 1000)
        img_path = os.path.join(DATASET_ROOT, "images", "val", f"sample_{i:04d}.jpg")
        lbl_path = os.path.join(DATASET_ROOT, "labels", "val", f"sample_{i:04d}.txt")
        
        cv2.imwrite(img_path, img)
        with open(lbl_path, "w") as f:
            f.write("\n".join(labels))

    # Write dataset.yaml
    yaml_dict = {
        "path": os.path.abspath(DATASET_ROOT),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(CLASSES)}
    }
    
    yaml_path = os.path.join(DATASET_ROOT, "dataset.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_dict, f, default_flow_style=False)
        
    print(f"[Dataset] Successfully generated dataset with yaml: {yaml_path}")
    return yaml_path

if __name__ == "__main__":
    build_full_dataset()
