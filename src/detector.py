"""
YOLOv8 Detection Engine for AegisFlow.
Performs high-reliability real-time detection of emergency vehicles and civilian traffic.
Supports custom fine-tuned weights, COCO baseline fallback with heuristic livery analysis,
and multi-model fusion.
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

@dataclass
class Detection:
    box: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str
    is_emergency: bool
    priority_weight: float
    color_bgr: Tuple[int, int, int]
    area: int
    center: Tuple[int, int]

# Emergency priority weighting
PRIORITY_WEIGHTS = {
    "ambulance": 1.0,
    "fire truck": 1.0,
    "fire_truck": 1.0,
    "police car": 0.8,
    "police_car": 0.8,
    "bus": 0.4,
    "truck": 0.3,
    "car": 0.2,
    "motorcycle": 0.1
}

CLASS_COLORS = {
    "ambulance": (0, 0, 255),       # Vibrant Red
    "fire truck": (0, 69, 255),     # Orange-Red
    "police car": (255, 140, 0),    # Cyan/Deep Blue (BGR)
    "bus": (0, 215, 255),           # Gold/Yellow
    "truck": (205, 133, 63),        # Steel Blue
    "car": (120, 200, 80),          # Green
    "motorcycle": (147, 20, 255)    # Purple
}

class EmergencyDetector:
    def __init__(self, model_path: Optional[str] = None, conf_threshold: float = 0.25, device: str = "auto"):
        self.conf_threshold = conf_threshold
        self.device = device
        self.custom_model = None
        self.base_model = None
        self.is_custom_model = False
        self.deep_classifier = None
        self.deep_preprocess = None
        
        self._init_models(model_path)

    def _init_models(self, model_path: Optional[str] = None):
        """Initializes YOLO models and deep vision backbone for high-precision vehicle classification."""
        if YOLO is None:
            print("[Detector] Warning: ultralytics is not available.")
            return

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        best_weights = os.path.join(base_dir, "models", "weights", "best.pt")
        nano_weights = os.path.join(base_dir, "models", "weights", "yolov8n.pt")

        # 1. Base COCO Model for real-world localization
        try:
            target_base = nano_weights if os.path.exists(nano_weights) else "yolov8n.pt"
            self.base_model = YOLO(target_base)
            print(f"[Detector] Loaded base YOLOv8 model for vehicle detection.")
        except Exception as e:
            print(f"[Detector] Could not load base model: {e}")

        # 2. Custom Model if available
        custom_path = model_path if (model_path and os.path.exists(model_path)) else (best_weights if os.path.exists(best_weights) else None)
        if custom_path and os.path.exists(custom_path):
            try:
                self.custom_model = YOLO(custom_path)
                self.is_custom_model = True
                print(f"[Detector] Loaded custom model: {custom_path}")
            except Exception as e:
                print(f"[Detector] Could not load custom model: {e}")

        # 3. Deep ImageNet backbone (MobileNetV3) for fine-grained vehicle verification
        try:
            import torchvision.models as models
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
            self.deep_classifier = models.mobilenet_v3_small(weights=weights).eval()
            self.deep_preprocess = weights.transforms()
            print("[Detector] Loaded deep vision backbone for fine-grained vehicle verification.")
        except Exception as e:
            print(f"[Detector] Deep classifier not loaded: {e}")

    def classify_vehicle_crop(self, crop_bgr: np.ndarray, base_class: str, base_conf: float) -> Tuple[str, float, bool]:
        """
        Accurately determines whether a cropped vehicle is an Ambulance, Fire Truck, Police Car,
        or a standard civilian vehicle (Car, Truck, Bus, Motorcycle).
        Prevents false-positive classification of civilian vehicles.
        """
        if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.shape[0] < 14 or crop_bgr.shape[1] < 14:
            return base_class, base_conf, False

        ch, cw = crop_bgr.shape[:2]
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)

        # 1. Roof Emergency Beacon / Lightbar Analysis (Strictly top 22% of vehicle)
        # Avoids brake lights / taillights which are at the bottom rear (> 65% height)
        roof_h = max(3, int(ch * 0.22))
        roof = hsv[:roof_h, :]
        roof_px = max(1, roof.shape[0] * roof.shape[1])

        # Saturated roof strobe lights (Red, Blue, Amber)
        strobe_r1 = cv2.inRange(roof, np.array([0, 110, 110]), np.array([10, 255, 255]))
        strobe_r2 = cv2.inRange(roof, np.array([170, 110, 110]), np.array([180, 255, 255]))
        strobe_b = cv2.inRange(roof, np.array([95, 110, 110]), np.array([135, 255, 255]))
        strobe_a = cv2.inRange(roof, np.array([12, 130, 130]), np.array([25, 255, 255]))

        roof_strobe_px = np.sum((strobe_r1 | strobe_r2 | strobe_b | strobe_a) > 0)
        roof_strobe_ratio = roof_strobe_px / roof_px

        # 2. Body Livery Profile (Mid-body: 20% to 80% height)
        body = hsv[int(ch * 0.20):int(ch * 0.80), :]
        body_px = max(1, body.shape[0] * body.shape[1])

        # White chassis (high value, low saturation) or high-visibility fluorescent yellow
        white_mask = cv2.inRange(body, np.array([0, 0, 165]), np.array([180, 55, 255]))
        white_ratio = np.sum(white_mask > 0) / body_px

        # Emergency red cross or red stripe in central body
        red_b1 = cv2.inRange(body, np.array([0, 110, 90]), np.array([10, 255, 255]))
        red_b2 = cv2.inRange(body, np.array([170, 110, 90]), np.array([180, 255, 255]))
        red_body_ratio = np.sum((red_b1 | red_b2) > 0) / body_px

        # Total fire-truck red coverage
        total_px = ch * cw
        all_red_mask = (cv2.inRange(hsv, np.array([0, 100, 80]), np.array([12, 255, 255])) |
                        cv2.inRange(hsv, np.array([168, 100, 80]), np.array([180, 255, 255])))
        all_red_ratio = np.sum(all_red_mask > 0) / total_px

        # 3. Deep Neural Classifier Verification
        p_amb, p_fire, p_police = 0.0, 0.0, 0.0
        if self.deep_classifier is not None and self.deep_preprocess is not None:
            try:
                import torch
                from PIL import Image
                rgb_crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                tensor = self.deep_preprocess(Image.fromarray(rgb_crop)).unsqueeze(0)
                with torch.no_grad():
                    probs = self.deep_classifier(tensor).squeeze(0).softmax(0)
                p_amb = float(probs[407].item())    # ImageNet: ambulance
                p_fire = float(probs[555].item())   # ImageNet: fire engine
                p_police = float(probs[734].item()) # ImageNet: police van
            except Exception:
                pass

        # 4. Accurate Classification Decision
        # --- AMBULANCE ---
        # An ambulance has:
        # - White / high-vis yellow chassis (white_ratio > 0.22) + emergency red stripe/cross (red_body_ratio > 0.02) + roof strobe (roof_strobe_ratio > 0.02)
        # OR Deep ImageNet model strongly predicts ambulance on real photograph (p_amb > 0.35 with white_ratio > 0.15 or roof_strobe > 0.01)
        is_amb_struct = (white_ratio > 0.22 and red_body_ratio > 0.02 and roof_strobe_ratio > 0.02)
        is_amb_deep = (p_amb > 0.35 and (white_ratio > 0.15 or roof_strobe_ratio > 0.01))
        
        if is_amb_struct or is_amb_deep:
            final_conf = max(base_conf, p_amb) if p_amb > 0.20 else base_conf
            return "ambulance", final_conf, True

        # --- FIRE TRUCK ---
        # Fire trucks are large heavy engines (truck or bus), NEVER passenger sedans or civilian cars!
        # Must have heavy red coverage (> 45%), roof emergency beacons, and be a truck/bus, OR deep classifier confirmation
        is_fire_struct = (all_red_ratio > 0.45 and base_class in ["truck", "bus"] and roof_strobe_ratio > 0.03)
        is_fire_deep = (p_fire > 0.35 and base_class in ["truck", "bus"])
        if is_fire_struct or is_fire_deep:
            final_conf = max(base_conf, p_fire) if p_fire > 0.20 else base_conf
            return "fire truck", final_conf, True

        # --- POLICE CAR ---
        has_dual_strobe = (np.sum((strobe_r1 | strobe_r2) > 0) / roof_px > 0.02 and np.sum(strobe_b > 0) / roof_px > 0.02)
        is_police_struct = (has_dual_strobe and roof_strobe_ratio > 0.04)
        is_police_deep = (p_police > 0.40 and roof_strobe_ratio > 0.02)
        if is_police_struct or is_police_deep:
            final_conf = max(base_conf, p_police) if p_police > 0.20 else base_conf
            return "police car", final_conf, True

        # --- CIVILIAN VEHICLE ---
        # If it doesn't meet emergency criteria, it is faithfully a civilian vehicle!
        return base_class, base_conf, False

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        """
        Runs robust detection on a frame:
        1. Localizes vehicles with YOLOv8.
        2. Accurately verifies each vehicle via deep visual features and livery analysis.
        """
        if frame_bgr is None:
            return []

        h, w = frame_bgr.shape[:2]
        detections: List[Detection] = []
        raw_candidates = []

        def calc_iou(b1, b2):
            xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
            xi2, yi2 = min(b1[2], b2[2]), min(b1[3], b2[3])
            inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
            area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
            union = area1 + area2 - inter
            return (inter / union) if union > 0 else 0.0

        # 1. Primary pass: Base COCO Model (Accurate localization of real cars, trucks, buses, motorcycles)
        if self.base_model is not None:
            try:
                results_base = self.base_model(frame_bgr, conf=self.conf_threshold, verbose=False)
                for r in results_base:
                    if r.boxes is None:
                        continue
                    for box_data in r.boxes:
                        x1, y1, x2, y2 = box_data.xyxy[0].cpu().numpy().astype(int)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w - 1, x2), min(h - 1, y2)
                        
                        conf = float(box_data.conf[0].cpu().numpy())
                        cls_id = int(box_data.cls[0].cpu().numpy())
                        raw_name = str(self.base_model.names.get(cls_id, cls_id)).lower()
                        
                        if raw_name in ["car", "bus", "truck", "motorcycle"]:
                            raw_candidates.append({
                                "box": (x1, y1, x2, y2),
                                "raw_class": raw_name,
                                "raw_conf": conf,
                                "cls_id": cls_id
                            })
            except Exception as e:
                print(f"[Detector] Base model inference error: {e}")

        # 2. Secondary pass: Custom Model (for synthetic simulation feeds)
        if self.custom_model is not None:
            try:
                results_custom = self.custom_model(frame_bgr, conf=self.conf_threshold, verbose=False)
                for r in results_custom:
                    if r.boxes is None:
                        continue
                    for box_data in r.boxes:
                        x1, y1, x2, y2 = box_data.xyxy[0].cpu().numpy().astype(int)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w - 1, x2), min(h - 1, y2)
                        new_box = (x1, y1, x2, y2)
                        
                        conf = float(box_data.conf[0].cpu().numpy())
                        cls_id = int(box_data.cls[0].cpu().numpy())
                        
                        # Only add if not already localized by base model
                        if not any(calc_iou(new_box, c["box"]) > 0.40 for c in raw_candidates):
                            raw_candidates.append({
                                "box": new_box,
                                "raw_class": "car",  # Initial hypothesis to be verified
                                "raw_conf": conf,
                                "cls_id": cls_id
                            })
            except Exception as e:
                print(f"[Detector] Custom inference error: {e}")

        # 3. Fine-Grained Classification on each candidate crop
        for cand in raw_candidates:
            x1, y1, x2, y2 = cand["box"]
            crop = frame_bgr[y1:y2, x1:x2]
            
            final_class, final_conf, is_emerg = self.classify_vehicle_crop(
                crop,
                base_class=cand["raw_class"],
                base_conf=cand["raw_conf"]
            )
            
            priority_w = PRIORITY_WEIGHTS.get(final_class, 0.2)
            color = CLASS_COLORS.get(final_class, (120, 200, 80))
            
            detections.append(Detection(
                box=(x1, y1, x2, y2),
                confidence=final_conf,
                class_id=cand["cls_id"],
                class_name=final_class,
                is_emergency=is_emerg,
                priority_weight=priority_w,
                color_bgr=color,
                area=(x2 - x1) * (y2 - y1),
                center=((x1 + x2) // 2, (y1 + y2) // 2)
            ))

        return detections

    def analyze_movement_scope(self, det: Detection, frame_shape: Tuple[int, int]) -> Dict[str, Any]:
        """
        Calculates the spatial scope of vehicle movement, approach lane corridor,
        distance to stop-line, and priority clearance advisory.
        """
        h, w = frame_shape[:2]
        cx, cy = det.center
        x1, y1, x2, y2 = det.box
        stop_line_y = int(h * 0.75)
        
        # 1. Lane Assignment
        if cx < w * 0.35:
            lane_name = "Lane 1 (Left / West Corridor)"
        elif cx > w * 0.65:
            lane_name = "Lane 3 (Right / East Corridor)"
        else:
            lane_name = "Lane 2 (Center Rapid Rescue Corridor)"
            
        # 2. Distance to Stop Line
        dist_px = max(0, stop_line_y - cy)
        dist_meters = round(dist_px * 0.18, 1) # Estimated distance scaling
        
        # 3. Proximity & Scope Status
        if det.is_emergency:
            if cy >= stop_line_y:
                scope_status = "INTERSECTION TRAVERSED / CLEARING"
                action_scope = "Emergency vehicle has crossed the preemption line. Transitioning to clearance hold."
                eta_sec = 2.0
            elif dist_px < 120:
                scope_status = "CRITICAL IMMINENT ARRIVAL"
                action_scope = f"Immediate Green Wave mandatory on {lane_name}. Keep all conflicting cross-lanes RED."
                eta_sec = 3.5
            elif dist_px < 280:
                scope_status = "RAPID APPROACH IN CORRIDOR"
                action_scope = f"Emergency vehicle approaching on {lane_name}. Safety yellow clearance active on opposing directions."
                eta_sec = 7.0
            else:
                scope_status = "INCOMING APPROACH DETECTED"
                action_scope = f"Emergency vehicle detected entering surveillance zone on {lane_name}. Preemption schedule queued."
                eta_sec = 12.0
        else:
            scope_status = "STANDARD CIVILIAN TRANSIT"
            action_scope = f"Civilian {det.class_name} operating under normal traffic flow on {lane_name}. Normal signal progression active."
            eta_sec = round(max(3.0, dist_px * 0.08), 1)

        return {
            "vehicle_type": det.class_name.upper(),
            "confidence_pct": round(det.confidence * 100, 1),
            "lane_corridor": lane_name,
            "distance_px": dist_px,
            "estimated_meters": dist_meters,
            "scope_status": scope_status,
            "action_scope": action_scope,
            "estimated_eta_sec": eta_sec,
            "is_emergency": det.is_emergency,
            "priority_weight": det.priority_weight
        }

    def draw_annotations(self, frame_bgr: np.ndarray, detections: List[Detection], draw_vectors: bool = True) -> np.ndarray:
        """Draws polished high-contrast bounding boxes, labels, movement vector arrows, and badges on frame."""
        annotated = frame_bgr.copy()
        h, w = frame_bgr.shape[:2]
        stop_line_y = int(h * 0.75)
        
        # Draw stop line
        cv2.line(annotated, (int(w*0.05), stop_line_y), (int(w*0.95), stop_line_y), (0, 0, 220), 2)
        cv2.putText(annotated, "VIRTUAL PREEMPTION STOP LINE", (int(w*0.08), stop_line_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)

        for det in detections:
            x1, y1, x2, y2 = det.box
            color = det.color_bgr
            cx, cy = det.center
            
            if det.is_emergency:
                # Glowing emergency border
                cv2.rectangle(annotated, (x1-1, y1-1), (x2+1, y2+1), (255, 255, 255), 1)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                
                # Corner HUD brackets
                corner_len = min(20, (x2 - x1) // 3)
                cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), (0, 255, 255), 3)
                cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), (0, 255, 255), 3)
                cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), (0, 255, 255), 3)
                cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), (0, 255, 255), 3)
                
                # Movement Direction Vector Arrow pointing towards stop line
                if draw_vectors:
                    target_y = min(stop_line_y, cy + int((y2 - y1) * 0.8))
                    cv2.arrowedLine(annotated, (cx, cy), (cx, target_y), (0, 255, 255), 3, tipLength=0.3)
                    cv2.circle(annotated, (cx, cy), 4, (0, 255, 255), -1)
                
                # Label tag
                tag = f"🚨 {det.class_name.upper()} ({det.confidence*100:.0f}%)"
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(annotated, (x1, max(0, y1 - th - 10)), (x1 + tw + 10, y1), (0, 0, 255), -1)
                cv2.putText(annotated, tag, (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                # Standard vehicle box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                tag = f"{det.class_name.capitalize()} {det.confidence*100:.0f}%"
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(annotated, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), (20, 20, 20), -1)
                cv2.putText(annotated, tag, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)

        return annotated
