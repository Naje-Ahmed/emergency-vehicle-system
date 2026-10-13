"""
High-Fidelity CCTV Stream & Ambulance Arrival Simulator for AegisFlow.
Renders sharp, photo-realistic vehicles with realistic shadows, flashing emergency LED beacons,
reflective livery, and multiple arrival positions & camera angles.
"""

import cv2
import numpy as np
import os
import math
from typing import Tuple, Optional

class TrafficStreamGenerator:
    def __init__(self, width: int = 854, height: int = 480, fps: int = 20):
        self.width = width
        self.height = height
        self.fps = fps

    def _draw_realistic_road(self, frame: np.ndarray, cam_angle: str = "CAM-01 (Center North Approach)"):
        """Draws textured asphalt road with perspective lane markings, stop line, and street elements."""
        h, w = self.height, self.width
        
        # Ground surroundings (Grass/pavement)
        frame[:] = (32, 42, 32)
        
        # Asphalt roadway with subtle texture
        if "Left" in cam_angle:
            poly = np.array([[int(w*0.10), 0], [int(w*0.75), 0], [int(w*0.92), h], [int(w*0.02), h]], np.int32)
        elif "Right" in cam_angle:
            poly = np.array([[int(w*0.25), 0], [int(w*0.90), 0], [int(w*0.98), h], [int(w*0.08), h]], np.int32)
        else:
            poly = np.array([[int(w*0.16), 0], [int(w*0.84), 0], [int(w*0.98), h], [int(w*0.02), h]], np.int32)
            
        cv2.fillPoly(frame, [poly], (48, 50, 54))
        
        # Sidewalk curbs
        cv2.polylines(frame, [poly], isClosed=False, color=(140, 140, 145), thickness=3)
        
        # Lane Dividers
        cv2.line(frame, (int(w * 0.5), 0), (int(w * 0.5), h), (240, 240, 240), 2)
        
        # Dashed lane markings
        for y in range(10, h, 40):
            cv2.line(frame, (int(w * 0.33), y), (int(w * 0.33), y + 18), (220, 220, 220), 2)
            cv2.line(frame, (int(w * 0.67), y), (int(w * 0.67), y + 18), (220, 220, 220), 2)

        # Virtual Preemption Stop Line
        stop_y = int(h * 0.75)
        cv2.line(frame, (int(w * 0.05), stop_y), (int(w * 0.95), stop_y), (0, 0, 230), 3)
        cv2.putText(frame, "VIRTUAL PREEMPTION SENSOR LINE", (int(w * 0.08), stop_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
        
        # Zebra Crosswalk
        for x in range(int(w * 0.1), int(w * 0.9), 32):
            cv2.rectangle(frame, (x, stop_y + 8), (x + 18, stop_y + 24), (220, 220, 220), -1)

    def _draw_realistic_ambulance(self, frame: np.ndarray, cx: int, cy: int, scale: float, frame_idx: int):
        """
        Renders a crisp, highly identifiable ambulance:
        - White van body with realistic drop shadow
        - Bright red horizontal stripe and Star of Life / Red Cross
        - High-visibility fluorescent yellow-green chevron markings
        - Dark tinted windshield and rear windows
        - Alternating flashing Red & Blue high-intensity LED emergency roof strobes
        - Headlights casting subtle glow forward
        """
        bw = int(120 * scale)
        bh = int(175 * scale)
        x1 = cx - bw // 2
        y1 = cy - bh // 2
        x2 = cx + bw // 2
        y2 = cy + bh // 2
        
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(self.width - 1, x2); y2 = min(self.height - 1, y2)
        
        # 1. Realistic Vehicle Drop Shadow
        shadow_offset = int(10 * scale)
        shadow_pts = np.array([
            [x1 - 4, y2 - 4], [x2 + 4, y2 - 4],
            [x2 + 8, y2 + shadow_offset], [x1 - 8, y2 + shadow_offset]
        ], np.int32)
        cv2.fillPoly(frame, [shadow_pts], (20, 20, 22))

        # 2. Main White Body Chassis
        cv2.rectangle(frame, (x1, y1), (x2, y2), (248, 250, 252), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 45, 50), 2)
        
        # 3. Red Livery & Reflective Chevron Stripes
        mid_y = (y1 + y2) // 2
        s_h = max(4, int(12 * scale))
        cv2.rectangle(frame, (x1, mid_y - s_h), (x2, mid_y + s_h), (0, 0, 220), -1)
        
        # Yellow-Green Hi-Vis Chevron Accents
        ch_w = max(4, int(8 * scale))
        for sx in range(x1 + 4, x2 - ch_w, ch_w * 2):
            cv2.rectangle(frame, (sx, mid_y - s_h), (sx + ch_w, mid_y + s_h), (50, 220, 220), -1)

        # 4. Red Cross Emblem on Roof / Hood
        cross_r = max(4, int(10 * scale))
        ccx, ccy = cx, y1 + int(bh * 0.42)
        cv2.rectangle(frame, (ccx - cross_r, ccy - cross_r//3), (ccx + cross_r, ccy + cross_r//3), (0, 0, 220), -1)
        cv2.rectangle(frame, (ccx - cross_r//3, ccy - cross_r), (ccx + cross_r//3, ccy + cross_r), (0, 0, 220), -1)

        # 5. Windshield & Cab Windows
        ws_h = int(28 * scale)
        cv2.rectangle(frame, (x1 + int(8*scale), y2 - ws_h - int(8*scale)), (x2 - int(8*scale), y2 - int(8*scale)), (50, 45, 40), -1)
        
        # 6. Roof Emergency LED Light Bar (Alternating Red & Blue Strobe Flash)
        strobe_phase = (frame_idx // 2) % 2
        roof_y = y1 + int(14 * scale)
        bar_w = int(36 * scale)
        bar_h = max(5, int(10 * scale))
        
        # Light bar housing
        cv2.rectangle(frame, (cx - bar_w, roof_y - bar_h//2), (cx + bar_w, roof_y + bar_h//2), (30, 30, 35), -1)
        
        c_left = (0, 0, 255) if strobe_phase == 0 else (255, 120, 0)
        c_right = (255, 120, 0) if strobe_phase == 0 else (0, 0, 255)
        
        # Glowing LED lenses
        cv2.circle(frame, (cx - int(18*scale), roof_y), max(4, int(9*scale)), c_left, -1)
        cv2.circle(frame, (cx + int(18*scale), roof_y), max(4, int(9*scale)), c_right, -1)
        
        # Strobe glow halo around lights
        halo_r = max(8, int(18 * scale))
        halo_overlay = frame.copy()
        cv2.circle(halo_overlay, (cx - int(18*scale), roof_y), halo_r, c_left, -1)
        cv2.circle(halo_overlay, (cx + int(18*scale), roof_y), halo_r, c_right, -1)
        cv2.addWeighted(halo_overlay, 0.35, frame, 0.65, 0, frame)

        # 7. Front Headlight Beams (Glow onto roadway)
        if scale > 0.45:
            hl_y = y2 - 2
            cv2.circle(frame, (x1 + int(12*scale), hl_y), max(3, int(6*scale)), (200, 255, 255), -1)
            cv2.circle(frame, (x2 - int(12*scale), hl_y), max(3, int(6*scale)), (200, 255, 255), -1)

        # 8. Text Badge "AMBULANCE"
        if scale > 0.55:
            cv2.putText(frame, "AMBULANCE", (x1 + int(10*scale), y1 + int(bh * 0.22)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.38 * scale, (20, 20, 20), 1, cv2.LINE_AA)

    def _draw_civilian_car(self, frame: np.ndarray, cx: int, cy: int, scale: float, color: Tuple[int, int, int]):
        """Renders realistic civilian vehicle for traffic context."""
        bw = int(100 * scale)
        bh = int(150 * scale)
        x1 = max(0, cx - bw // 2); y1 = max(0, cy - bh // 2)
        x2 = min(self.width - 1, cx + bw // 2); y2 = min(self.height - 1, cy + bh // 2)
        
        # Shadow
        cv2.rectangle(frame, (x1 - 2, y2 - 2), (x2 + 2, y2 + int(6*scale)), (20, 20, 20), -1)
        # Body
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 30, 30), 2)
        # Windows
        cv2.rectangle(frame, (x1 + int(8*scale), y1 + int(24*scale)), (x2 - int(8*scale), y2 - int(24*scale)), (45, 45, 50), -1)

    def generate_frame(self, frame_idx: int, arrival_position: str = "Center Rapid Approach") -> np.ndarray:
        """
        Renders continuous CCTV surveillance frame with the Ambulance arriving
        at configurable lanes/trajectories.
        """
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self._draw_realistic_road(frame, arrival_position)
        
        cycle_len = 180 # 9 seconds per approach loop
        t = (frame_idx % cycle_len) / float(cycle_len)
        
        # 1. Background Civilian Traffic
        # Left Lane Civilian Car (Slow moving)
        t_car1 = (t * 0.75 + 0.3) % 1.0
        y_car1 = int(self.height * (0.05 + 0.85 * t_car1))
        s_car1 = 0.38 + 0.62 * (y_car1 / self.height)
        x_car1 = int(self.width * 0.26)
        self._draw_civilian_car(frame, x_car1, y_car1, s_car1, (180, 110, 40)) # Metallic Blue Car

        # Right Lane Civilian Car
        t_car2 = (t * 0.85 + 0.6) % 1.0
        y_car2 = int(self.height * (0.05 + 0.85 * t_car2))
        s_car2 = 0.38 + 0.62 * (y_car2 / self.height)
        x_car2 = int(self.width * 0.74)
        self._draw_civilian_car(frame, x_car2, y_car2, s_car2, (160, 160, 165)) # Silver Car

        # 2. Focus Target: Fast Approaching Ambulance
        # Computes dynamic position based on selected arrival mode
        t_amb = (t * 1.35) % 1.0
        y_amb = int(self.height * (0.05 + 0.90 * t_amb))
        s_amb = 0.40 + 0.70 * (y_amb / self.height)
        
        if "Left" in arrival_position or "Overtake" in arrival_position:
            # Swerving/overtaking on left lane
            x_amb = int(self.width * 0.32 + 20 * math.sin(t_amb * math.pi))
            cam_label = "CCTV-02 [WEST CORRIDOR - LEFT LANE BYPASS]"
        elif "Right" in arrival_position:
            # Inbound right lane
            x_amb = int(self.width * 0.66 - 15 * math.sin(t_amb * math.pi))
            cam_label = "CCTV-03 [EAST CORRIDOR - RIGHT INBOUND]"
        elif "Multi-Lane" in arrival_position:
            # Cycles between left, center, and right lane each cycle
            sub_c = (frame_idx // cycle_len) % 3
            lane_offsets = [0.50, 0.34, 0.66]
            x_amb = int(self.width * lane_offsets[sub_c])
            cam_label = f"CCTV-04 [DYNAMIC CORRIDOR - LANE {sub_c+1}]"
        else:
            # Center North Fast Corridor
            x_amb = int(self.width * 0.50)
            cam_label = "CCTV-01 [INTERSECTION NORTH - AMBULANCE CORRIDOR]"

        self._draw_realistic_ambulance(frame, x_amb, y_amb, s_amb, frame_idx)

        # 3. Live CCTV HUD Metadata
        cv2.putText(frame, f"{cam_label} | 20 FPS | REC ●", (18, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 240, 200), 1, cv2.LINE_AA)
        
        return frame

    def export_sample_video(self, output_path: str, num_seconds: int = 10, arrival_mode: str = "Center Rapid Approach"):
        """Exports high quality MP4 sample video."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        
        total_frames = self.fps * num_seconds
        for i in range(total_frames):
            frame = self.generate_frame(i, arrival_mode)
            out.write(frame)
            
        out.release()
        print(f"[StreamGenerator] Saved CCTV video: {output_path}")

if __name__ == "__main__":
    gen = TrafficStreamGenerator()
    gen.export_sample_video("data/samples/cctv_ambulance_approach.mp4", 10)
