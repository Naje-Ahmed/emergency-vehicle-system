"""
Vehicle Tracking and Approach Vector Estimation for AegisFlow.
Tracks detected vehicles frame-to-frame, computes velocity vectors,
estimates distance/proximity to intersection stop-lines, and calculates growth rate.
"""

import time
import math
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from src.detector import Detection

@dataclass
class TrackedVehicle:
    track_id: int
    class_name: str
    is_emergency: bool
    priority_weight: float
    box: Tuple[int, int, int, int]
    center: Tuple[int, int]
    confidence: float
    trajectory: List[Tuple[int, int]] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    areas: List[int] = field(default_factory=list)
    
    velocity_px_s: float = 0.0
    approach_vector: Tuple[float, float] = (0.0, 0.0)
    approaching_intersection: bool = False
    area_growth_rate: float = 0.0
    distance_to_stopline_px: float = 9999.0
    lane_id: str = "Lane 1"
    last_updated: float = field(default_factory=time.time)
    missed_frames: int = 0

class VehicleTracker:
    def __init__(self, max_distance_threshold: float = 85.0, max_missed_frames: int = 10):
        self.max_distance_threshold = max_distance_threshold
        self.max_missed_frames = max_missed_frames
        self.next_track_id = 1
        self.tracks: Dict[int, TrackedVehicle] = {}

    def _calculate_iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """Calculates Intersection over Union between two bounding boxes."""
        x1_a, y1_a, x2_a, y2_a = box1
        x1_b, y1_b, x2_b, y2_b = box2
        
        xi1 = max(x1_a, x1_b)
        yi1 = max(y1_a, y1_b)
        xi2 = min(x2_a, x2_b)
        yi2 = min(y2_a, y2_b)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        area_a = (x2_a - x1_a) * (y2_a - y1_a)
        area_b = (x2_b - x1_b) * (y2_b - y1_b)
        
        union_area = area_a + area_b - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def _assign_lane(self, center: Tuple[int, int], frame_shape: Tuple[int, int]) -> str:
        """Assigns approach lane based on center position in frame."""
        h, w = frame_shape[:2]
        cx, cy = center
        
        # 4-Lane quadrant mapping or 2-lane roadway
        if cx < w * 0.5:
            return "Lane 1 (North-Bound)" if cy < h * 0.5 else "Lane 2 (South-Bound)"
        else:
            return "Lane 3 (East-Bound)" if cy < h * 0.5 else "Lane 4 (West-Bound)"

    def update(
        self,
        detections: List[Detection],
        frame_shape: Tuple[int, int],
        stopline_y: Optional[int] = None,
        current_time: Optional[float] = None
    ) -> List[TrackedVehicle]:
        """
        Updates tracked vehicles with current frame detections.
        """
        now = current_time if current_time is not None else time.time()
        h, w = frame_shape[:2]
        ref_stopline = stopline_y if stopline_y is not None else int(h * 0.75)
        
        # Match current tracks with incoming detections
        track_ids = list(self.tracks.keys())
        assigned_tracks = set()
        assigned_dets = set()
        
        # 1. First pass: Match by IOU & Euclidean distance
        for det_idx, det in enumerate(detections):
            best_track_id = None
            best_score = float('inf')
            
            for tid in track_ids:
                if tid in assigned_tracks:
                    continue
                track = self.tracks[tid]
                
                # Distance between centers
                dist = math.hypot(det.center[0] - track.center[0], det.center[1] - track.center[1])
                iou = self._calculate_iou(det.box, track.box)
                
                # Score combines distance and IOU
                match_score = dist - (iou * 100.0)
                
                if dist < self.max_distance_threshold and match_score < best_score:
                    best_score = match_score
                    best_track_id = tid
            
            if best_track_id is not None:
                assigned_tracks.add(best_track_id)
                assigned_dets.add(det_idx)
                
                # Update existing track
                t = self.tracks[best_track_id]
                t.box = det.box
                t.center = det.center
                t.confidence = det.confidence
                t.is_emergency = det.is_emergency or t.is_emergency
                t.priority_weight = max(t.priority_weight, det.priority_weight)
                t.class_name = det.class_name if det.is_emergency else t.class_name
                t.trajectory.append(det.center)
                t.timestamps.append(now)
                t.areas.append(det.area)
                t.missed_frames = 0
                t.last_updated = now
                t.lane_id = self._assign_lane(det.center, frame_shape)
                
                # Trim histories to last 20 frames
                if len(t.trajectory) > 20:
                    t.trajectory = t.trajectory[-20:]
                    t.timestamps = t.timestamps[-20:]
                    t.areas = t.areas[-20:]
                
                # Velocity & vector estimation
                if len(t.trajectory) >= 3:
                    dt = t.timestamps[-1] - t.timestamps[0]
                    if dt > 0.001:
                        dx = t.trajectory[-1][0] - t.trajectory[0][0]
                        dy = t.trajectory[-1][1] - t.trajectory[0][1]
                        dist_moved = math.hypot(dx, dy)
                        t.velocity_px_s = dist_moved / dt
                        t.approach_vector = (dx / dt, dy / dt)
                        
                        # Approaching intersection if moving downward/towards stopline
                        t.approaching_intersection = dy > 2.0 or (abs(dy) > 1.0 and t.center[1] < ref_stopline)
                        
                        # Area growth rate (scale change indicates approaching camera)
                        d_area = t.areas[-1] - t.areas[0]
                        t.area_growth_rate = d_area / (t.areas[0] * dt + 1e-5)
                
                # Distance to stop line
                t.distance_to_stopline_px = max(0.0, float(ref_stopline - t.center[1]))
        
        # 2. Add new tracks for unassigned detections
        for det_idx, det in enumerate(detections):
            if det_idx not in assigned_dets:
                tid = self.next_track_id
                self.next_track_id += 1
                
                lane = self._assign_lane(det.center, frame_shape)
                dist_stop = max(0.0, float(ref_stopline - det.center[1]))
                
                new_track = TrackedVehicle(
                    track_id=tid,
                    class_name=det.class_name,
                    is_emergency=det.is_emergency,
                    priority_weight=det.priority_weight,
                    box=det.box,
                    center=det.center,
                    confidence=det.confidence,
                    trajectory=[det.center],
                    timestamps=[now],
                    areas=[det.area],
                    velocity_px_s=0.0,
                    distance_to_stopline_px=dist_stop,
                    lane_id=lane,
                    last_updated=now,
                    missed_frames=0
                )
                self.tracks[tid] = new_track
        
        # 3. Handle unassigned existing tracks
        expired_ids = []
        for tid, track in self.tracks.items():
            if tid not in assigned_tracks:
                track.missed_frames += 1
                if track.missed_frames > self.max_missed_frames:
                    expired_ids.append(tid)
        
        for tid in expired_ids:
            del self.tracks[tid]
            
        return list(self.tracks.values())
