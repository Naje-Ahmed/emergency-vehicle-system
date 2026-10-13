"""
Dynamic Priority Engine & Adaptive Signal State Machine for AegisFlow.
Computes Urgency Index, evaluates lane preemption, manages signal state transitions,
and handles multi-lane emergency conflict resolution.
"""

import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from src.tracker import TrackedVehicle

@dataclass
class SignalPhase:
    lane_id: str
    state: str  # 'RED', 'YELLOW', 'GREEN'
    timer_sec: float
    is_preempted: bool = False

class TrafficSignalController:
    # State Machine Constants
    STATE_NORMAL_CYCLE = "NORMAL_CYCLE"
    STATE_PREEMPTION_PENDING = "PREEMPTION_PENDING"
    STATE_GREEN_WAVE = "GREEN_WAVE_OVERRIDE"
    STATE_CLEARANCE_HOLD = "CLEARANCE_HOLD"

    def __init__(
        self,
        urgency_threshold: float = 0.65,
        yellow_duration: float = 2.5,
        clearance_hold_duration: float = 4.0,
        normal_green_duration: float = 12.0
    ):
        self.urgency_threshold = urgency_threshold
        self.yellow_duration = yellow_duration
        self.clearance_hold_duration = clearance_hold_duration
        self.normal_green_duration = normal_green_duration

        # Lanes: Lane 1 (North), Lane 2 (South), Lane 3 (East), Lane 4 (West)
        self.lanes = ["Lane 1 (North-Bound)", "Lane 2 (South-Bound)", "Lane 3 (East-Bound)", "Lane 4 (West-Bound)"]
        
        # State tracker
        self.current_state = self.STATE_NORMAL_CYCLE
        self.active_preempted_lane: Optional[str] = None
        self.active_emergency_vehicle: Optional[TrackedVehicle] = None
        
        # Signal Lights state for all 4 lanes
        self.signals: Dict[str, SignalPhase] = {
            lane: SignalPhase(lane_id=lane, state="RED", timer_sec=0.0) for lane in self.lanes
        }
        
        # Set default normal cycle: Lanes 1 & 2 GREEN, Lanes 3 & 4 RED
        self.normal_cycle_phase = 0  # 0: N-S Green, 1: N-S Yellow, 2: E-W Green, 3: E-W Yellow
        self.state_entered_time = time.time()
        self.preemption_start_time: Optional[float] = None
        self.last_logged_incident_id: Optional[int] = None
        
        # Metrics
        self.total_preemptions = 0
        self.current_urgency_index = 0.0

    def compute_urgency_index(
        self,
        vehicle: TrackedVehicle,
        frame_shape: Tuple[int, int],
        siren_detected: bool = False
    ) -> float:
        """
        Calculates Urgency Index U:
        U = w_type * conf * (1.0 + 0.4 * (v / v_max) + 0.6 * (area / total_area)) + siren_bonus
        """
        h, w = frame_shape[:2]
        total_frame_area = h * w
        current_area = vehicle.box[2] * vehicle.box[3] if len(vehicle.box) == 4 else (vehicle.box[2] - vehicle.box[0]) * (vehicle.box[3] - vehicle.box[1])
        area_ratio = min(1.0, current_area / (total_frame_area * 0.4 + 1e-5))
        
        # Velocity ratio normalized to max 300 px/s
        v_ratio = min(1.0, vehicle.velocity_px_s / 300.0)
        
        # Siren acoustic confirmation adds 0.15 boost
        siren_bonus = 0.15 if siren_detected else 0.0
        
        base_urgency = vehicle.priority_weight * vehicle.confidence
        proximity_factor = 1.0 + (0.4 * v_ratio) + (0.6 * area_ratio)
        
        urgency = (base_urgency * proximity_factor) + siren_bonus
        return round(float(min(1.0, urgency)), 3)

    def evaluate_priority(
        self,
        tracked_vehicles: List[TrackedVehicle],
        frame_shape: Tuple[int, int],
        siren_detected: bool = False,
        now: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Main cycle execution: updates urgency, checks preemption triggers,
        and switches signal phases safely.
        """
        current_time = now if now is not None else time.time()
        time_in_state = current_time - self.state_entered_time
        
        # Filter for approaching emergency vehicles
        emergency_candidates = [
            v for v in tracked_vehicles if v.is_emergency and v.approaching_intersection
        ]
        
        # Fallback: any emergency vehicle in frame
        if not emergency_candidates:
            emergency_candidates = [v for v in tracked_vehicles if v.is_emergency]
            
        top_urgency = 0.0
        top_vehicle = None
        
        if emergency_candidates:
            # Score each candidate and resolve conflicts
            scored_candidates = []
            for v in emergency_candidates:
                score = self.compute_urgency_index(v, frame_shape, siren_detected)
                scored_candidates.append((score, v))
                
            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            top_urgency, top_vehicle = scored_candidates[0]
            
        self.current_urgency_index = top_urgency

        # --- State Machine Transitions ---
        if self.current_state == self.STATE_NORMAL_CYCLE:
            if top_urgency >= self.urgency_threshold and top_vehicle is not None:
                # Emergency detected! Check if target lane is already green
                target_lane = top_vehicle.lane_id
                target_is_green = (
                    (self.normal_cycle_phase == 0 and ("Lane 1" in target_lane or "Lane 2" in target_lane)) or
                    (self.normal_cycle_phase == 2 and ("Lane 3" in target_lane or "Lane 4" in target_lane))
                )
                
                self.active_emergency_vehicle = top_vehicle
                self.active_preempted_lane = target_lane
                self.preemption_start_time = current_time
                self.total_preemptions += 1
                
                if target_is_green:
                    # Switch immediately to Green Wave
                    self.current_state = self.STATE_GREEN_WAVE
                    self.state_entered_time = current_time
                else:
                    # Need transition yellow on conflicting lanes
                    self.current_state = self.STATE_PREEMPTION_PENDING
                    self.state_entered_time = current_time
            else:
                # Normal cyclic progression
                self._update_normal_cycle(time_in_state, current_time)

        elif self.current_state == self.STATE_PREEMPTION_PENDING:
            # Show Yellow on currently green non-emergency lanes
            for lane in self.lanes:
                if lane != self.active_preempted_lane:
                    self.signals[lane].state = "YELLOW"
                    self.signals[lane].is_preempted = False
                else:
                    self.signals[lane].state = "RED"  # Wait for yellow clearance
                    
            if time_in_state >= self.yellow_duration:
                # Switch to Green Wave Corridor
                self.current_state = self.STATE_GREEN_WAVE
                self.state_entered_time = current_time

        elif self.current_state == self.STATE_GREEN_WAVE:
            # Target approach is forced GREEN, all other approaches RED
            for lane in self.lanes:
                if lane == self.active_preempted_lane:
                    self.signals[lane].state = "GREEN"
                    self.signals[lane].is_preempted = True
                else:
                    self.signals[lane].state = "RED"
                    self.signals[lane].is_preempted = False
                    
            # Check if emergency vehicle has cleared or is no longer tracked
            vehicle_still_present = any(
                v.track_id == self.active_emergency_vehicle.track_id for v in tracked_vehicles
            ) if self.active_emergency_vehicle else False
            
            # If vehicle passed stop-line or disappeared, enter Clearance Hold
            if (not vehicle_still_present and time_in_state >= 3.0) or time_in_state > 25.0:
                self.current_state = self.STATE_CLEARANCE_HOLD
                self.state_entered_time = current_time

        elif self.current_state == self.STATE_CLEARANCE_HOLD:
            # Keep green for a few safety seconds to allow complete intersection traversal
            if time_in_state >= self.clearance_hold_duration:
                # Return to Normal Cycle
                self.current_state = self.STATE_NORMAL_CYCLE
                self.state_entered_time = current_time
                self.active_preempted_lane = None
                self.active_emergency_vehicle = None

        return {
            "state": self.current_state,
            "preempted_lane": self.active_preempted_lane,
            "urgency_index": self.current_urgency_index,
            "target_vehicle": self.active_emergency_vehicle,
            "signals": {k: v.state for k, v in self.signals.items()},
            "time_in_state": round(time_in_state, 1)
        }

    def _update_normal_cycle(self, time_in_state: float, current_time: float):
        """Standard 4-phase signal timing (N-S Green -> N-S Yellow -> E-W Green -> E-W Yellow)."""
        if self.normal_cycle_phase == 0:  # N-S Green
            self.signals["Lane 1 (North-Bound)"].state = "GREEN"
            self.signals["Lane 2 (South-Bound)"].state = "GREEN"
            self.signals["Lane 3 (East-Bound)"].state = "RED"
            self.signals["Lane 4 (West-Bound)"].state = "RED"
            if time_in_state >= self.normal_green_duration:
                self.normal_cycle_phase = 1
                self.state_entered_time = current_time

        elif self.normal_cycle_phase == 1:  # N-S Yellow
            self.signals["Lane 1 (North-Bound)"].state = "YELLOW"
            self.signals["Lane 2 (South-Bound)"].state = "YELLOW"
            self.signals["Lane 3 (East-Bound)"].state = "RED"
            self.signals["Lane 4 (West-Bound)"].state = "RED"
            if time_in_state >= self.yellow_duration:
                self.normal_cycle_phase = 2
                self.state_entered_time = current_time

        elif self.normal_cycle_phase == 2:  # E-W Green
            self.signals["Lane 1 (North-Bound)"].state = "RED"
            self.signals["Lane 2 (South-Bound)"].state = "RED"
            self.signals["Lane 3 (East-Bound)"].state = "GREEN"
            self.signals["Lane 4 (West-Bound)"].state = "GREEN"
            if time_in_state >= self.normal_green_duration:
                self.normal_cycle_phase = 3
                self.state_entered_time = current_time

        elif self.normal_cycle_phase == 3:  # E-W Yellow
            self.signals["Lane 1 (North-Bound)"].state = "RED"
            self.signals["Lane 2 (South-Bound)"].state = "RED"
            self.signals["Lane 3 (East-Bound)"].state = "YELLOW"
            self.signals["Lane 4 (West-Bound)"].state = "YELLOW"
            if time_in_state >= self.yellow_duration:
                self.normal_cycle_phase = 0
                self.state_entered_time = current_time
