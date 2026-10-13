"""
Dual-Factor Siren Acoustic Analyzer for AegisFlow.
Performs FFT spectral and harmonic analysis to detect acoustic emergency siren signals
(e.g., European Yelp/Wail, US Hi-Lo sirens in 700Hz - 1600Hz range).
"""

import numpy as np
from scipy.signal import find_peaks
from typing import Tuple, Optional, Dict, Any

class SirenAudioAnalyzer:
    def __init__(self, sample_rate: int = 22050, siren_freq_band: Tuple[int, int] = (650, 1650)):
        self.sample_rate = sample_rate
        self.siren_freq_band = siren_freq_band
        self.history_scores = []
        
    def analyze_audio_chunk(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """
        Analyzes an audio time series chunk and determines the siren probability.
        """
        if audio_data is None or len(audio_data) < 256:
            return {
                "siren_detected": False,
                "confidence": 0.0,
                "peak_frequency_hz": 0.0,
                "spectral_energy_ratio": 0.0
            }
            
        # If stereo, average to mono
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
            
        # Apply Hanning window
        window = np.hanning(len(audio_data))
        windowed = audio_data * window
        
        # FFT computation
        fft_vals = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(len(windowed), d=1.0 / self.sample_rate)
        
        # Band filtering for siren range (650 - 1650 Hz)
        in_band_mask = (freqs >= self.siren_freq_band[0]) & (freqs <= self.siren_freq_band[1])
        total_energy = np.sum(fft_vals ** 2) + 1e-9
        siren_energy = np.sum(fft_vals[in_band_mask] ** 2)
        
        energy_ratio = float(siren_energy / total_energy)
        
        # Find dominant peak within the siren band
        band_fft = fft_vals.copy()
        band_fft[~in_band_mask] = 0
        peak_idx = np.argmax(band_fft)
        peak_freq = float(freqs[peak_idx]) if np.max(band_fft) > 0 else 0.0
        
        # Peaks analysis
        peaks, properties = find_peaks(band_fft, height=np.max(band_fft) * 0.4 if np.max(band_fft) > 0 else 0)
        has_prominent_peaks = len(peaks) >= 1 and len(peaks) <= 6
        
        # Confidence calculation
        confidence = min(1.0, max(0.0, (energy_ratio * 1.8) if has_prominent_peaks else energy_ratio * 0.8))
        siren_detected = confidence >= 0.45 and peak_freq >= self.siren_freq_band[0]
        
        return {
            "siren_detected": siren_detected,
            "confidence": round(float(confidence), 3),
            "peak_frequency_hz": round(peak_freq, 1),
            "spectral_energy_ratio": round(energy_ratio, 3)
        }

    def generate_simulated_siren_signal(self, duration_sec: float = 1.0, is_siren: bool = True) -> np.ndarray:
        """Generates synthetic siren audio (wailing frequency modulation) or background city noise."""
        t = np.linspace(0, duration_sec, int(self.sample_rate * duration_sec), endpoint=False)
        if is_siren:
            # Frequency modulated siren (sweeping between 700Hz and 1400Hz at 0.5Hz rate)
            mod_freq = 0.8
            f_inst = 1000 + 350 * np.sin(2 * np.pi * mod_freq * t)
            phase = 2 * np.pi * np.cumsum(f_inst) / self.sample_rate
            siren_wave = 0.7 * np.sin(phase) + 0.2 * np.sin(2 * phase)
            noise = 0.1 * np.random.normal(0, 1, len(t))
            return (siren_wave + noise).astype(np.float32)
        else:
            # Ambient urban noise (low-frequency rumble + white noise)
            low_rumble = 0.4 * np.sin(2 * np.pi * 120 * t) + 0.3 * np.sin(2 * np.pi * 240 * t)
            noise = 0.3 * np.random.normal(0, 1, len(t))
            return (low_rumble + noise).astype(np.float32)
