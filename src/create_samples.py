"""
Utility script to generate sample video and audio assets for testing AegisFlow.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.stream_generator import TrafficStreamGenerator
from src.audio_analyzer import SirenAudioAnalyzer
import scipy.io.wavfile as wav

def generate_all_samples():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    samples_dir = os.path.join(base_dir, "data", "samples")
    audio_dir = os.path.join(base_dir, "data", "audio")
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    print("Generating synthetic CCTV sample video...")
    gen = TrafficStreamGenerator(width=960, height=540, fps=20)
    video_path = os.path.join(samples_dir, "cctv_ambulance_approach.mp4")
    gen.export_sample_video(video_path, num_seconds=12)

    print("Generating synthetic siren audio WAV file...")
    audio_gen = SirenAudioAnalyzer(sample_rate=22050)
    siren_samples = audio_gen.generate_simulated_siren_signal(duration_sec=6.0, is_siren=True)
    audio_path = os.path.join(audio_dir, "emergency_siren_sample.wav")
    wav.write(audio_path, 22050, siren_samples)
    print(f"Sample audio saved to {audio_path}")

if __name__ == "__main__":
    generate_all_samples()
