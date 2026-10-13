"""
YOLOv8 Emergency Vehicle Fine-Tuning & Training Pipeline for AegisFlow.
Executes end-to-end dataset generation, training, validation, and weight deployment.
"""

import os
import sys
import shutil
from ultralytics import YOLO

# Add root path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from models.prepare_dataset import build_full_dataset, DATASET_ROOT

def run_training(
    epochs: int = 10,
    img_size: int = 640,
    batch_size: int = 16,
    device: str = "cpu"
):
    """
    Runs end-to-end YOLOv8 fine-tuning on the emergency vehicle dataset.
    """
    yaml_path = os.path.join(DATASET_ROOT, "dataset.yaml")
    if not os.path.exists(yaml_path):
        print("[Train] Dataset not detected. Generating fresh annotated dataset...")
        yaml_path = build_full_dataset(num_train=120, num_val=30)
        
    print(f"\n=======================================================")
    print(f"🚀 Launching AegisFlow YOLOv8 Emergency Vehicle Training")
    print(f"   Epochs: {epochs} | Image Size: {img_size} | Device: {device}")
    print(f"   Dataset Config: {yaml_path}")
    print(f"=======================================================\n")
    
    # Load base weights
    model = YOLO("yolov8n.pt")
    
    # Run training
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        project=os.path.join(ROOT_DIR, "models", "runs"),
        name="emergency_yolov8",
        exist_ok=True,
        plots=True,
        save=True,
        verbose=True
    )
    
    # Find and copy best.pt
    best_weights_src = os.path.join(ROOT_DIR, "models", "runs", "emergency_yolov8", "weights", "best.pt")
    dest_weights = os.path.join(ROOT_DIR, "models", "weights", "best.pt")
    
    if os.path.exists(best_weights_src):
        os.makedirs(os.path.dirname(dest_weights), exist_ok=True)
        shutil.copy(best_weights_src, dest_weights)
        print(f"\n✅ Training Successful! Deployed best weights to: {dest_weights}")
    else:
        print(f"[Train] Model saved inside runs directory.")
        
    return results

if __name__ == "__main__":
    run_training(epochs=10, img_size=640, batch_size=16)
