# evaluate.py
from pathlib import Path
from ultralytics import YOLO

def evaluate_test_set() -> None:
    """Evaluates the trained YOLO model on the test dataset split and generates visual predictions."""
    project_root = Path(__file__).resolve().parent
    
    # 1. Point to your finalized weights and your original data map
    weights_path = project_root / "runs" / "detect" / "models" / "helmet-detection-5" / "weights" / "best.pt"
    data_yaml_path = project_root / "data" / "Motorcycle-Helmet-Detection-2" / "data.yaml"
    
    # 2. Point to the physical images for the visual prediction phase
    test_images_path = project_root / "data" / "Motorcycle-Helmet-Detection-2" / "test" / "images"

    print(f"Loading weights from: {weights_path.name}")
    model = YOLO(str(weights_path))

    # Phase 1: Quantitative Metrics (The Math)
    print("\n--- Calculating final metrics on the test split ---")
    metrics = model.val(
        data=str(data_yaml_path),
        split="test",               # Instructs YOLO to evaluate the test set specifically
        project="runs/detect",
        name="test_evaluation_metrics"
    )
    print(f"Final Test mAP50: {metrics.box.map50:.3f}")

    # Phase 2: Qualitative Visuals (The Boxes)
    print("\n--- Drawing bounding boxes on test images ---")
    model.predict(
        source=str(test_images_path),
        conf=0.5,
        save=True,                  # Saves annotated images to disk
        project="runs/detect",
        name="test_evaluation_visuals"
    )

if __name__ == "__main__":
    evaluate_test_set()