# inference.py
from pathlib import Path
from ultralytics import YOLO

def test_model():
    # 1. Point to the EXACT nested path where YOLO saved your weights
    project_root = Path(__file__).resolve().parent
   
   
    weights_path = (
        project_root / "runs" / "detect" / "models" / "helmet-detection-5" / "weights" / "best.pt"
    )
    
    # 2. Point to the folder where you put your personal images
    source_images = project_root / "test_images"
    

    # 3. Load your custom-trained model
    print(f"Loading custom weights from: {weights_path.name}")
    model = YOLO(str(weights_path))

    # 4. Run prediction on your personal images folder
    print(f"Predicting on images in: {source_images}")
    model.predict(
        source=str(source_images),
        conf=0.4,           # Lowered slightly to 40% to catch more helmets on your first test
        save=True,          # Saves the annotated image
        project="detect/", # Saves cleanly inside your existing runs folder
        name="my_personal_tests" 
    )
    
    print("Inference complete! Look inside 'runs/detect/my_personal_tests/' to see the results!")

if __name__ == "__main__":
    test_model()