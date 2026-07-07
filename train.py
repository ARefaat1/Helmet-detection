# train.py
import yaml
from pathlib import Path
from ultralytics import YOLO

def load_config():
    config_path = Path(__file__).resolve().parent / "config" / "train_config.yaml"
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def train_detector():
    # 1. Load your settings
    config = load_config()
    
    # 2. Setup Paths
    project_root = Path(__file__).resolve().parent
    data_yaml_path = project_root / "data" / "Motorcycle-Helmet-Detection-2" / "data.yaml"
    
    # 3. Initialize Model dynamically
    model = YOLO(config["model"]["name"])
    
    print(f"Starting training for {config['training']['epochs']} epochs...")
    
    # 4. Pass the config values into the train function
    model.train(
        data=str(data_yaml_path),
        epochs=config["training"]["epochs"],
        imgsz=config["model"]["img_size"],
        batch=config["training"]["batch_size"],
        patience=config["training"]["patience"],
        lr0=config["hyperparameters"]["lr0"],
        weight_decay=config["hyperparameters"]["weight_decay"],
        mosaic=config["hyperparameters"]["mosaic"],
        hsv_v=config["hyperparameters"]["hsv_v"],
        project="models",
        name="helmet-detection"
    )

if __name__ == "__main__":
    train_detector()