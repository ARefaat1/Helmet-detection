Here is a complete, professional README.md file tailored specifically to your project’s architecture, configuration layout, and the environment constraints you discovered during testing. This is structured exactly like a production-grade machine learning repository for a software engineering portfolio.Helmet Detector AI (YOLOv26)An applied machine learning pipeline built to detect motorcycle riders and classify helmet usage directly from elevated traffic camera streams. The project utilizes the Ultralytics YOLOv26 object detection framework, managed via a modern, reproducible uv Python workspace.📌 Project Overview & Operational ConstraintsUnlike standard classification models that require manual image cropping, this pipeline operates as an End-to-End Object Detection System. It simultaneously handles bounding box localization ($x, y, w, h$) and multi-class classification (nc=4, identifying specific attributes like full-faced helmets) directly from full-resolution frames.⚠️ Important: Environment & Distribution ShiftTarget Environment: This model is trained explicitly on elevated, wide-angle CCTV/security intersection footage. It expects a high-angle perspective where motorcycles are small-to-medium objects within a wider traffic flow.Known Limitations: Due to Distribution Shift, the model will exhibit low confidence or fail entirely on close-up images (e.g., smartphone photos taken from a 2-meter distance) or highly contrasted close-up night shots. For maximum performance, deployment feeds must match the baseline CCTV viewpoint.📂 Project ArchitectureThe repository enforces a clean separation of concerns between raw data, environment configurations, reusable utility code, and execution entry points:Plaintexthelmet_detection/
├── config/
│   └── train_config.yaml         # Centralized hyperparameters dashboard
├── data/
│   └── dataset/                  # Native YOLO formatted data split
│       ├── train/                # Training images & bounding box labels
│       ├── valid/                # Validation / evaluation holdout
│       ├── test/                 # Test set for post-training verification
│       └── data.yaml             # Master dataset mapping (classes, paths)
├── runs/
│   └── detect/                   # Training artifacts & serialized weights (.pt)
├── src/
│   ├── __init__.py               # Marks directory as a python package
│   └── utils.py                  # Standalone helper functions and metrics
├── train.py                      # Training pipeline execution engine
├── inference.py                  # Production inference/prediction script
└── pyproject.toml                # Managed dependency manifest (PyTorch CUDA locked)





🛠️ Setup & InstallationThis project utilizes uv for blazing-fast package management and deterministic environment synchronization. It is locked to PyTorch with CUDA local GPU acceleration.Clone the Repository:Bashgit clone https://github.com/yourusername/helmet-detector-ai.git
cd helmet-detector-ai
Synchronize the Virtual Environment:uv will automatically read the pyproject.toml file, set up a local virtual environment, and fetch the correct GPU-enabled PyTorch wheels directly from the isolated server index:Bashuv sync
🚀 How to Run1. Training the ModelHyperparameters are separated from the code logic. If you need to tweak the initial learning rate (lr0), change the batch size, or adjust real-world data augmentations (like shifting brightness hsv_v or enabling mosaic), edit config/train_config.yaml.To launch the training pipeline on your local GPU:Bashuv run train.py
Note for Windows users: The configuration defaults to workers: 0 inside the training profile to prevent multi-threaded file-locking conflicts (labels.cache errors).2. Running Inference on Personal ImagesDrop your test images (CCTV or wide-angle street shots) into a folder named test_images/ in the project root, then run the inference engine to generate predictions:Bashuv run inference.py
The model will load your finalized mathematical weights (best.pt) from your latest run, parse the images, and output annotated frames with tight bounding boxes and confidence parameters directly to runs/detect/my_personal_tests/.📊 Sample Inference ResultsThe model demonstrates excellent bounding box localization and high confidence thresholds when processing native high-angle street configurations, gracefully filtering out stationary background vehicles (such as empty, parked scooters) to minimize industrial false-alarm triggers.