from roboflow import Roboflow
import os

# Read the Roboflow API key from the environment
roboflow_api = os.environ.get("ROBOFLOW_API_KEY")
if not roboflow_api:
	raise RuntimeError("Environment variable ROBOFLOW_API_KEY is not set")

# Download dataset into the data/ directory
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
os.makedirs(data_dir, exist_ok=True)

rf = Roboflow(api_key=roboflow_api)
project = rf.workspace("data-science-173-msp9r").project("motorcycle-helmet-detection")
version = project.version(2)
dataset = version.download("yolo26", location=data_dir)