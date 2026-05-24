import os
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(ROOT_DIR, "data"))
MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(ROOT_DIR, "models"))
REGISTRY_PATH = os.getenv("REGISTRY_PATH", os.path.join(ROOT_DIR, "registry.json"))



