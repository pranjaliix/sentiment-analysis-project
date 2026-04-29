from huggingface_hub import HfApi
from dotenv import load_dotenv
import os

# Load variables from .env
load_dotenv()

hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError("HF_TOKEN not found in environment variables.")

api = HfApi()
api.upload_folder(
    folder_path="model",
    path_in_repo="",
    repo_id="Pranjalii/sentiment-analysis-bert",
    repo_type="model",
    token=hf_token
)

print("Upload complete! Check your Hugging Face repo online.")