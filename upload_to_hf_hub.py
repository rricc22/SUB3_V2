#!/usr/bin/env python3
"""
Script to upload model to Hugging Face Hub (OPTION 1)

Author: Riccardo
Date: 2026-01-14
"""

from huggingface_hub import HfApi, create_repo
import os

# Initialize API
api = HfApi()

# Repository details
repo_id = "rricc22/heart-rate-prediction-lstm"
repo_type = "model"

print("=" * 60)
print("OPTION 1: Upload Model to Hugging Face Hub")
print("=" * 60)

# Try to create repository (will skip if exists)
try:
    print(f"\n1. Creating repository: {repo_id}")
    create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)
    print("   ✓ Repository created/exists")
except Exception as e:
    print(f"   ! Note: {e}")
    print("   ! You may need to create the repository manually at:")
    print(f"   ! https://huggingface.co/new?repo_type=model&repo_name=heart-rate-prediction-lstm")
    create_manually = input("\n   Press Enter after creating the repository manually, or 'q' to quit: ")
    if create_manually.lower() == 'q':
        exit(0)

# Files to upload
files_to_upload = [
    ("Model/checkpoints/best_model.pt", "best_model.pt"),
    ("Model/lstm.py", "lstm.py"),
    ("Model/loss.py", "loss.py"),
    ("Model/feature_engineering.py", "feature_engineering.py"),
    ("HF_MODEL_CARD.md", "README.md"),
    ("requirements_hf.txt", "requirements.txt"),
]

print(f"\n2. Uploading files to {repo_id}...")
for local_path, remote_path in files_to_upload:
    try:
        full_path = os.path.join("/home/riccardo/Documents/SUB3_V2", local_path)
        if os.path.exists(full_path):
            print(f"   Uploading: {local_path} -> {remote_path}")
            api.upload_file(
                path_or_fileobj=full_path,
                path_in_repo=remote_path,
                repo_id=repo_id,
                repo_type=repo_type
            )
            print(f"   ✓ Uploaded {remote_path}")
        else:
            print(f"   ! Skipped {local_path} (not found)")
    except Exception as e:
        print(f"   ✗ Error uploading {local_path}: {e}")

print("\n" + "=" * 60)
print("UPLOAD COMPLETE!")
print("=" * 60)
print(f"\nYour model is now available at:")
print(f"https://huggingface.co/{repo_id}")
print("\nUsers can download it with:")
print(f"""
from huggingface_hub import hf_hub_download

checkpoint_path = hf_hub_download(
    repo_id="{repo_id}",
    filename="best_model.pt"
)
""")
