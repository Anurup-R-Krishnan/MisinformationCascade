"""Root submission entrypoint for OpenEnv hackathon evaluation.

Environment variables used by this baseline:
  API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
  MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
  HF_TOKEN     = os.getenv("HF_TOKEN")
  LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # optional
"""

import os

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

from misinformation_cascade_env.inference import main


if __name__ == "__main__":
    main()
