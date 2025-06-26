import os
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

def download_and_extract():
    dest = "data/toxic"
    os.makedirs(dest, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    print("📥 Downloading dataset...")
    api.competition_download_files(
        'jigsaw-toxic-comment-classification-challenge',
        path=dest
    )

    zip_path = os.path.join(dest, "jigsaw-toxic-comment-classification-challenge.zip")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest)

    print(f"✅ Extracted to {dest}")
