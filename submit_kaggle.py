from kaggle.api.kaggle_api_extended import KaggleApi

def submit_to_kaggle(file_path: str, message: str):
    api = KaggleApi()
    api.authenticate()

    print("🚀 Submitting to Kaggle...")
    api.competition_submit(
        file_name=file_path,
        message=message,
        competition='jigsaw-toxic-comment-classification-challenge'
    )
    print("✅ Submission uploaded successfully!")