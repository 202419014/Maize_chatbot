from huggingface_hub import HfApi, login

login(token="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")  # ← paste your token here

api = HfApi()

api.upload_file(
    path_or_fileobj="D:/genai/models/maize_check_model_fixed.pth",
    path_in_repo="maize_check_model.pth",
    repo_id="jhaanasreya/maize-advisory-models",  # ← change this
    repo_type="model"
)

api.upload_file(
    path_or_fileobj="D:/genai/models/maize_disease_model_fixed.pth",
    path_in_repo="maize_disease_model.pth",
    repo_id="jhaanasreya/maize-advisory-models",  # ← change this
    repo_type="model"
)

api.upload_file(
    path_or_fileobj="D:/genai/models/maize_pest_model_fixed.pth",
    path_in_repo="maize_pest_model.pth",
    repo_id="jhaanasreya/maize-advisory-models",  # ← change this
    repo_type="model"
)

print("All models uploaded successfully!")




