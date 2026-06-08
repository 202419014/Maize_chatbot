#  Maize Chatbot  
> AI-Powered Disease & Pest Detection System with Multilingual Advisory - built with Streamlit, ResNet18, CLIP and Sentence Transformers

---

## What this bot does

The Maize Advisory Chatbot helps farmers and agronomists identify diseases and pests in maize crops, and get actionable advice in English or Hindi. Users upload a maize leaf image or ask questions via text or voice, and the system provides instant diagnosis and treatment recommendations.

**Key capabilities:**
- Maize leaf image upload — classifies disease or pest using ResNet18 deep learning models
- CLIP-based maize verification — rejects non-maize images before running classification
- Multilingual support — English and Hindi queries and answers
- Voice input — speak your question and get an answer
- Text-to-speech output — listen to the advice in your preferred language
- Semantic search over a curated maize advisory dataset using Sentence Transformers
- Common quick-question buttons for the most frequent farmer queries

---

## Quick Start


### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/maize-advisory-chatbot.git
cd maize-advisory-chatbot
```

### 2. Install Dependencies
Requires Python 3.9+. It is strongly recommended to use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### 3. Download or Train Models
Place the trained model files under the `models/` directory:
```
models/
├── maize_disease_model.pth
└── maize_pest_model.pth
```
To train the models yourself, run the training scripts (see Training section below).

### 4. Prepare the Dataset
Place `dataset.csv` in the root directory. It must contain the following columns:

| Column | Description |
|---|---|
| `English_Language`	| Question in English (used for semantic indexing) |
| `Answers_from_ChatGpt_in_english` |	English answer |
| `Hindi_Language_Answers` |	Hindi answer |
---

### 5. Run the App
```bash
streamlit run chatbot.py
```
The app will open automatically at http://localhost:8501
---

##Folder Structure

```
project-root/
│
├── chatbot.py                     # Streamlit entry point — UI, voice, Q&A
├── predict.py                     # CLIP maize check + ResNet18 inference
│
├── training/
│   ├── train_disease.py           # Train disease model (CPU, with augmentation)
│   ├── train_disease_model.py     # Alternate disease training script (GPU support)
│   ├── train_pest.py              # Train pest classification model
│   └── train_maize_check.py       # Train binary maize/non-maize classifier
│
├── models/
│   ├── maize_disease_model.pth    # Trained disease classifier (ResNet18, 4 classes)
│   └── maize_pest_model.pth       # Trained pest classifier (ResNet18, 4 classes)
│
├── dataset/
│   ├── maize_disease/             # Image dataset for disease training
│   ├── maize_pest/                # Image dataset for pest training
│   └── maize_check_balanced/      # Balanced dataset for maize/non-maize check
│
├── images/
│   └── maize_side.jpg             # Sidebar illustration
│
├── dataset.csv                    # Q&A advisory dataset (English + Hindi)
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```
---

## Tech Stack & Libraries

| Library | Purpose |
|---|---|
| `streamlit` |	Web UI framework |
| `torch` / `torchvision`	| Deep learning — ResNet18 training & inference |
| `transformers`  (HuggingFace)	| CLIP model for maize verification |
| `sentence-transformers`	| Semantic similarity for Q&A retrieval |
| `deep-translator`	| Hindi → English translation for search |
| `gTTS`	| Text-to-speech output |
| `SpeechRecognition`	| Voice input recognition |
| `Pillow` |	Image loading and preprocessing |
| `pandas`	| Dataset handling |
| `scikit-learn`	| Cosine similarity for semantic search |

## Classification Labels
> Disease model (4 classes):

| Label | Display Name |
|---|---|
| `blight` |	Blight |
| `grey_leaf_spot` |	Grey Leaf Spot |
| `healthy`	| Healthy |
| `rust` |	Rust |
---

> Pest model (4 classes):

| Label | Display Name |
|---|---|
| `aphids` |	Aphids |
| `corn_rootworm` |	Corn Rootworm |
| `fall_army_worm` |	Fall Army Worm |
| `stalk_borer`	| Stalk Borer |
---

## Training
> All training scripts are in the `training/` folder. Update the dataset paths before running.

## Train Disease Model
```bash
python training/train_disease.py
```
> Trains a ResNet18 model on 4 disease classes. Saves to `models/maize_disease_model.pth`.

## Train Pest Model
```bash
python training/train_pest.py
```
> Trains a ResNet18 model on 4 pest classes. Saves to `models/maize_pest_model.pth`.

### Train Maize Verification Model (Optional)
```bash
python training/train_maize_check.py
```
Trains a binary classifier (maize vs. non-maize). This is a fallback; the default `predict.py` uses CLIP for zero-shot maize verification.

## Training configuration:
| Parameter | Value |
|---|---|
| Architecture	| ResNet18 (pretrained on ImageNet) |
| Optimizer |	Adam (lr = 0.0001) |
| Loss |	CrossEntropyLoss |
| Epochs |	5 |
| Batch size | 16 |
| Input size | 	224 × 224 |
---

## How It Works

1. Image upload — user uploads a maize leaf photo
2. Maize check — CLIP scores the image against maize and non-maize text descriptors; non-maize images are rejected
3. Classification — both the disease and pest ResNet18 models run inference; the higher-confidence prediction wins
4. Auto-query — the detected label triggers an automatic advisory search (e.g. "How to control Rust in maize?")
5. Semantic search — the query (translated to English if Hindi) is encoded and matched against the dataset using cosine similarity
6. Answer display — the best-matching answer is shown in the selected language, with optional text-to-speech playback
---

##  Team Members

| Name | Role |
|---|---|
| D. Sushma | Developed image processing pipeline including maize detection (CLIP) and disease/pest prediction models. Integrated audio input, bilingual support, and handled overall system integration, testing, and final presentation.
| V. Jhaana Sreya | Developed chatbot interface using Streamlit, including language selection, Also implemented text-based query system using MiniLM and cosine similarity and contributed to report writing. |
| D. Sweta | Curated and prepared the synthetic dataset for chatbot responses, common questions, and user interaction (text input). Also handled PowerPoint presentation and part of the project report. |
---


## Notes
> The CLIP model (`openai/clip-vit-base-patch32`) is downloaded automatically from HuggingFace on first run (~600 MB).
> `train_disease_model.py` and `train_disease.py` are two versions of the disease training script. `train_disease.py` is the final version with augmentation and accuracy logging.
> Model `.pth` files are not included in this repository due to size. Train them locally or request access from the team.
---

## License
This project was developed as an academic submission. Image datasets are sourced from publicly available agricultural image repositories.

