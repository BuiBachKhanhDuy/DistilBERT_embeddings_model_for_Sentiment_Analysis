# Sentiment Analysis with Context using DistilBERT and Transformers

This project implements a **sentiment analysis model** that classifies text into **Negative, Neutral, or Positive** sentiment.  
It supports two input streams:  
- **Main Text** (e.g., a review, post, or comment)  
- **Context** (surrounding text, metadata, or additional description)  

The model can be trained in different configurations:  
- **Pretrained DistilBERT (HuggingFace)** with or without cross-attention  
- **Custom Transformer Encoder (from scratch)** with or without cross-attention  


## 📂 Project Structure
```plaintext
├── data.py # Data loading, cleaning, tokenization
├── model.py # Model architectures (DistilBERT + Transformer Encoder)
├── train_eval.py # Training, evaluation, visualization, and prediction loop
├── requirements.txt # Dependencies
├── sentiment_data.csv (not included, provide your own dataset)
```
## ⚙️ Installation

### 1. Clone the repository
git clone https://github.com/yourusername/sentiment-analysis-model.git
cd sentiment-analysis-model

###2. Create a virtual environment (Python 3.8–3.11 recommended)
python -m venv env
env\Scripts\activate   # On Windows
source env/bin/activate  # On Linux/Mac

###3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

###4. 🛠️ Requirements

Main libraries:
- TensorFlow 2.12
- HuggingFace Transformers 4.30.2
- PyTorch (for tokenizer dependencies)
- scikit-learn, pandas, numpy, nltk, matplotlib

## 📊 Dataset

This repository includes a dataset file: **sentiment_data.csv** with **1000+ synthetic samples** generated using AI.  

Each row has:
- **text**: the main input sentence (e.g., a review, opinion, or statement)  
- **context**: additional information related to the text (e.g., category, situation, or metadata)  
- **label**: the sentiment class (Negative, Neutral, Positive)  

### Notes
- The dataset is **AI-generated** and intended for **educational and research purposes only**.  
- While it’s balanced across the 3 sentiment classes, performance on real-world data may vary.  
- You can replace this dataset with your own by keeping the same column format.

## Model Evaluation

![Accuracy over epochs](https://github.com/user-attachments/assets/6fd238f6-4f0b-4453-99c4-ea66f201cc01)

![Loss over epochs](https://github.com/user-attachments/assets/950c9836-ddb4-41e4-bc92-50e326a61a7b)

![Model Comparison](https://github.com/user-attachments/assets/ff0426e0-decf-4977-bf4e-9d95156d15e6)
