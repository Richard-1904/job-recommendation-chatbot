import torch
import pickle
import numpy as np
from torch import nn
from sklearn.base import BaseEstimator, ClassifierMixin

class JobClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_size=1768, num_classes=100, hidden1=1024, hidden2=512):
        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._build_model()

    def _build_model(self):
        self.model = nn.Sequential(
            nn.Linear(self.input_size, self.hidden1),
            nn.LayerNorm(self.hidden1),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(self.hidden1, self.hidden2),
            nn.LayerNorm(self.hidden2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(self.hidden2, self.num_classes)
        ).to(self.device)

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            preds = torch.argmax(self.model(X_tensor), dim=1)
        return preds.cpu().numpy()
    
def load_model():
    from sentence_transformers import SentenceTransformer
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.feature_extraction.text import TfidfVectorizer

    embedder = SentenceTransformer('all-mpnet-base-v2')
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    with open("cluster_to_label.pkl", "rb") as f:
        cluster_to_label = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    model = JobClassifier(input_size=1768, num_classes=len(label_encoder.classes_))
    model.model.load_state_dict(torch.load("job_model.pth", map_location=torch.device('cpu')))
    model.model.eval()

    return model, embedder, label_encoder, cluster_to_label, scaler, vectorizer

def predict_job_title(resume_text, model, embedder, vectorizer, label_encoder, cluster_to_label, scaler):
    input_embed = embedder.encode([resume_text])
    input_tfidf = vectorizer.transform([resume_text]).toarray()
    input = np.hstack((input_tfidf, input_embed))
    input = scaler.transform(input)
    pred_label = model.predict(input)[0]
    decoded_label = label_encoder.inverse_transform([pred_label])[0]
    return decoded_label