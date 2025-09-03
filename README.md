# **NLP-Driven Job Recommendation Chatbot**  
## **📌 Overview**  
This project is an **NLP_Driven Job Recommendation Chatbot** developed during my internship at **Cochin University of Science and Technology (CUSAT)** under the guidance of **Dr. Jeena Kleenankandy** as part of the **IEEE CIS Kerala Section Summer Internship 2025**.  

The chatbot interacts with users through a dynamic, conversational interface, collects their profile details, and predicts the most suitable job role using a **Hybrid NLP + Machine Learning model**.  

---
## **✨ Features**  
- **Integration with Hugging Face Inference API (LLaMA 3.1)** – Uses the **LLaMa LLM** to generate context-aware questions.  
- **Hybrid NLP Model** – Combines **TF-IDF vectorization** and **SentenceTransformer embeddings** for accurate predictions.  
- **PyTorch Classifier** – A neural network model trained on job posting data.  
- **Interactive Interface** – Built using **Gradio** for real-time chatbot interaction.  
- **Session Logging** – Stores user chats and predictions for analysis and model improvement.  

---

## **🛠️ Tech Stack**  

### **Languages & Frameworks**  
- Python  
- PyTorch  
- scikit-learn  
- SentenceTransformers  
- Gradio  

### **NLP & AI Tools**  
- TF-IDF Vectorizer  
- Sentence Embeddings (`all-mpnet-base-v2`)  
- LLaMa LLM 

---

## 📂 Project Structure  
project/  
│── app.py  
│── model.py  
│── profile_builder.py  
│── job_recommender.ipynb  
│── requirements.txt  
│── README.md  
│── .gitignore  

---

## ⚙️ Installation & Setup

1️⃣ **Clone the repository**
```bash
git clone https://github.com/your-username/job-recommendation-chatbot.git
cd job-recommendation-chatbot
```

2️⃣ Create a virtual environment & activate it
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
```

3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

4️⃣ Run the chatbot
```bash
python app.py
```  
---  

## **📊 How It Works**
1. User Interaction – The chatbot asks initial questions to gather user details.  
2. Dynamic Questioning – The Mistral LLM adapts its next question based on previous answers.  
3. Profile Processing – Responses are converted into numerical vectors using TF-IDF and sentence embeddings.  
4. Prediction – The trained PyTorch model predicts the most suitable job title.  
<<<<<<< HEAD
5. Result Display – The chatbot shows the predicted job and logs the session.  
=======
5. Result Display – The chatbot shows the predicted job and logs the session.  
