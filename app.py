import gradio as gr
import csv
import os
from gradio.themes.soft import Soft
from model import load_model, predict_job_title
from profile_builder import ProfileBuilder
from openai import OpenAI

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN")
)

def llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct:fireworks-ai",
        messages=[{"role": "system", "content": "You are a helpful assistant that asks polite, short questions to understand a user's background for job recommendations."},
                  {"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

builder = ProfileBuilder(use_llm=True,llm=llm)
model, embedder, label_encoder, cluster_to_label, scaler, vectorizer = load_model()

chat_history = []

def log_session_to_csv(name, full_chat, prediction, filename="chat_sessions.csv"):
    file_exists = os.path.isfile(filename)
    with open(filename, mode="a", newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Name", "Chat", "Predicted Job"])
        writer.writerow([name, full_chat.strip(), prediction])

class CustomTheme(Soft):
    def __init__(self):
        super().__init__()

def chat_interface(user_input, history):
    if history is None:
        history = []
    
    builder.receive_response(user_input)

    if builder.is_complete():
        combined_text = builder.combined_text()
        predicted_title = predict_job_title(combined_text, model, embedder, vectorizer, label_encoder, cluster_to_label, scaler)
        response = f"Thanks! Let me analyze your inputs and predict your ideal job.\nBased on your inputs, your ideal job would be: {predicted_title}"

        session_log = ""
        for u, b in history:
            session_log += f"User: {u}\nBot: {b}\n"
        session_log += f"User: {user_input}\nBot: {response}"

        user_name = builder.name
        log_session_to_csv(user_name, session_log, predicted_title)
        
    else:
        response = builder.next_question()
        
    history.append((user_input,response))
    return history, history, ""

def start_chat():
    first_msg = builder.next_question()
    return [("",first_msg)]

def reset_button():
    global builder
    builder = ProfileBuilder(use_llm=True,llm=llm)
    first_msg = builder.next_question()
    return [("", first_msg)], [("", first_msg)]

with gr.Blocks(theme=CustomTheme()) as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Your message")
    clear = gr.Button("Reset Chat")

    state = gr.State([])

    msg.submit(chat_interface, [msg, state], [chatbot, state, msg])

    clear.click(reset_button, None, [chatbot, state])

    demo.load(start_chat, None, chatbot)
    
demo.launch()
