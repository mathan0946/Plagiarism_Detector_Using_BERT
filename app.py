from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import torch
from keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import os

app = Flask(__name__)

# Load BERT model and tokenizer
model_path = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path, output_attentions=False, output_hidden_states=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the relative path to the CSV file
data_path = os.path.join(script_dir, 'data', 'database.csv')
# Global variables
sample_size = 100
vector_index = None
data_loaded = False

def preprocess_data(data_path, sample_size):
    data = pd.read_csv(data_path, low_memory=False)
    data = data.dropna(subset=['abstract']).reset_index(drop=True)
    if 'paper_id' not in data.columns:
        raise ValueError("The input data must contain a 'paper_id' column.")
    data = data.sample(min(sample_size, len(data)))[['abstract', 'paper_id']]
    return data

@app.route('/check_plagiarism_database', methods=['GET'])
def check_plagiarism_database_page():
    return render_template('check_database.html')

@app.route('/check_plagiarism_text', methods=['GET'])
def check_plagiarism_text_page():
    return render_template('check_text.html')


def create_vector_from_text(text, MAX_LEN=510):
    # Tokenization and encoding
    input_ids = tokenizer.encode(text, add_special_tokens=True, max_length=MAX_LEN)
    results = pad_sequences([input_ids], maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    input_ids = torch.tensor(results[0]).unsqueeze(0).to(device)
    attention_mask = torch.tensor([int(i > 0) for i in input_ids[0]]).unsqueeze(0).to(device)
    
    # Model evaluation and vector extraction
    model.eval()
    with torch.no_grad():
        logits, encoded_layers = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
    vector = encoded_layers[-1][0][0].cpu().numpy()  # Extract the last layer [CLS] token embedding
    return vector

def create_vector_index(data):
    vectors = []
    for text in tqdm(data['abstract'].values, desc="Generating vectors"):
        vector = create_vector_from_text(text)
        vectors.append(vector)
    data["vectors"] = vectors
    return data[['abstract', 'paper_id', 'vectors']]

@app.route('/')
def home():
    global data_loaded, vector_index
    if not data_loaded:
        preprocessed_data = preprocess_data(data_path, sample_size)
        vector_index = create_vector_index(preprocessed_data)
        data_loaded = True
        print("Dataset loaded and processed.")
    return render_template('home.html')

@app.route('/plagiarism_choice', methods=['POST'])
def plagiarism_choice():
    choice = request.form['choice']
    if choice == 'database':
        return render_template('check_database.html')
    elif choice == 'text':
        return render_template('check_text.html')

@app.route('/check_plagiarism_database', methods=['POST'])
def check_plagiarism_database():
    user_text = request.form['text']
    user_vector = create_vector_from_text(user_text)
    
    # Compute cosine similarity with all abstracts in the vector index
    similarities = [
        (row['paper_id'], np.dot(user_vector, row['vectors']) / (np.linalg.norm(user_vector) * np.linalg.norm(row['vectors'])))
        for _, row in vector_index.iterrows()
    ]
    
    # Identify the most similar paper
    most_similar = max(similarities, key=lambda x: x[1])
    similarity_score = float(most_similar[1])
    
    # Define plagiarism threshold
    threshold = 0.9  
    result = "Plagiarized" if similarity_score >= threshold else "Not Plagiarized"
    
    return jsonify({"most_similar_paper_id": most_similar[0], "similarity_score": similarity_score, "result": result})

@app.route('/check_plagiarism_text', methods=['POST'])
def check_plagiarism_text():
    text1 = request.form['text1']
    text2 = request.form['text2']
    vector1 = create_vector_from_text(text1)
    vector2 = create_vector_from_text(text2)
    
    # Calculate cosine similarity
    similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    similarity_score = float(similarity)
    
    # Define plagiarism threshold
    threshold = 0.9
    result = "Plagiarized" if similarity_score >= threshold else "Not Plagiarized"
    
    return jsonify({"similarity_score": similarity_score, "result": result})

if __name__ == '__main__':
    app.run(debug=True)
