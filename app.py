import os
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from docx import Document
from flask import Flask, request, jsonify, send_from_directory, send_file
from collections import defaultdict
from fuzzywuzzy import fuzz
import pandas as pd
import tempfile
import openpyxl

app = Flask(__name__)

# Directory to save uploaded files
UPLOAD_FOLDER = 'uploaded_cvs'
PUBLIC_FOLDER = os.path.join('static', UPLOAD_FOLDER)
os.makedirs(PUBLIC_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = PUBLIC_FOLDER

# Storage for matched CVs and positions
matched_cvs_storage = {}
positions_storage = {}

def download_nltk_data():
    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords')
    try:
        word_tokenize("test")
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/wordnet.zip')
    except LookupError:
        nltk.download('wordnet')

# Ensure NLTK data is available
download_nltk_data()

# Function to get synonyms using WordNet
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return synonyms

# Function to preprocess CV text
def preprocess_cv(cv_text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    skills = []
    contact_info = {}

    for line in cv_text:
        words = word_tokenize(line)
        words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
        words = [lemmatizer.lemmatize(word) for word in words]

        if line.startswith("3."):
            skill_name = line.split(":")[1].split('(')[0].strip()  # Extract SkillName without SkillLevel
            skills.append(skill_name)
        elif line.lower().startswith("full name:"):
            contact_info['full_name'] = line.split(":")[1].strip()
        elif line.lower().startswith("address:"):
            contact_info['address'] = line.split(":")[1].strip()
        elif line.lower().startswith("phone number:"):
            contact_info['phone_number'] = line.split(":")[1].strip()
        elif line.lower().startswith("email:"):
            contact_info['email'] = line.split(":")[1].strip()

    return skills, contact_info

# Function to load CV data from folder
def load_cv_data(folder_path):
    file_names = os.listdir(folder_path)
    cv_data = []
    for file_name in file_names:
        if file_name.endswith('.docx'):
            file_path = os.path.join(folder_path, file_name)
            doc = Document(file_path)
            cv_text = [paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()]
            skills, contact_info = preprocess_cv(cv_text)
            cv_data.append((skills, contact_info, file_name, file_path))
    return cv_data

# Function to preprocess entered skills
def preprocess_entered_skills(entered_skills):
    lemmatizer = WordNetLemmatizer()
    processed_skills = []
    for skill in entered_skills:
        words = word_tokenize(skill)
        words = [word.lower() for word in words if word.isalpha()]
        words = [lemmatizer.lemmatize(word) for word in words]
        processed_skills.append(' '.join(words))
    return processed_skills

# Function to detect similarity based on entered skills
def detect_similarity(entered_skills, cv_data):
    entered_skills_processed = preprocess_entered_skills(entered_skills)  # Preprocess the entered skills

    cv_matches = defaultdict(list)

    for skills, contact_info, file_name, file_path in cv_data:
        cv_skills = [skill.lower() for skill in skills]  # Lower case the skills from CV

        common_skills = []
        for entered_skill in entered_skills_processed:
            entered_synonyms = get_synonyms(entered_skill)
            for cv_skill in cv_skills:
                cv_synonyms = get_synonyms(cv_skill)
                if (fuzz.partial_ratio(entered_skill, cv_skill) >= 80) or (entered_synonyms & set(cv_synonyms)):
                    common_skills.append(cv_skill)
                    break  # Move to the next entered skill once a match is found

        match_score = len(common_skills) / len(entered_skills)  # Score based on overlap with entered skills

        cv_matches[match_score].append((contact_info, file_name, file_path, common_skills))

    sorted_scores = sorted(cv_matches.keys(), reverse=True)
    recommended_cvs = []

    for score in sorted_scores:
        if score > 0:
            for cv_info in cv_matches[score]:
                recommended_cvs.append(cv_info)

    return recommended_cvs

# Endpoint to upload CVs
@app.route('/upload-cv', methods=['POST'])
def upload_cv():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    files = request.files.getlist('file')
    if not files:
        return jsonify({"error": "No selected file"}), 400

    uploaded_files = []
    for file in files:
        if file and file.filename.endswith('.docx'):
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            uploaded_files.append({
                "message": "File uploaded successfully",
                "file_path": os.path.join(app.config['UPLOAD_FOLDER'], filename)
            })

    if not uploaded_files:
        return jsonify({"error": "Invalid file format. Only .docx files are allowed."}), 400

    return jsonify(uploaded_files), 200

# Endpoint to enter positions
@app.route('/enter-positions', methods=['POST'])
def enter_positions():
    data = request.json
    position = data.get('position')

    if not position:
        return jsonify({"error": "Position is required"}), 400

    positions_storage['position'] = position
    return jsonify({"message": "Position stored successfully", "position": position}), 200

# Endpoint to get matching CVs based on skills
@app.route('/match-cvs', methods=['POST'])
def match_cvs():
    data = request.json
    skills = data.get('skills')

    if not skills:
        return jsonify({"error": "Skills are required"}), 400

    position = positions_storage.get('position')
    if not position:
        return jsonify({"error": "Position is not entered. Please enter position first."}), 400

    cv_data = load_cv_data(app.config['UPLOAD_FOLDER'])
    if not cv_data:
        return jsonify({"error": "No CV data found in the uploaded folder"}), 404

    similar_cvs = detect_similarity(skills, cv_data)
    if not similar_cvs:
        return jsonify({"message": "No CVs found matching the entered skills"}), 404

    # Store the results in the global storage
    matched_cvs_storage['cvs'] = similar_cvs
    matched_cvs_storage['position'] = position

    results = []
    for cv_info in similar_cvs:
        contact_info = cv_info[0]
        recommended_cv_file = cv_info[1]
        recommended_cv_path = cv_info[2]
        matched_skills = cv_info[3]

        results.append({
            "position": position,
            "file_name": recommended_cv_file,
            "file_path": f"/{app.config['UPLOAD_FOLDER']}/{recommended_cv_file}",
            "contact_info": {
                "email": contact_info.get('email', 'N/A'),
                "phone_number": contact_info.get('phone_number', 'N/A')
            },
            "matched_skills": matched_skills
        })

    return jsonify(results)

# Endpoint to get the matched CVs
@app.route('/get-matched-cvs', methods=['GET'])
def get_matched_cvs():
    if 'cvs' not in matched_cvs_storage or not matched_cvs_storage['cvs']:
        return jsonify({"error": "No matched CVs found. Please run the match-cvs endpoint first."}), 400

    position = matched_cvs_storage.get('position', 'N/A')

    results = []
    for cv_info in matched_cvs_storage['cvs']:
        contact_info = cv_info[0]
        recommended_cv_file = cv_info[1]
        recommended_cv_path = cv_info[2]
        matched_skills = cv_info[3]

        results.append({
            "position": position,
            "file_name": recommended_cv_file,
            "file_path": f"/{app.config['UPLOAD_FOLDER']}/{recommended_cv_file}",
            "contact_info": {
                "email": contact_info.get('email', 'N/A'),
                "phone_number": contact_info.get('phone_number', 'N/A')
            },
            "matched_skills": matched_skills
        })

    return jsonify(results)

# Endpoint to export matched CVs to Excel
@app.route('/export-to-excel', methods=['GET'])
def export_to_excel():
    if 'cvs' not in matched_cvs_storage or not matched_cvs_storage['cvs']:
        return jsonify({"error": "No matched CVs found. Please run the match-cvs endpoint first."}), 400

    position = matched_cvs_storage.get('position', 'N/A')

    # Extract email and phone number data
    data = matched_cvs_storage['cvs']
    export_data = []
    for cv_info in data:
        contact_info = cv_info[0]
        email = contact_info.get('email', 'N/A')
        phone_number = contact_info.get('phone_number', 'N/A')
        export_data.append({
            "Position": position,
            "Email": email,
            "Phone Number": phone_number
        })

    # Create DataFrame
    df = pd.DataFrame(export_data)

    # Export to Excel
    excel_filename = "matched_cvs.xlsx"
    excel_path = os.path.join(app.config['UPLOAD_FOLDER'], excel_filename)
    df.to_excel(excel_path, index=False)

    return send_file(excel_path, as_attachment=True)

# Endpoint to serve uploaded files
@app.route(f'/{UPLOAD_FOLDER}/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)