import os
import shutil
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from docx import Document
from flask import Flask, request, jsonify, send_from_directory, send_file
from collections import defaultdict
from fuzzywuzzy import fuzz
import pandas as pd
from flask_cors import CORS
import nlp
import spacy

app = Flask(__name__)
CORS(app)  # Enable CORS for the Flask app
#NLP : Tokenization,Lemmatization
#Named Entity Recognition (NER) 
#Fuzzy Matching
#Synonym Extraction using WordNet
# Directory to save uploaded files
UPLOAD_FOLDER = 'uploaded_cvs'
PUBLIC_FOLDER = os.path.join('static', UPLOAD_FOLDER)
os.makedirs(PUBLIC_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = PUBLIC_FOLDER

nlp = spacy.load('en_core_web_md')


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

def preprocess_cv(cv_text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    skills = []
    contact_info = {}

    for line in cv_text:
        # Tokenize the line into words
        words = word_tokenize(line)
        
        # Lowercase and filter out non-alphabetic words and stopwords
        words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
        
        # Lemmatize the words
        words = [lemmatizer.lemmatize(word) for word in words]

        # Check for specific sections or headers where skills are listed
        if line.startswith("3."):  # Assuming skills are listed under section starting with "3."
            # Extract skills using a specific format or pattern
            if "SkillName" in line:
                # Split the line to get the skill name
                skill_info = line.split(":")
                if len(skill_info) >= 2:
                    skill_name = skill_info[1].strip()
                    skills.append(skill_name)
        
        # Extract contact information (example: email, phone number)
        elif line.lower().startswith("full name:"):
            contact_info['full_name'] = line.split(":")[1].strip()
        elif line.lower().startswith("address:"):
            contact_info['address'] = line.split(":")[1].strip()
        elif line.lower().startswith("phone number:"):
            contact_info['phone_number'] = line.split(":")[1].strip()
        elif line.lower().startswith("email:"):
            contact_info['email'] = line.split(":")[1].strip()

    # Print extracted skills for debugging
    print(f"Extracted Skills: {skills}")

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
# Function to preprocess entered skills using spaCy embeddings
def preprocess_entered_skills(entered_skills):
    processed_skills = []
    for skill in entered_skills:
        doc = nlp(skill.lower())  # Process skill text with spaCy
        processed_skills.append(doc)
    return processed_skills


# Function to detect similarity based on entered skills using spaCy embeddings
def detect_similarity(entered_skills, cv_data):
    entered_skills_processed = preprocess_entered_skills(entered_skills)  # Preprocess the entered skills

    cv_matches = defaultdict(list)

    for skills, contact_info, file_name, file_path in cv_data:
        cv_skills = [skill.lower() for skill in skills]  # Lowercase the skills from CV

        common_skills = []
        for entered_skill in entered_skills_processed:
            found_match = False
            for cv_skill in cv_skills:
                doc_cv_skill = nlp(cv_skill)
                similarity_score = entered_skill.similarity(doc_cv_skill)
                if similarity_score >= 0.7:  # Adjust similarity threshold as needed
                    common_skills.append(cv_skill)
                    found_match = True
                    break
            
            if not found_match:
                # Fallback to fuzzy matching if no strong embedding similarity found
                for cv_skill in cv_skills:
                    if fuzz.partial_ratio(entered_skill.text, cv_skill) >= 80:
                        common_skills.append(cv_skill)
                        found_match = True
                        break

        match_score = len(common_skills) / len(entered_skills)  # Score based on overlap with entered skills

        cv_matches[match_score].append((skills, contact_info, file_name, file_path, common_skills))

    sorted_scores = sorted(cv_matches.keys(), reverse=True)
    recommended_cvs = []

    for score in sorted_scores:
        if score > 0:
            for cv_info in cv_matches[score]:
                skills, contact_info, recommended_cv_file, recommended_cv_path, matched_skills = cv_info
                
                # Ensure unique skills are included
                unique_matched_skills = list(set(matched_skills))
                
                recommended_cvs.append({
                    "position": positions_storage['position'],
                    "file_name": recommended_cv_file,
                    "file_path": f"/{app.config['UPLOAD_FOLDER']}/{recommended_cv_file}",
                    "contact_info": {
                        "email": contact_info.get('email', 'N/A'),
                        "phone_number": contact_info.get('phone_number', 'N/A')
                    },
                    "matched_skills": unique_matched_skills
                })

    # Debug print to check the processed skills and matched output
    print(f"Processed entered skills: {[skill.text for skill in entered_skills_processed]}")
    print(f"Matched CVs: {recommended_cvs}")

    return recommended_cvs


# Endpoint to upload CVs
@app.route('/upload-cv', methods=['POST'])
def upload_cv():
    global matched_cvs_storage
    global positions_storage
    
    matched_cvs_storage = {}  # Clear the matched CVs storage
    positions_storage = {}  # Clear the positions storage
    
    # Clear the upload directory
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            return jsonify({"error": f"Failed to delete {file_path}. Reason: {e}"}), 500

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
        # Ensure to access cv_info correctly based on its structure
        contact_info = cv_info['contact_info']
        recommended_cv_file = cv_info['file_name']
        matched_skills = cv_info['matched_skills']

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

if __name__ == '__main__':
    app.run(debug=True)
