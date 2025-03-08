import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from first_original import PDFResearchAnalyzer

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

analyzer = None  # Store chatbot instance
PREDEFINED_QUERIES = [
    "What can be used to improve the model?",
    "Summarize the result",
    "Describe the experimental approach",
    "What were the key findings?",
    "Which dataset is used?"
]

@app.route("/")
def home():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload_pdf():
    global analyzer
    if "pdf" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["pdf"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(file.filename))
    file.save(filepath)

    analyzer = PDFResearchAnalyzer(filepath)
    predefined_responses = {q: analyzer.advanced_query_response(q) for q in PREDEFINED_QUERIES}

    return jsonify({"message": "PDF uploaded successfully", "predefined_responses": predefined_responses})

@app.route("/chat", methods=["POST"])
def chat():
    global analyzer
    if not analyzer:
        return jsonify({"error": "Upload a PDF first"}), 400

    data = request.json
    user_message = data.get("message")

    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    response = analyzer.advanced_query_response(user_message)
    return jsonify({"reply": response["direct_answer"]})

if __name__ == "__main__":
    app.run(debug=True)
