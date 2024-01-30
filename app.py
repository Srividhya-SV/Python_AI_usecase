import PyPDF2
from flask import Flask, render_template, request
import tempfile, os
from transformers import pipeline

app = Flask(__name__)

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as pdf_file:
        # pdf_file_data = pdf_file.read
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

question_answerer = pipeline("question-answering", model='distilbert-base-uncased-distilled-squad')

# Using route to display the form
@app.route("/")
def index():
    return render_template("index.html")

# Route to handle form submission
@app.route("/submit", methods=["POST"])
def submit():
    pdf_file = request.files["pdf_file"]
    question = request.form["question"]
    pdf_file_path = os.path.abspath(pdf_file.name)
    
    if not pdf_file or not question:
        return "Please provide both a PDF file and a question."
    print(pdf_file)
    
    pdf_file.save(pdf_file_path)
    text = extract_text_from_pdf(pdf_file_path)
    
    # search_results = query_llm(question)
    os.remove(pdf_file_path)
    result = question_answerer(question=question, context=text)
    result1 = result['answer']

    # print(f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}")
    # print(result1)
                                                                                                              
    return render_template("result.html", extraction_results=result1)

if __name__ == "__main__":
    app.run(debug=True)
