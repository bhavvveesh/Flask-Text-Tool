from flask import Flask, render_template, request, send_file
from transformers import pipeline
from deep_translator import GoogleTranslator
import torch
import os

app = Flask(__name__)

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the summarization model (optimized)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if device == "cuda" else -1)

translator = GoogleTranslator(source="auto", target="kn")  # Kannada translation

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()

            # Summarize the text
            summary = summarizer(text, max_length=150, min_length=50, do_sample=False)[0]["summary_text"]

            # Translate full text & summary
            full_kannada = translator.translate(text)
            summary_kannada = translator.translate(summary)

            # Save results
            full_trans_path = os.path.join(UPLOAD_FOLDER, "kannada_translation.txt")
            summary_trans_path = os.path.join(UPLOAD_FOLDER, "kannada_summarized_translation.txt")

            with open(full_trans_path, "w", encoding="utf-8") as f:
                f.write(full_kannada)
            with open(summary_trans_path, "w", encoding="utf-8") as f:
                f.write(summary_kannada)

            return render_template("index.html", full_trans=full_trans_path, summary_trans=summary_trans_path)

    return render_template("index.html")

@app.route("/download/<filename>")
def download(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename), as_attachment=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
