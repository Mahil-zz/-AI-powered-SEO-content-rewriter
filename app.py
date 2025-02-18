import sys
import os
from flask import Flask, request, jsonify, render_template
import requests
from bs4 import BeautifulSoup
from flask_cors import CORS  # Importing CORS module for cross-origin requests
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Initialize Flask app
app = Flask(__name__, template_folder='templates')
CORS(app, resources={r"/analyze": {"origins": "*"}})  # Enable CORS for /analyze route

# Load the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize FAISS index for storing vectors
dimension = 384
index = faiss.IndexFlatL2(dimension)

# API key for Gemini or other AI services
API_KEY = 'AIzaSyApScRuzNqVPpqF1-FPX6vgLVhtoaGbRus'
GEMINI_API_URL = 'https://api.gemini.ai/v1/rewrite'


@app.route('/')
def home():
    """Serve the HTML file"""
    return render_template("index.html")


def get_page_content(url):
    """Scrape the webpage and get its content (Title, Meta, Body Text)."""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    title = soup.title.string if soup.title else ""
    meta_description = ""
    for meta_tag in soup.find_all("meta"):
        if 'name' in meta_tag.attrs and meta_tag.attrs['name'].lower() == 'description':
            meta_description = meta_tag.attrs['content']
            break

    paragraphs = soup.find_all('p')
    body_text = ' '.join([p.get_text() for p in paragraphs])

    return {
        "title": title,
        "meta_description": meta_description,
        "body_text": body_text
    }


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    url = data.get('url')
    keywords = data.get('keywords')

    if not url or not keywords:
        return jsonify({"error": "URL and keywords are required"}), 400

    # Scrape the page content
    page_content = get_page_content(url)

    return jsonify({
        "title": page_content['title'],
        "meta_description": page_content['meta_description'],
        "body_text": page_content['body_text']
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
