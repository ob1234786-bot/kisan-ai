import os
import json
from flask import Flask, render_template, request, jsonify
from google import genai
from google.genai import types
from PIL import Image

app = Flask(__name__)

api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/analyze", methods=["POST"])
def analyze():
    if not client:
        return jsonify({"error": "GEMINI_API_KEY Render settings mein set nahi hai!"}), 500

    if "file" not in request.files:
        return jsonify({"error": "Koi photo upload nahi hui"}), 400

    try:
        file = request.files["file"]
        img = Image.open(file.stream)

        prompt = """
        You are an expert agricultural scientist. Analyze this leaf image and return a strict JSON response with these exact keys:
        {
            "crop_name": "Name of the plant or crop",
            "health_status": "Healthy or Mild Infection or Critical Disease",
            "confidence_score": 92,
            "disease_name": "Exact disease name or None - Plant is Healthy",
            "severity_metrics": {
                "leaf_damage": 20,
                "fungal_risk": 15,
                "nutrient_deficiency": 10,
                "hydration_level": 85
            },
            "roman_urdu_summary": "Simple, clear explanation in Roman Urdu for a Pakistani farmer",
            "organic_treatment": "Step-by-step low-cost desi or organic cure in Roman Urdu",
            "chemical_treatment": "Recommended agricultural spray or fertilizer with dosage in Roman Urdu",
            "prevention_tips": ["Tip 1 in Roman Urdu", "Tip 2 in Roman Urdu", "Tip 3 in Roman Urdu"]
        }
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[img, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )

        result_data = json.loads(response.text)
        return jsonify(result_data)

    except Exception as e:
        return jsonify({"error": f"Analysis fail ho gayi: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
