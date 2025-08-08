# app/app.py

from flask import Flask, request, jsonify, render_template
import joblib

# Load model and vectorizer
model = joblib.load("../model/emotion_model.pkl")
vectorizer = joblib.load("../model/vectorizer.pkl")

# Initialize Flask app
app = Flask(__name__)

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        text = request.form["user_input"]
        text_clean = text.lower()  # same cleaning as training
        vector = vectorizer.transform([text_clean])
        prediction = model.predict(vector)
        return render_template("index.html", user_input=text, prediction=prediction[0])

if __name__ == "__main__":
    app.run(debug=True)
