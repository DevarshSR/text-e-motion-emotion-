# test_model.py

import joblib

# Load the saved model and vectorizer
model = joblib.load("model/emotion_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# Sample input
sample_text = "wow i am charged as hell"

# Preprocess it same as training (lowercase only, since model was trained like that)
sample_text_clean = sample_text.lower()

# Transform into TF-IDF vector
sample_vector = vectorizer.transform([sample_text_clean])

# Predict
prediction = model.predict(sample_vector)

print(f"\nYour text: {sample_text}")
print(f"Predicted Emotion: {prediction[0]}")
