import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Step 1: Load cleaned dataset
df = pd.read_csv("data/train.txt", sep=";", header=None, names=["text", "emotion"])
df["text_clean"] = df["text"].str.lower().str.replace('[^\w\s]', '', regex=True)

# Step 2: TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["text_clean"])

# Step 3: Labels (emotion classes)
y = df["emotion"]

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 6: Evaluate
y_pred = model.predict(X_test)
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# Step 7: Save model and vectorizer
joblib.dump(model, "model/emotion_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("\nModel and vectorizer saved successfully! 🎉")
