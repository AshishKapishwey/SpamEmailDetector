from flask import Flask, request, render_template
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

# Load trained model and vectorizer
model_path = os.path.join("model", "spam_model.pkl")
vectorizer_path = os.path.join("model", "vectorizer.pkl")

with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

with open(vectorizer_path, "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        email_text = request.form["email_text"]
        transformed_text = vectorizer.transform([email_text])
        prediction = model.predict(transformed_text)[0]
        result = "Spam" if prediction == 1 else "Not Spam"
        return render_template("index.html", result=result, email_text=email_text)

    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(debug=True)
