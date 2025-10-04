import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import gradio as gr
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("/content/spam_dataset (1).csv")

# Ensure 'text' column exists
if 'text' not in df.columns:
    df.rename(columns={df.columns[0]: "text"}, inplace=True)

# 🔹 Create dummy labels for testing (replace with real labels if available)
df["label"] = [0 if i % 2 == 0 else 1 for i in range(len(df))]

# Features and target
X = df["text"].astype(str)
y = df["label"]

# Vectorize text
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_vec = vectorizer.fit_transform(X)

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_vec, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# 🔹 Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("===== Evaluation Metrics =====")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", conf_matrix)



# Function for Gradio
def predict_message(message):
    vec = vectorizer.transform([message])
    prediction = model.predict(vec)[0]
    return "🚨 Harassment Detected" if prediction == 1 else "✅ Safe Message"

# Gradio interface
iface = gr.Interface(
    fn=predict_message,
    inputs=gr.Textbox(lines=3, placeholder="Type a message here..."),
    outputs="text",
    title="HarassFreeWork - Harassment Detection",
    description="Type a workplace message or email text to check if it contains harassment patterns."
)

iface.launch()
