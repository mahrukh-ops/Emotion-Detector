import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
data = pd.read_csv("dataset.csv" , sep=";" , header=None)

data.columns = ["text" , "label"]

# Show first 5 rows
print(data.head())

# Text column
X = data["text"]

# Emotion labels
y = data["label"]

# Convert text into numbers
vectorizer = CountVectorizer()

X_vectorized = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = MultinomialNB()

# Train model
model.fit(X_train, y_train)

# predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Save model

with open("emotion_model.pkl" , "wb") as f:
    pickle.dump(model,f)

with open("vectorization.pkl ", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model trained successfully!")

#Test prediction

user_text = ["i am feeling very happy today"]

#Convert text into numbers
user_text_vectorizer = vectorizer.transform(user_text)

#prediction
prediction = model.predict(user_text_vectorizer)

print("Predicted Emotion:" , prediction[0])