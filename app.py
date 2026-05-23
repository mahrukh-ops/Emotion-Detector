import streamlit as st
import pickle

#load model and vectorizer
model = pickle.load(open("emotion_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("Emotion Detector App")
st.write("Predict emotions from text using NLP and Machine Learning.")

user_input =  st.text_input("Enter a sentence:")

if st.button("Predict"):
    data = vectorizer.transform([user_input])
    result = model.predict(data)
    st.success("Emotion: " + result[0])