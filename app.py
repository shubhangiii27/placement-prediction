import os
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Placement Predictor", layout="centered")

# ---- Background Styling ----
def set_bg(style):
    if style == "Solid Color":
        st.markdown(
            """<style>
            .stApp { background-color: #0f172a; color: white; }
            </style>""", unsafe_allow_html=True)
    elif style == "Gradient":
        st.markdown(
            """<style>
            .stApp {
                background: linear-gradient(135deg, #1e3c72, #2a5298);
                color: white;
            }
            </style>""", unsafe_allow_html=True)
    elif style == "Texture":
        st.markdown(
            """<style>
            .stApp {
                background-image: url('https://www.transparenttextures.com/patterns/cubes.png');
                background-color: #111827;
                color: white;
            }
            </style>""", unsafe_allow_html=True)

bg_style = st.sidebar.selectbox("Choose Website Background", 
                                ["Solid Color", "Gradient", "Texture"])
set_bg(bg_style)

st.title("🎓 Placement Predictor Model")

# ---- Load or Train Model ----
MODEL_PATH = "model.pkl"

def train_model():
    data = {
        "cgpa": [6.5,7.2,8.1,9.0,5.8,7.9,8.5,6.0],
        "aptitude": [60,70,85,90,50,78,88,55],
        "coding": [65,72,88,92,48,80,90,52],
        "communication": [55,65,75,85,50,70,82,58],
        "internships": [0,1,2,3,0,1,2,0],
        "backlogs": [2,1,0,0,3,1,0,2],
        "placed": [0,1,1,1,0,1,1,0]
    }
    df = pd.DataFrame(data)
    X = df.drop("placed", axis=1)
    y = df["placed"]
    model = LogisticRegression()
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    return model

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = train_model()

# ---- User Input ----
st.subheader("Enter Student Details")

cgpa = st.slider("CGPA", 0.0, 10.0, 7.0)
aptitude = st.slider("Aptitude Score", 0, 100, 70)
coding = st.slider("Coding Skill", 0, 100, 70)
communication = st.slider("Communication Skill", 0, 100, 65)
internships = st.number_input("Internships", 0, 10, 1)
backlogs = st.number_input("Backlogs", 0, 10, 0)

if st.button("Predict Placement"):
    features = np.array([[cgpa, aptitude, coding, communication, internships, backlogs]])
    prediction = model.predict(features)[0]
    if prediction == 1:
        st.success("✅ High Chance of Getting Placed")
    else:
        st.error("❌ Low Chance of Getting Placed")
