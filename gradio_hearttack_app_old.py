import gradio as gr
import xgboost as xgb
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the model and the scaler
model = joblib.load('best_XGB.pkl')
scaler = joblib.load('scaler.pkl')  # Load the scaler that was saved during training
cutoff = 0.42  # Custom cutoff probability

# Load SHAP explainer based on your XGBoost model
explainer = shap.Explainer(model)

# Define the prediction function with preprocessing, scaling, and SHAP analysis
def predict_heart_attack(Gender, age, cigsPerDay, BPMeds, prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose):
    # Define feature names in the same order as the training data
    feature_names = ['Gender', 'age', 'cigsPerDay', 'BPMeds', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
    
    # Create a DataFrame with the correct feature names for prediction
    features = pd.DataFrame([[Gender, age, cigsPerDay, BPMeds, prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose]], columns=feature_names)
    
    # Standardize the features (scaling)
    scaled_features = scaler.transform(features)
    
    # Predict probabilities
    proba = model.predict_proba(scaled_features)[:, 1]  # Probability of class 1 (heart attack)
    
    # Apply custom cutoff
    if proba[0] >= cutoff:
        prediction_class = 1
    else:
        prediction_class = 0
    
    # Generate SHAP values for the prediction
    shap_values = explainer(scaled_features)
    
    # Plot SHAP values
    plt.figure(figsize=(8, 6))
    shap.waterfall_plot(shap_values[0])
    plt.savefig('shap_plot.png')  # Save SHAP plot to a file
    
    result = f"Predicted Probability: {proba[0]*100:.2f}%. Predicted Class with cutoff {cutoff}: {prediction_class}"
    
    return result, 'shap_plot.png'  # Return the prediction and SHAP plot

# Create the Gradio interface with preprocessing, prediction, and SHAP visualization
with gr.Blocks() as app:
    with gr.Row():
        with gr.Column():
            Gender = gr.Radio([0, 1], label="Gender (0=Female, 1=Male)")
            cigsPerDay = gr.Slider(0, 40, step=1, label="Cigarettes per Day")
            prevalentHyp = gr.Radio([0, 1], label="Prevalent Hypertension (0=No, 1=Yes)")
            totChol = gr.Slider(100, 400, step=1, label="Total Cholesterol in mg/dl")
            diaBP = gr.Slider(60, 120, step=1, label="Diastolic/Lower BP")
            heartRate = gr.Slider(50, 120, step=1, label="Heart Rate")
        
        with gr.Column():
            age = gr.Slider(20, 80, step=1, label="Age (years)")
            BPMeds = gr.Radio([0, 1], label="On BP Medications (0=No, 1=Yes)")
            diabetes = gr.Radio([0, 1], label="Diabetes (0=No, 1=Yes)")
            sysBP = gr.Slider(90, 200, step=1, label="Systolic BP/Higher BP")
            BMI = gr.Slider(15, 40, step=0.1, label="Body Mass Index  (weight in kg/ height in meter squared)(BMI) in kg/m2")
            glucose = gr.Slider(50, 250, step=1, label="Fasting Glucose Level")
    
    # Center-aligned prediction output
    with gr.Row():
        gr.HTML("<div style='text-align: center; width: 100%'>Heart Attack Prediction</div>")
    
    with gr.Row():
        prediction_output = gr.Textbox(label="", interactive=False, elem_id="prediction_output")
    
    with gr.Row():
        shap_plot_output = gr.Image(label="SHAP Analysis")

    # Link inputs and prediction output
    submit_btn = gr.Button("Submit")
    submit_btn.click(fn=predict_heart_attack, inputs=[Gender, age, cigsPerDay, BPMeds, prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose], outputs=[prediction_output, shap_plot_output])

app.launch()
