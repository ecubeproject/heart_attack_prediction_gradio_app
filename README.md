# heart_attack_prediction_gradio_app
A gradio app based on XGBoost classifier with 97% F1-Score on real data for predicting probability of heart-attack. Its basic versionis deployed on hugging face at [https://huggingface.co/spaces/minusquare/gradio_app]

# 🫀 Cardiac Risk Prediction App – Clinical AI for Heart Attack Prevention

**Organization**: Ecube Analytics  
**Deployment**: [Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/minusquare/gradio_app)  
**Status**: Deployed (prototype); selected for enterprise integration with stroke and diabetes modules.

---

## 🚀 Project Description

This is a real-world AI-driven healthcare application that predicts the risk of heart failure (cardiac arrest) based on clinical and lifestyle features. The app not only classifies heart-attack risk but also **explains** the predictions using SHAP-based interpretability, making it a powerful tool for **both patients and clinicians**.

### 🔍 Key Highlights:

- **Real clinical dataset** used for model development to ensure reliability and generalizability.
- Built with a **powerful XGBoost classifier**, optimized using GridSearchCV and validated with 5-fold cross-validation.
- Custom probability threshold (42%) used to maximize the F1-score by reducing both false positives and false negatives.
- **Gradio-powered UI** for user-friendly web interaction with:
  - Interactive risk-meter display
  - SHAP analysis plot for individualized explanations
  - Instant prediction with clear medical advisories

---

## ✅ Project Outcome

- Achieved **97% F1-score** on the test set.
- Successfully deployed on Hugging Face Spaces and receiving real-world engagement.
- Signed for enterprise integration with additional modules (Stroke & Diabetes).
- **SHAP-based visual explanations** help:
  - **Patients** modify high-risk lifestyle factors
  - **Doctors** tailor treatment plans for individuals
- Showcased as a **clinical-grade AI solution** bridging data science and preventive cardiology.

---

## 🧰 Tech Stack

| Layer             | Tools / Libraries Used                                        |
|------------------|---------------------------------------------------------------|
| **Language**      | Python 3.x                                                   |
| **Modeling**      | XGBoost (with SMOTE for class balancing)                     |
| **Evaluation**    | Accuracy, Precision, Recall, F1 Score, AUC                   |
| **Explainability**| SHAP (force plots and bar plots for individual prediction)   |
| **Front-End**     | Gradio (interactive app with sliders, risk-meter, SHAP plot) |
| **Deployment**    | Hugging Face Spaces (current); Docker + FastAPI (planned)    |

---

## 📂 Folder Structure

```plaintext
.
├── app.py                     # Gradio application code
├── model/
│   ├── xgb_model.pkl          # Trained XGBoost model
│   └── scaler.pkl             # Feature scaler object
├── shap/
│   └── explainer.pkl          # SHAP explainer object
├── data/
│   └── heart_dataset.csv      # Preprocessed training dataset
├── utils.py                   # Helper functions
├── requirements.txt           # Python dependencies
├── README.md                  # This file

🧪 Sample Features Used

    Gender, Age, CigarettesPerDay, BPMeds, PrevalentHypertension, Diabetes

    TotalCholesterol, SystolicBP, DiastolicBP, BMI, HeartRate, FastingGlucose

📈 Output Example

    Predicted probability of heart attack: 84.19%

    SHAP chart shows glucose, BP, and heart rate as top contributing factors

    Clear interpretability for personalized lifestyle and treatment guidance

👨‍⚕️ Disclaimer

    This application is designed for educational and demonstrational purposes. It is not a substitute for professional medical advice, diagnosis, or treatment. Please consult your doctor for medical concerns.

🤝 Contributors

    Author: Tejas Desai
    email [aimldstejas@gmail.com]
    LinkedIn: [https://www.linkedin.com/in/tejasddesaiindia] 
    Organization: Ecube Analytics
