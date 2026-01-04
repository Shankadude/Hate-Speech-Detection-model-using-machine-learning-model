# Hate-Speech-Detection-model-using-machine-learning-model-
Developed a machine learning model to detect hate speech in text data. Its a very simple implementation of machine learning.

# Hate Speech Detection using Machine Learning & Power BI

This project implements a **Hate Speech Detection system** using **Machine Learning (Logistic Regression pipeline)** in Python and provides an interactive **Power BI dashboard** for analysis and visualization of predictions and model performance.

The goal is to classify text into:
- **Hate Speech**
- **Offensive Language**
- **Neither**

The project includes training/inference logic, prediction output files, performance metrics, and dashboard-ready datasets.

---

## Project Highlights

 Text classification using NLP + Machine Learning  
 Pre-trained Logistic Regression pipeline saved as `.joblib`  
 Prediction outputs stored as CSV for reporting  
 Model evaluation metrics stored in JSON  
 Power BI dashboard for interactive visualization  
 Confusion matrix data in long format for easy charting  

---

## Project Structure

hate_speech_detection/
│
├── hate_speech_detection.py # Main script (training/inference/evaluation)
├── hate_speech_lr_pipeline.joblib # Saved ML pipeline (TF-IDF + Logistic Regression)
├── model_metrics.json # Model performance metrics
│
├── labeled_data.csv # Input labeled dataset used for training
├── hate_speech_predictions.csv # Prediction output file
├── confusion_matrix_long.csv # Confusion matrix in long format (for dashboard)
│
├── HateSpeechDashboard.pbix # Power BI dashboard file
└── .gitignore
