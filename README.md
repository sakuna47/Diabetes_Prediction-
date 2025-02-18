# Diabetes Prediction Web App

This repository contains a machine learning-based web application for predicting the likelihood of diabetes. The application is built using Python, Streamlit, Scikit-learn, and Imbalanced-learn (SMOTE) to handle class imbalance.

## Features
- Loads a pre-trained **Random Forest Classifier** for diabetes prediction.
- Uses **KNN Imputer** for missing value handling.
- Applies **SMOTE** to balance the dataset before training.
- Standardizes input data using **StandardScaler**.
- Provides a **Streamlit** interface for user input and prediction.

## Dataset
The model is trained on the **Pima Indians Diabetes Dataset**, which includes the following features:
- **Pregnancies**
- **Glucose**
- **Blood Pressure**
- **Skin Thickness**
- **Insulin**
- **BMI** (Body Mass Index)
- **Diabetes Pedigree Function**
- **Age**
- **Outcome** (Target variable: 0 = No Diabetes, 1 = Diabetes)

## Installation
Clone the repository and install the required dependencies:

```bash
git clone https://github.com/sakuna47/diabetes-prediction-app.git
cd diabetes-prediction-app
pip install -r requirements.txt
```

## Running the Application
To launch the Streamlit web app, run:

```bash
streamlit run app.py
```

## Model Training
If you want to retrain the model, run the `train_model.py` script:

```bash
python train_model.py
```

## File Structure
```
├── app.py                 # Streamlit app
├── train_model.py         # Model training script
├── diabetes_model.pkl     # Trained model
├── scaler.pkl             # Fitted scaler for input data
├── diabetes.csv           # Dataset (if applicable)
├── requirements.txt       # Dependencies
├── README.md              # Documentation
```

## Dependencies
Ensure the following Python libraries are installed:

```bash
pandas
numpy
scikit-learn
imblearn
streamlit
matplotlib
seaborn
pickle
```

Install dependencies using:
```bash
pip install -r requirements.txt
```

## License
This project is licensed under the MIT License.

## Author
[sakuna sankalpa](https://github.com/sakuna47)

