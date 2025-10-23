# Kidney Disease Detection using Machine Learning

Welcome to the Kidney Disease Detection project! This repository contains a comprehensive pipeline to detect Chronic Kidney Disease (CKD) using machine learning models trained on a real-world clinical dataset. The project includes data preprocessing, exploratory data analysis (EDA), model training, hyperparameter tuning, and a user-friendly Streamlit web app for live prediction.

***

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Data Preprocessing & EDA](#data-preprocessing--eda)
- [Model Training & Evaluation](#model-training--evaluation)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Streamlit App Usage](#streamlit-app-usage)
- [Future Improvements](#future-improvements)
- [Author](#author)
- [License](#license)

***

## Project Overview

Chronic Kidney Disease is a serious medical condition affecting millions worldwide. Early detection is critical to managing health outcomes. This project uses clinical data features such as blood pressure, serum creatinine, hemoglobin levels, and more to build predictive models identifying CKD risk.

The main goal is:

- To preprocess and analyze the dataset effectively
- Train machine learning models with high accuracy and robustness
- Provide an interactive web interface for easy predictions without coding

***

## Dataset

The dataset used here is the Chronic Kidney Disease Dataset from Kaggle, consisting of 400 patient records with 24 features including demographic and clinical attributes. It contains some missing values that are handled during preprocessing.

Dataset link: [https://www.kaggle.com/datasets/mansoordaku/ckdisease](https://www.kaggle.com/datasets/mansoordaku/ckdisease)

***

## Technologies Used

- Python 3.8+
- Pandas, NumPy for data manipulation
- Matplotlib, Seaborn for EDA and visualization
- Scikit-learn for machine learning models
- XGBoost for powerful boosted classifiers
- Streamlit for developing the interactive web app

***

## Getting Started

### Prerequisites

Install required packages using:

```bash
pip install -r requirements.txt
```

### Running Jupyter Notebooks

Explore data and models using the provided notebook files:

- `data_preprocessing.ipynb` - Data cleaning and imputation
- `eda_visualizations.ipynb` - Exploratory data analysis and plots
- `model_training.ipynb` - Model training and evaluation including Random Forest, SVM, XGBoost
- `hyperparameter_tuning.ipynb` - Grid search for best model parameters

### Running Streamlit App

To run the interactive app:

```bash
streamlit run app.py
```

***

## Project Structure

```
kidney-disease-prediction/
│
├── data/
│   └── kidney_disease.csv         # Raw dataset file
│
├── models/
│   └── best_rf_model.pkl          # Saved best trained model
│
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── eda_visualizations.ipynb
│   ├── model_training.ipynb
│   └── hyperparameter_tuning.ipynb
│
├── app.py                        # Streamlit web app code
├── requirements.txt              # Required Python packages
├── README.md                    # This file
```

***

## Data Preprocessing & EDA

- Missing values replaced carefully using median (numerical) and mode (categorical).
- Columns with complete missing data dropped.
- Visualizations include age and blood pressure distributions, correlation heatmaps, and classification distributions.
- Key insights drawn for feature importance aiding model selection.

***

## Model Training & Evaluation

- Models trained: Random Forest, Support Vector Machine, XGBoost.
- Performance metrics include accuracy, precision, recall, f1-score, and confusion matrices.
- Cross-validation used to ensure robustness and prevent overfitting.
- Tuned Random Forest showed excellent balance with mean CV accuracy > 98%.

***

## Hyperparameter Tuning

- GridSearchCV used on Random Forest with parameters such as number of estimators, max depth, and minimum samples split.
- Optimal parameters found and best model saved for deployment.

***

## Streamlit App Usage

- User inputs clinical parameters through an intuitive UI.
- Real-time predictions with probability scores displayed.
- Helps healthcare professionals or users assess kidney disease risk quickly.

***

## Future Improvements

- Integration of SHAP values for explainability.
- Incorporation of more clinical features or multi-modal data.
- Deployment on cloud platforms for wider accessibility.
- Building a mobile-friendly version of the app.

***

## Author

Vishal Dubey – Data Science Enthusiast, Machine Learning Practitioner

GitHub: [github.com/Vishaldubey2210](https://github.com/Vishaldubey2210)

