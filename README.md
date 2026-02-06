# House Price Prediction – End-to-End Machine Learning Project

## Project Overview
This project focuses on building and deploying an end-to-end **House Price Prediction** system using Machine Learning.  
The model predicts house prices based on key property features and is exposed as a REST API using FastAPI, which is deployed on the cloud.

The goal of this project is to demonstrate the **complete ML lifecycle** — from data preprocessing and model selection to deployment.

---

## Problem Statement
Predict the **sale price of a house** using historical housing data and relevant property attributes such as size, quality, garage capacity, basement area, and year built.

This is a **regression problem**.

---

## Dataset
- Source: Kaggle – *House Prices: Advanced Regression Techniques*
- Rows: ~1,460
- Features: Numeric + Categorical
- Target Variable: `SalePrice`

---

## Data Preprocessing & Feature Engineering
- Handled missing values using domain-aware strategies:
  - Structural absence filled with `"None"`
  - Numerical absence filled with `0` or median
  - Neighborhood-wise median imputation for `LotFrontage`
- Log-transformed the target variable (`SalePrice`) to reduce skewness
- One-hot encoded categorical features
- Removed noisy or non-informative columns
- Final feature count increased due to encoding (handled via regularization)

---

## Model Training & Evaluation
Multiple regression models were trained and evaluated:

| Model | Test RMSE | Test R² |
|------|---------|--------|
| Linear Regression | Higher error | Lower generalization |
| Ridge Regression | **Best** | **Best** |
| Lasso Regression | Good | Slightly lower |
| Random Forest | Overfitting | Lower test stability |

### Final Model Chosen: **Ridge Regression**
**Reason:**  
Ridge Regression provided the best balance between bias and variance, achieving the lowest test RMSE and highest test R², while handling multicollinearity caused by one-hot encoding.

---

## Final Model Performance
- Test RMSE ≈ **0.138**
- Test R² ≈ **0.89**
- Strong generalization on unseen data

---

## Deployment
- API Framework: **FastAPI**
- Model Serialization: `joblib`
- Cloud Platform: **Render (Free Tier)**

### Live API URL
<a href="https://house-price-api-gim6.onrender.com/docs">Live URL</a>


---
## Datasets

<a href="https://github.com/Kalai-2216/House_Price_Prediction/blob/4d60fe67bf5bbd7e4423d464cd5af49abbc93710/app/main.py">API File</a>

<a href="https://github.com/Kalai-2216/House_Price_Prediction/blob/4d60fe67bf5bbd7e4423d464cd5af49abbc93710/.ipynb_checkpoints/House_price_prediction-checkpoint.ipynb">Juypter Notebook</a>

<a href="https://github.com/Kalai-2216/House_Price_Prediction/blob/4d60fe67bf5bbd7e4423d464cd5af49abbc93710/Data/data_description.txt">Dataset column description file</a>

---
## Sample input
{
  "OverallQual": 7,
  "GrLivArea": 1800,
  "GarageCars": 2,
  "TotalBsmtSF": 900,
  "YearBuilt": 2005
}

- Sample Output
{
  "predicted_house_price": 9832877.75
}


  

## Future Improvements

- Add input validation using Pydantic

- Add model explainability using SHAP

- Build a frontend interface using Streamlit

- Implement logging and monitoring

- Improve API security and request validation

- Automate deployment using CI/CD pipelines


