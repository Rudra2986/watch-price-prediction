# 👑 Chronos AI — Watch Price Intelligence & Prediction

[![Python Version](https://img.shields.io/badge/python-3.11-emerald?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-blue?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Model Performance](https://img.shields.io/badge/R%C2%B2%20Score-71.2%25-gold?style=for-the-badge&logo=boost&logoColor=white)](#-model-performance-summary)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

An end-to-end Machine Learning pipeline and interactive web application designed to predict watch prices (in INR) from product attributes scraped from the **WatchVine** Indian e-commerce platform.

Named **Chronos AI**, the system features a high-performance predictive model backed by a state-of-the-art Streamlit dashboard styled after luxury horology brands like Rolex, complete with interactive 3D elements, dynamic charts, and AI explainability.

---

## 🌟 Key Features

*   **Custom Text-Mining Feature Extractor**: Solved data sparsity by extracting watch brands and automatic movement statuses directly from product titles. Consolidates brand names to 49 unique categories, reducing brand sparsity from **87.5% to 0.07%** and recovering hidden features.
*   **Leakage-Free Preprocessing**: Applied target encoding for high-cardinality features (color, dial color, strap color) and one-hot encoding for categorical variables with a strict split-before-encoding protocol to avoid data leakage.
*   **Optuna-Tuned Predictive Power**: Compares 6 regression models (Linear Regression, Ridge, Random Forest, XGBoost, LightGBM, CatBoost) to find the absolute best predictor.
*   **Luxury Web Experience**: Custom-designed, mobile-responsive dark glassmorphic UI styled with a **Rolex-inspired Emerald Green, Seafoam Mint, and Champagne Gold** palette.
*   **3D Interactive Watch Face**: Embedded a mouse-interactive, real-time-synchronized 3D watch wireframe using **Three.js** inside the Streamlit Hero layout.
*   **SHAP Explainability (XAI)**: Includes interactive SHAP graphs explaining how individual attributes (dial color, strap material, brand) affect the predicted price.
*   **Bento Grid Analytics**: High-end corporate dashboard detailing model comparison charts, dataset insights, and feature importance tables.

---

## 🚀 Model Performance Summary

By engineering brand and movement attributes from raw product text, the model's test performance saw a massive leap:
*   **Baseline R² Score**: ~0.450
*   **Optimized R² Score**: **0.7117** 📈 (A **58% improvement** in variance explained)
*   **Root Mean Squared Error (RMSE)**: Reduced by **30%** to **INR 466**
*   **Mean Absolute Error (MAE)**: **INR 294**

| Model | Baseline R² | Tuned R² (with Text Parsing) | Test RMSE (INR) | Test MAE (INR) |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost (Tuned)** | 0.446 | **0.7117** | **₹466** | **₹294** |
| **CatBoost (Tuned)** | 0.450 | 0.6980 | ₹512 | ₹311 |
| **Random Forest** | 0.427 | 0.6890 | ₹525 | ₹320 |
| **LightGBM** | 0.431 | 0.6800 | ₹540 | ₹331 |
| **Ridge Regression** | 0.285 | 0.4200 | ₹790 | ₹580 |
| **Linear Regression** | 0.281 | 0.4180 | ₹792 | ₹584 |

---

## 🛠️ Step-by-Step ML Pipeline

The project is structured sequentially into 7 pipeline scripts to make it modular and easy to follow:

```
watch-price-prediction/
│
├── .github/workflows/
│   └── lint.yml                    # Automated code linting check
├── .gitignore                      # Environment and cache rules
├── LICENSE                         # MIT Open Source License
├── README.md                       # Beautiful repo overview (this file)
├── PROJECT_DOCUMENTATION.txt       # Detailed project design documentation
├── requirements.txt                # Package dependencies
│
├── step1_Data_cleaning.py          # Cleans raw scraped data down to 18 core columns
├── step2_eda.py                    # Generates 25 analytical plots for dataset patterns
├── step3_feature_engineering.py    # Standardizes categories, encodes variables, log transforms target
├── step4_model_building.py          # Trains 6 baseline machine learning estimators
├── step5_model_evaluation.py       # Benchmark reports and 5 performance metrics/plots
├── step6_hyperparameter_tuning.py  # Optuna Bayesian hyperparameter search
├── step7_streamlit_app.py          # Chronos AI Ultimate dashboard application
│
├── eda_plots/                      # Folder containing 25 EDA plots
├── evaluation_plots/               # Benchmark comparison plots
└── tuning_plots/                   # Optuna and SHAP visualization exports
```

### 1. Data Cleaning (`step1_Data_cleaning.py`)
Cleans the raw, noisy dataset containing 3,000+ scraped columns to 18 clean columns. It normalizes text, standardizes number formats, handles pricing details, and addresses empty inputs.

### 2. Exploratory Data Analysis (`step2_eda.py`)
Generates 25 plots analyzing target distribution, brand proportions, categories, and correlation matrices. Identifies right-skewness of price (justifying log transformation) and severe brand sparsity (87.5% missing).

### 3. Feature Engineering (`step3_feature_engineering.py`)
*   Drops redundant attributes and engineers custom binary flag variables.
*   Performs one-hot encoding for lower-cardinality features and target encoding for high-cardinality features (brands, colors).
*   Applies a strict split before encoding to prevent test leakage.
*   Saves encoder artifacts (`target_encoders.pkl` and `train_columns.pkl`) for deployment.

### 4. Model Building (`step4_model_building.py`)
Trains 6 distinct machine learning algorithms with 5-fold cross-validation. Employs scaling pipelines for linear models while feeding raw engineered matrices to tree models.

### 5. Model Evaluation (`step5_model_evaluation.py`)
Validates model outputs on a held-out test dataset using R², RMSE, and MAE metrics. Plots residual distributions and actual vs. predicted curves.

### 6. Hyperparameter Tuning (`step6_hyperparameter_tuning.py`)
Optimizes hyperparameters using **Optuna** (100 trials) for tree-based architectures. Generates SHAP impact plots and exports the production-ready model artifact (`final_model.pkl`).

### 7. Chronos AI Dashboard (`step7_streamlit_app.py`)
A gorgeous interactive Streamlit web dashboard. It features a Rolex-themed dark UI, responsive Bento grid widgets, real-time predictions, value-range estimates, local SHAP explanation plots, and an interactive Three.js 3D watch wireframe.

---

## 💻 Installation & Usage

Follow these steps to run Chronos AI locally on your system:

### 1. Clone the Repository
```bash
git clone https://github.com/Rudra2986/watch-price-prediction.git
cd watch-price-prediction
```

### 2. Create a Virtual Environment
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Dashboard
```bash
streamlit run step7_streamlit_app.py
```

### 5. Run the Entire Pipeline (Optional)
To retrain the models and generate new metadata, run the scripts in sequence:
```bash
python step1_Data_cleaning.py
python step2_eda.py
python step3_feature_engineering.py
python step4_model_building.py
python step5_model_evaluation.py
python step6_hyperparameter_tuning.py
```

---

## 🎨 Aesthetic Highlights (Dashboard UI/UX)

*   **Rolex-Inspired Luxury Palette**: Deep Emerald Green (`#0B2B1B`), Seafoam Mint (`#10b981`), and Champagne Gold (`#d4af37`) accents.
*   **Three.js Watch Wireframe**: An interactive 3D watch rotating in real-time, matching user cursor coordinates.
*   **Bento Grid & Glassmorphism**: High-end grid layout constructed with CSS container structures and floating glassy panels.
*   **Shimmering Animations**: Micro-interactions, pulsing glows, and smooth transitions on card hovering and buttons.
