# Developing a Weather-Integrated Ensemble Machine Learning Model for Last-Mile Delivery Delay Prediction

This repository contains a lightweight, low-cost, and high-accuracy ensemble machine learning framework designed to predict last-mile food delivery delay times (ETA).
The system integrates **spatial feature engineering (Haversine distance)**, **temporal feature engineering**, and **monthly-hourly historical weather climatology profiles** compiled from 10 years of historical local measurements.

The modeling pipeline implements a **Stacking Ensemble Regressor**:
- **Layer 1 (Base Estimators)**: Random Forest (Bagging), LightGBM (Boosting), XGBoost (Boosting), and CatBoost (Boosting)
- **Layer 2 (Meta-Estimator)**: Ridge Regression (L2 Regularized)

Model decisions are explained globally and locally using **SHAP (Explainable AI - XAI)**.

---

## 📂 Project Structure
```text
delivery_delay_prediction/
├── data/
│   ├── raw/                      # Put raw CSV dataset files here
│   └── processed/                # Generated merged datasets
├── models/                       # Saved trained ensemble models and plots
├── src/                          # Modular Python source code
│   ├── preprocess.py             # Data cleaning and weather profile integration
│   ├── models.py                 # Estimators and hyperparameter grids
│   ├── train.py                  # Grid tuning and training orchestrator
│   └── predict.py                # Single inference demo
├── consolidated_pipeline.py       # VS Code Cell-by-Cell Interactive Python script
├── consolidated_notebook.ipynb   # Jupyter Notebook for Google Colab/local Jupyter
├── requirements.txt              # Project packages dependency list
└── README.md                     # Project documentation (this file)
```

---

## 🚀 Running the Project

### Option A: Google Colab (Zero Local Setup)

This is the easiest way to run and test the project without installing Python locally.

1. Open [Google Colab](https://colab.research.google.com/).
2. Click **Upload** and select `consolidated_notebook.ipynb` from this project folder.
3. **Run Cell 1 first**: This allocates the runtime machine and installs the required packages (`geopy`, `catboost`, `shap`).
4. **Upload Datasets**: Once connected, click the **Folder icon** on the left sidebar. Drag and upload all raw CSV files (the delivery dataset `train.csv` and weather cities datasets like `delhi.csv`, `bengaluru.csv`, etc.) directly into the `/content/` directory.
5. **Run remaining cells**: Execute Cell 2 through Cell 7 step-by-step.

---

### Option B: Local Python Environment (Using Virtual Environment)

To run the project locally on your machine, you must use the local virtual environment (`venv`).

#### 1. Place Raw Datasets
Ensure your raw CSV dataset files are placed in the `data/raw/` directory inside the project folder.

#### 2. Create and Activate Virtual Environment
Create the virtual environment:
```bash
python -m venv venv
```

Activate the virtual environment:
- **On Windows**:
  ```cmd
  venv\Scripts\activate
  ```
- **On macOS/Linux**:
  ```bash
  source venv/bin/activate
  ```

#### 3. Install Dependencies
Install all required libraries inside the activated virtual environment:
```bash
pip install -r requirements.txt
```

#### 4. Execute Scripts
You can run the modular training orchestrator directly using the venv interpreter:
```bash
# To run preprocess and train the models
python src/train.py
```
Alternatively, you can run the single-file pipeline script:
```bash
python consolidated_pipeline.py
```
After training completes, the model is saved to `models/ensemble_model.joblib` and the performance comparison summary is printed in your terminal.

---

### Option C: VS Code (Interactive Execution)

1. Open the project folder `delivery_delay_prediction` in VS Code.
2. Ensure you have the **Python** and **Jupyter** extensions installed in VS Code.
3. Make sure raw datasets are placed in `data/raw/`.
4. Open `consolidated_pipeline.py`.
5. Click **"Run Cell"** above any block starting with `# %%`.
6. When prompted to select a Python interpreter, select the interpreter inside the local virtual environment (`venv/Scripts/python.exe`).
7. Run the cells sequentially to visualize output tables, LightGBM feature importances, and SHAP graphs.

---

## 📊 Model Evaluation Results Summary

Our Stacking Ensemble blends diverse Bagging (Random Forest) and Boosting (LightGBM, XGBoost, CatBoost) algorithms, yielding highly precise predictions:

* **$R^2$ Score**: **0.8303** (Explains 83.03% of factors causing urban delivery delay)
* **MAE (Mean Absolute Error)**: **3.044 minutes**
* **RMSE (Root Mean Squared Error)**: **3.835 minutes**
