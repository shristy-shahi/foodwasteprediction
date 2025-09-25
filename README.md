# **🍽️ Food Waste Prediction**















##  📌 Project Overview

This project focuses on predicting food waste in retail stores using machine learning models. By analyzing past data, the goal is to minimize waste and optimize inventory management.

The workflow includes data preprocessing, exploratory data analysis (EDA), and machine learning model training to build an effective prediction system.

##  ✨ Features

Data Preprocessing & Cleaning 🛠️
Cleans and prepares raw data for analysis.

Exploratory Data Analysis (EDA) 📊
Uses visualizations like heatmaps and scatter plots to uncover data patterns and relationships.

Machine Learning Modeling 🤖
Trains and tests various ML algorithms to find the best predictive model.

Dimensionality Reduction 🔻
Applies PCA (Principal Component Analysis) to reduce features while retaining important information.

Regularization & Optimization ⚡
Implements methods to prevent overfitting and fine-tune model performance.

Cross-Validation & Performance Metrics 📉
Evaluates models using RMSE, R², and confusion matrix.

##  📂 Dataset

The project utilizes retail store data:

foodwastedata.csv → Raw, unprocessed data

final_food_waste_data.csv → Processed & cleaned dataset

2021_population.csv → Additional reference data (if applicable)

untitled(3).ipynb → Main project notebook

##  🏗️ Tech Stack

Programming Language: Python 🐍

Libraries & Frameworks: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

Tools: Jupyter Notebook, GitHub

##  🧩 Modeling Approach

Data Preprocessing

Handling missing values

Encoding categorical variables

Feature selection

Exploratory Data Analysis (EDA)

Heatmaps, scatter plots, and correlation analysis

Model Training

K-Nearest Neighbors (KNN)

Random Forest

XGBoost

Performance Evaluation

Metrics: RMSE, R²

Optimization: Gradient Descent

## ⚙️ How to Run the Project

Follow these steps to set up and run the project locally:

1️⃣ **Clone the Repository**
```bash
git clone https://github.com/your-username/food-waste-prediction.git
cd food-waste-prediction
2️⃣ Create a Virtual Environment (Optional but Recommended)

bash
Copy code
# Create virtual environment
python -m venv venv

# Activate on Mac/Linux
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
3️⃣ Install Dependencies

bash
Copy code
pip install -r requirements.txt
4️⃣ Run the Jupyter Notebook

bash
Copy code
jupyter notebook untitled(3).ipynb
