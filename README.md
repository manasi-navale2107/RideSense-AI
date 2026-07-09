# 🚖 RideSense AI – Intelligent Taxi Trip Duration Prediction

RideSense AI is an end-to-end Machine Learning project that predicts the duration of taxi trips in New York City using historical trip data. The project leverages spatio-temporal feature engineering, regression models, and exploratory data analysis to estimate travel time accurately and generate meaningful transportation insights.

The objective is to solve a real-world business problem faced by ride-hailing platforms such as Uber, Lyft, and Ola by providing accurate trip duration estimates that can improve customer experience, driver allocation, route planning, and operational efficiency.

---

# 📌 Business Problem

Ride-hailing platforms receive thousands of trip requests every minute.

Providing an accurate Estimated Time of Arrival (ETA) is essential because it helps:

- Improve customer satisfaction
- Optimize driver allocation
- Reduce waiting time
- Improve route planning
- Support dynamic pricing strategies
- Enhance operational efficiency

RideSense AI predicts the expected duration of a taxi trip using historical trip information and engineered geographical and temporal features.

---

# 📂 Dataset

**Dataset Name**

NYC Taxi Trip Duration

**Source**

Kaggle Playground Competition

https://www.kaggle.com/competitions/nyc-taxi-trip-duration

The dataset contains historical taxi trips in New York City, including:

- Pickup datetime
- Pickup coordinates
- Dropoff coordinates
- Passenger count
- Vendor ID
- Store and Forward flag
- Trip duration (Target Variable)

---

# 🎯 Objective

Build a Machine Learning regression model capable of accurately predicting taxi trip duration based on trip-related information.

---

# 📊 Project Workflow

```text
NYC Taxi Dataset
        │
        ▼
Data Cleaning
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Feature Engineering
        │
        ▼
Data Preprocessing
        │
        ▼
Train-Test Split
        │
        ▼
Regression Model Training
        │
        ▼
Model Comparison
        │
        ▼
Hyperparameter Tuning
        │
        ▼
Model Evaluation
        │
        ▼
Best Model Selection
        │
        ▼
Trip Duration Prediction
```

---

# ⚙️ Feature Engineering

Several meaningful features are created from the original dataset to improve prediction performance.

Examples include:

- Pickup Hour
- Pickup Day
- Pickup Month
- Day of Week
- Weekend Indicator
- Trip Distance
- Rush Hour Indicator
- Coordinate-based Features

These engineered features help the model capture both temporal and geographical travel patterns.

---

# 🤖 Machine Learning Models

The following regression algorithms will be compared:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor
- LightGBM Regressor

The best-performing model will be selected based on evaluation metrics.

---

# 📈 Evaluation Metrics

The models will be evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

---

# 🛠️ Tech Stack

### Programming Language

- Python

### Libraries

- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- LightGBM
- Matplotlib
- Seaborn
- Joblib

### Tools

- Jupyter Notebook
- Git
- GitHub

---

# 🚀 Installation

## Clone Repository

```bash
git clone https://github.com/manasi-navale2107/RideSense-AI.git
cd RideSense-AI
```

## Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Project

Open the notebooks in the following order:

1. Data Cleaning & EDA
2. Feature Engineering
3. Model Training
4. Model Evaluation
5. Prediction

---

# 📌 Future Improvements

- Interactive Streamlit Dashboard
- FastAPI Prediction API
- Real-time Trip Prediction
- Interactive Route Visualization
- Docker Support
- MLflow Experiment Tracking
- Cloud Deployment
- CI/CD Pipeline

---

# 💡 Key Learning Outcomes

- End-to-end Machine Learning Pipeline
- Exploratory Data Analysis
- Feature Engineering
- Regression Algorithms
- Hyperparameter Tuning
- Model Evaluation
- Urban Mobility Data Analysis
- Transportation Analytics


# 👩‍💻 Author

**Manasi Navale**

Artificial Intelligence & Data Science Undergraduate

GitHub: https://github.com/manasi-navale2107

LinkedIn: https://linkedin.com/in/manasinavale07
