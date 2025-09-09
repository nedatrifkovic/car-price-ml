# 🚗 Car Price Prediction API

Machine Learning project for predicting used car prices with XGBoost and serving predictions via a FastAPI REST API.

🔗 Interactive docs: http://127.0.0.1:8000/docs

### 📂 Project Overview

📊 EDA & Feature Engineering → notebooks/eda.ipynb

🤖 Trained Model → XGBoost with preprocessing pipeline

⚡ API → built with FastAPI

🛠 Deployment Ready → uvicorn api.app:app --reload

### 🚀 Quickstart

#### Clone the repo
git clone https://github.com/your-username/car-price-ml.git
cd car-price-ml

#### Install dependencies
Install dependencies using Poetry:
poetry install


Activate the virtual environment:
poetry shell

#### Training the Models
To train the machine learning models, run:
poetry run python -m src.train


#### Run the API
poetry run uvicorn api.app:app --reload

Visit: 👉 http://127.0.0.1:8000/docs

### 📡 Example Prediction

Request:
{
    "manufacturer": "Toyota",
    "model": "Corolla",
    "prod_year": 2015,
    "category": "Sedan",
    "leather_interior": "No",
    "fuel_type": "Petrol",
    "engine_volume": 1.6,
    "mileage": 120000,
    "cylinders": 4,
    "gear_box_type": "Automatic",
    "drive_wheels": "front",
    "doors": 5,
    "wheel": "Left wheel",
    "color": "White",
    "airbags": 6,
    "is_Turbo": "No",
}


Response:
{
  "predictions": [10500.45]
}


### 🧠 Model

Tested models: DecisionTree, RandomForest, XGBoost

Best model: XGBoost (tuned)

Stored in:

models/car_price_model_XGBoost_Tuned.pkl

models/feature_config.json

### 👩‍💻 Author

✨ Developed by Neda Stanojević