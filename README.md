# Car Price Prediction API

Machine Learning project for predicting used car prices with XGBoost and serving predictions via a FastAPI REST API.
Built with **Python, Scikit-learn, XGBoost, FastAPI, and Poetry**. 

Interactive docs: http://127.0.0.1:8000/docs

### Project Structure
```
car-price-ml/
├── api/
│   └── app.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── car_price_model_XGBoost_Tuned.pkl
│   └── feature_config.json
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
├── pyproject.toml
└── README.md
```

### Project Overview

EDA & Feature Engineering → notebooks/eda.ipynb

Trained Model → XGBoost with preprocessing pipeline

API → built with FastAPI

Deployment Ready → uvicorn api.app:app --reload

### Quickstart

#### Clone the repo
```
git clone https://github.com/your-username/car-price-ml.git
cd car-price-ml
```

#### Install dependencies
Install dependencies using Poetry:
```
poetry install
```

Activate the virtual environment:
```
poetry shell
```

#### Training the Models (optional)
To train the machine learning models, run:
```
poetry run python -m src.train
```

#### Run the API locally
```
poetry run uvicorn api.app:app --reload
```

Visit: http://127.0.0.1:8000/docs



## Run with Docker
Build the image
```
docker build -t car-price-api .
```

Run the container
```
docker run -p 8000:8000 car-price-api
```

Visit: http://0.0.0.0:8000/docs


### Example Prediction

Request:
```json
[
  {
    "manufacturer": "Toyota",
    "model": "Corolla",
    "prod_year": 2017,
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
    "is_Turbo": "No"
  }
]
```

Response:
```json
{
  "predictions": [
    25608.908203125
  ]
}
```

### Model

Tested models: DecisionTree, RandomForest, XGBoost

Best model: XGBoost (tuned)

Stored in:

models/car_price_model_XGBoost_Tuned.pkl

models/feature_config.json