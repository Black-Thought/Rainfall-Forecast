# 🌧️ Rainfall Forecast API

A FastAPI-based service to forecast rainfall using a trained XGBoost model.


## 🚀 Features

* 📊 ML-based rainfall prediction (XGBoost)
* ⚡ FastAPI backend
* 🐳 Dockerized deployment
* 🔁 Time-series forecasting (supports future dates)
* 📦 Clean modular architecture


## Setup (Local)

### 1. Create virtual environment

```bash
python -m venv .venv
```

Activate:

```bash
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```


### 2. Install dependencies

```bash
pip install -r requirements.txt
```


### 3. Run FastAPI

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```


### 4. Open API Docs

```
http://localhost:8000/docs
```

---

## 🐳 Run with Docker


```bash
docker compose up --build
```

```bash
docker compose down
```

## 🌐 API Endpoints

### ✅ Health Check

```http
GET /
```

Response:

```json
{
  "status": "API is running"
}
```

---

### 🌧️ Forecast Rainfall

```http
POST /forecast/rainfall
```

#### Request Body

```json
{
  "station_name": "Belgaum",
  "start_date": "2026-05-01",
  "num_days": 5
}
```

---

#### Response

```json
{
  "station_name": "Belgaum",
  "start_date": "2026-05-01",
  "num_days": 5,
  "predictions": [
    {
      "date_of_record": "2026-05-01",
      "predicted_rainfall": 2.34
    }
  ]
}
```

---

## 🧪 Running Tests

```bash
pytest
```

---

## ⚠️ Notes

* Ensure model file exists:

```bash
app/models/xgb_rainfall_model.pkl
```

* Dataset path:

```bash
app/data/processed_weather_data.csv
```