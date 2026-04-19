# 🌧️ Rainfall Forecasting using Zone-wise XGBoost Models

## 📌 Overview

This study implements **zone-specific XGBoost regression models** to forecast rainfall across India. Stations are classified into three monsoon regimes:

* **SW_MONSOON** (South-West dominant)
* **NE_MONSOON** (North-East dominant)
* **LOW_MONSOON** (Low rainfall regions)

Each zone has an independent model trained using a **station-based split**:

* **70% stations → training**
* **30% stations → testing**

---

## ⚙️ Model Training Summary

| Zone        | Training Stations | Testing Stations |
| ----------- | ----------------- | ---------------- |
| SW_MONSOON  | 202               | 87               |
| LOW_MONSOON | 57                | 25               |
| NE_MONSOON  | 24                | 11               |

---

## 📊 Evaluation Metrics

The following metrics are used:

* **RMSE** → Root Mean Squared Error
* **MAE** → Mean Absolute Error
* **MSE** → Mean Squared Error
* **NSE** → Nash–Sutcliffe Efficiency
* **R²** → Coefficient of Determination

---

# 🌧️ Zone-wise Results

---

## 🌊 SW_MONSOON Zone

### 📍 Sample Station Performance

| Station           | RMSE  | MAE  | NSE   | R²    |
| ----------------- | ----- | ---- | ----- | ----- |
| Jamshedpur        | 5.02  | 2.60 | 0.629 | 0.629 |
| Hyderabad Airport | 10.66 | 4.89 | 0.183 | 0.183 |
| Varanasi          | 9.93  | 4.15 | 0.378 | 0.378 |
| Jhansi            | 10.10 | 3.90 | 0.357 | 0.357 |
| Raxaul            | 7.09  | 3.25 | 0.671 | 0.671 |

---

### 🔷 Zone Summary

| Metric | Value    |
| ------ | -------- |
| RMSE   | **8.61** |
| MAE    | **3.87** |
| MSE    | 82.43    |
| NSE    | 0.393    |
| R²     | 0.393    |

---

### 🧠 Insight

* Moderate performance
* Struggles with **high variability rainfall regions**
* NSE ~0.39 → acceptable but not strong

---

## 🌵 LOW_MONSOON Zone

### 📍 Sample Station Performance

| Station   | RMSE | MAE  | NSE   | R²    |
| --------- | ---- | ---- | ----- | ----- |
| Vidisha   | 2.79 | 1.11 | 0.819 | 0.819 |
| Rohtak    | 4.17 | 1.60 | 0.494 | 0.494 |
| Bellary   | 2.99 | 1.29 | 0.642 | 0.642 |
| Narnaul   | 3.59 | 1.30 | 0.599 | 0.599 |
| Jaisalmer | 7.04 | 2.33 | 0.188 | 0.188 |

---

### 🔷 Zone Summary

| Metric | Value    |
| ------ | -------- |
| RMSE   | **5.13** |
| MAE    | **2.01** |
| MSE    | 29.18    |
| NSE    | 0.536    |
| R²     | 0.536    |

---

### 🧠 Insight

* **Best performing zone**
* Rainfall is **less volatile → easier to predict**
* NSE ~0.54 → good model reliability

---

## 🌧️ NE_MONSOON Zone

### 📍 Sample Station Performance

| Station       | RMSE  | MAE  | NSE    | R²     |
| ------------- | ----- | ---- | ------ | ------ |
| Pondicherry   | 7.53  | 4.06 | 0.792  | 0.792  |
| Tirupati      | 5.17  | 3.01 | 0.560  | 0.560  |
| Cuddalore     | 13.82 | 5.90 | 0.479  | 0.479  |
| Kodaikanal    | 11.69 | 9.75 | -1.032 | -1.032 |
| Kanniyakumari | 12.10 | 6.04 | -0.023 | -0.023 |

---

### 🔷 Zone Summary

| Metric | Value     |
| ------ | --------- |
| RMSE   | **10.22** |
| MAE    | **5.18**  |
| MSE    | 113.54    |
| NSE    | 0.272     |
| R²     | 0.272     |

---

### 🧠 Insight

* **Worst performing zone**
* Highly seasonal and erratic rainfall
* Negative NSE at some stations → poor generalization

---

# 🔥 Overall Model Performance

| Metric | Value    |
| ------ | -------- |
| RMSE   | **7.99** |
| MAE    | **3.69** |
| MSE    | 75.05    |
| NSE    | 0.400    |
| R²     | 0.400    |

---

# 📌 Final Zone Comparison

| Zone        | RMSE     | MAE      | NSE       |
| ----------- | -------- | -------- | --------- |
| LOW_MONSOON | **5.13** | **2.01** | **0.536** |
| SW_MONSOON  | 8.61     | 3.87     | 0.393     |
| NE_MONSOON  | 10.22    | 5.18     | 0.272     |

---

# 🧠 Key Insights

### ✅ 1. Low rainfall regions are easiest to predict

* Stable rainfall patterns
* Lower variance
* High NSE

---

### ⚠️ 2. SW monsoon is moderately predictable

* Large spatial variability
* Orographic effects (Western Ghats)

---

### ❌ 3. NE monsoon is hardest

* Short duration, intense bursts
* High unpredictability
* Poor model generalization


# 🚀 Conclusion

* Zone-based modeling significantly improves realism
* However:

  * Performance varies **strongly by climatic regime**
* Future improvements:

  * Add **seasonal features**
  * Use **sequence models (LSTM/Transformer)**
  * Incorporate **geospatial features**