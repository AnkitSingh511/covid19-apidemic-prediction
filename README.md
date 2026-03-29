# 🦠 COVID-19 Epidemic Spread Prediction
### CodeCure AI Hackathon — Track C

## 📌 Overview
A machine learning pipeline that predicts COVID-19 epidemic 
spread and visualizes outbreak risk across countries 
using multi-source epidemiological data.

## 🚀 Features
- Prophet 30-day case forecasting per country
- Per-country risk classification (High / Medium / Low)
- Interactive Folium risk map with heatmap layer
- Multi-source data pipeline (OWID + JHU + Vaccination + Testing)
- Streamlit dashboard with 5 interactive tabs
- Random Forest feature importance analysis

## 🗂️ Project Structure
├── prediction.py       # ML pipeline — run this first
├── dashboard.py        # Streamlit dashboard — run this second
├── ml_risk_map.html    # Pre-generated interactive risk map
├── requirements.txt    # All dependencies
└── README.md

## ⚙️ How to Run

### 1. Install dependencies
pip install -r requirements.txt

### 2. Download datasets (see below) and place in project folder

### 3. Run ML pipeline first
python prediction.py

### 4. Launch dashboard
python -m streamlit run dashboard.py

## 📥 Dataset Download Links
- [OWID COVID-19 Data](https://github.com/owid/covid-19-data)
- [Johns Hopkins CSSE](https://github.com/CSSEGISandData/COVID-19)
- [Our World in Data — Vaccination](https://github.com/owid/covid-19-data/tree/master/public/data/vaccinations)
- [Our World in Data — Testing](https://github.com/owid/covid-19-data/tree/master/public/data/testing)

## 🛠️ Tech Stack
Python · Streamlit · Prophet · Scikit-learn · 
Plotly · Folium · Pandas · NumPy · OpenPyXL

## 📊 Models Used
| Model | Purpose |
| Facebook Prophet | 30-day case forecasting |
| Linear Regression | 14-day per-country prediction |
| Random Forest | Feature importance analysis |

https://github.com/shadow12546-sketch/covid19-epidemic-prediction
