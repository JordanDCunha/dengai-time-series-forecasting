# DengAI: Disease Spread Prediction (Machine Learning)

## Overview

This project was developed as part of **HackML 2026**, a Kaggle-style machine learning competition focused on predicting disease spread.

The goal is to build a model that predicts weekly dengue fever cases using environmental and climate data.

The dataset includes weather, precipitation, humidity, and vegetation features for two cities:

- San Juan (Puerto Rico)
- Iquitos (Peru)

## Problem Type

Time-series regression

## Approach

### Feature Engineering

- Seasonal encoding (sin/cos transformation of week of year)
- Lag features (previous dengue case counts)
- Rolling statistics (moving averages and standard deviation)

### Model

- XGBoost Regressor
- Separate models trained for each city to capture location-specific patterns

### Validation Strategy

- TimeSeriesSplit (to prevent data leakage)
- Evaluated using Mean Absolute Error (MAE)

## Results

- Cross-validation MAE: ~9.26
- Performance is competitive with top HackML 2026 leaderboard scores (~368–700 range)

## How to Run

```bash
pip install -r requirements.txt
python train.py
