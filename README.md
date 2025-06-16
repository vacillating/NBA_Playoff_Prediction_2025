# NBA Playoff Prediction 2025

This project uses data from the 2024-25 NBA season (fetched via the NBA API) to predict the outcomes of playoff games. Two models were used:

- **Random Forest Classifier**
- **Logistic Regression**

The dataset was automatically obtained via API, and rolling features like average points, win rate, and plus-minus were calculated. The models were trained on regular-season games and tested on playoff games.

## Run Instructions

1. Make sure `nba_api`, `pandas`, and `scikit-learn` are installed.
2. Run the main script:

```bash
python nba_prediction.py
```
