from nba_api.stats.endpoints import leaguegamefinder
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Fetch data from NBA API
def fetch_pacers_data(season='2024-25'):
    PACERS_TEAM_ID = '1610612754'
    gamefinder = leaguegamefinder.LeagueGameFinder(
        team_id_nullable=PACERS_TEAM_ID,
        season_nullable=season
    )
    games_df = gamefinder.get_data_frames()[0]
    return games_df

# Prepare and clean data, generate rolling features
def preprocess_data(df):
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values('GAME_DATE').reset_index(drop=True)
    df['IS_HOME'] = df['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
    df['WIN'] = df['WL'].apply(lambda x: 1 if x == 'W' else 0)
    df['POINT_DIFF'] = df['PTS'] - df['PTS'].shift(1)
    df['ROLLING_PTS'] = df['PTS'].rolling(window=5, min_periods=1).mean()
    df['ROLLING_PLUS_MINUS'] = df['PLUS_MINUS'].rolling(window=5, min_periods=1).mean()
    df['ROLLING_WIN_RATE'] = df['WIN'].rolling(window=5, min_periods=1).mean()
    df['IS_PLAYOFF'] = df['GAME_ID'].astype(str).str.startswith('004').astype(int)
    df = df.dropna()
    return df

# Split data into training and testing
def split_data(df):
    features = ['IS_HOME', 'POINT_DIFF', 'ROLLING_PTS', 'ROLLING_PLUS_MINUS', 'ROLLING_WIN_RATE']
    X = df[features]
    y = df['WIN']
    X_train = X[df['IS_PLAYOFF'] == 0]
    y_train = y[df['IS_PLAYOFF'] == 0]
    X_test = X[df['IS_PLAYOFF'] == 1]
    y_test = y[df['IS_PLAYOFF'] == 1]
    return X_train, X_test, y_train, y_test

# Train Random Forest model
def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    return rf

# Run Logistic Regression with exact match to notebook
def run_logistic_regression(X_train, y_train, X_test, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print("📈 Logistic Regression Accuracy on Playoff Games:", accuracy)
    print("\n📊 Classification Report:")
    print(report)
    return model

# Evaluate and print model results
def evaluate_models(rf_model, lr_model, X_test, y_test):
    for name, model in [('Random Forest', rf_model), ('Logistic Regression', lr_model)]:
        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds)
        print(f"\nModel: {name}\nAccuracy: {accuracy:.2f}\n{report}")

def predict_latest_game(X_test, rf_model, lr_model):
    latest_game = X_test.tail(1)
    rf_pred = rf_model.predict(latest_game)[0]
    lr_pred = lr_model.predict(latest_game)[0]

    print("\n🎯 Game 5 Prediction (Pacers vs Thunder):")
    print(f"Random Forest Prediction: {'Win' if rf_pred == 1 else 'Lose'}")
    print(f"Logistic Regression Prediction: {'Win' if lr_pred == 1 else 'Lose'}")

# Main execution
def main():
    raw_data = fetch_pacers_data()
    processed_data = preprocess_data(raw_data)
    X_train, X_test, y_train, y_test = split_data(processed_data)
    rf_model = train_random_forest(X_train, y_train)
    lr_model = run_logistic_regression(X_train, y_train, X_test, y_test)
    evaluate_models(rf_model, lr_model, X_test, y_test)
    predict_latest_game(X_test, rf_model, lr_model)

if __name__ == '__main__':
    main()

