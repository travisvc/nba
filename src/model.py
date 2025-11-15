from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from feature_engineer import get_feature_columns


def train_model(games_df):
    """Train model with simple train/test split."""
    print("  Training model...")
    
    # Encode team IDs and game IDs
    le_team = LabelEncoder()
    le_opp = LabelEncoder()
    le_game = LabelEncoder()
    
    games_df["TEAM_ID"] = le_team.fit_transform(games_df["TEAM_ID"])
    games_df["OPP_TEAM_ID"] = le_opp.fit_transform(games_df["OPP_TEAM_ID"])
    games_df["GAME_ID"] = le_game.fit_transform(games_df["GAME_ID"])
    
    # Get features (only use columns that exist)
    feature_cols = get_feature_columns()
    available_cols = [col for col in feature_cols if col in games_df.columns]
    if len(available_cols) < len(feature_cols):
        missing = set(feature_cols) - set(available_cols)
        print(f"  Warning: Missing columns {missing}, using {len(available_cols)} features")
    X = games_df[available_cols]
    y = games_df["WIN"]
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=175, max_depth=25, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"  Test accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Add predictions to dataframe
    games_df["IS_TEST_SET"] = 0
    test_indices = X_test.index
    games_df.loc[test_indices, "IS_TEST_SET"] = 1
    games_df.loc[test_indices, "PREDICTED_WIN"] = y_pred
    games_df.loc[test_indices, "PREDICTED_WIN_PROB"] = model.predict_proba(X_test)[:, 1]
    games_df.loc[test_indices, "CORRECT_PREDICTION"] = (y_pred == y_test).astype(int)
    
    # Fill training set predictions
    train_indices = X_train.index
    games_df.loc[train_indices, "PREDICTED_WIN"] = model.predict(X_train)
    games_df.loc[train_indices, "PREDICTED_WIN_PROB"] = model.predict_proba(X_train)[:, 1]
    games_df.loc[train_indices, "CORRECT_PREDICTION"] = (model.predict(X_train) == y_train).astype(int)
    
    return games_df
