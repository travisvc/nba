import pandas as pd

def filter_modern_seasons(games_df):
    """Filter games to modern seasons (2019-2024)."""
    print("\n[3/7] Filtering games for modern seasons (2019-2024)...")
    games_modern = games_df[
        (games_df.SEASON_ID.str[-4:] == "2019")
        | (games_df.SEASON_ID.str[-4:] == "2020")
        | (games_df.SEASON_ID.str[-4:] == "2021")
        | (games_df.SEASON_ID.str[-4:] == "2022")
        | (games_df.SEASON_ID.str[-4:] == "2023")
        | (games_df.SEASON_ID.str[-4:] == "2024")
    ].copy()
    print(f"  Filtered to {len(games_modern)} games from modern seasons")
    return games_modern

def clean_data(games_df):
    """Clean and process the game data."""
    print("\n[4/7] Processing and cleaning data...")
    print("  Converting GAME_DATE to datetime...")
    games_df["GAME_DATE"] = pd.to_datetime(games_df["GAME_DATE"])
    print("  Sorting games by date...")
    games_df.sort_values(by="GAME_DATE", inplace=True)

    # Add binary "WIN" column
    print("  Creating WIN column...")
    games_df["WIN"] = games_df["WL"].apply(lambda x: 1 if x == "W" else 0)

    # Convert stat columns to float (only those used in features)
    print("  Converting stat columns to float...")
    stat_columns = [
        "PTS", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
        "REB", "OREB", "DREB", "AST", "BLK", "TOV"
    ]
    for col in stat_columns:
        games_df[col] = games_df[col].astype(float)
    
    return games_df
