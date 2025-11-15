def get_opponent_id(matchup, team_abbr_to_id, team_id):
    """Extract opponent team ID from matchup string."""
    if "@" in matchup:
        opponent_abbr = matchup.split(" @ ")[-1]
    else:
        opponent_abbr = matchup.split(" vs. ")[-1]
    return team_abbr_to_id.get(opponent_abbr, team_id)


def add_features(games_df, team_abbr_to_id):
    """Add all features."""
    print("\n[5/7] Engineering features...")
    
    print("  Adding opponent team ID...")
    games_df["OPP_TEAM_ID"] = games_df.apply(
        lambda row: get_opponent_id(row["MATCHUP"], team_abbr_to_id, row["TEAM_ID"]), axis=1
    )
    
    print("  Adding Home Game Advantage...")
    games_df["HGA"] = games_df["MATCHUP"].apply(lambda x: 0 if "@" in x else 1)
    
    print("  Adding Last Game Outcome...")
    games_df["LAST_GAME_OUTCOME"] = (
        games_df.groupby("TEAM_ID")["WIN"].shift(1).fillna(0)
    )
    
    print("  Adding EFG%...")
    games_df["EFG%"] = (
        games_df["FGM"] + (0.5 * games_df["FG3M"])
    ) / games_df["FGA"]
    
    print("  Adding TOV%...")
    games_df["TOV%"] = games_df["TOV"] / (
        games_df["FGA"] + 0.44 * games_df["FTA"] + games_df["TOV"]
    )
    
    print("  Adding FTR...")
    games_df["FTR"] = games_df["FTA"] / games_df["FGA"]
    
    print("  Adding TS%...")
    games_df["TS%"] = games_df["PTS"] / (
        2 * (games_df["FGA"] + (0.44 * games_df["FTA"]))
    )
    
    games_df["EFG%"] = games_df["EFG%"].fillna(0)
    games_df["TOV%"] = games_df["TOV%"].fillna(0)
    games_df["FTR"] = games_df["FTR"].fillna(0)
    games_df["TS%"] = games_df["TS%"].fillna(0)
    
    print("  Feature engineering complete")
    return games_df


def get_feature_columns():
    """Return list of feature columns."""
    return [
        "TEAM_ID",
        "OPP_TEAM_ID",
        "PTS",
        "OREB",
        "DREB",
        "REB",
        "AST",
        "BLK",
        "TOV",
        "EFG%",
        "TOV%",
        "FTR",
        "TS%",
        "HGA",
        "LAST_GAME_OUTCOME",
    ]
