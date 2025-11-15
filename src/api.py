from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from data_fetcher import fetch_all_games
from data_processor import filter_modern_seasons, clean_data
from feature_engineer import add_features
from model import train_model

app = FastAPI(title="NBA Game Prediction API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage for pipeline results
pipeline_data = {
    "games": None,
    "model_results": None,
    "team_abbr_to_id": None,
    "id_to_abbr": None,
    "status": "not_run"
}

class PipelineResponse(BaseModel):
    status: str
    message: str
    games_count: Optional[int] = None
    model_accuracy: Optional[float] = None


@app.get("/")
def read_root():
    return {"message": "NBA Game Prediction API"}


@app.post("/run-pipeline", response_model=PipelineResponse)
async def run_pipeline():
    """Run the full prediction pipeline."""
    try:
        print("=" * 60)
        print("NBA Game Prediction Model - Starting")
        print("=" * 60)

        # Fetch data
        all_games, team_abbr_to_id = fetch_all_games()
        id_to_abbr = {v: k for k, v in team_abbr_to_id.items()}

        # Process data
        games_modern = filter_modern_seasons(all_games)
        games_modern = clean_data(games_modern)
        games_modern = add_features(games_modern, team_abbr_to_id)

        # Train model
        print("\n[6/7] Training model...")
        games_modern = train_model(games_modern)
        
        # Calculate test set accuracy
        test_set = games_modern[games_modern["IS_TEST_SET"] == 1]
        accuracy = test_set["CORRECT_PREDICTION"].mean() if len(test_set) > 0 else 0.0
        correct_predictions = test_set["CORRECT_PREDICTION"].sum() if len(test_set) > 0 else 0
        
        # Store results
        pipeline_data["games"] = games_modern
        pipeline_data["team_abbr_to_id"] = team_abbr_to_id
        pipeline_data["id_to_abbr"] = id_to_abbr
        pipeline_data["model_results"] = {
            "accuracy": float(accuracy),
            "test_size": len(test_set),
            "correct_predictions": int(correct_predictions)
        }
        pipeline_data["status"] = "completed"

        return PipelineResponse(
            status="completed",
            message="Pipeline completed successfully",
            games_count=len(games_modern),
            model_accuracy=float(accuracy)
        )
    except Exception as e:
        pipeline_data["status"] = "error"
        import traceback
        error_trace = traceback.format_exc()
        print(f"ERROR in pipeline: {e}")
        print(error_trace)
        raise HTTPException(status_code=500, detail=f"{str(e)}\n\n{error_trace}")


@app.get("/teams")
def get_teams():
    """Get list of all teams with their IDs and abbreviations."""
    if pipeline_data["id_to_abbr"] is None:
        raise HTTPException(status_code=404, detail="Pipeline not run yet. Call /run-pipeline first.")
    
    teams_list = [
        {"team_id": team_id, "team_abbr": abbr}
        for team_id, abbr in sorted(pipeline_data["id_to_abbr"].items())
    ]
    
    return {"teams": teams_list}


@app.get("/model-results")
def get_model_results():
    """Get model training results."""
    if pipeline_data["model_results"] is None:
        raise HTTPException(status_code=404, detail="Pipeline not run yet. Call /run-pipeline first.")
    
    return pipeline_data["model_results"]


@app.get("/games")
def get_games(limit: int = 100, offset: int = 0):
    """Get game data with predictions."""
    if pipeline_data["games"] is None:
        raise HTTPException(status_code=404, detail="Pipeline not run yet. Call /run-pipeline first.")
    
    games_df = pipeline_data["games"]
    id_to_abbr = pipeline_data["id_to_abbr"]
    
    # Convert dates
    games_df_copy = games_df.copy()
    if games_df_copy["GAME_DATE"].dtype != 'object':
        games_df_copy["GAME_DATE"] = games_df_copy["GAME_DATE"].dt.strftime("%Y-%m-%d")
    
    # Get paginated games
    games_data = games_df_copy.iloc[offset:offset+limit]
    
    # Format response
    result = []
    for _, row in games_data.iterrows():
        result.append({
            "game_date": row["GAME_DATE"],
            "team_id": int(row["TEAM_ID"]),
            "team_abbr": id_to_abbr.get(int(row["TEAM_ID"]), f"Team {int(row['TEAM_ID'])}"),
            "opp_team_id": int(row["OPP_TEAM_ID"]),
            "opp_team_abbr": id_to_abbr.get(int(row["OPP_TEAM_ID"]), f"Team {int(row['OPP_TEAM_ID'])}"),
            "is_home": int(row.get("HGA", 0)),
            "actual_win": int(row["WIN"]),
            "predicted_win": int(row.get("PREDICTED_WIN", 0)),
            "predicted_win_prob": float(row.get("PREDICTED_WIN_PROB", 0.0)),
            "correct_prediction": int(row.get("CORRECT_PREDICTION", 0)),
            "is_test_set": int(row.get("IS_TEST_SET", 0)),
            "points": float(row.get("PTS", 0)),
        })
    
    return {
        "total": len(games_df),
        "limit": limit,
        "offset": offset,
        "games": result
    }


@app.get("/status")
def get_status():
    """Get pipeline status."""
    return {
        "status": pipeline_data["status"],
        "has_data": pipeline_data["games"] is not None,
        "has_results": pipeline_data["model_results"] is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
