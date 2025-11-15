from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder
import pandas as pd
import time
from requests.exceptions import ReadTimeout, RequestException


def fetch_team_games(team_id, team_name, max_retries=3, delay=2):
    """Fetch games for a team with retry logic and delays."""
    for attempt in range(max_retries):
        try:
            gamefinder = leaguegamefinder.LeagueGameFinder(
                team_id_nullable=team_id, 
                season_type_nullable="Regular Season"
            )
            games = gamefinder.get_data_frames()[0]
            return games
        except (ReadTimeout, RequestException) as e:
            if attempt < max_retries - 1:
                wait_time = delay * (2 ** attempt)  # Exponential backoff
                print(f"  Timeout for {team_name}, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                print(f"  Failed to fetch games for {team_name} after {max_retries} attempts")
                return None
        except Exception as e:
            print(f"  Error fetching games for {team_name}: {e}")
            return None
    return None


def fetch_all_games():
    """Fetch all NBA games for all teams."""
    print("\n[1/7] Loading NBA teams...")
    nba_teams = teams.get_teams()
    team_abbr_to_id = {team["abbreviation"]: team["id"] for team in nba_teams}
    print(f"  Found {len(nba_teams)} teams")
    
    all_games = pd.DataFrame()
    print(f"Fetching games for {len(nba_teams)} teams...")
    successful = 0
    failed = 0

    for idx, team in enumerate(nba_teams, 1):
        team_id = team["id"]
        team_name = team["abbreviation"]
        print(f"[{idx}/{len(nba_teams)}] Fetching games for {team_name} (ID: {team_id})...")
        
        games = fetch_team_games(team_id, team_name)
        if games is not None and not games.empty:
            all_games = pd.concat([all_games, games], ignore_index=True)
            successful += 1
            print(f"  Successfully fetched {len(games)} games")
        else:
            failed += 1
        
        # Add delay between requests to avoid rate limiting
        if idx < len(nba_teams):
            time.sleep(1)  # 1 second delay between requests

    print(f"\n[2/7] Completed fetching: {successful} teams successful, {failed} teams failed")
    print(f"  Total games fetched: {len(all_games)}")

    if all_games.empty:
        raise ValueError("No games were fetched. Please check your network connection and try again.")
    
    return all_games, team_abbr_to_id

