"use client";

import { useState, useEffect } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface TeamInfo {
  team_id: number;
  team_abbr: string;
}

interface ModelResults {
  accuracy: number;
  test_size: number;
  correct_predictions: number;
}

interface Game {
  game_date: string;
  team_id: number;
  team_abbr: string;
  opp_team_id: number;
  opp_team_abbr: string;
  is_home: number;
  actual_win: number;
  predicted_win: number;
  predicted_win_prob: number;
  correct_prediction: number;
  is_test_set: number;
  points: number;
}

export default function Home() {
  const [status, setStatus] = useState<string>("not_run");
  const [loading, setLoading] = useState(false);
  const [teamsList, setTeamsList] = useState<TeamInfo[]>([]);
  const [modelResults, setModelResults] = useState<ModelResults | null>(null);
  const [games, setGames] = useState<Game[]>([]);
  const [gamesLoading, setGamesLoading] = useState(false);
  const [gamesOffset, setGamesOffset] = useState(0);
  const [gamesTotal, setGamesTotal] = useState(0);

  const checkStatus = async () => {
    try {
      const response = await fetch(`${API_URL}/status`);
      const data = await response.json();
      setStatus(data.status);
      return data.has_data;
    } catch (error) {
      console.error("Error checking status:", error);
      return false;
    }
  };

  const runPipeline = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/run-pipeline`, {
        method: "POST",
      });
      const data = await response.json();
      setStatus(data.status);
      if (data.status === "completed") {
        await loadData();
      }
    } catch (error) {
      console.error("Error running pipeline:", error);
      alert("Error running pipeline. Check console for details.");
    } finally {
      setLoading(false);
    }
  };

  const loadData = async () => {
    try {
      // Load teams list
      const teamsResponse = await fetch(`${API_URL}/teams`);
      const teamsData = await teamsResponse.json();
      setTeamsList(teamsData.teams || []);

      // Load model results
      const resultsResponse = await fetch(`${API_URL}/model-results`);
      const resultsData = await resultsResponse.json();
      setModelResults(resultsData);

      // Load games
      await loadGames(0);
    } catch (error) {
      console.error("Error loading data:", error);
    }
  };

  const loadGames = async (offset: number = 0, limit: number = 100) => {
    setGamesLoading(true);
    try {
      const response = await fetch(
        `${API_URL}/games?limit=${limit}&offset=${offset}`
      );
      const data = await response.json();
      setGames(data.games || []);
      setGamesTotal(data.total || 0);
      setGamesOffset(offset);
    } catch (error) {
      console.error("Error loading games:", error);
    } finally {
      setGamesLoading(false);
    }
  };

  useEffect(() => {
    checkStatus().then((hasData) => {
      if (hasData) {
        loadData();
      }
    });
  }, []);

  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-black p-8">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-4xl font-bold mb-8 text-black dark:text-zinc-50">
          NBA Game Prediction Dashboard
        </h1>

        {/* Control Panel */}
        <div className="bg-white dark:bg-zinc-900 rounded-lg shadow-lg p-6 mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-2xl font-semibold mb-2 text-black dark:text-zinc-50">
                Pipeline Status
              </h2>
              <p className="text-zinc-600 dark:text-zinc-400">
                Status: <span className="font-semibold">{status}</span>
              </p>
            </div>
            <button
              onClick={runPipeline}
              disabled={loading}
              className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? "Running..." : "Run Pipeline"}
            </button>
          </div>
        </div>

        {/* Model Results */}
        {modelResults && (
          <div className="bg-white dark:bg-zinc-900 rounded-lg shadow-lg p-6 mb-8">
            <h2 className="text-2xl font-semibold mb-4 text-black dark:text-zinc-50">
              Model Results
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
                <p className="text-sm text-zinc-600 dark:text-zinc-400">
                  Accuracy
                </p>
                <p className="text-3xl font-bold text-blue-600 dark:text-blue-400">
                  {(modelResults.accuracy * 100).toFixed(2)}%
                </p>
              </div>
              <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg">
                <p className="text-sm text-zinc-600 dark:text-zinc-400">
                  Correct Predictions
                </p>
                <p className="text-3xl font-bold text-green-600 dark:text-green-400">
                  {modelResults.correct_predictions}
                </p>
              </div>
              <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg">
                <p className="text-sm text-zinc-600 dark:text-zinc-400">
                  Test Set Size
                </p>
                <p className="text-3xl font-bold text-purple-600 dark:text-purple-400">
                  {modelResults.test_size}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Games with Predictions */}
        {games.length > 0 && (
          <div className="bg-white dark:bg-zinc-900 rounded-lg shadow-lg p-6">
            <h2 className="text-2xl font-semibold mb-4 text-black dark:text-zinc-50">
              Games with Predictions ({gamesTotal} total)
            </h2>

            {/* Pagination */}
            <div className="flex items-center justify-between mb-4">
              <div className="text-sm text-zinc-600 dark:text-zinc-400">
                Showing {gamesOffset + 1} -{" "}
                {Math.min(gamesOffset + games.length, gamesTotal)} of{" "}
                {gamesTotal}
              </div>
              <div className="flex gap-2">
                <button
                  onClick={() => loadGames(Math.max(0, gamesOffset - 100))}
                  disabled={gamesOffset === 0 || gamesLoading}
                  className="px-4 py-2 bg-zinc-200 dark:bg-zinc-800 text-black dark:text-zinc-50 rounded hover:bg-zinc-300 dark:hover:bg-zinc-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Previous
                </button>
                <button
                  onClick={() => loadGames(gamesOffset + 100)}
                  disabled={
                    gamesOffset + games.length >= gamesTotal || gamesLoading
                  }
                  className="px-4 py-2 bg-zinc-200 dark:bg-zinc-800 text-black dark:text-zinc-50 rounded hover:bg-zinc-300 dark:hover:bg-zinc-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Next
                </button>
              </div>
            </div>

            {/* Games Table */}
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-zinc-200 dark:border-zinc-700">
                    <th className="text-left p-2 text-zinc-600 dark:text-zinc-400">
                      Date
                    </th>
                    <th className="text-left p-2 text-zinc-600 dark:text-zinc-400">
                      Matchup
                    </th>
                    <th className="text-center p-2 text-zinc-600 dark:text-zinc-400">
                      Points
                    </th>
                    <th className="text-center p-2 text-zinc-600 dark:text-zinc-400">
                      Predicted
                    </th>
                    <th className="text-center p-2 text-zinc-600 dark:text-zinc-400">
                      Actual
                    </th>
                    <th className="text-center p-2 text-zinc-600 dark:text-zinc-400">
                      Prob
                    </th>
                    <th className="text-center p-2 text-zinc-600 dark:text-zinc-400">
                      Result
                    </th>
                    <th className="text-center p-2 text-zinc-600 dark:text-zinc-400">
                      Set
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {games.map((game, idx) => (
                    <tr
                      key={idx}
                      className={`border-b border-zinc-100 dark:border-zinc-800 ${
                        game.correct_prediction
                          ? "bg-green-50/50 dark:bg-green-900/10"
                          : "bg-red-50/50 dark:bg-red-900/10"
                      }`}
                    >
                      <td className="p-2 text-zinc-700 dark:text-zinc-300">
                        {new Date(game.game_date).toLocaleDateString()}
                      </td>
                      <td className="p-2">
                        <span className="font-medium text-black dark:text-zinc-50">
                          {game.team_abbr}
                        </span>
                        <span className="text-zinc-500 dark:text-zinc-400 mx-1">
                          {game.is_home ? "vs" : "@"}
                        </span>
                        <span className="font-medium text-black dark:text-zinc-50">
                          {game.opp_team_abbr}
                        </span>
                      </td>
                      <td className="p-2 text-center text-zinc-700 dark:text-zinc-300">
                        {game.points}
                      </td>
                      <td className="p-2 text-center">
                        <span
                          className={`px-2 py-1 rounded ${
                            game.predicted_win
                              ? "bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-300"
                              : "bg-gray-100 dark:bg-gray-800 text-gray-800 dark:text-gray-300"
                          }`}
                        >
                          {game.predicted_win ? "Win" : "Loss"}
                        </span>
                      </td>
                      <td className="p-2 text-center">
                        <span
                          className={`px-2 py-1 rounded ${
                            game.actual_win
                              ? "bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-300"
                              : "bg-red-100 dark:bg-red-900/30 text-red-800 dark:text-red-300"
                          }`}
                        >
                          {game.actual_win ? "Win" : "Loss"}
                        </span>
                      </td>
                      <td className="p-2 text-center text-zinc-700 dark:text-zinc-300">
                        {(game.predicted_win_prob * 100).toFixed(1)}%
                      </td>
                      <td className="p-2 text-center">
                        {game.correct_prediction ? (
                          <span className="text-green-600 dark:text-green-400">
                            ✓
                          </span>
                        ) : (
                          <span className="text-red-600 dark:text-red-400">
                            ✗
                          </span>
                        )}
                      </td>
                      <td className="p-2 text-center">
                        <span
                          className={`px-2 py-1 rounded text-xs font-medium ${
                            game.is_test_set
                              ? "bg-purple-100 dark:bg-purple-900/30 text-purple-800 dark:text-purple-300"
                              : "bg-yellow-100 dark:bg-yellow-900/30 text-yellow-800 dark:text-yellow-300"
                          }`}
                        >
                          {game.is_test_set ? "Test" : "Train"}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
