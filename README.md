# Forecasting NBA match results using multivariate Generalized Autoregressive Score Models

Travis van Cornewal
(VU Student Number: 2731231)

supervisor: dr. Janneke van Brummelen
co-reader: dr. Jan Bauer

This folder contains the code and data used for my thesis, focused on forecasting NBA match using multivariate Generalized Autoregressive Score (GAS) models by modeling team scores as a pair of counts. We evaluate the use of the theoretically more robust Negative Binomial against the Poisson in an overdispersed data environment.

## Project Structure

```
- functions.py            # GAS model functions and utilities
- main.ipynb              # Main notebook for model training and forecasting
- nba_game.csv            # Historical NBA game data
- visualizations.ipynb    # Notebook for plotting and analysis
```

## Dataset

- **nba_game.csv**: Contains historical NBA match data used for training and testing the model.
- Columns: `game_date`, `team_name_home`, `team_name_away`, `pts_home`, `pts_away`, `season_type`

## Methodology

- Poisson and Negative Binomial GAS models are implemented.
- Time-varying attack and defense strengths are estimated for each team.
- SExpanding forecasts.
- Evaluation by Average Ranked Probability Score and Log Score, and model comparisons by the Diebold-Mariano Test.

## How to Use

1. Run the `main.ipynb` file in Jupyter to obtain all the forecast results.
2. Run the `visualizations.ipynb` file in Jupyter to create plots.
