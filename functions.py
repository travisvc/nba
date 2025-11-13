import numpy as np
from scipy.special import gammaln
from scipy.optimize import minimize
from scipy.stats import norm, poisson, nbinom
from collections import defaultdict

def diebold_mariano(loss_benchmark, loss_candidate):
    d = loss_candidate - loss_benchmark  # benchmark - candidate
    T = len(d)
    d_mean = np.mean(d)
    d_var = np.var(d, ddof=1)
    dm_stat = d_mean / np.sqrt(d_var / T)
    p_value = 2 * (1 - norm.cdf(np.abs(dm_stat)))
    return dm_stat, p_value
    
def compute_toto_probs(mu_home, mu_away, distr='negbin', r=None, max_pts=200):
    """ Compute win probabilities for NBA based on score distributions. """
    x_vals = np.arange(0, max_pts + 1)
    y_vals = np.arange(0, max_pts + 1)

    if distr == 'poisson':
        pmf_home = poisson.pmf(x_vals[:, None], mu_home)
        pmf_away = poisson.pmf(y_vals[None, :], mu_away)
    elif distr == 'negbin':
        if r is None:
            raise ValueError("Negative binomial requires parameter 'r'")
        p_home = r / (r + mu_home)
        p_away = r / (r + mu_away)
        pmf_home = nbinom.pmf(x_vals[:, None], r, p_home)
        pmf_away = nbinom.pmf(y_vals[None, :], r, p_away)
    else:
        raise ValueError("distr must be either 'poisson' or 'negbin'")

    joint_pmf = pmf_home * pmf_away
    np.fill_diagonal(joint_pmf, 0)
    joint_pmf /= joint_pmf.sum()

    p_home_win = np.sum(np.tril(joint_pmf, -1))
    p_away_win = np.sum(np.triu(joint_pmf, 1))

    return p_home_win, p_away_win, joint_pmf

def run_gas_forecast_loop(train_test_data, test_data, all_teams, f1_estimate, model_class, compute_probs_fn, distr="negbin"):
    """ Expanding window forecast. """
    forecast_results = []
    latent_history = defaultdict(list)
    season_groups = test_data.groupby("season")

    for season, season_data in season_groups:
        print(f"\n=== Training for season {season} ===")
        season_start_date = season_data["game_date"].min()

        history_data = train_test_data[
            (train_test_data["game_date"] < season_start_date) &
            (train_test_data["season_type"].isin(["Regular Season", "Pre Season"]))
        ]

        model = model_class(all_teams, f1_estimate=f1_estimate)
        model.fit(history_data)
        f = model.f.copy()

        # Preseason 
        preseason_games = season_data[season_data["season_type"] == "Pre Season"]
        for date in sorted(preseason_games["game_date"].unique()):
            games_on_date = preseason_games[preseason_games["game_date"] == date]
            f = model.update_state(games_on_date, f)

            for team in all_teams:
                idx = model.team_idx[team]
                latent_history[team].append({
                    "team_idx": idx,
                    "date": date,
                    "attack": f[idx],
                    "defense": f[model.N + idx],
                    "season_type": "Pre Season"
                })

        # Regular season
        regular_games = season_data[season_data["season_type"] == "Regular Season"]
        print(f"\n=== Forecasting for season {season} ===")
        for forecast_date in sorted(regular_games["game_date"].unique()):
            games_today = regular_games[regular_games["game_date"] == forecast_date]

            for row in games_today.itertuples():
                i = model.team_idx.get(row.team_name_home)
                j = model.team_idx.get(row.team_name_away)
                if i is None or j is None:
                    continue

                fij = model._get_f_vector(i, j, f)
                mu_home, mu_away = model._means(fij)

                if distr == "negbin":
                    p_home_win, p_away_win, _ = compute_probs_fn(mu_home, mu_away, distr=distr, r=model.r)
                else:
                    p_home_win, p_away_win, _ = compute_probs_fn(mu_home, mu_away, distr="poisson")

                true_vec = [1, 0] if row.pts_home > row.pts_away else [0, 1]
                pred_vec = [p_home_win, p_away_win]
                rps = 0.5 * np.sum((np.cumsum(pred_vec) - np.cumsum(true_vec)) ** 2)
                log_score = -np.log(p_home_win if row.pts_home > row.pts_away else p_away_win)

                forecast_results.append({
                    "date": forecast_date,
                    "home_team": row.team_name_home,
                    "away_team": row.team_name_away,
                    "true_home": row.pts_home,
                    "true_away": row.pts_away,
                    "mu_home": mu_home,
                    "mu_away": mu_away,
                    "p_home_win": p_home_win,
                    "p_away_win": p_away_win,
                    "rps": rps,
                    "log_score": log_score
                })

            f = model.update_state(games_today, f)
            for team in all_teams:
                idx = model.team_idx[team]
                latent_history[team].append({
                    "team_idx": idx,
                    "date": forecast_date,
                    "attack": f[idx],
                    "defense": f[model.N + idx],
                    "season_type": "Regular Season"
                })

    print("Finished forecasting.")
    return forecast_results, latent_history


# Negative Binomial without intercept
class GASNegBin:
    def __init__(self, all_teams, param_guess=None, f1_estimate=None):
        self.teams = list(all_teams)
        self.team_idx = {team: i for i, team in enumerate(self.teams)}
        self.N = len(self.teams)

        self.a1 = 0.0005
        self.a2 = 0.0005
        self.b1 = 0.995
        self.b2 = 0.995
        self.delta = 0.025
        self.r = 1182

        self.f = np.zeros(2 * self.N)
        self.f1_estimate = f1_estimate if f1_estimate is not None else np.zeros(2 * self.N)
        self.omega = self._compute_omega()

        self.param_guess = param_guess if param_guess is not None else [
            self.a1, self.a2, self.b1, self.b2, self.delta, np.log(self.r)
        ]

    def _get_f_vector(self, i, j, f):
        return np.array([f[i], f[j], f[self.N + i], f[self.N + j]])

    def _means(self, fij):
        mu_home = np.exp(self.delta + fij[0] - fij[3])
        mu_away = np.exp(fij[1] - fij[2])
        return mu_home, mu_away

    def _neg_bin_logpmf(self, x, mu):
        r = self.r
        p = r / (r + mu)
        return gammaln(x + r) - gammaln(r) - gammaln(x + 1) + r * np.log(p) + x * np.log(1 - p)

    def _score(self, x, y, fij):
        mu1, mu2 = self._means(fij)
        r = self.r
        s1 = (x - mu1) / (1 + mu1 / r)
        s2 = (y - mu2) / (1 + mu2 / r)
        s3 = -(y - mu2) / (1 + mu2 / r)
        s4 = -(x - mu1) / (1 + mu1 / r)
        return np.array([s1, s2, s3, s4])

    def _set_f_vector(self, i, j, fij, f):
        f_new = f.copy()
        f_new[i] = fij[0]
        f_new[j] = fij[1]
        f_new[self.N + i] = fij[2]
        f_new[self.N + j] = fij[3]
        return f_new

    def _update_fij(self, fij, s_t, i, j):
        A = np.diag([self.a1, self.a1, self.a2, self.a2])
        B = np.diag([self.b1, self.b1, self.b2, self.b2])
        ω = np.array([self.omega[i], self.omega[j], self.omega[self.N + i], self.omega[self.N + j]])
        return ω + B @ fij + A @ s_t

    def _log_likelihood(self, data):
        f = self.f1_estimate.copy()
        loglik = 0.0
        for row in data.itertuples():
            i = self.team_idx[row.team_name_home]
            j = self.team_idx[row.team_name_away]
            x, y = row.pts_home, row.pts_away

            fij = self._get_f_vector(i, j, f)
            mu1, mu2 = self._means(fij)
            loglik += self._neg_bin_logpmf(x, mu1) + self._neg_bin_logpmf(y, mu2)

            s_t = self._score(x, y, fij)
            fij_new = self._update_fij(fij, s_t, i, j)
            f = self._set_f_vector(i, j, fij_new, f)

        self.f = f.copy()
        return -loglik

    def _compute_omega(self):
        B_diag = np.concatenate([np.full(self.N, self.b1), np.full(self.N, self.b2)])
        return self.f1_estimate * (1 - B_diag)

    def _objective(self, params, data):
        self.a1, self.a2, self.b1, self.b2, self.delta, log_r = params
        self.r = np.exp(log_r)
        self.omega = self._compute_omega()
        return self._log_likelihood(data)

    def fit(self, data):
        bounds = [
            (0.00001, 0.2),   # a1
            (0.00001, 0.2),   # a2
            (0.95, 0.9999),   # b1
            (0.95, 0.9999),   # b2
            (0.001, 0.1),     # delta
            (np.log(1), np.log(5000)),  # log(r)
        ]

        result = minimize(
            self._objective,
            x0=self.param_guess,
            args=(data,),
            method='L-BFGS-B',
            bounds=bounds,
            options={"disp": False, "gtol": 1e-8, "maxiter": 2000}
        )
        self.a1, self.a2, self.b1, self.b2, self.delta, log_r = result.x
        self.r = np.exp(log_r)
        self.param_guess = result.x
        self.omega = self._compute_omega()

        print("==== Optimization Results ====")
        print(f"a1={self.a1:.4f}, a2={self.a2:.4f}")
        print(f"b1={self.b1:.4f}, b2={self.b2:.4f}")
        print(f"delta={self.delta:.4f}, r={self.r:.4f}")
        if result.success:
            print("Optimization converged successfully.")
        else:
            print("Optimization failed to converge.")

    def update_state(self, round_data, f_t=None):
        f = self.f.copy() if f_t is None else f_t.copy()
        s = np.zeros(2 * self.N)
        teams_playing = set()

        for row in round_data.itertuples():
            i = self.team_idx[row.team_name_home]
            j = self.team_idx[row.team_name_away]
            fij = self._get_f_vector(i, j, f)
            score = self._score(row.pts_home, row.pts_away, fij)

            s[i] += score[0]
            s[j] += score[1]
            s[self.N + i] += score[2]
            s[self.N + j] += score[3]
            teams_playing.update([i, j])

        for idx in range(self.N):
            if idx in teams_playing:
                f[idx] = self.omega[idx] + self.b1 * f[idx] + self.a1 * s[idx]
                f[self.N + idx] = self.omega[self.N + idx] + self.b2 * f[self.N + idx] + self.a2 * s[self.N + idx]
            else:
                f[idx] = self.omega[idx] + self.b1 * f[idx]
                f[self.N + idx] = self.omega[self.N + idx] + self.b2 * f[self.N + idx]

        return f

# Negative Binomial with intercept
class GASNegBinIntercept:
    def __init__(self, all_teams, param_guess=None, f1_estimate=None):
        self.teams = list(all_teams)
        self.team_idx = {team: i for i, team in enumerate(self.teams)}
        self.N = len(self.teams)

        self.a1 = 0.0005
        self.a2 = 0.0005
        self.b1 = 0.995
        self.b2 = 0.995
        self.delta = 0.025
        self.intercept = 4.6
        self.r = 1920 

        self.f = np.zeros(2 * self.N)
        self.f1_estimate = f1_estimate if f1_estimate is not None else np.zeros(2 * self.N)
        self.omega = self._compute_omega()

        self.param_guess = param_guess if param_guess is not None else [
            self.a1, self.a2, self.b1, self.b2, self.delta, self.intercept, np.log(self.r)
        ]

    def _get_f_vector(self, i, j, f):
        return np.array([f[i], f[j], f[self.N + i], f[self.N + j]])

    def _means(self, fij):
        mu_home = np.exp(self.intercept + self.delta + fij[0] - fij[3])
        mu_away = np.exp(self.intercept + fij[1] - fij[2])
        return mu_home, mu_away

    def _neg_bin_logpmf(self, x, mu):
        r = self.r
        p = r / (r + mu)
        return gammaln(x + r) - gammaln(r) - gammaln(x + 1) + r * np.log(p) + x * np.log(1 - p)

    def _score(self, x, y, fij):
        mu1, mu2 = self._means(fij)
        r = self.r
        s1 = (x - mu1) / (1 + mu1 / r)
        s2 = (y - mu2) / (1 + mu2 / r)
        s3 = -(y - mu2) / (1 + mu2 / r)
        s4 = -(x - mu1) / (1 + mu1 / r)
        return np.array([s1, s2, s3, s4])

    def _set_f_vector(self, i, j, fij, f):
        f_new = f.copy()
        f_new[i] = fij[0]
        f_new[j] = fij[1]
        f_new[self.N + i] = fij[2]
        f_new[self.N + j] = fij[3]
        return f_new

    def _update_fij(self, fij, s_t, i, j):
        A = np.diag([self.a1, self.a1, self.a2, self.a2])
        B = np.diag([self.b1, self.b1, self.b2, self.b2])
        ω = np.array([self.omega[i], self.omega[j], self.omega[self.N + i], self.omega[self.N + j]])
        return ω + B @ fij + A @ s_t
    
    def _log_likelihood(self, data):
        f = self.f1_estimate.copy()
        loglik = 0.0
        for row in data.itertuples():
            i = self.team_idx[row.team_name_home]
            j = self.team_idx[row.team_name_away]
            x, y = row.pts_home, row.pts_away

            fij = self._get_f_vector(i, j, f)
            mu1, mu2 = self._means(fij)
            loglik += self._neg_bin_logpmf(x, mu1) + self._neg_bin_logpmf(y, mu2)
            
            s_t = self._score(x, y, fij)
            fij_new = self._update_fij(fij, s_t, i, j)
            f = self._set_f_vector(i, j, fij_new, f)

        self.f = f.copy()
        return -loglik 

    def _compute_omega(self):
        B_diag = np.concatenate([np.full(self.N, self.b1), np.full(self.N, self.b2)])
        return self.f1_estimate * (1 - B_diag)

    def _objective(self, params, data):
        self.a1, self.a2, self.b1, self.b2, self.delta, self.intercept, log_r = params
        self.r = np.exp(log_r)
        self.omega = self._compute_omega()        
        return self._log_likelihood(data)

    def fit(self, data):
        bounds = [
            (0.00001, 0.2),   # a1
            (0.00001, 0.2),   # a2
            (0.95, 0.9999),  # b1
            (0.95, 0.9999),  # b2
            (0.001, 0.1),    # delta
            (1.0, 6.0),   # intercept
            (np.log(1), np.log(5000)),  # r
        ]

        result = minimize(
            self._objective,
            x0=self.param_guess,
            args=(data,),
            method='L-BFGS-B',
            bounds=bounds,
            options={"disp": False, "gtol": 1e-8, "maxiter": 2000}
        )
        self.a1, self.a2, self.b1, self.b2, self.delta, self.intercept, log_r = result.x
        self.r = np.exp(log_r)
        self.param_guess = result.x
        self.omega = self._compute_omega()

        print("\n==== Optimization Results ====")
        print(f"a1={self.a1:.4f}, a2={self.a2:.4f}")
        print(f"b1={self.b1:.4f}, b2={self.b2:.4f}")
        print(f"delta={self.delta:.4f}, intercept={self.intercept:.4f}, r={self.r:.4f}")
        if result.success:
            print("Optimization converged successfully.")
        else:
            print("Optimization failed to converge.")

    def update_state(self, round_data, f_t=None):
        f = self.f.copy() if f_t is None else f_t.copy()
        s = np.zeros(2 * self.N)
        teams_playing = set()

        for row in round_data.itertuples():
            i = self.team_idx[row.team_name_home]
            j = self.team_idx[row.team_name_away]
            fij = self._get_f_vector(i, j, f)
            score = self._score(row.pts_home, row.pts_away, fij)

            s[i] += score[0]
            s[j] += score[1]
            s[self.N + i] += score[2]
            s[self.N + j] += score[3]
            teams_playing.update([i, j])

        for idx in range(self.N):
            if idx in teams_playing:
                f[idx] = self.omega[idx] + self.b1 * f[idx] + self.a1 * s[idx]
                f[self.N + idx] = self.omega[self.N + idx] + self.b2 * f[self.N + idx] + self.a2 * s[self.N + idx]
            else:
                f[idx] = self.omega[idx] + self.b1 * f[idx]
                f[self.N + idx] = self.omega[self.N + idx] + self.b2 * f[self.N + idx]

        return f

# Poisson without intercept
class GASPoisson:
    def __init__(self, all_teams, param_guess=None, f1_estimate=None):
        self.teams = list(all_teams)
        self.team_idx = {team: i for i, team in enumerate(self.teams)}
        self.N = len(self.teams)

        self.a1 = 0.0005
        self.a2 = 0.0005
        self.b1 = 0.995
        self.b2 = 0.995
        self.delta = 0.035

        self.f = np.zeros(2 * self.N)
        self.f1_estimate = f1_estimate if f1_estimate is not None else np.zeros(2 * self.N)
        self.omega = self._compute_omega()

        self.param_guess = param_guess if param_guess is not None else [
            self.a1, self.a2, self.b1, self.b2, self.delta
        ]

    def _get_f_vector(self, i, j, f):
        return np.array([f[i], f[j], f[self.N + i], f[self.N + j]])

    def _means(self, fij):
        mu_home = np.exp(self.delta + fij[0] - fij[3])
        mu_away = np.exp(fij[1] - fij[2])
        return mu_home, mu_away

    def _poisson_logpmf(self, x, mu):
        return x * np.log(mu) - mu - gammaln(x + 1)

    def _score(self, x, y, fij):
        mu1, mu2 = self._means(fij)
        s1 = x - mu1
        s2 = y - mu2
        s3 = -(y - mu2)
        s4 = -(x - mu1)
        return np.array([s1, s2, s3, s4])

    def _set_f_vector(self, i, j, fij, f):
        f_new = f.copy()
        f_new[i] = fij[0]
        f_new[j] = fij[1]
        f_new[self.N + i] = fij[2]
        f_new[self.N + j] = fij[3]
        return f_new

    def _update_fij(self, fij, s_t, i, j):
        A = np.diag([self.a1, self.a1, self.a2, self.a2])
        B = np.diag([self.b1, self.b1, self.b2, self.b2])
        ω = np.array([self.omega[i], self.omega[j], self.omega[self.N + i], self.omega[self.N + j]])
        return ω + B @ fij + A @ s_t

    def _log_likelihood(self, data):
        f = self.f1_estimate.copy()
        loglik = 0.0
        for row in data.itertuples():
            i = self.team_idx[row.team_name_home]
            j = self.team_idx[row.team_name_away]
            x, y = row.pts_home, row.pts_away

            fij = self._get_f_vector(i, j, f)
            mu1, mu2 = self._means(fij)
            loglik += self._poisson_logpmf(x, mu1) + self._poisson_logpmf(y, mu2)

            s_t = self._score(x, y, fij)
            fij_new = self._update_fij(fij, s_t, i, j)
            f = self._set_f_vector(i, j, fij_new, f)

        self.f = f.copy()
        return -loglik

    def _compute_omega(self):
        B_diag = np.concatenate([np.full(self.N, self.b1), np.full(self.N, self.b2)])
        return self.f1_estimate * (1 - B_diag)

    def _objective(self, params, data):
        self.a1, self.a2, self.b1, self.b2, self.delta = params
        self.omega = self._compute_omega()
        return self._log_likelihood(data)

    def fit(self, data):
        bounds = [
            (0.00001, 0.2),   # a1
            (0.00001, 0.2),   # a2
            (0.95, 0.9999),   # b1
            (0.95, 0.9999),   # b2
            (0.001, 0.1),     # delta
        ]

        result = minimize(
            self._objective,
            x0=self.param_guess,
            args=(data,),
            method='L-BFGS-B',
            bounds=bounds,
            options={"disp": False, "gtol": 1e-8, "maxiter": 2000}
        )

        self.a1, self.a2, self.b1, self.b2, self.delta = result.x
        self.param_guess = result.x
        self.omega = self._compute_omega()

        print("\n==== Optimization Results ====")
        print(f"a1={self.a1:.4f}, a2={self.a2:.4f}")
        print(f"b1={self.b1:.4f}, b2={self.b2:.4f}")
        print(f"delta={self.delta:.4f}")
        if result.success:
            print("Optimization converged successfully.")
        else:
            print("Optimization failed to converge.")

    def update_state(self, round_data, f_t=None):
        f = self.f.copy() if f_t is None else f_t.copy()
        s = np.zeros(2 * self.N)
        teams_playing = set()

        for row in round_data.itertuples():
            i = self.team_idx[row.team_name_home]
            j = self.team_idx[row.team_name_away]
            fij = self._get_f_vector(i, j, f)
            score = self._score(row.pts_home, row.pts_away, fij)

            s[i] += score[0]
            s[j] += score[1]
            s[self.N + i] += score[2]
            s[self.N + j] += score[3]
            teams_playing.update([i, j])

        for idx in range(self.N):
            if idx in teams_playing:
                f[idx] = self.omega[idx] + self.b1 * f[idx] + self.a1 * s[idx]
                f[self.N + idx] = self.omega[self.N + idx] + self.b2 * f[self.N + idx] + self.a2 * s[self.N + idx]
            else:
                f[idx] = self.omega[idx] + self.b1 * f[idx]
                f[self.N + idx] = self.omega[self.N + idx] + self.b2 * f[self.N + idx]

        return f

# Poisson with intercept
class GASPoissonIntercept:
    def __init__(self, all_teams, param_guess=None, f1_estimate=None):
        self.teams = list(all_teams)
        self.team_idx = {team: i for i, team in enumerate(self.teams)}
        self.N = len(self.teams)

        self.a1 = 0.0005
        self.a2 = 0.0005
        self.b1 = 0.995
        self.b2 = 0.995
        self.delta = 0.025
        self.intercept = 4.6

        self.f = np.zeros(2 * self.N)
        self.f1_estimate = f1_estimate if f1_estimate is not None else np.zeros(2 * self.N)
        self.omega = self._compute_omega()

        self.param_guess = param_guess if param_guess is not None else [
            self.a1, self.a2, self.b1, self.b2, self.delta, self.intercept
        ]

    def _get_f_vector(self, i, j, f):
        return np.array([f[i], f[j], f[self.N + i], f[self.N + j]])

    def _means(self, fij):
        mu_home = np.exp(self.intercept + self.delta + fij[0] - fij[3])
        mu_away = np.exp(self.intercept + fij[1] - fij[2])
        return mu_home, mu_away

    def _poisson_logpmf(self, x, mu):
        return x * np.log(mu) - mu - gammaln(x + 1)

    def _score(self, x, y, fij):
        mu1, mu2 = self._means(fij)
        s1 = x - mu1
        s2 = y - mu2
        s3 = -(y - mu2)
        s4 = -(x - mu1)
        return np.array([s1, s2, s3, s4])

    def _set_f_vector(self, i, j, fij, f):
        f_new = f.copy()
        f_new[i] = fij[0]
        f_new[j] = fij[1]
        f_new[self.N + i] = fij[2]
        f_new[self.N + j] = fij[3]
        return f_new

    def _update_fij(self, fij, s_t, i, j):
        A = np.diag([self.a1, self.a1, self.a2, self.a2])
        B = np.diag([self.b1, self.b1, self.b2, self.b2])
        ω = np.array([self.omega[i], self.omega[j], self.omega[self.N + i], self.omega[self.N + j]])
        return ω + B @ fij + A @ s_t

    def _log_likelihood(self, data):
        f = self.f1_estimate.copy()
        loglik = 0.0
        for row in data.itertuples():
            i = self.team_idx[row.team_name_home]
            j = self.team_idx[row.team_name_away]
            x, y = row.pts_home, row.pts_away

            fij = self._get_f_vector(i, j, f)
            mu1, mu2 = self._means(fij)
            loglik += self._poisson_logpmf(x, mu1) + self._poisson_logpmf(y, mu2)

            s_t = self._score(x, y, fij)
            fij_new = self._update_fij(fij, s_t, i, j)
            f = self._set_f_vector(i, j, fij_new, f)

        self.f = f.copy()
        return -loglik

    def _compute_omega(self):
        B_diag = np.concatenate([np.full(self.N, self.b1), np.full(self.N, self.b2)])
        return self.f1_estimate * (1 - B_diag)

    def _objective(self, params, data):
        self.a1, self.a2, self.b1, self.b2, self.delta, self.intercept = params
        self.omega = self._compute_omega()
        return self._log_likelihood(data)

    def fit(self, data):
        bounds = [
            (0.00001, 0.2),   # a1
            (0.00001, 0.2),   # a2
            (0.95, 0.9999),   # b1
            (0.95, 0.9999),   # b2
            (0.001, 0.1),     # delta
            (1.0, 6.0),       # intercept
        ]

        result = minimize(
            self._objective,
            x0=self.param_guess,
            args=(data,),
            method='L-BFGS-B',
            bounds=bounds,
            options={"disp": False, "gtol": 1e-8, "maxiter": 2000}
        )

        self.a1, self.a2, self.b1, self.b2, self.delta, self.intercept = result.x
        self.param_guess = result.x
        self.omega = self._compute_omega()

        print("\n==== Optimization Results ====")
        print(f"a1={self.a1:.4f}, a2={self.a2:.4f}")
        print(f"b1={self.b1:.4f}, b2={self.b2:.4f}")
        print(f"delta={self.delta:.4f}, intercept={self.intercept:.4f}")
        if result.success:
            print("Optimization converged successfully.")
        else:
            print("Optimization failed to converge.")

    def update_state(self, round_data, f_t=None):
        f = self.f.copy() if f_t is None else f_t.copy()
        s = np.zeros(2 * self.N)
        teams_playing = set()

        for row in round_data.itertuples():
            i = self.team_idx[row.team_name_home]
            j = self.team_idx[row.team_name_away]
            fij = self._get_f_vector(i, j, f)
            score = self._score(row.pts_home, row.pts_away, fij)

            s[i] += score[0]
            s[j] += score[1]
            s[self.N + i] += score[2]
            s[self.N + j] += score[3]
            teams_playing.update([i, j])

        for idx in range(self.N):
            if idx in teams_playing:
                f[idx] = self.omega[idx] + self.b1 * f[idx] + self.a1 * s[idx]
                f[self.N + idx] = self.omega[self.N + idx] + self.b2 * f[self.N + idx] + self.a2 * s[self.N + idx]
            else:
                f[idx] = self.omega[idx] + self.b1 * f[idx]
                f[self.N + idx] = self.omega[self.N + idx] + self.b2 * f[self.N + idx]

        return f
