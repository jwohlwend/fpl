from copy import deepcopy
from typing import Set, NamedTuple, Dict
import os

import torch
import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp


DIR = os.path.join(os.path.dirname(__file__), "data")
# SEASONS = ["2016-17", "2017-18", "2018-19", "2019-20"]
SEASONS = ["2017-18"]
TEST_SEASON = "2020-21"
POS_DICT = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


class Action(NamedTuple):
    players_out: Set[str]
    players_in: Set[str]
    captain: str
    vice_captain: str
    bench_0: str
    bench_1: str
    bench_2: str
    bench_3: str


class State(NamedTuple):
    points: int
    players: Set[str]
    bank: float
    gw: int
    free_transfers: int
    player_bought_value: Dict[str, float]


def lp_team_solver(
    points: Dict[str, float],
    costs: Dict[str, float],
    revenues: Dict[str, float],
    on_team: Dict[str, bool],
    positions: Dict[str, str],
    teams: Dict[str, str],
    budget: float,
):
    """Solves the FPL linear program team selection:

    max ijk x(team=i, position=j, player=k) * point(team=i, position=j, player=k)
    sum ijk x(team=i, position=j, player=k) * cost(team=i, position=j, player=k) <= budget
    sum ik x(team=i, position=keeper, player=k) = 2
    sum ik x(team=i, position=defendr, player=k) = 5
    sum ik x(team=i, position=midfielder, player=k) = 5
    sum ik x(team=i, position=attacker, player=k) = 3
    sum jk x(team=i, position=j, player=k) <= 3 for all i
    0 <= x(team=i, position=j, player=k) <= 1  for all ijk

    """
    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver("SCIP")

    X = dict()
    players = list(points.keys())
    for player in players:
        X[player] = solver.IntVar(0.0, 1.0, f"x_{player}")

    # Sum of player costs should be less than budget
    # contribution of players on team: - sum((1-x)*revenue)
    # contribution of players not on team: sum( x * cost)
    cost = [(1 - X[p]) * -revenues[p] for p in players if p in on_team]
    cost += [X[p] * costs[p] for p in players if p not in on_team]
    solver.Add(sum(cost) <= budget)

    # Number of keepers = 2
    keepers = [p for p in players if positions[p] == "GK"]
    solver.Add(sum(X[player] for player in keepers) == 2)

    # Number of defenders = 5
    defenders = [p for p in players if positions[p] == "DEF"]
    solver.Add(sum(X[player] for player in defenders) == 5)

    # Number of midfielders = 5
    midfielders = [p for p in players if positions[p] == "MID"]
    solver.Add(sum(X[player] for player in midfielders) == 5)

    # Number of attackers = 3
    forwards = [p for p in players if positions[p] == "FWD"]
    solver.Add(sum(X[player] for player in forwards) == 3)

    # For each team, sum of x <= 3 (K constraints, K = number of teams)
    all_teams = set(teams.values())
    for team in all_teams:
        roster = [p for p in players if teams[p] == team]
        solver.Add(sum(X[player] for player in roster) <= 3)

    # Maximize sum of player values
    solver.Maximize(sum(X[player] * points[player] for player in players))
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        players_in = {
            p
            for p in players
            if (not on_team[p]) and (round(X[p].solution_value()) == 1)
        }
        players_out = {
            p for p in players if (on_team[p]) and (round(X[p].solution_value()) == 0)
        }
    else:
        raise ValueError("The problem does not have an optimal solution.")

    return players_in, players_out


def lp_formation_solver(points, positions):
    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver("SCIP")

    X = dict()
    players = list(points.keys())
    for player in players:
        X[player] = solver.IntVar(0.0, 1.0, f"x_{player}")

    # Solver
    solver.Add(sum(X[player] for player in players) == 11)

    # Number of keepers = 1
    keepers = [p for p in players if positions[p] == "GK"]
    solver.Add(sum(X[player] for player in keepers) == 1)

    # Number of defenders = 3
    defenders = [p for p in players if positions[p] == "DEF"]
    solver.Add(sum(X[player] for player in defenders) >= 3)

    # (Number of midfielders = 2)
    midfielders = [p for p in players if positions[p] == "MID"]
    solver.Add(sum(X[player] for player in midfielders) >= 2)

    # Number of attackers = 1
    forwards = [p for p in players if positions[p] == "FWD"]
    solver.Add(sum(X[player] for player in forwards) >= 1)

    # Maximize sum of player values
    solver.Maximize(sum(X[p] * points[p] for p in players))
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        benched = {p for p in players if round(X[p].solution_value()) == 0}
        bench_0 = [p for p in benched if (positions[p] == "GK")][0]
        benched.remove(bench_0)
        benched = sorted(benched, key=lambda x: points[x], reverse=True)
        return [bench_0] + benched
    else:
        raise ValueError("The problem does not have an optimal solution.")


class FPLEnvironment(object):
    """"Evniornment object for the oracle FPL game."""

    state: State

    def __init__(self):
        self.feature_dim = 43
        self.reset()

    def reset(self, test: bool = False):
        """Reset the environment."""
        # Randomly pick a season
        if test:
            season = TEST_SEASON
        else:
            idx = np.random.randint(len(SEASONS))
            season = SEASONS[idx]

        # Load data and team
        self.init_state()
        self.load_db(season)

    def init_state(self):
        init_state = State(
            players=set(),
            bank=1000,
            gw=1,
            free_transfers=15,  # wildcard
            player_bought_value=dict(),
            points=0,
        )
        self.state = init_state

    def load_db(self, season):
        # Load season data
        path = os.path.join(DIR, f"fpl_season_{season}v1.csv")
        self.db = pd.read_csv(path, encoding="ISO-8859-1")
        self.players = list(self.db["name"].unique())        

    def features(self, device) -> torch.Tensor:
        """
        Return the feature to pass to the model for prediction.
        `features`: torch.Tensor (N,d) for N players and d features
            Per player features:
            - points for future games
            - revenue
            - cost
            - on team boolean
            - value in the bank
            - number of free transfers
        """
        on_team = {p: p in self.state.players for p in self.players}
        # TODO: check if iterate on GW is okay here
        db_as_dict = (
            self.db[self.db["GW"] == (self.state.gw)].set_index("name").to_dict()
        )
        values = db_as_dict["value"]

        costs = {}
        revenues = {}
        for player in self.players:
            if on_team[player]:
                costs[player] = 0.0
                # Profit / 2 or loss
                old_value = self.state.player_bought_value[player]
                new_value = values[player]
                if new_value >= old_value:
                    revenues[player] = old_value + (new_value - old_value) // 2
                else:
                    revenues[player] = new_value
            elif player not in values:
                costs[player] = 1000.0
                revenues[player] = 0.0
            else:
                costs[player] = values[player]
                revenues[player] = 0.0

        total_features = []
        for player in self.players:
            features = [
                costs[player],
                revenues[player],
                on_team[player],
                self.state.free_transfers,
                self.state.bank,
            ]
            # player_points = list(self.db[self.db["name"] == player]["total_points"])
            player_points = list(
                self.db[(self.db["name"] == player) & (self.db["GW"] >= self.state.gw)][
                    "total_points"
                ]
            )
            features = features + player_points
            if len(features) > 43:
                import pdb

                pdb.set_trace()
            # Pad if needed
            features += [0] * (43 - len(features))
            total_features.append(features)

        total_features = torch.tensor(total_features, dtype=torch.float, device=device)
        # total_features[:self.state.gw] = 0
        return total_features

    def sample_action(self, priority: Dict[str, float]) -> Action:
        """Sample a valid action given the model priority prediction."""
        on_team = {p: p in self.state.players for p in self.players}

        self.db_as_dict = (
            self.db[self.db["GW"] == self.state.gw].set_index("name").to_dict()
        )
        positions = self.db_as_dict["position"]
        teams = self.db_as_dict["team"]
        values = self.db_as_dict["value"]
        points = self.db_as_dict["total_points"]

        costs = {}
        revenues = {}
        for player in self.players:
            if on_team[player]:
                costs[player] = 0.0
                # Profit / 2 or loss
                old_value = self.state.player_bought_value[player]
                new_value = values[player]
                if new_value >= old_value:
                    revenues[player] = old_value + (new_value - old_value) // 2
                else:
                    revenues[player] = new_value
            # TODO: ideally remove this
            elif player not in values:
                costs[player] = 1000
                revenues[player] = 0.0
                positions[player] = "DEF"
                teams[player] = 1
                points[player] = 0
            else:
                costs[player] = values[player]
                revenues[player] = 0.0

        # Solve with LP
        players_in, players_out = lp_team_solver(
            points=priority,
            costs=costs,
            revenues=revenues,
            on_team=on_team,
            positions=positions,
            teams=teams,
            budget=self.state.bank,
        )

        # Choose formation
        new_players = (self.state.players | players_in) - players_out
        points = {k: v for k, v in points.items() if k in new_players}
        positions = {k: v for k, v in positions.items() if k in new_players}
        benched = lp_formation_solver(points, positions)

        # Choose captains
        captain = max(new_players, key=lambda x: points[x])
        vice_captain = max(new_players - {captain}, key=lambda x: points[x])

        # Return action
        action = Action(
            players_out=players_out,
            players_in=players_in,
            captain=captain,
            vice_captain=vice_captain,
            bench_0=benched[0],
            bench_1=benched[1],
            bench_2=benched[2],
            bench_3=benched[3],
        )
        return action

    def update(self, action: Action):
        """Update team."""
        current = self.state

        # Compute profit from selling players
        net = 0
        players = deepcopy(current.players)
        bought = deepcopy(current.player_bought_value)
        for player in action.players_out:
            # Use value of the next starting gw
            new_value = self.db[
                (self.db["name"] == player) & (self.db["GW"] == current.gw)
            ].iloc[0]["value"]

            # Update player list
            players.remove(player)
            old_value = bought.pop(player)

            # Profit / 2 or loss
            if new_value >= old_value:
                net += old_value + (new_value - old_value) // 2
            else:
                net += new_value

        for player in action.players_in:
            # Get new value
            new_value = self.db[
                (self.db["name"] == player) & (self.db["GW"] == current.gw)
            ].iloc[0]["value"]

            players.add(player)
            bought[player] = new_value
            net -= new_value

        if current.free_transfers == 15:  # WC
            new_free = 1
            points = 0
        else:
            # Compute new transfer count
            num = len(action.players_in)
            free = current.free_transfers
            new_free = min(max(1, free - num + 1), 2)
            points = max(num - free, 0) * -4

        # Non bench first, and potentially sub in players
        positions = self.db_as_dict["position"]
        minutes_dict = self.db_as_dict["minutes"]

        # Compute whether captain played any minutes
        captain_minutes = minutes_dict[action.captain]
        captain_played = captain_minutes > 0

        # Compute points for fielded players
        players_used = set()
        keeper_played = False
        fielded_not_played = set()
        fielded_per_positon = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
        benched = {action.bench_0, action.bench_1, action.bench_2, action.bench_3}
        field = {p for p in players if (p not in benched)}
        for player in field:
            # Minutes played
            minutes = minutes_dict[player]
            fielded_per_positon[positions[player]] += 1

            # Keep tracker
            if minutes == 0:
                fielded_not_played.add(player)
                continue

            if positions[player] == "GK":
                keeper_played = True

            points_given = self.db[
                (self.db["name"] == player) & (self.db["GW"] == current.gw)
            ].iloc[0]["total_points"]

            if action.captain == player and captain_played:
                points_given *= 2
            elif action.vice_captain == player and not captain_played:
                points_given *= 2

            points += points_given
            players_used.add(player)

        # Sub-in keeper if needed
        if not keeper_played:
            points_given = self.db[
                (self.db["name"] == action.bench_0) & (self.db["GW"] == current.gw)
            ].iloc[0]["total_points"]
            points += points_given
            players_used.add(action.bench_0)

        # Sub-in outfield players if needed
        candidates = [action.bench_1, action.bench_2, action.bench_3]
        min_per_position = {"DEF": 3, "MID": 2, "FWD": 1}
        for candidate in candidates:

            removed = None
            can_pos = positions[candidate]
            for player in fielded_not_played:
                pos = positions[player]
                if pos == can_pos:
                    removed = player
                    break
                elif fielded_per_positon[can_pos] > min_per_position[can_pos]:
                    removed = player
                    break

            if removed is not None:
                fielded_not_played.remove(removed)
                fielded_per_positon[can_pos] -= 1
                points_given = self.db[
                    (self.db["name"] == candidate) & (self.db["GW"] == current.gw)
                ].iloc[0]["total_points"]
                points += points_given
                players_used.add(candidate)

        # Update state
        new_gw = current.gw + 1
        new_bank = current.bank + net
        new_points = current.points + points
        new_state = State(
            players=players,
            bank=new_bank,
            gw=new_gw,
            free_transfers=new_free,
            player_bought_value=bought,
            points=new_points,
        )
        self.state = new_state
        mask = [1 if p in players_used else 0 for p in self.players]
        mask = torch.tensor(mask)  # type: ignore
        return points, mask
