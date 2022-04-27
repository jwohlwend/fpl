import os

import numpy as np
import pandas as pd


DIR = ""
SEASONS = [
    "2016-2017",
    "2017-2018",
    "2018-2019",
    "2019-2020"
]
TEST_SEASON = "2020-2021"


class FPLEnvironment(object):

    def __init__(self):
        self.reset()

    def step(self, action):
        """Returns the next state, reward and done."""
        pass

    def reset(self, test: bool = False):
        """Reset the environment."""
        # Randomly pick a season
        if test:
            season = TEST_SEASON
        else:
            idx = np.random.randint(len(SEASONS))
            season = SEASONS[idx]

        # Set the player representation
        data = pd.read_csv(os.path.join(DIR, season, "gws", "merged_gw.csv"))
        data = data[["name", "value", "total_points", "GW"]]

        self.players = []
        self.on_team = []
        self.benched_keeper = []
        self.benched_first = []
        self.benched_second = []
        self.benched_third = []
        self.bank = 0
        self.free_transfers = 0
        self.captain = []
        self.vice_captain = []
