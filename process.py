import pandas as pd
import os
import numpy as np
from collections import defaultdict

POS_DICT = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
COVID_SEASON = { j:j-9 for j in range(39,48) }
DIR = "data"

for season in ["2016-17", "2017-18", "2018-19", "2019-20", "2020-21", "2021-22"]:
    path = os.path.join(DIR, season, "gws", "merged_gw.csv")
    db = pd.read_csv(path, encoding="ISO-8859-1")
    db = db[["name", "value", "total_points", "minutes", "GW", "element"]]
    db["name_str"] = db["name"].str.replace("_[0-9]+", "", regex=True)
    db["name"] = db["element"]
    # sort by GW, name
    db = db.sort_values(by=["name", "GW"])

    # Merge multi-game GW's
    db = db.groupby(["name", "GW"]).agg(
        {"total_points": sum, "minutes": sum, "value": max}
    )
    db = db.reset_index()

    # covid weeks are shifted
    if season == "2019-20":
        db = db.replace({"GW": COVID_SEASON})

    # Add missing seasons
    missing_df = defaultdict(list)
    for i in set(db["name"]):
        missing_gws = [
            j for j in range(1, 39) if not (j in list(db[db["name"] == i]["GW"]))
        ]
        for j in missing_gws:
            missing_df["name"].append(i)
            missing_df["GW"].append(j)
            missing_df["total_points"].append(0)
            missing_df["minutes"].append(0)
            missing_df["value"].append(np.nan)

    missing_df = pd.DataFrame(missing_df)
    db = pd.concat([db, missing_df])

    db = db.sort_values(by=["name", "GW"])
    db = db.reset_index(drop=True)
    # interpolate
    db = db.groupby("name").apply(
        lambda group: group.interpolate(method="index", limit_direction="both")
    )

    # Add team & position info
    teams = pd.read_csv(os.path.join(DIR, season, "players_raw.csv"))
    teams["name"] = teams["first_name"] + "_" + teams["second_name"]
    teams = teams[["name", "team", "element_type", "id"]]
    teams = teams.replace({"element_type": POS_DICT})
    teams["position"] = teams["element_type"]
    teams["name_str"] = teams["name"]
    teams["name"] = teams["id"]

    # sort by GW, name
    db = db.sort_values(by=["GW", "name"])

    db = db.merge(teams, on="name", how="left")
    db = db[
        [
            "name",
            "GW",
            "total_points",
            "minutes",
            "value",
            "team",
            "position",
            "name_str",
        ]
    ]
    db.to_csv("data/fpl_season_{}v1.csv".format(season), index=False)
