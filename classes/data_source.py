from pandas.core.api import DataFrame as DataFrame
import streamlit as st
import requests
import pandas as pd
import numpy as np
import copy
import json
import datetime
from scipy.stats import zscore
import os
from statsmodels.api import load
import torch
import joblib
import toml
import dill

from itertools import accumulate
from pathlib import Path
import sys
import pyarrow.parquet as pq
#importing necessary libraries
from mplsoccer import Sbopen
import pandas as pd
import numpy as np
import warnings
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os
import random as rn
#warnings not visible on the course webpage
pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')
import pickle 
from joblib import load
import shap
import xgboost as xgb
import joblib
from utils.utils import SimplerNet
import cloudpickle
#from dtreeviz import dtreeviz
from classes.bayes_model import BayesianClassificationTree,Node

import json
from pathlib import Path
from classes.bayes_model import BayesianClassificationTree

import dice_ml
from dice_ml import Dice

#from dtreeviz.trees import dtreeviz
import streamlit.components.v1 as components
from pytorch_tabnet.tab_model import TabNetClassifier



import classes.data_point as data_point




import math
from scipy.special import betaln



# from classes.wyscout_api import WyNot

@st.cache_data(show_spinner=False)
def build_bayes_tree(X_df, y, *,
                     alpha=1.0, beta=1.0,
                     split_prior_decay=0.9,
                     min_samples=20, max_depth=4,
                     split_precision=1e-6):
    """Train your Bayes tree once per identical (X_df, y) pair."""
    tree = BayesianClassificationTree(
        alpha=alpha, beta=beta,
        split_prior_decay=split_prior_decay,
        min_samples=min_samples,
        max_depth=max_depth,
        split_precision=split_precision
    )
    return tree.fit(X_df, y)



# Base class for all data
class Data:
    """
    Get, process, and manage various forms of data.
    """

    data_point_class = None

    def __init__(self):
        self.df = self.get_processed_data()

    def get_raw_data(self) -> pd.DataFrame:
        raise NotImplementedError("Child class must implement get_raw_data(self)")

    def process_data(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError(
            "Child class must implement process_data(self, df_raw)"
        )

    def get_processed_data(self, match_id):

        raw = self.get_raw_data(match_id)
        return self.process_data(raw)

    def select_and_filter(self, column_name, label, default_index=0):

        df = self.df
        selected_id = st.selectbox(label, df[column_name].unique(), index=default_index)
        self.df = df[df[column_name] == selected_id]


# Base class for stat related data sources
# Calculates zscores, ranks and pct_ranks
class Stats(Data):
    """
    Builds upon DataSource for data sources which have metrics and info
    """

    def __init__(self):
        # Dataframe specs:
        # df_info: index = player, columns = basic info
        # df_metrics: index = player/team_id, columns = multiindex (Raw, Z, Rank), (metrics)
        self.df = self.get_processed_data()
        self.metrics = []
        self.negative_metrics = []

    def get_metric_zscores(self, df):

        df_z = df.apply(zscore, nan_policy="omit")

        # Rename every column to include "Z" at the end
        df_z.columns = [f"{col}_Z" for col in df_z.columns]

        # Here we get opposite value of metrics if their weight is negative
        for metric in set(self.negative_metrics).intersection(self.metrics):
            df_z[metric] = df_z[metric] * -1
        return df_z

    def get_ranks(self, df):
        df_ranks = df.rank(ascending=False)

        # Rename every column to include "Ranks" at the end
        df_ranks.columns = [f"{col}_Ranks" for col in df_ranks.columns]

        return df_ranks

    def get_pct_ranks(self, df):
        df_pct = df.rank(pct=True) * 100
        # Rename every column to include "Pct_Ranks" at the end
        df_pct.columns = [f"{col}_Pct_Ranks" for col in df_pct.columns]

        return df_pct

    def calculate_statistics(self, metrics, negative_metrics=[]):
        self.metrics = metrics
        self.negative_metrics = negative_metrics

        df = self.df
        # Add zscores, rankings and qualities
        df_metric_zscores = self.get_metric_zscores(df[metrics])
        # Here we want to use df_metric_zscores to get the ranks and pct_ranks due to negative metrics
        df_metric_ranks = self.get_ranks(df[metrics])

        # Add ranks and pct_ranks as new columns
        self.df = pd.concat([df, df_metric_zscores, df_metric_ranks], axis=1)


class PlayerStats(Stats):
    data_point_class = data_point.Player
    # This can be used if some metrics are not good to perform, like tackles lost.
    negative_metrics = []

    def __init__(self, minimal_minutes=300):
        self.minimal_minutes = minimal_minutes

        super().__init__()

    def get_raw_data(self):

        df = pd.read_csv("data/events/Forwards.csv", encoding="unicode_escape")

        return df

    def process_data(self, df_raw):
        df_raw = df_raw.rename(columns={"shortName": "player_name"})

        df_raw = df_raw.replace({-1: np.nan})
        # Remove players with low minutes
        df_raw = df_raw[(df_raw.Minutes >= self.minimal_minutes)]

        if len(df_raw) < 10:  # Or else plots won't work
            raise Exception("Not enough players with enough minutes")

        return df_raw

    def to_data_point(self, gender, position) -> data_point.Player:

        id = self.df.index[0]

        # Reindexing dataframe
        self.df.reset_index(drop=True, inplace=True)

        name = self.df["player_name"][0]
        minutes_played = self.df["Minutes"][0]
        self.df = self.df.drop(columns=["player_name", "Minutes"])

        # Convert to series
        ser_metrics = self.df.squeeze()

    



class CountryStats(Stats):
    data_point_class = data_point.Country
    # This can be used if some metrics are not good to perform, like tackles lost.
    negative_metrics = []

    def __init__(self):

        self.drill_down = self.get_drill_down_dict()
        self.drill_down_threshold = 1

        super().__init__()

    def get_drill_down_data(self, file_path):
        df = self.process_data(self.get_z_scores(pd.read_csv(file_path)))

        return dict(zip(df.country.values, df.drill_down_metric.values))

    def get_drill_down_data_values(self, file_path, metric_name):

        df = self.process_data(pd.read_csv(file_path))

        # create a value column that has the values from the columns is given by the dict self.drill_down_metric_country_question
        # where the dict has format {country: question}
        df["value_low"] = df.apply(
            lambda x: x[
                self.drill_down_metric_country_question[metric_name][x["country"]][0]
            ],
            axis=1,
        )

        df["value_high"] = df.apply(
            lambda x: x[
                self.drill_down_metric_country_question[metric_name][x["country"]][1]
            ],
            axis=1,
        )

        values = [
            (floor(l), ceil(h)) for l, h in zip(df["value_low"], df["value_high"])
        ]

        return dict(zip(df.country.values, values))

    def get_drill_down_dict(
        self,
    ):

        # read all .csv files from path ending in _pre.csv
        path = "data/wvs/intermediate_data/"
        all_files = os.listdir(path)

        self.drill_down_metric_country_question = dict(
            (
                "_".join(file.split("_")[:-1]),
                self.get_drill_down_data(path + file),
            )
            for file in all_files
            if file.endswith("_pre.csv")
        )

        drill_down_data_raw = dict(
            (
                "_".join(file.split("_")[:-1]),
                self.get_drill_down_data_values(
                    path + file, "_".join(file.split("_")[:-1])
                ),
            )
            for file in all_files
            if file.endswith("_raw.csv")
        )

        metrics = [m for m in self.drill_down_metric_country_question.keys()]
        countries = [
            k for k in self.drill_down_metric_country_question[metrics[0]].keys()
        ]

        drill_down = [
            (
                country,
                dict(
                    [
                        (
                            metric,
                            (
                                self.drill_down_metric_country_question[metric][
                                    country
                                ],
                                drill_down_data_raw[metric][country],
                            ),
                        )
                        for metric in metrics
                    ]
                ),
            )
            for country in countries
        ]

        return dict(drill_down)

    def get_z_scores(self, df, metrics=None, negative_metrics=[]):

        if metrics is None:
            metrics = [m for m in df.columns if m not in ["country"]]

        df_z = df[metrics].apply(zscore, nan_policy="omit")

        # Rename every column to include "Z" at the end
        df_z.columns = [f"{col}_Z" for col in df_z.columns]

        # Here we get opposite value of metrics if their weight is negative
        for metric in set(negative_metrics).intersection(metrics):
            df_z[metric] = df_z[metric] * -1

        # find the columns that has greatest magnitude
        drill_down_metrics_high = df[metrics].idxmax(axis=1)
        drill_down_metrics_low = df[metrics].idxmin(axis=1)

        drill_down_metrics = [
            (l, h) for l, h in zip(drill_down_metrics_low, drill_down_metrics_high)
        ]

        ### Using the question with the greatest raw magnitude vs using the question with the greatest z-score magnitude

        # # find the columns that has greatest magnitude
        # drill_down_metrics = df[metrics].abs().idxmax(axis=1)

        # # find the columns that end with "_Z" and has greatest magnitude
        # drill_down_metrics = (
        #     df_z[df_z.columns[df_z.columns.str.endswith("_Z")]]
        #     .abs()
        #     .idxmax(axis=1)
        #     .apply(lambda x: "_".join(x.split("_")[:-1]))
        # )

        ###

        df_z["drill_down_metric"] = drill_down_metrics

        # # Here we want to use df_metric_zscores to get the ranks and pct_ranks due to negative metrics
        # df_ranks = df[metrics].rank(ascending=False)

        # # Rename every column to include "Ranks" at the end
        # df_ranks.columns = [f"{col}_Ranks" for col in df_ranks.columns]

        # Add ranks and pct_ranks as new columns
        # return pd.concat([df, df_z, df_ranks], axis=1)
        return pd.concat([df, df_z], axis=1)

    def select_random(self):
        # return the index of the random sample
        return self.df.sample(1).index[0]

    def get_raw_data(self):

        df = pd.read_csv("data/wvs/wave_7.csv")

        return df

    def process_data(self, df_raw):

        # raise error if df_raw["country"] contains any NaN values
        if df_raw["country"].isnull().values.any():
            raise ValueError("Country column contains NaN values")
        # raise error if df_raw["country"] contains any empty strings
        if (df_raw["country"] == "").values.any():
            raise ValueError("Country column contains empty strings")

        # raise error if df_raw["country"] contains any duplicates
        if df_raw["country"].duplicated().any():
            raise ValueError("Country column contains duplicates")

        # # raise warning is any nan values are present in the dataframe
        # if df_raw.isnull().values.any():
        #     st.warning("Data contains NaN values")

        if len(df_raw) < 10:  # Or else plots won't work
            raise Exception("Not enough data points")

        return df_raw

    def to_data_point(self) -> data_point.Country:

        id = self.df.index[0]

        # Reindexing dataframe
        self.df.reset_index(drop=True, inplace=True)

        name = self.df["country"][0]
        self.df = self.df.drop(columns=["country"])

        # Convert to series
        ser_metrics = self.df.squeeze()

        # get the names of columns in ser_metrics than end in "_Z" with abs value greater than 1.5
        drill_down_metrics = ser_metrics[
            ser_metrics.index.str.endswith("_Z")
            & (ser_metrics.abs() >= self.drill_down_threshold)
        ].index.tolist()
        drill_down_metrics = [
            "_".join(x.split("_")[:-1]).lower() for x in drill_down_metrics
        ]

        drill_down_values = dict(
            [
                (key, value)
                for key, value in self.drill_down[name].items()
                if key.lower() in drill_down_metrics
            ]
        )

        return self.data_point_class(
            id=id,
            name=name,
            ser_metrics=ser_metrics,
            relevant_metrics=self.metrics,
            drill_down_metrics=drill_down_values,
        )






class Shots(Data):

    data_point_class = data_point.Individual

    def __init__(self, competition, match_id):
        self.df_shots = self.get_processed_data(match_id)  # Process the raw data directly
        self.xG_Model = self.load_model(competition)  # Load the model once
        self.parameters = self.read_model_params(competition)
        self.df_contributions = self.weight_contributions()
    #@st.cache_data(hash_funcs={"classes.data_source.Shots": lambda self: hash(self.raw_hash_attrs)}, ttl=5*60)


    def get_raw_data(self, match_id=None):
        parser = Sbopen()
        shot_df = pd.DataFrame()
        track_df = pd.DataFrame()
        #store data in one dataframe
        df_event = parser.event(match_id)[0]
            #open 360 data
        df_track = parser.event(match_id)[2]
            #get shots
        shots = df_event.loc[df_event["type_name"] == "Shot"]
        shots.x = shots.x.apply(lambda cell: cell*105/120)
        shots.y = shots.y.apply(lambda cell: cell*68/80)
        shots.end_x = shots.end_x.apply(lambda cell: cell*105/120)
        shots.end_y = shots.end_y.apply(lambda cell: cell*68/80)
        df_track.x = df_track.x.apply(lambda cell: cell*105/120)
        df_track.y = df_track.y.apply(lambda cell: cell*68/80)
            #append event and trackings to a dataframe
        shot_df = pd.concat([shot_df, shots], ignore_index = True)
        track_df = pd.concat([track_df, df_track], ignore_index = True)

        #reset indicies
        shot_df.reset_index(drop=True, inplace=True)
        track_df.reset_index(drop=True, inplace=True)
        #filter out non open-play shots
        shot_df = shot_df.loc[shot_df["sub_type_name"] == "Open Play"]
        shot_df = shot_df.loc[shot_df["body_part_name"] != "Head"]

        #filter out shots where goalkeeper was not tracked
        gks_tracked = track_df.loc[track_df["teammate"] == False].loc[track_df["position_name"] == "Goalkeeper"]['id'].unique()
        shot_df = shot_df.loc[shot_df["id"].isin(gks_tracked)]

        df_raw = (shot_df, track_df) 

        return df_raw
    
    def filter_by_match(self, match_id):
        """
        Filter shots and tracking data for a specific match ID.
        """
        self.df_shots = self.df_shots[self.df_shots['match_id'] == match_id]
        self.df_contributions = self.df_contributions[self.df_contributions[['match_id'] == match_id]]


    def process_data(self, df_raw: tuple) -> pd.DataFrame:

        test_shot, track_df = df_raw

        #ball_goalkeeper distance
        def dist_to_gk(test_shot, track_df):
            #get id of the shot to search for tracking data using this index
            test_shot_id = test_shot["id"]
            #check goalkeeper position
            gk_pos = track_df.loc[track_df["id"] == test_shot_id].loc[track_df["teammate"] == False].loc[track_df["position_name"] == "Goalkeeper"][["x", "y"]]
            #calculate distance from event to goalkeeper position
            dist = np.sqrt((test_shot["x"] - gk_pos["x"])**2 + (test_shot["y"] - gk_pos["y"])**2)
            return dist.iloc[0]

        #ball goalkeeper y axis
        def y_to_gk(test_shot, track_df):
            #get id of the shot to search for tracking data using this index
            test_shot_id = test_shot["id"]
            #calculate distance from event to goalkeeper position
            gk_pos = track_df.loc[track_df["id"] == test_shot_id].loc[track_df["teammate"] == False].loc[track_df["position_name"] == "Goalkeeper"][["y"]]
            #calculate distance from event to goalkeeper position in y axis
            dist = abs(test_shot["y"] - gk_pos["y"])
            return dist.iloc[0]

        #number of players less than 3 meters away from the ball
        def three_meters_away(test_shot, track_df):
            #get id of the shot to search for tracking data using this index
            test_shot_id = test_shot["id"]
            #get all opposition's player location
            player_position = track_df.loc[track_df["id"] == test_shot_id].loc[track_df["teammate"] == False][["x", "y"]]
            #calculate their distance to the ball
            dist = np.sqrt((test_shot["x"] - player_position["x"])**2 + (test_shot["y"] - player_position["y"])**2)
            #return how many are closer to the ball than 3 meters
            return len(dist[dist<3])

        #number of players inside a triangle
        def players_in_triangle(test_shot, track_df):
            #get id of the shot to search for tracking data using this index
            test_shot_id = test_shot["id"]
            #get all opposition's player location
            player_position = track_df.loc[track_df["id"] == test_shot_id].loc[track_df["teammate"] == False][["x", "y"]]
            #checking if point inside a triangle
            x1 = 105
            y1 = 34 - 7.32/2
            x2 = 105
            y2 = 34 + 7.32/2
            x3 = test_shot["x"]
            y3 = test_shot["y"]
            xp = player_position["x"]
            yp = player_position["y"]
            c1 = (x2-x1)*(yp-y1)-(y2-y1)*(xp-x1)
            c2 = (x3-x2)*(yp-y2)-(y3-y2)*(xp-x2)
            c3 = (x1-x3)*(yp-y3)-(y1-y3)*(xp-x3)
            #get number of players inside a triangle
            return len(player_position.loc[((c1<0) & (c2<0) & (c3<0)) | ((c1>0) & (c2>0) & (c3>0))])

        #goalkeeper distance to goal
        def gk_dist_to_goal(test_shot, track_df):
            #get id of the shot to search for tracking data using this index
            test_shot_id = test_shot["id"]
            #get goalkeeper position
            gk_pos = track_df.loc[track_df["id"] == test_shot_id].loc[track_df["teammate"] == False].loc[track_df["position_name"] == "Goalkeeper"][["x", "y"]]
            #calculate their distance to goal
            dist = np.sqrt((105 -gk_pos["x"])**2 + (34 - gk_pos["y"])**2)
            return dist.iloc[0]


        # Distance to the nearest opponent
        def nearest_opponent_distance(test_shot, track_df):
            # Get the ID of the shot to search for tracking data using this index
            test_shot_id = test_shot["id"]
            
            # Get all opposition's player locations (non-teammates)
            opponent_position = track_df.loc[track_df["id"] == test_shot_id].loc[track_df["teammate"] == False][["x", "y"]]
            
            # Calculate the Euclidean distance to each opponent
            distances = np.sqrt((test_shot["x"] - opponent_position["x"])**2 + (test_shot["y"] - opponent_position["y"])**2)
            
            # Return the minimum distance (i.e., nearest opponent)
            return distances.min() if len(distances) > 0 else np.nan

        # Calculate the angle to the nearest opponent
        def nearest_opponent_angle(test_shot, track_df):
            # Get the ID of the shot to search for tracking data using this index
            test_shot_id = test_shot["id"]
            
            # Get all opposition's player locations (non-teammates)
            opponent_position = track_df.loc[track_df["id"] == test_shot_id].loc[track_df["teammate"] == False][["x", "y"]]
            
            # Check if there are any opponents
            if opponent_position.empty:
                return np.nan
            
            # Calculate the Euclidean distance to each opponent
            distances = np.sqrt((test_shot["x"] - opponent_position["x"])**2 + (test_shot["y"] - opponent_position["y"])**2)
            
            # Find the index of the nearest opponent
            nearest_index = distances.idxmin()
            
            # Get the coordinates of the nearest opponent
            nearest_opponent = opponent_position.loc[nearest_index]
            
            # Calculate the angle to the nearest opponent using arctan2
            angle = np.degrees(np.arctan2(nearest_opponent["y"] - test_shot["y"], nearest_opponent["x"] - test_shot["x"]))
            
            # Normalize angles to be within 0-360 degrees
            angle = angle % 360
            
            return angle



        def angle_to_gk(test_shot, track_df):
            #get id of the shot to search for tracking data using this index
            test_shot_id = test_shot["id"]
            #check goalkeeper position
            gk_pos = track_df.loc[track_df["id"] == test_shot_id].loc[track_df["teammate"] == False].loc[track_df["position_name"] == "Goalkeeper"][["x", "y"]]
            if gk_pos.empty:
                return np.nan
            angle = -np.degrees(np.arctan2(gk_pos["y"] - test_shot["y"], gk_pos["x"] - test_shot["x"]))
            return angle.iloc[0]



        # Distance to the nearest teammate
        def nearest_teammate_distance(test_shot, track_df):
            # Get the ID of the shot to search for tracking data using this index
            test_shot_id = test_shot["id"]
            
            # Get all teammates' locations (excluding the shooter)
            teammate_position = track_df.loc[track_df["id"] == test_shot_id].loc[track_df["teammate"] == True][["x", "y"]]
            
            # Calculate the Euclidean distance to each teammate
            distances = np.sqrt((test_shot["x"] - teammate_position["x"])**2 + (test_shot["y"] - teammate_position["y"])**2)
            
            # Return the minimum distance (i.e., nearest teammate)
            return distances.min() if len(distances) > 0 else np.nan


        # Transposing teammates' and opponents' coordinates into the dataframe
        def transpose_player_positions(test_shot, track_df):
            test_shot_id = test_shot["id"]

            # Separate teammates and opponents
            teammates = track_df[(track_df["id"] == test_shot_id) & (track_df["teammate"] == True)]
            opponents = track_df[(track_df["id"] == test_shot_id) & (track_df["teammate"] == False)]

            # Transpose and flatten coordinates for teammates
            teammate_x = teammates["x"].values
            teammate_y = teammates["y"].values
            teammate_coords = {
                f"teammate_{i+1}_x": x for i, x in enumerate(teammate_x)
            }
            teammate_coords.update({
                f"teammate_{i+1}_y": y for i, y in enumerate(teammate_y)
            })

            # Transpose and flatten coordinates for opponents
            opponent_x = opponents["x"].values
            opponent_y = opponents["y"].values
            opponent_coords = {
                f"opponent_{i+1}_x": x for i, x in enumerate(opponent_x)
            }
            opponent_coords.update({
                f"opponent_{i+1}_y": y for i, y in enumerate(opponent_y)
            })

            # Combine both dictionaries into a single row dictionary
            return {**teammate_coords, **opponent_coords}
        
        model_vars = test_shot[["match_id", "id", "player_name", "team_name", "index", "x", "y", "end_x", "end_y", "minute", "play_pattern_name"]].copy()
        
        #model_vars["header"] = test_shot.body_part_name.apply(lambda cell: 1 if cell == "Head" else 0)
        model_vars["left_foot"] = test_shot.body_part_name.apply(lambda cell: 1 if cell == "Left Foot" else 0)
        model_vars["throw in"] = test_shot.play_pattern_name.apply(lambda cell: 1 if cell == "From Throw In" else 0)
        model_vars["corner"] = test_shot.play_pattern_name.apply(lambda cell: 1 if cell == "From Corner" else 0)
        model_vars["regular_play"] = test_shot.play_pattern_name.apply(lambda cell: 1 if cell == "Regular Play" else 0)
        model_vars["free_kick"] = test_shot.play_pattern_name.apply(lambda cell: 1 if cell == "From Free Kick" else 0)
        model_vars["goal"] = test_shot.outcome_name.apply(lambda cell: 1 if cell == "Goal" else 0)

        # Add necessary features and correct transformations
        model_vars["goal_smf"] = model_vars["goal"].astype(object)
        model_vars['start_x'] = model_vars.x
        model_vars['start_y'] = model_vars.y
        model_vars["x"] = model_vars.x.apply(lambda cell: 105 - cell)  # Adjust x for goal location
        model_vars["c"] = model_vars.y.apply(lambda cell: abs(34 - cell))
        model_vars["end_x"] = model_vars.end_x.apply(lambda cell: 105 - cell)

        # Calculate angle and distance
        model_vars["angle_to_goal"] = np.where(np.arctan(7.32 * model_vars["x"] / (model_vars["x"]**2 + model_vars["c"]**2 - (7.32/2)**2)) >= 0,
                                    np.arctan(7.32 * model_vars["x"] / (model_vars["x"]**2 + model_vars["c"]**2 - (7.32/2)**2)),
                                    np.arctan(7.32 * model_vars["x"] / (model_vars["x"]**2 + model_vars["c"]**2 - (7.32/2)**2)) + np.pi) * 180 / np.pi

        model_vars["distance to goal"] = np.sqrt(model_vars["x"]**2 + model_vars["c"]**2)

        # Add other features (assuming your earlier functions return correct results)
        model_vars["dist_to_gk"] = test_shot.apply(dist_to_gk, track_df=track_df, axis=1)
        model_vars["gk_distance_y"] = test_shot.apply(y_to_gk, track_df=track_df, axis=1)
        model_vars["close_players"] = test_shot.apply(three_meters_away, track_df=track_df, axis=1)
        model_vars["triangle zone"] = test_shot.apply(players_in_triangle, track_df=track_df, axis=1)
        model_vars["gk distance to goal"] = test_shot.apply(gk_dist_to_goal, track_df=track_df, axis=1)
        model_vars["distance to nearest opponent"] = test_shot.apply(nearest_opponent_distance, track_df=track_df, axis=1)
        model_vars["angle_to_nearest_opponent"] = test_shot.apply(nearest_opponent_angle, track_df=track_df, axis=1)
        model_vars["angle_to_gk"] = test_shot.apply(angle_to_gk, track_df=track_df, axis=1)
        model_vars['distance from touchline']= model_vars['c']**2
        model_vars['angle to gk and opponent']= model_vars['angle_to_goal']*model_vars['angle_to_gk']*model_vars['angle_to_nearest_opponent']
        # Merge player position data by applying the transpose_player_positions function
        player_position_features = test_shot.apply(transpose_player_positions, track_df=track_df, axis=1)

        # Convert player_position_features (which is a Series of dicts) into a DataFrame and concatenate with model_vars
        player_position_df = pd.DataFrame(list(player_position_features))
        model_vars = pd.concat([model_vars.reset_index(drop=True), player_position_df.reset_index(drop=True)], axis=1)



        # Binary features
        model_vars["is_closer"] = np.where(model_vars["gk distance to goal"] > model_vars["distance to goal"], 1, 0)
        #model_vars["header"] = test_shot.body_part_name.apply(lambda cell: 1 if cell == "Head" else 0)
        model_vars['goal'] = test_shot.outcome_name.apply(lambda cell: 1 if cell == "Goal" else 0)
        model_vars['Intercept'] = 1
        #model_vars = sm.add_constant(model_vars)
        #model_vars.rename(columns={'const': 'Intercept'}, inplace=True)  # Rename 'const' to 'Intercept'

        # model_vars.dropna(inplace=True)
        variable_names = {
                        #'x': 'horizontal_distance_to_goal',
                        'c': 'vertical_distance_to_center',
                        'distance to goal': 'euclidean_distance_to_goal',
                        'close_players': 'nearby_opponents_in_3_meters',
                        'triangle zone': 'opponents_in_triangle',
                        'gk distance to goal': 'goalkeeper_distance_to_goal',
                        'header': 'header',
                        'distance to nearest opponent': 'distance_to_nearest_opponent',
                        'angle_to_gk': 'angle_to_goalkeeper',
                        'left_foot': 'shot_with_left_foot',
                        'throw in': 'shot_after_throw_in',
                        'corner': 'shot_after_corner',
                        'free_kick': 'shot_after_free_kick',
                        'regular_play': 'shot_during_regular_play'
                    }
        model_vars = model_vars.rename(columns=variable_names)
        model_vars.fillna(0, inplace=True) 


        return model_vars
    

    def read_model_params(self, competition):
        competitions_dict_prams = {
        "EURO Men 2024": "data/model_params_EURO_2024.xlsx",
        "National Women's Soccer League (NWSL) 2018": "data/model_params_NWSL.xlsx",
        "FIFA 2022": "data/model_params_FIFA_2022.xlsx",
        "Women's Super League (FAWSL) 2017-18": "data/model_params_FAWSL.xlsx",
        "EURO Men 2020": "data/model_params_EURO_2020.xlsx",
        "Africa Cup of Nations (AFCON) 2023": "data/model_params_AFCON_2023.xlsx",}

        file_path = competitions_dict_prams.get(competition)

        if not file_path:
            st.error("Parameter file not found for the selected competition.")
            return None
        try:
            parameters = pd.read_excel(file_path)
            return parameters

        except FileNotFoundError:
            st.error(f"File not found: {file_path}")
            return None
        except Exception as e:
            st.error(f"Error reading parameter file: {e}")
            return None



    def weight_contributions(self):
        df_shots = self.df_shots  # Work with a copy to avoid altering the original dataframe
        parameters = self.parameters
        self.intercept = parameters[parameters['Parameter'] == 'Intercept']['Value'].values[0]
        
        # Exclude intercept from parameter calculations
        self.parameters = parameters[parameters['Parameter'] != 'Intercept']

        # Calculate contributions for all shots (mean-centering requires all data)
        for _, row in self.parameters.iterrows():
            param_name = row['Parameter']
            param_value = row['Value']
            contribution_col = f"{param_name}_contribution"

            # Calculate the contribution
            df_shots[contribution_col] = df_shots[param_name] * param_value

            # Mean-center the contributions
            #df_shots[contribution_col] -= df_shots[contribution_col].mean()

        # Prepare contributions dataframe
        df_contribution = df_shots[['id', 'match_id'] + [col for col in df_shots.columns if 'contribution' in col]]

        # Calculate xG for each shot individually
        xG_values = []
        for _, shot in df_shots.iterrows():
            linear_combination = self.intercept

            # Add contributions from all parameters for this shot
            for _, param in self.parameters.iterrows():
                param_name = param['Parameter']
                param_value = param['Value']
                linear_combination += shot[param_name] * param_value
            # Apply logistic function to calculate xG
            xG = 1 / (1 + np.exp(-linear_combination))
            xG_values.append(xG)

        # Add xG values to df_shots and df_contribution
        df_shots['xG'] = xG_values
        df_contribution['xG'] = xG_values

        return df_contribution
    
    @staticmethod
    def load_model(competition):

        competitions_dict = {
        "EURO Men 2024": "data/xg_model_EURO_2024.sav",
        "National Women's Soccer League (NWSL) 2018": "data/xg_model_NWSL.sav",
        "FIFA 2022": "data/xg_model_FIFA_2022.sav",
        "Women's Super League (FAWSL) 2017-18": "data/xg_model_FAWSL.sav",
        "EURO Men 2020": "data/xg_model_EURO_2020.sav",
        "Africa Cup of Nations (AFCON) 2023": "data/xg_model_AFCON_2023.sav",}
        
        saved_model_path = competitions_dict.get(competition)
    
        if not saved_model_path:
            st.error("Model file not found for the selected competition.")
            return None
        
        try:
            model = load(saved_model_path)
            with st.expander(f"Model Summary for {competition}"):
                st.write(model.summary())  
            
            return model
        
        except FileNotFoundError:
            st.error(f"Model file not found at: {saved_model_path}")
            return None
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None



    def to_data_point(self) -> data_point.Individual:
        
        id = self.df_shots['id'].iloc[0]
        self.df_shots.reset_index(drop=True, inplace=True)
        self.df_shots=self.df_shots.drop(columns=["id","xG"])
        ser_metrics = self.df_shots.squeeze()

        return self.data_point_class(id=id,ser_metrics=ser_metrics)


class Passes(Data): 
    def __init__(self,competition,match_id):
        self.match_id = match_id
        self.df_pass,self.df_tracking = self.get_data(match_id)



    # drop columns to match features used during training
        orig_drop = ['id','match_id','player_id','season','passer_name','pass_recipient_id',
                'receiver_name','team_name','forward pass','h2','h3','team_id',
                'possession_team_id','passer_x','passer_y','backward pass','lateral pass',
                'possession_xg','possession_goal','h1','h4','start_x','start_y','end_x',
                'end_y','possession_xG_target','pressure level passer'
        ]
        cols_to_drop = [c for c in orig_drop if c in self.df_pass.columns]
        X_df = self.df_pass.drop(columns=cols_to_drop)
        y = self.df_pass["possession_xG_target"]

            # Load model from JSON
        json_path = Path(__file__).resolve().parent.parent / "data" / "bayes_structure.json"
        if not json_path.exists():
                raise FileNotFoundError(f"Trained Bayesian tree not found at {json_path}")

        with open(json_path, "r") as f:
                tree_data = json.load(f)

        self.bayes_tree = BayesianClassificationTree.from_dict(tree_data)

        # Compute predictions
        bayes_probs = self.bayes_tree.predict_proba(X_df)
        bayes_contrib = self.bayes_tree.path_contributions(X_df)

        self.df_bayes_preds = pd.DataFrame({
            "id": self.df_pass["id"],
            "match_id": self.df_pass["match_id"],
            "xT_predicted_bayes": bayes_probs
        })

        self.df_contributions_bayes = (
                bayes_contrib
                .assign(id=self.df_pass["id"], match_id=self.df_pass["match_id"])
                .merge(self.df_bayes_preds[["id", "xT_predicted_bayes"]], on="id", how="left")
            )

        self.pass_df_bayes = self.df_pass.copy()
 


        
        #initialzing for logistic mdoel
        self.xT_Model = self.load_model_logistic(competition,show_summary=False)
        self.parameters = self.read_model_params(competition)
        self.df_contributions = self.weight_contributions_logistic()
        
        #initializing for xg_boost model
        drop_cols = ['possession_xG_target','speed_difference', 'end_distance_to_sideline_contribution','h1', 'h2', 'h3', 'h4','start_distance_to_goal_contribution', 'packing_contribution', 'pass_angle_contribution', 'pass_length_contribution', 'end_distance_to_goal_contribution', 'start_angle_to_goal_contribution', 'start_distance_to_sideline_contribution', 'teammates_beyond_contribution', 'opponents_beyond_contribution', 'teammates_nearby_contribution', 'opponents_between_contribution', 'opponents_nearby_contribution', 'speed_difference_contribution','end_angle_to_goal_contribution', 'pressure_on_passer_contribution','xT']
        self.pass_df_xgboost = self.df_pass.drop(columns=[col for col in drop_cols if col in self.df_pass.columns])
        xGB_model = self.load_xgboost_model(competition)
        self.feature_contrib_df = self.get_feature_contributions(self.pass_df_xgboost,xGB_model)

        #initializing for xNN model
        drop_cols_xNN = ['possession_xG_target','speed_difference', 'end_distance_to_sideline_contribution','start_distance_to_goal_contribution', 'packing_contribution', 'pass_angle_contribution', 'pass_length_contribution', 'end_distance_to_goal_contribution', 'start_angle_to_goal_contribution', 'start_distance_to_sideline_contribution', 'teammates_beyond_contribution', 'opponents_beyond_contribution', 'teammates_nearby_contribution', 'opponents_between_contribution', 'opponents_nearby_contribution', 'speed_difference_contribution','end_angle_to_goal_contribution', 'pressure_on_passer_contribution','xT']
        self.pass_df_xNN = self.df_pass.drop(columns=[col for col in drop_cols_xNN if col in self.df_pass.columns])
        new_cols = {
                "h1": "pressure based",
                "h2": "speed based",
                "h3": "position based",
                "h4": "event based"
            }
        self.pass_df_xNN.rename(columns=new_cols, inplace=True)
        self.contributions_xNN = self.get_feature_contributions_xNN(self.pass_df_xNN,competition) 
        self.model_contribution_xNN = self.get_model_contributions_xNN(self.pass_df_xNN,competition)

        # initializing for TabNet model
        drop_cols_tabnet = ['possession_xG_target','speed_difference','start_distance_to_goal_contribution', 'packing_contribution', 'pass_angle_contribution', 'pass_length_contribution', 'end_distance_to_goal_contribution', 'start_angle_to_goal_contribution', 'start_distance_to_sideline_contribution', 'teammates_beyond_contribution', 'opponents_beyond_contribution', 'teammates_nearby_contribution', 'opponents_between_contribution', 'opponents_nearby_contribution', 'speed_difference_contribution', 'xT']
        self.pass_df_tabnet = self.df_pass.drop(columns=[col for col in drop_cols_tabnet if col in self.df_pass.columns])
        self.tabnet_model = self.load_tabnet_model(competition)
        self.scaler = self.load_scaler_tabnet()
        self.contributions_tabnet = self.get_feature_contributions_tabnet(self.pass_df_tabnet,self.tabnet_model,self.scaler)
#         # self.contributions_tabnet = self.get_feature_contributions_tabnet(self.pass_df_tabnet, self.tabnet_model, self.scaler)


#         self.pass_df_tabnet = self.df_pass.drop(columns=[col for col in drop_cols_tabnet if col in self.df_pass.columns])
#         self.tabnet_model = self.load_tabnet_model(competition)
#         self.scaler = self.load_scaler_tabnet()



#                 # Initial feature list
#         feature_cols = list(self.pass_df_tabnet.columns)

#         # Filter to numeric only
#         feature_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(self.pass_df_tabnet[col])]

#         # Call function
#         self.contributions_tabnet = self.get_feature_contributions_tabnet(
#             self.pass_df_tabnet,
#             self.tabnet_model,
#             self.scaler,
#             feature_cols
# )




        # # Initializing for TabNet model
        # drop_cols_tabnet = [
        #     'possession_xG_target', 'speed_difference', 'start_distance_to_goal_contribution',
        #     'packing_contribution', 'pass_angle_contribution', 'pass_length_contribution',
        #     'end_distance_to_goal_contribution', 'start_angle_to_goal_contribution',
        #     'start_distance_to_sideline_contribution', 'teammates_beyond_contribution',
        #     'opponents_beyond_contribution', 'teammates_nearby_contribution',
        #     'opponents_between_contribution', 'opponents_nearby_contribution',
        #     'speed_difference_contribution', 'xT'
        # ]
        # self.pass_df_tabnet = self.df_pass.drop(columns=[col for col in drop_cols_tabnet if col in self.df_pass.columns])
        # self.tabnet_model = self.load_tabnet_model(competition)
        # self.scaler = self.load_scaler_tabnet()

        # # Load training feature names from saved artifact
        # import pickle
        # with open("data/feature_names.pkl", "rb") as f:
        #     feature_cols = pickle.load(f)

        # # Verify feature count
        # if len(feature_cols) != 19:
        #     raise ValueError(f"Loaded feature_names.pkl contains {len(feature_cols)} features, expected 19")

        # # Call function
        # self.contributions_tabnet = self.get_feature_contributions_tabnet(
        #     self.pass_df_tabnet,
        #     self.tabnet_model,
        #     self.scaler,
        #     feature_cols
        # )
                
        
        #load pressure based model
        self.pressure_df = (self.df_pass.loc[:, ["id","pressure_on_passer","opponents_nearby","teammates_nearby","packing"]]
            .copy())        
        self.df_contributions_pressure = self.contributions_xNN[["id","pressure_on_passer","opponents_nearby","teammates_nearby","packing"]]


        #load speed based model
        self.speed_df = (self.df_pass.loc[:, ["id","match_id","average_speed_of_teammates","average_speed_of_opponents"]]
            .copy())        
        self.df_contributions_speed = self.contributions_xNN[["id","average_speed_of_teammates","average_speed_of_opponents"]]

        #position based model
        self.position_df = (self.df_pass.loc[:, ["id","match_id","teammates_behind","teammates_beyond","opponents_behind","opponents_beyond","opponents_between"]]
            .copy())        
        self.df_contributions_position = self.contributions_xNN[["id","teammates_behind","teammates_beyond","opponents_behind","opponents_beyond","opponents_between"]]

        #event based model
        self.event_df = (self.df_pass.loc[:, ["id","match_id","start_distance_to_goal","end_distance_to_goal","start_distance_to_sideline","end_distance_to_sideline","start_angle_to_goal","end_angle_to_goal","pass_angle","pass_length"]]
            .copy())        
        self.df_contributions_event = self.contributions_xNN[["id","start_distance_to_goal","end_distance_to_goal","start_distance_to_sideline","end_distance_to_sideline","start_angle_to_goal","end_angle_to_goal","pass_angle","pass_length"]]

       #mimic_features = self.load_mimic_feature_names()
        self.tree = self.load_mimic_tree(competition)
        self.leaf_models = self.load_leaf_models(competition)
        self.feature_names = self.load_mimic_feature_names(competition)
        self.leaf_feature_means = self.load_leaf_feature_means(competition)

        
# check for mathcing features
        missing = set(self.feature_names) - set(self.df_pass.columns)
        if missing:
            st.error(f"Missing required mimic features: {missing}")
            self.pass_df_mimic = pd.DataFrame()
            self.df_contributions_mimic = pd.DataFrame()
        else:
            self.pass_df_mimic = self.df_pass[self.feature_names + ["id", "match_id"]]
            self.df_contributions_mimic = self.weight_contributions_mimic(self.pass_df_mimic,self.tree,self.leaf_models,self.feature_names,self.leaf_feature_means)

        self.tree = self.load_mimic_tree(competition)
        self.leaf_models = self.load_leaf_models(competition)
        self.feature_names = self.load_mimic_feature_names(competition)
        self.leaf_feature_means = self.load_leaf_feature_means(competition)
    
    def get_data(self, match_id=None):
        self.df_pass = pd.read_csv("data/df_passes.csv")
        self.df_tracking = pd.read_csv("data/tracking.csv")
        

        if match_id is not None:
            match_id = int(match_id)
            self.df_pass["match_id"] = self.df_pass["match_id"].astype(int)
            self.df_pass = self.df_pass[self.df_pass["match_id"] == match_id].reset_index(drop=True)

        return self.df_pass,self.df_tracking
    

    def read_model_params(self, competition):
        competitions_dict_prams = {"Allsevenskan 2022": "data/params_logistic.csv",
    "Allsevenskan 2023": "data/params_logistic.csv"}

        file_path = competitions_dict_prams.get(competition)

        if not file_path:
            st.error("Parameter file not found for the selected competition.")
            return None
        try:
            parameters = pd.read_csv(file_path)
            parameters = parameters.rename(columns={
            'beta_coeff': 'Parameter',
            'value': 'Value'
        })
            return parameters

        except FileNotFoundError:
            st.error(f"File not found: {file_path}")
            return None
        except Exception as e:
            st.error(f"Error reading parameter file: {e}")
            return 
    

    def read_position_model_params(self, competition):
        competitions_dict_prams = {"Allsevenskan 2022": "data/position_model_params.csv",
    "Allsevenskan 2023": "data/position_model_params.csv"}

        file_path = competitions_dict_prams.get(competition)

        if not file_path:
            st.error("Parameter file not found for the selected competition.")
            return None
        try:
            parameters = pd.read_csv(file_path)
            parameters = parameters.rename(columns={
            'model3_coeff': 'Parameter',
            'value_model3_coeff': 'Value'
        })
            return parameters

        except FileNotFoundError:
            st.error(f"File not found: {file_path}")
            return None
        except Exception as e:
            st.error(f"Error reading parameter file: {e}")
            return None
        

    def read_speed_model_params(self, competition):
        competitions_dict_prams = {"Allsevenskan 2022": "data/speed_model_params.csv",
    "Allsevenskan 2023": "data/speed_model_params.csv"}

        file_path = competitions_dict_prams.get(competition)

        if not file_path:
            st.error("Parameter file not found for the selected competition.")
            return None
        try:
            parameters = pd.read_csv(file_path)
            parameters = parameters.rename(columns={
            'model2_coeff': 'Parameter',
            'value_model2_coeff': 'Value'
        })
            return parameters

        except FileNotFoundError:
            st.error(f"File not found: {file_path}")
            return None
        except Exception as e:
            st.error(f"Error reading parameter file: {e}")
            return None
        
    
    def read_pressure_model_params(self, competition):
        competitions_dict_prams = {"Allsevenskan 2022": "data/pressure_model_params.csv",
    "Allsevenskan 2023": "data/pressure_model_params.csv"}

        file_path = competitions_dict_prams.get(competition)

        if not file_path:
            st.error("Parameter file not found for the selected competition.")
            return None
        try:
            parameters = pd.read_csv(file_path)
            parameters = parameters.rename(columns={
            'model1_coeff': 'Parameter',
            'value_model1_coeff': 'Value'
        })
            return parameters

        except FileNotFoundError:
            st.error(f"File not found: {file_path}")
            return None
        except Exception as e:
            st.error(f"Error reading parameter file: {e}")
            return None
        
    def read_event_model_params(self, competition):
        competitions_dict_prams = {"Allsevenskan 2022": "data/event_model_params.csv",
    "Allsevenskan 2023": "data/event_model_params.csv"}

        file_path = competitions_dict_prams.get(competition)

        if not file_path:
            st.error("Parameter file not found for the selected competition.")
            return None
        try:
            parameters = pd.read_csv(file_path)
            parameters = parameters.rename(columns={
            'model4_coeff': 'Parameter',
            'value_model4_coeff': 'Value'
        })
            return parameters

        except FileNotFoundError:
            st.error(f"File not found: {file_path}")
            return None
        except Exception as e:
            st.error(f"Error reading parameter file: {e}")
            return None


    def weight_contributions_logistic(self):
        df_pass = self.df_pass # Work with a copy to avoid altering the original dataframe
        parameters = self.parameters
        self.intercept = parameters[parameters['Parameter'] == 'Intercept']['Value'].values[0]
        
        # Exclude intercept from parameter calculations
        self.parameters = parameters[parameters['Parameter'] != 'Intercept']

        # Calculate contributions for all shots (mean-centering requires all data)
        for _, row in self.parameters.iterrows():
            param_name = row['Parameter']
            param_value = row['Value']
            contribution_col = f"{param_name}_contribution"

             # Calculate the contribution
            df_pass[contribution_col] = df_pass[param_name] * param_value

             # Mean-center the contributions
            df_pass[contribution_col] -= df_pass[contribution_col].mean()

         # Prepare contributions dataframe
        df_contribution = df_pass[['id', 'match_id'] + [col for col in df_pass.columns if 'contribution' in col]]

         # Calculate xG for each shot individually
        xG_values = []
        for _, shot in df_pass.iterrows():
            linear_combination = self.intercept

            # Add contributions from all parameters for this pass
            for _, param in self.parameters.iterrows():
                param_name = param['Parameter']
                param_value = param['Value']
                linear_combination += shot[param_name] * param_value
             # Apply logistic function to calculate xG
            xG = 1 / (1 + np.exp(-linear_combination))
            xG_values.append(xG)

         # Add xG values to df_shots and df_contribution
        df_pass['xT'] = xG_values
        df_contribution['xT'] = xG_values

        return df_contribution
    
    @staticmethod
    def load_mimic_tree(competition):
        competitions_dict = {
            "Allsevenskan 2022": "data/mimic_tree.pkl",
            "Allsevenskan 2023": "data/mimic_tree.pkl",
            
            
        }
        saved_model_path = competitions_dict.get(competition)

        if not saved_model_path:
            st.error("Model file not found for the selected competition.")
            return None

        try:
            model = load(saved_model_path)
            return model
        

        except FileNotFoundError:
            st.error(f"Model file not found at: {saved_model_path}")
            return None
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    @staticmethod
    def load_leaf_models(competition):
        competitions_dict = {
            "Allsevenskan 2022": "data/leaf_models.pkl",
            "Allsevenskan 2023": "data/leaf_models.pkl",
            
            
        }
        saved_model_path = competitions_dict.get(competition)

        if not saved_model_path:
            st.error("Model file not found for the selected competition.")
            return None

        try:
            model = load(saved_model_path)
            return model
        

        except FileNotFoundError:
            st.error(f"Model file not found at: {saved_model_path}")
            return None
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    
    @staticmethod
    def load_mimic_feature_names(competition):
        competitions_dict = {
            "Allsevenskan 2022": "data/mimic_feature_names.pkl",
            "Allsevenskan 2023": "data/mimic_feature_names.pkl",
            
            
        }
        saved_model_path = competitions_dict.get(competition)

        if not saved_model_path:
            st.error("Model file not found for the selected competition.")
            return None

        try:
            model = load(saved_model_path)
            return model
        

        except FileNotFoundError:
            st.error(f"Model file not found at: {saved_model_path}")
            return None
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    
    @staticmethod
    def load_leaf_feature_means(competition):
        competitions_dict = {
            "Allsevenskan 2022": "data/leaf_feature_means.pkl",
            "Allsevenskan 2023": "data/leaf_feature_means.pkl",
            
            
        }
        saved_model_path = competitions_dict.get(competition)

        if not saved_model_path:
            st.error("Model file not found for the selected competition.")
            return None

        try:
            model = load(saved_model_path)
            return model
        

        except FileNotFoundError:
            st.error(f"Model file not found at: {saved_model_path}")
            return None
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
                    
    @staticmethod
    def load_model_logistic(competition, show_summary=False):

        competitions_dict = {
            "Allsevenskan 2022": "data/logistic_model_joblib.sav",
            "Allsevenskan 2023": "data/logistic_model_joblib.sav",
            
            
        }

        saved_model_path = competitions_dict.get(competition)

        if not saved_model_path:
            st.error("Model file not found for the selected competition.")
            return None

        try:
            model = load(saved_model_path)
            if show_summary:
                with st.expander(f"Model Summary for {competition}"):
                    st.text(model.summary().as_text())
            return model
        

        except FileNotFoundError:
            st.error(f"Model file not found at: {saved_model_path}")
            return None
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
                    

    @staticmethod
    def load_pressure_model(competition, show_summary=False):

        competitions_dict = {
            "Allsevenskan 2022": "data/pressure_based_model.sav",
            "Allsevenskan 2023": "data/pressure_based_model.sav"
        }

        saved_model_path = competitions_dict.get(competition)

        if not saved_model_path:
            st.error("Model file not found for the selected competition.")
            return None

        try:
            model = load(saved_model_path)
            if show_summary:
                with st.expander(f"Model Summary for h1 : Pressure based model {competition}"):
                    st.text(model.summary().as_text())
            return model
        

        except FileNotFoundError:
            st.error(f"Model file not found at: {saved_model_path}")
            return None
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None   
        
    
    @staticmethod
    def load_speed_model(competition, show_summary=False):

        competitions_dict = {
            "Allsevenskan 2022": "data/speed_based_model.sav",
            "Allsevenskan 2023": "data/speed_based_model.sav"
        }

        saved_model_path = competitions_dict.get(competition)

        if not saved_model_path:
            st.error("Model file not found for the selected competition.")
            return None

        try:
            model = load(saved_model_path)
            if show_summary:
                with st.expander(f"Model Summary for h2 : Speed based model {competition}"):
                    st.text(model.summary().as_text())
            return model
        

        except FileNotFoundError:
            st.error(f"Model file not found at: {saved_model_path}")
            return None
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None   
        
    @staticmethod
    def load_position_model(competition, show_summary=False):

        competitions_dict = {
            "Allsevenskan 2022": "data/position_based_model.sav",
            "Allsevenskan 2023": "data/position_based_model.sav"
        }

        saved_model_path = competitions_dict.get(competition)

        if not saved_model_path:
            st.error("Model file not found for the selected competition.")
            return None

        try:
            model = load(saved_model_path)
            if show_summary:
                with st.expander(f"Model Summary for h3 : Position based model {competition}"):
                    st.text(model.summary().as_text())
            return model
        

        except FileNotFoundError:
            st.error(f"Model file not found at: {saved_model_path}")
            return None
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None  
        

    @staticmethod
    def load_event_model(competition, show_summary=False):

        competitions_dict = {
            "Allsevenskan 2022": "data/event_based_model.sav",
            "Allsevenskan 2023": "data/event_based_model.sav"
        }

        saved_model_path = competitions_dict.get(competition)

        if not saved_model_path:
            st.error("Model file not found for the selected competition.")
            return None

        try:
            model = load(saved_model_path)
            if show_summary:
                with st.expander(f"Model Summary for h4 : Event based model {competition}"):
                    st.text(model.summary().as_text())
            return model
        

        except FileNotFoundError:
            st.error(f"Model file not found at: {saved_model_path}")
            return None
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None   


    def load_xNN(self,competition):
        competitions_dict = {
            "Allsevenskan 2022": "data/simplernet_full_model.pth",
            "Allsevenskan 2023": "data/simplernet_full_model.pth"
        }

        saved_model_path = competitions_dict.get(competition)

        if not saved_model_path:
            st.error("Model file not found for the selected competition.")
            return None

        try:
            import torch.serialization
            torch.serialization.add_safe_globals({'SimplerNet': SimplerNet})
            model = torch.load(saved_model_path,weights_only=False)
            model.eval()
            return model
        
        except Exception as e:
            st.error(f"Error loading xNN model: {e}")
            return None  

    def load_scaler(self):
        try:
            scaler = joblib.load("data/scaler.pkl")
            return scaler
        except Exception as e:
            st.error(f"Error loading scaler: {e}")
            return None
    
    ## contributions of xNN input
    def get_model_contributions_xNN(self,pass_df_xNN,competition):
        # Load model and scaler
        model = self.load_xNN(competition)
        scaler = self.load_scaler()
        if model is None or scaler is None:
            return None
        
        features_xNN = ['pressure based','speed based','position based','event based']
        xnn_df = pass_df_xNN[features_xNN]
        X_h = pass_df_xNN[features_xNN]
        
        X_scaled = scaler.transform(X_h)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        # Predict xT
        with torch.no_grad():
            logits = model(X_tensor)
            xT_probs = torch.sigmoid(logits).numpy().flatten()
        
        #features_xNN = ['h1','h2','h3','h4']
        gamma_weights = model.model[0].weight.detach().numpy()        # (16, 4)
        final_weights = model.model[2].weight.detach().numpy().flatten()  # (16,)
        # Effective γ values from each h_k to output
        effective_gamma = final_weights @ gamma_weights  # (4,)

        # Add contributions to DataFrame
        for i, h_col in enumerate(['pressure based', 'speed based', 'position based','event based']):
            xnn_df[f"{h_col}_contrib"] = xnn_df[f"{h_col}"] * effective_gamma[i]
            xnn_df[f"{h_col}_contrib"] -= xnn_df[f"{h_col}_contrib"].mean()
        xnn_contribution = xnn_df[['pressure based_contrib','speed based_contrib',
       'position based_contrib','event based_contrib']]
        
        xnn_contribution.insert(0, 'id', pass_df_xNN['id'].values)
        xnn_contribution.insert(1, "xT_predicted", xT_probs)
        
        return xnn_contribution


    def get_feature_contributions_xNN(self,pass_df_xNN,competition):
        # Load model and scaler
        model = self.load_xNN(competition)
        scaler = self.load_scaler()
        if model is None or scaler is None:
            return None

        # Prepare data
        df_features_contrib = pass_df_xNN[['start_distance_to_goal', 'end_distance_to_goal', 'pass_length',
       'pass_angle', 'start_angle_to_goal', 'end_angle_to_goal',
       'start_distance_to_sideline', 'end_distance_to_sideline','teammates_behind',
       'teammates_beyond', 'opponents_beyond', 'opponents_behind',
       'opponents_between', 'packing', 'pressure_on_passer',
       'average_speed_of_teammates', 'average_speed_of_opponents',
       'opponents_nearby', 'teammates_nearby']]
        X_h = pass_df_xNN[['pressure based', 'speed based', 'position based', 'event based']]
        X_scaled = scaler.transform(X_h)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        # Predict xT
        with torch.no_grad():
            logits = model(X_tensor)
            xT_probs = torch.sigmoid(logits).numpy().flatten()


        # Load logistic coefficients from multiple sources
        logistic_dfs = [
            self.read_pressure_model_params(competition),
            self.read_speed_model_params(competition),
            self.read_position_model_params(competition),
            self.read_event_model_params(competition)
        ]

        logistic_coeffs = {
        h: dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
            for h, df in zip(X_h, logistic_dfs)
            }

        #gamma_weights = model.model[0].weight.detach().cpu().numpy().flatten()
        gamma_weights = model.model[0].weight.detach().numpy()        # (16, 4)
        final_weights = model.model[2].weight.detach().numpy().flatten()  # (16,)
        # Effective γ values from each h_k to output
        effective_gamma = final_weights @ gamma_weights  # (4,)

        for i, h in enumerate(X_h):
            beta_dict = logistic_coeffs[h]
            γ = effective_gamma[i]
            for feature, β in beta_dict.items():
                if feature in df_features_contrib.columns:
                    df_features_contrib[feature] = df_features_contrib[feature] * β * γ
                    df_features_contrib[feature] -= df_features_contrib[feature].mean()

        # Add IDs and xT prediction to result
        df_features_contrib.insert(0, "xT_predicted", xT_probs)
        df_features_contrib.insert(0, "id", pass_df_xNN["id"].values)

        return df_features_contrib

    
    def load_xgboost_model(self,competition):
        competitions_dict = {
            "Allsevenskan 2022": "data/XGBoost_Model_joblib.sav",
            "Allsevenskan 2023": "data/XGBoost_Model_joblib.sav"
        }

        saved_model_path = competitions_dict.get(competition)

        if not saved_model_path:
            st.error("Model file not found for the selected competition.")
            return None

        try:
            model = load(saved_model_path)
            return model
        
        except FileNotFoundError:
            st.error(f"Model file not found at: {saved_model_path}")
            return None
        
        except Exception as e:
            st.error(f"Error loading XGBoost model: {e}")
            return None  


    
    def get_feature_contributions(self,pass_df_xgboost,xGB_model):
    

        # 1. Define features to be used for prediction (exclude non-feature columns)
        feature_cols = [col for col in pass_df_xgboost.columns if col not in ['id', 'player_id', 'match_id', 'team_id', 'possession_team_id',
       'passer_x', 'passer_y', 'start_x', 'start_y', 'end_x', 'end_y', 'pressure level passer', 'forward pass', 'backward pass', 'lateral pass', 'season','pass_recipient_id','passer_name','receiver_name','team_name','possession_xg','possession_goal']]
        
        # 2. Extract X (feature matrix)
        X = pass_df_xgboost[feature_cols]

            # 3. Predict xT probabilities using the classifier
        xT_probabilities = xGB_model.predict_proba(X)[:,1]

        # 4. Compute SHAP values
        explainer = shap.Explainer(xGB_model, X)
        shap_values = explainer(X)

        # 5. Build SHAP DataFrame
        shap_df = pd.DataFrame(shap_values.values, columns=feature_cols, index=pass_df_xgboost.index)

        # 6. Add id, match_id, and xT prediction at the beginning
        shap_df.insert(0, 'xT_predicted', xT_probabilities)
        shap_df.insert(0, 'match_id', pass_df_xgboost['match_id'].values)
        shap_df.insert(0, 'id', pass_df_xgboost['id'].values)
        
        # 7. Reorder columns: id, match_id, all SHAP features, xT_predicted
        ordered_cols = ['id', 'match_id'] + feature_cols + ['xT_predicted']
        shap_df = shap_df[ordered_cols]

        return shap_df 
    
    def generate_pass_counterfactuals_by_id(self,
    selected_pass_id,
    pass_df_xgboost,
    xGB_model,
    threshold: float = 0.3,
    total_CFs: int = 5
    
):
    

        # Check ID is valid
        if selected_pass_id not in pass_df_xgboost["id"].values:
            raise ValueError(f"pass_id {selected_pass_id} not found in dataframe.")

        # Find row index
        query_index = pass_df_xgboost[pass_df_xgboost["id"] == selected_pass_id].index[0]

        # Select feature columns
        feature_cols = [col for col in pass_df_xgboost.columns if col not in [
        'id', 'player_id', 'match_id', 'team_id', 'possession_team_id',
        'passer_x', 'passer_y', 'start_x', 'start_y', 'end_x', 'end_y',
        'pressure level passer', 'forward pass', 'backward pass', 'lateral pass',
        'season', 'pass_recipient_id', 'passer_name', 'receiver_name',
        'team_name', 'possession_xg', 'possession_goal']]

        X = pass_df_xgboost[feature_cols]
        y = self.df_pass.loc[pass_df_xgboost.index, 'possession_xG_target']

        # Select the instance to explain
        query_instance = X.iloc[[query_index]]
        pred_prob = xGB_model.predict_proba(query_instance)[0][1]
        print(f"[DEBUG] Predicted xT for pass {selected_pass_id}: {pred_prob}")

        # ✅ EARLY EXIT before doing anything with DiCE
        if pred_prob > threshold:
            return pd.DataFrame(), pred_prob

        # ✅ Now safe to create DiCE objects
        data_dice = dice_ml.Data(
        dataframe=pd.concat([X, y], axis=1),
        continuous_features=feature_cols,
        outcome_name='possession_xG_target')
        model_dice = dice_ml.Model(model=xGB_model, backend="sklearn")
        exp = Dice(data_dice, model_dice)

        # Generate counterfactuals
        #cf = exp.generate_counterfactuals(query_instance, total_CFs=total_CFs, desired_class=1)

        try:
            # ❗ Specify features to keep fixed
            features_to_keep_fixed = [
                'start_distance_to_goal', 'start_distance_to_sideline', 'pressure_on_passer',
                'opponents_beyond', 'opponents_between', 'opponents_nearby', 'teammates_nearby'
                ]

            # ✅ Allow DiCE to vary only the remaining features
            features_to_vary = [f for f in feature_cols if f not in features_to_keep_fixed]

            cf = exp.generate_counterfactuals(query_instance, total_CFs=1, desired_class=1, features_to_vary=features_to_vary)
            df_attempted = cf.cf_examples_list[0].final_cfs_df

            if df_attempted.empty:
                print(f"[DEBUG] DiCE ran but returned no valid counterfactuals.")
            else:
                print("[DEBUG] Raw counterfactual explanation:")
                print(cf.visualize_as_dataframe())

        except Exception as e:
            print(f"[WARNING] DiCE failed to generate counterfactuals: {e}")
            return pd.DataFrame(), pred_prob, pd.DataFrame()


        cf_df = cf.cf_examples_list[0].final_cfs_df[feature_cols]
        cf_df["predicted_prob"] = xGB_model.predict_proba(cf_df)[:, 1]
        print(cf_df[["predicted_prob"]])

        #print(f"Generated {len(cf_df)} counterfactuals, kept {len(filtered_cf)} after threshold filtering.")


        filtered_cf = cf_df[cf_df["predicted_prob"] > threshold].reset_index(drop=True)

                # ✅ SHAP for counterfactual (only if one exists)
        shap_df_cf = pd.DataFrame()
        if not filtered_cf.empty:
            cf_instance = filtered_cf.iloc[[0]][feature_cols]
            explainer = shap.Explainer(xGB_model, X)
            shap_values_cf = explainer(cf_instance)
            shap_df_cf = pd.DataFrame([shap_values_cf.values[0]], columns=feature_cols)
            


        # Add metadata
        filtered_cf["pass_id"] = str(selected_pass_id)
        filtered_cf["type"] = "counterfactual"
        original_with_id = query_instance.copy()
        original_with_id["predicted_prob"] = pred_prob
        original_with_id["pass_id"] = str(selected_pass_id)
        original_with_id["type"] = "original"

        result_df = pd.concat([original_with_id, filtered_cf], ignore_index=True)
        return result_df, pred_prob, shap_df_cf

    

    
    
    @staticmethod
    def mimic_predict(X_input, tree, leaf_models):
        leaf_indices = tree.apply(X_input)
        preds = np.zeros(X_input.shape[0])
        for i, leaf in enumerate(leaf_indices):
            if leaf in leaf_models:
                preds[i] = leaf_models[leaf].predict(X_input[i].reshape(1, -1))[0]
            else:
                preds[i] = tree.predict(X_input[i].reshape(1, -1))[0]
        return preds

    

    def weight_contributions_mimic(self,
                               pass_df_mimic: pd.DataFrame,
                               tree,
                               leaf_models: dict,
                               feature_names: list[str],
                               leaf_feature_means: dict[int, np.ndarray]
                               ) -> pd.DataFrame:
    

    
        if not (tree and leaf_models and feature_names and leaf_feature_means):
            st.error("Some mimic components are missing.")
            return pd.DataFrame()

        missing = set(feature_names) - set(pass_df_mimic.columns)
        if missing:
            st.error(f"Missing features in data: {missing}")
            return pd.DataFrame()

    #  feature matrix same as  training 
        X_pass = pass_df_mimic[feature_names].values.astype(np.float32)

    # mimic predictions for each row (pass)
        xT_preds = self.mimic_predict(X_pass, tree, leaf_models)  # <- 20 cols only

    #  leaf IDs
        leaf_ids = tree.apply(X_pass)

    #  loop and build rows(pass)
        rows = []
        for i, (x_vec, leaf_id) in enumerate(zip(X_pass, leaf_ids)):
            lin_model = leaf_models.get(leaf_id)
            if lin_model is None:           # should not happen
                continue

            mean_x        = leaf_feature_means[leaf_id]
            contrib       = lin_model.coef_ * (x_vec - mean_x)

            row = {
            "id":            pass_df_mimic.loc[i, "id"],
            "match_id":      pass_df_mimic.loc[i, "match_id"],
            "leaf_id":       leaf_id,
            "leaf_intercept": lin_model.intercept_,
            "mimic_xT":      float(xT_preds[i]),
        }
        # add individual feature contributions
            for feat_val, feat_name in zip(contrib, feature_names):
                row[f"{feat_name}_contribution_mimic"] = feat_val

            rows.append(row)

    # final df
        core_cols = ["id", "match_id", "leaf_id", "leaf_intercept", "mimic_xT"]
        contrib_cols = [f"{f}_contribution_mimic" for f in feature_names]
        return pd.DataFrame(rows)[core_cols + contrib_cols]
    
    #loading TabNet model

    def load_tabnet_model(self, competition):
            competitions_dict = {
                "Allsevenskan 2022": "data/tabnet_model.zip",
                "Allsevenskan 2023": "data/tabnet_model.zip"
            }
            
            network_path = competitions_dict.get(competition)

            if not network_path or not os.path.exists(network_path):
                st.error(f"Network file not found at: {network_path}")
                return None

            try:
                # Initialize model with same parameters used during training
                model = TabNetClassifier(
                    n_d=16,
                    n_a=16,
                    n_steps=5,
                    gamma=1.5,
                    lambda_sparse=1e-3,
                    mask_type="entmax"
                )
                
                # Load the saved model
                model.load_model(network_path)
                model.network.eval()  # Corrected: Use model.network.eval()

                # st.success("TabNet model loaded successfully!")
                return model

            except Exception as e:
                st.error(f"Failed to load TabNet model: {e}")
                return None
    
    def load_scaler_tabnet(self):
        scaler_path_tabnet = "data/scaler_tabnet.pkl"
        if not os.path.exists(scaler_path_tabnet):
            st.error(f"Scaler file not found at: {scaler_path_tabnet}")
            return None
        try:
            with open(scaler_path_tabnet, "rb") as f:
                scaler = pickle.load(f)
            # st.success("Scaler loaded successfully!")
            return scaler
        except Exception as e:
            st.error(f"Failed to load scaler: {e}")
            return None
    def load_feature_names(self):
            feature_names_path = "data/feature_names.pkl"
            if not os.path.exists(feature_names_path):
                st.error(f"Feature names file not found at: {feature_names_path}")
                return None
            try:
                with open(feature_names_path, "rb") as f:
                    feature_names = pickle.load(f)
                # st.success("Feature names loaded successfully!")
                return feature_names
            except Exception as e:
                st.error(f"Failed to load feature names: {e}")
                return None        


    def get_feature_contributions_tabnet(self, pass_df_tabnet, tabnet_model, scaler):
        # Validate inputs first
        if pass_df_tabnet is None or pass_df_tabnet.empty:
            raise ValueError("Input DataFrame is empty or None")
        if tabnet_model is None:
            raise ValueError("TabNet model is not loaded")
        if scaler is None:
            raise ValueError("Scaler is not loaded")

        try:
            # Get feature names
            feature_cols = self.load_feature_names()
            if not feature_cols:
                raise ValueError("Failed to load feature names")

            # Validate DataFrame columns
            missing_features = [col for col in feature_cols if col not in pass_df_tabnet.columns]
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")

            # Preprocess data
            X = pass_df_tabnet[feature_cols].values.astype(np.float32)
            X_scaled = scaler.transform(X)
            
            # Convert to tensor
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32, requires_grad=True)

            # Calculate gradients
            tabnet_model.network.eval()
            with torch.set_grad_enabled(True):
                output = tabnet_model.network(X_tensor)[0][:, 1]
                output.sum().backward()
                gradients = X_tensor.grad.detach().numpy()

            # Calculate contributions
            gradient_input = gradients * X_scaled

            # Create results DataFrame
            contributions_df = pd.DataFrame(gradient_input, columns=feature_cols)
            # contributions_df.insert(0, 'id', pass_df_tabnet['id'].values)
            #contributions_df['Predicted_Probability'] = output.detach().numpy()
            y_pred_proba = tabnet_model.predict_proba(X_scaled)[:, 1]
            contributions_df['Predicted_Probability'] = y_pred_proba


            contributions_df.insert(0, 'match_id', pass_df_tabnet['match_id'].values)
            contributions_df.insert(0, 'id', pass_df_tabnet['id'].values)

            return contributions_df

        except Exception as e:
            st.error(f"Feature contribution error: {str(e)}")
            raise RuntimeError(f"Feature calculation failed: {e}") from e        
       ##working^^^^

    # def get_feature_contributions_tabnet(self, pass_df_tabnet, tabnet_model, scaler):
    #     import torch
    #     import numpy as np
    #     import pandas as pd
    #     import shap

    #     # Validate inputs
    #     if pass_df_tabnet is None or pass_df_tabnet.empty:
    #         raise ValueError("Input DataFrame is empty or None")
    #     if tabnet_model is None:
    #         raise ValueError("TabNet model is not loaded")
    #     if scaler is None:
    #         raise ValueError("Scaler is not loaded")

    #     try:
    #         # Load and validate feature names
    #         feature_cols = self.load_feature_names()
    #         if not feature_cols:
    #             raise ValueError("Failed to load feature names")

    #         missing_features = [col for col in feature_cols if col not in pass_df_tabnet.columns]
    #         if missing_features:
    #             raise ValueError(f"Missing required features: {missing_features}")

    #         # Preprocess input
    #         X = pass_df_tabnet[feature_cols].values.astype(np.float32)
    #         X_scaled = scaler.transform(X)

    #         # Convert to tensor
    #         X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    #         # Set model to eval mode
    #         tabnet_model.network.eval()

    #         # Initialize SHAP GradientExplainer
    #         explainer = shap.GradientExplainer(tabnet_model.network, X_tensor)

    #         # Compute SHAP values (feature contributions)
    #         # Use a subset if dataset is large to reduce computation time
    #         shap_values = explainer.shap_values(X_tensor, nsamples=100)  # nsamples controls approximation

    #         # Extract SHAP values for the positive class (class 1)
    #         # TabNet outputs [M_explain, output], we assume output[:, 1] is the positive class
    #         shap_values_class1 = shap_values[0][:, :, 1]  # Shape: (n_samples, n_features)

    #         # Get predicted probabilities
    #         with torch.no_grad():
    #             output = tabnet_model.network(X_tensor)[0][:, 1].numpy()

    #         # Prepare output DataFrame
    #         contributions_df = pd.DataFrame(shap_values_class1, columns=feature_cols)
    #         contributions_df.insert(0, 'id', pass_df_tabnet['id'].values)
    #         contributions_df['Predicted_Probability'] = output

    #         return contributions_df

    #     except Exception as e:
    #         import streamlit as st
    #         st.error(f"Feature contribution error: {str(e)}")
    #         raise RuntimeError(f"Feature calculation failed: {e}") from e

##### SHAP^^
    # def get_feature_contributions_tabnet(self, pass_df_tabnet, tabnet_model, scaler):
    #     import torch
    #     import numpy as np
    #     import pandas as pd
    #     from captum.attr import IntegratedGradients

    #     if pass_df_tabnet is None or pass_df_tabnet.empty:
    #         raise ValueError("Input DataFrame is empty or None")
    #     if tabnet_model is None:
    #         raise ValueError("TabNet model is not loaded")
    #     if scaler is None:
    #         raise ValueError("Scaler is not loaded")

    #     try:
    #         feature_cols = self.load_feature_names()
    #         if not feature_cols:
    #             raise ValueError("Failed to load feature names")

    #         missing_features = [col for col in feature_cols if col not in pass_df_tabnet.columns]
    #         if missing_features:
    #             raise ValueError(f"Missing required features: {missing_features}")

    #         # Preprocess input
    #         X = pass_df_tabnet[feature_cols].values.astype(np.float32)
    #         X_scaled = scaler.transform(X)
    #         X_tensor = torch.tensor(X_scaled, dtype=torch.float32, requires_grad=True)

    #         # ✅ Use the internal PyTorch model directly
    #         model = tabnet_model.network
    #         model.eval()

    #         # ✅ Define the forward function for Captum using torch model
    #         def forward_func(inputs):
    #             logits, _ = model(inputs)
    #             probs = torch.softmax(logits, dim=1)
    #             return probs[:, 1]  # return positive class probability

    #         # Integrated Gradients
    #         ig = IntegratedGradients(forward_func)
    #         baseline = torch.zeros_like(X_tensor)

    #         attributions, delta = ig.attribute(
    #             X_tensor,
    #             baselines=baseline,
    #             n_steps=50,
    #             return_convergence_delta=True
    #         )

    #         attributions = attributions.detach().numpy()
    #         y_pred_proba = tabnet_model.predict_proba(X_scaled)[:, 1]

    #         if attributions.shape[0] != y_pred_proba.shape[0]:
    #             raise ValueError(f"Mismatch between attributions ({attributions.shape[0]}) and output ({y_pred_proba.shape[0]})")

    #         contributions_df = pd.DataFrame(attributions, columns=feature_cols)
    #         contributions_df.insert(0, 'id', pass_df_tabnet['id'].values)
    #         contributions_df['Predicted_Probability'] = y_pred_proba

    #         if np.any(np.abs(delta.detach().numpy()) > 1e-3):
    #             print("⚠️ Convergence delta is high, results may be approximate.")

    #         return contributions_df

    #     except Exception as e:
    #         import streamlit as st
    #         st.error(f"Feature contribution error: {str(e)}")
    #         raise RuntimeError(f"Feature calculation failed: {e}") from e


###working ^^^

    # def get_feature_contributions_tabnet(self, pass_df_tabnet, tabnet_model, scaler):
    #     import shap
    #     import numpy as np
    #     import pandas as pd

    #     if pass_df_tabnet is None or pass_df_tabnet.empty:
    #         raise ValueError("Input DataFrame is empty or None")
    #     if tabnet_model is None:
    #         raise ValueError("TabNet model is not loaded")
    #     if scaler is None:
    #         raise ValueError("Scaler is not loaded")

    #     try:
    #         feature_cols = self.load_feature_names()
    #         if not feature_cols:
    #             raise ValueError("Failed to load feature names")

    #         missing_features = [col for col in feature_cols if col not in pass_df_tabnet.columns]
    #         if missing_features:
    #             raise ValueError(f"Missing required features: {missing_features}")

    #         # Preprocess input
    #         X = pass_df_tabnet[feature_cols].values.astype(np.float32)
    #         X_scaled = scaler.transform(X)

    #         # Use a small random sample as background data
    #         background = X_scaled[np.random.choice(X_scaled.shape[0], min(100, X_scaled.shape[0]), replace=False)]

    #         # Define prediction function
    #         def predict_fn(data):
    #             return tabnet_model.predict_proba(data)

    #         # Initialize Kernel SHAP
    #         explainer = shap.KernelExplainer(predict_fn, background)
    #         shap_values = explainer.shap_values(X_scaled)

    #         # Use shap_values[1] for the positive class (xT = 1)
    #         contribution_matrix = shap_values[1]  # shape: (n_samples, n_features)

    #         # Mean-center contributions like β×x - mean(β×x)
    #         contribution_matrix_centered = contribution_matrix - np.mean(contribution_matrix, axis=0)

    #         # Predicted probabilities
    #         y_pred_proba = tabnet_model.predict_proba(X_scaled)[:, 1]

    #         # Assemble DataFrame
    #         contributions_df = pd.DataFrame(contribution_matrix_centered, columns=feature_cols)
    #         contributions_df.insert(0, 'id', pass_df_tabnet['id'].values)
    #         contributions_df['Predicted_Probability'] = y_pred_proba

    #         return contributions_df

    #     except Exception as e:
    #         import streamlit as st
    #         st.error(f"Feature contribution error: {str(e)}")
    #         raise RuntimeError(f"Feature calculation failed: {e}") from e
























    # def get_feature_contributions_tabnet(self, pass_df_tabnet, tabnet_model, scaler, feature_cols):
    #     import numpy as np
    #     import pandas as pd
    #     from lime.lime_tabular import LimeTabularExplainer

    #     # Step 1: Input validation
    #     if pass_df_tabnet is None or pass_df_tabnet.empty:
    #         raise ValueError("Input DataFrame is empty or None")
    #     if tabnet_model is None:
    #         raise ValueError("TabNet model is not loaded")
    #     if scaler is None:
    #         raise ValueError("Scaler is not loaded")
    #     if not feature_cols:
    #         raise ValueError("Feature column list is empty")

    #     # Expected number of features (from training)
    #     expected_features = 19  # From training code
    #     if hasattr(scaler, 'n_features_in_'):
    #         if scaler.n_features_in_ != expected_features:
    #             raise ValueError(f"Scaler expects {scaler.n_features_in_} features, but training expects {expected_features}")

    #     # Debugging: Print feature counts and columns
    #     print(f"Expected features: {expected_features}")
    #     print(f"Provided feature_cols ({len(feature_cols)}): {feature_cols}")
    #     print(f"DataFrame columns ({len(pass_df_tabnet.columns)}): {pass_df_tabnet.columns.tolist()}")

    #     # Validate feature count
    #     if len(feature_cols) != expected_features:
    #         raise ValueError(f"Feature count mismatch: Provided {len(feature_cols)} features, but scaler expects {expected_features}")

    #     # Step 2: Prepare clean input data
    #     if not all(col in pass_df_tabnet.columns for col in feature_cols):
    #         missing_cols = [col for col in feature_cols if col not in pass_df_tabnet.columns]
    #         raise ValueError(f"Some feature columns are missing in pass_df_tabnet: {missing_cols}")
    #     X = pass_df_tabnet[feature_cols].values.astype(np.float32)

    #     # Step 3: Define prediction function
    #     def predict_proba_fn(x):
    #         x = np.array(x, dtype=np.float32)
    #         if x.shape[1] != len(feature_cols):
    #             raise ValueError(f"Input shape mismatch: Expected {len(feature_cols)} features, got {x.shape[1]}")
    #         x_scaled = scaler.transform(x)
    #         return tabnet_model.predict_proba(x_scaled)

    #     # Step 4: Initialize LIME explainer
    #     explainer = LimeTabularExplainer(
    #         training_data=X,
    #         mode="classification",
    #         feature_names=feature_cols,
    #         discretize_continuous=True,
    #         verbose=False,
    #         random_state=42
    #     )

    #     # Step 5: Run LIME on each sample
    #     all_contributions = []
    #     all_probabilities = []

    #     for i in range(X.shape[0]):
    #         exp = explainer.explain_instance(X[i], predict_proba_fn, num_features=len(feature_cols))
    #         contrib_dict = dict(exp.as_list())
    #         contrib_row = [contrib_dict.get(f, 0.0) for f in feature_cols]
    #         all_contributions.append(contrib_row)

    #         # Get probability of class 1
    #         x_i = X[i].reshape(1, -1)
    #         pred_prob = predict_proba_fn(x_i)[0][1]
    #         all_probabilities.append(pred_prob)

    #     # Step 6: Assemble results
    #     contributions_df = pd.DataFrame(all_contributions, columns=feature_cols)

    #     if 'id' in pass_df_tabnet.columns:
    #         contributions_df.insert(0, 'id', pass_df_tabnet['id'].values)

    #     contributions_df['Predicted_Probability'] = all_probabilities
    #     contributions_df['Sum_Contribution'] = contributions_df[feature_cols].sum(axis=1)

    #     return contributions_df

    

#new approch



    # def get_feature_contributions_tabnet(self, pass_df_tabnet, tabnet_model, scaler):
    #         # Validate inputs first
    #         if pass_df_tabnet is None or pass_df_tabnet.empty:
    #             raise ValueError("Input DataFrame is empty or None")
    #         if tabnet_model is None:
    #             raise ValueError("TabNet model is not loaded")
    #         if scaler is None:
    #             raise ValueError("Scaler is not loaded")

    #         try:
    #             # Get feature names
    #             feature_cols = self.load_feature_names()
    #             if not feature_cols:
    #                 raise ValueError("Failed to load feature names")

    #             # Validate DataFrame columns
    #             missing_features = [col for col in feature_cols if col not in pass_df_tabnet.columns]
    #             if missing_features:
    #                 raise ValueError(f"Missing required features: {missing_features}")

    #             # Preprocess data
    #             X = pass_df_tabnet[feature_cols].values.astype(np.float32)
    #             X_scaled = scaler.transform(X)
    #             print(f"Shape of X_scaled: {X_scaled.shape}")

    #             # Convert to tensor
    #             X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    #             # Get attention masks
    #             tabnet_model.eval()
    #             _, masks = tabnet_model.explain(X_tensor)  # Shape: (n_samples, n_features, n_steps)
    #             mask_weights = np.mean(masks, axis=2)  # Shape: (n_samples, n_features)
    #             print(f"Shape of mask_weights: {mask_weights.shape}")

    #             # Compute baseline predictions
    #             with torch.no_grad():
    #                 predicted_probs = tabnet_model.predict_proba(X_scaled)[:, 1]  # Probability for class 1
    #             print(f"Shape of predicted_probs: {predicted_probs.shape}")

    #             # Initialize signed contributions
    #             signed_contributions = np.zeros_like(X_scaled)

    #             # Perturbation parameters
    #             perturbation_scale = 0.05  # Perturb by ±5% of feature range
    #             feature_ranges = np.ptp(X, axis=0)  # Range of original (unscaled) features
    #             top_n_features = 5  # Perturb only top 5 features per sample based on attention

    #             for sample_idx in range(X.shape[0]):
    #                 # Select top N features with highest attention for this sample
    #                 attention_scores = mask_weights[sample_idx]
    #                 top_features = np.argsort(attention_scores)[-top_n_features:]

    #                 for feature_idx in top_features:
    #                     # Create positive and negative perturbations
    #                     X_pos = X_scaled.copy()
    #                     X_neg = X_scaled.copy()
    #                     perturbation = perturbation_scale * feature_ranges[feature_idx]
    #                     perturbation_scaled = perturbation / scaler.scale_[feature_idx]
    #                     X_pos[sample_idx, feature_idx] += perturbation_scaled
    #                     X_neg[sample_idx, feature_idx] -= perturbation_scaled

    #                     # Compute perturbed predictions
    #                     with torch.no_grad():
    #                         pos_probs = tabnet_model.predict_proba(X_pos)[sample_idx, 1]
    #                         neg_probs = tabnet_model.predict_proba(X_neg)[sample_idx, 1]

    #                     # Calculate signed contribution
    #                     contribution = (pos_probs - neg_probs) / (2 * perturbation_scaled)
    #                     signed_contributions[sample_idx, feature_idx] = contribution

    #             # Weight contributions by attention masks
    #             weighted_contributions = signed_contributions * mask_weights
    #             print(f"Shape of weighted_contributions: {weighted_contributions.shape}")

    #             # Normalize contributions for interpretability
    #             max_abs = np.abs(weighted_contributions).max(axis=1, keepdims=True)
    #             max_abs[max_abs == 0] = 1  # Avoid division by zero
    #             weighted_contributions = weighted_contributions / max_abs

    #             # Create results DataFrame
    #             contributions_df = pd.DataFrame(weighted_contributions, columns=feature_cols)

    #             # Add ID column safely
    #             if 'id' in pass_df_tabnet.columns:
    #                 contributions_df.insert(0, 'id', pass_df_tabnet['id'].values)
    #             else:
    #                 print("⚠️ Warning: 'id' column missing in input data.")

    #             # Add predicted probability
    #             contributions_df['Predicted_Probability'] = predicted_probs

    #             # Final debug: ensure column exists
    #             if 'Predicted_Probability' not in contributions_df.columns:
    #                 print("❌ Predicted_Probability column not found in final DataFrame.")
    #             else:
    #                 print("✅ Predicted_Probability column added successfully.")

    #             return contributions_df

    #         except Exception as e:
    #             st.error(f"Feature contribution error: {str(e)}")
    #             raise RuntimeError(f"Feature calculation failed: {e}") from e

    # Other methods like load_scaler_tabnet, load_feature_names, load_tabnet_model...








##saba^^
##----------------------------------------------------##


    # def weight_contributions_mimic(self,pass_df_mimic,tree,leaf_models,feature_names,leaf_feature_means):
    # # Load mimic model parts
    #     # tree = self.load_mimic_tree()
    #     # leaf_models = self.load_leaf_models()
    #     # feature_names = self.load_mimic_feature_names()
    #     # leaf_feature_means = self.load_leaf_feature_means()
    #     X_pass = pass_df_mimic[feature_names].values.astype(np.float32)
    #     xT = Passes.mimic_predict(pass_df_mimic,tree,leaf_models)

    #     if not tree or not leaf_models or not feature_names or not leaf_feature_means:
    #         st.error("Mimic model components are missing or failed to load.")
    #         return pd.DataFrame()
    #     df_pass = pass_df_mimic.copy()

    #     missing = set(feature_names) - set(df_pass.columns)
    #     if missing:
    #         st.error(f"Missing features in data: {missing}")
    #         return pd.DataFrame()

    #     #X_pass = df_pass[feature_names].values.astype(np.float32)
    #     leaf_ids = tree.apply(X_pass)
    
    #     results = []
    #     for i, x in enumerate(X_pass):
    #         leaf_id = leaf_ids[i]
    #         if leaf_id not in leaf_models:
    #             continue

    #         lin_model = leaf_models[leaf_id]
    #         mean_x = leaf_feature_means[leaf_id]
    #         x_centered = x - mean_x
    #         contributions = lin_model.coef_ * x_centered
    #         #raw = lin_model.intercept_ + contributions.sum()
    #         #xT  = 1 / (1 + np.exp(-raw))      # instead of np.clip(raw, 0, 1)

    #         #xT = np.clip(lin_model.intercept_ + contributions.sum(), 0, 1)

    #         row = {
    #             "id": df_pass.loc[i, "id"],
    #             "match_id": df_pass.loc[i, "match_id"],
    #             "leaf_id": leaf_id,
    #             "leaf_intercept": lin_model.intercept_,
    #             "mimic_xT": xT
    #         }

    #         for j, feat in enumerate(feature_names):
    #             row[f"{feat}_contribution_mimic"] = contributions[j]

    #         results.append(row)

    #     df_result = pd.DataFrame(results)

    #     core_cols = ["id", "match_id", "leaf_id",'leaf_intercept', "mimic_xT"]
    #     contrib_cols = [f"{f}_contribution_mimic" for f in feature_names]

    #     return df_result[core_cols + contrib_cols]


        
    
#     def show_mimic_tree_in_streamlit(
#             tree,
#             feature_names,
#             x_train, y_train,          # can be a 100‑row sample to keep the SVG small
#             x_selected_row,            # 1‑D numpy array of the current pass (20 cols)
#             height_px: int = 650
#             ):
#             """
#             Render the whole DecisionTreeRegressor **and** highlight the
#             decision‑path for `x_selected_row` inside a Streamlit container.
#             """
#             viz = dtreeviz(
#                 model          = tree,
#                 X_train        = x_train,
#                 y_train        = y_train,
#                 feature_names  = feature_names,
#                 target_name    = "xT",
#                 X              = x_selected_row,   # <- highlight this pass
#                 precision      = 2,
#                 fancy          = False             # True => gradient bars, slower
#             )
#             components.html(viz._repr_html_(), height=height_px, scrolling=True)
# # -----------------------------------------------------------------



# def show_mimic_tree_in_streamlit(
#             tree,
#             feature_names,
#             x_train, y_train,          # can be a 100‑row sample to keep the SVG small
#             x_selected_row,            # 1‑D numpy array of the current pass (20 cols)
#             height_px: int = 650
#             ):
#             """
#             Render the whole DecisionTreeRegressor **and** highlight the
#             decision‑path for `x_selected_row` inside a Streamlit container.
#             """
#             viz = dtreeviz(
#                 tree_model          = tree,
#                 X_train        = x_train,
#                 y_train        = y_train,
#                 feature_names  = feature_names,
#                 target_name    = "xT",
#                 X              = x_selected_row,   # <- highlight this pass
#                 precision      = 2,
#                 fancy          = False             # True => gradient bars, slower
#             )
#             components.html(viz._repr_html_(), height=height_px, scrolling=True)
    





        












