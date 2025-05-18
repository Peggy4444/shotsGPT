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
#from dtreeviz import dtreeviz

#from dtreeviz.trees import dtreeviz
import streamlit.components.v1 as components



import classes.data_point as data_point

# from classes.wyscout_api import WyNot


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

        #load pressure based model
        self.pressure_df = (self.df_pass.loc[:, ["id","pressure_on_passer","opponents_nearby","teammates_nearby","packing"]]
            .copy())        
        # self.parameters_pressure = self.read_pressure_model_params(competition)
        # self.df_contributions_pressure = self.contributions_logistic_pressure(self.pressure_df,self.pass_df_xNN)
        self.df_contributions_pressure = self.contributions_xNN[["id","pressure_on_passer","opponents_nearby","teammates_nearby","packing"]]


        #load speed based model
        self.speed_df = (self.df_pass.loc[:, ["id","match_id","average_speed_of_teammates","average_speed_of_opponents"]]
            .copy())        
        self.parameters_speed = self.read_speed_model_params(competition)
        self.df_contributions_speed = self.contributions_logistic_speed(self.speed_df,self.pass_df_xNN)

        #position based model
        self.position_df = (self.df_pass.loc[:, ["id","match_id","teammates_behind","teammates_beyond","opponents_behind","opponents_beyond","opponents_between"]]
            .copy())        
        self.parameters_position = self.read_position_model_params(competition)
        self.df_contributions_position = self.contributions_logistic_position(self.position_df,self.pass_df_xNN)

        #event based model
        self.event_df = (self.df_pass.loc[:, ["id","match_id","start_distance_to_goal","end_distance_to_goal","start_distance_to_sideline","end_distance_to_sideline","start_angle_to_goal","end_angle_to_goal","pass_angle","pass_length"]]
            .copy())        
        self.parameters_event = self.read_event_model_params(competition)
        self.df_contributions_event = self.contributions_logistic_event(self.event_df,self.pass_df_xNN)

        #self.X_train_for_viz = self.pass_df_mimic[self.feature_names].values.astype(np.float32)
        #self.y_train_for_viz = self.df_contributions_mimic["mimic_xT"].values.astype(np.float32)



        #Initializing for mimic
        ''''
        drop_cols_mimic = [
            'possession_xG_target', 'speed_difference', 'xT',
            'h1', 'h2', 'h3', 'h4',
            'start_distance_to_goal_contribution', 'packing_contribution',
            'pass_angle_contribution', 'pass_length_contribution',
            'end_distance_to_goal_contribution', 'start_angle_to_goal_contribution',
            'start_distance_to_sideline_contribution', 'teammates_beyond_contribution',
            'opponents_beyond_contribution', 'teammates_nearby_contribution',
            'opponents_between_contribution', 'opponents_nearby_contribution',
            'speed_difference_contribution'
        ]
        self.pass_df_mimic = self.df_pass.drop(columns=[col for col in drop_cols_mimic if col in self.df_pass.columns])
        self.df_contributions_mimic = self.weight_contributions_mimic(self.pass_df_mimic)
        '''
        
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
        self.X_train_for_viz = self.pass_df_mimic[self.feature_names].values.astype(np.float32)
        self.y_train_for_viz = self.df_contributions_mimic["mimic_xT"].values.astype(np.float32)

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
    
    ## contribution pressure based logistic models
    def contributions_logistic_pressure(self, pressure_df, pass_df_xnn):
        
        df = pressure_df.copy()

        #pre‐loaded params
        params = self.parameters_pressure

        #compute and mean‐center each contribution
        for _, row in params.iterrows():
            name = row['Parameter']
            val  = row['Value']
            col  = f"{name}_contribution"

            df[col] = df[name] * val
            df[col] -= df[col].mean()

        #the id/match_id + contributions and merge
        contrib_cols = [c for c in df.columns if c.endswith("_contribution")]
        result = (
            pass_df_xnn[['id','match_id']]
            .merge(df[['id','match_id'] + contrib_cols], on=['id','match_id'])
        )

        return result


    ### contribution for speed based model
    def contributions_logistic_speed(self,speed_df,pass_df_xnn):
        df = speed_df.copy()

        # 2) grab your pre‐loaded params
        params = self.parameters_speed

        # 3) compute and mean‐center each contribution
        for _, row in params.iterrows():
            name = row['Parameter']
            val  = row['Value']
            col  = f"{name}_contribution"

            df[col] = df[name] * val
            df[col] -= df[col].mean()

        # 4) pick out just the id/match_id + contributions and merge
        contrib_cols = [c for c in df.columns if c.endswith("_contribution")]
        result = (
            pass_df_xnn[['id','match_id']]
            .merge(df[['id','match_id'] + contrib_cols], on=['id','match_id'])
        )

        return result
    
    #contribution of position based 
    def contributions_logistic_position(self,position_df,pass_df_xnn):
        df = position_df.copy()

        # 2) grab your pre‐loaded params
        params = self.parameters_position

        # 3) compute and mean‐center each contribution
        for _, row in params.iterrows():
            name = row['Parameter']
            val  = row['Value']
            col  = f"{name}_contribution"

            df[col] = df[name] * val
            df[col] -= df[col].mean()

        # 4) pick out just the id/match_id + contributions and merge
        contrib_cols = [c for c in df.columns if c.endswith("_contribution")]
        result = (
            pass_df_xnn[['id','match_id']]
            .merge(df[['id','match_id'] + contrib_cols], on=['id','match_id'])
        )

        return result

    ## contributions of event based model 
    def contributions_logistic_event(self,event_df,pass_df_xnn):
        df = event_df.copy()

        # 2) grab your pre‐loaded params
        params = self.parameters_event

        # 3) compute and mean‐center each contribution
        for _, row in params.iterrows():
            name = row['Parameter']
            val  = row['Value']
            col  = f"{name}_contribution"

            df[col] = df[name] * val
            df[col] -= df[col].mean()

        # 4) pick out just the id/match_id + contributions and merge
        contrib_cols = [c for c in df.columns if c.endswith("_contribution")]
        result = (
            pass_df_xnn[['id','match_id']]
            .merge(df[['id','match_id'] + contrib_cols], on=['id','match_id'])
        )

        return result

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
            xnn_df[f"{h_col}"] -= xnn_df[f"{h_col}"].mean()
            xnn_df[f"{h_col}_contrib"] = xnn_df[f"{h_col}"] * effective_gamma[i]
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
    
    # def load_mimic_tree(self):
    #     try:
    #         return joblib.load("data/mimic_tree.pkl")
    #     except Exception as e:
    #         st.error(f"Error loading mimic tree: {e}")
    #         return None

    # def load_leaf_models(self):
    #     try:
    #         return joblib.load("data/leaf_models.pkl")
    #     except Exception as e:
    #         st.error(f"Error loading leaf models: {e}")
    #         return None

    # def load_mimic_feature_names(self):
    #     try:
    #         return joblib.load("data/mimic_feature_names.pkl")
    #     except Exception as e:
    #         st.error(f"Error loading mimic feature names: {e}")
    #         return []

    # def load_leaf_feature_means(self):
    #     try:
    #         return joblib.load("data/leaf_feature_means.pkl")
    #     except Exception as e:
    #         st.error(f"Error loading leaf feature means: {e}")
    #         return {}
        
    # def mimic_predict(X_input,tree,leaf_models):
    #     leaf_indices = tree.apply(X_input)
    #     preds = np.zeros(X_input.shape[0])
    #     for i, leaf in enumerate(leaf_indices):
    #         if leaf in leaf_models:
    #             preds[i] = leaf_models[leaf].predict(X_input[i].reshape(1, -1))[0]
    #         else:
    #             preds[i] = tree.predict(X_input[i].reshape(1, -1))[0]
    #     return preds

    
    
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
    





        












