# Library imports
from pathlib import Path
import sys

#importing necessary libraries
from mplsoccer import Sbopen
import pandas as pd
import numpy as np
import json
import warnings
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import os
import random as rn
#warnings not visible on the course webpage
pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')


#setting random seeds so that the results are reproducible on the webpage
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
np.random.seed(1)
rn.seed(1)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import streamlit as st
import pandas as pd
import numpy as np
import argparse
import tiktoken
import os
from utils.utils import normalize_text

from classes.data_source import Passes
from classes.visual import DistributionPlot,PassContributionPlot_Logistic, PassContributionPlot_XGBoost
from classes.data_source import Passes
from classes.visual import DistributionPlot,PassContributionPlot_Logistic,PassVisual_logistic

# Function to load and inject custom CSS from an external file
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)



from utils.page_components import (
    add_common_page_elements
)


from classes.chat import PlayerChat

from utils.page_components import add_common_page_elements
from utils.utils import select_player, create_chat

sidebar_container = add_common_page_elements()
page_container = st.sidebar.container()
sidebar_container = st.sidebar.container()

st.divider()

st.markdown("## Passes commentator")

competitions = {
    "Allsevenskan 2022": "data/matches_2022.json",
    "Allsevenskan 2023": "data/matches_2023.json"
}

# Select a competition
selected_competition = st.sidebar.selectbox("Select a Competition", options=competitions.keys())

# Load the JSON file corresponding to the selected competition
file_path = competitions[selected_competition]

with open(file_path, 'r') as f:
    id_to_match_name = json.load(f)


selected_match_name = st.sidebar.selectbox(
    "Select a Match", 
    options=id_to_match_name.values())

match_name_to_id = {v: k for k, v in id_to_match_name.items()}
selected_match_id = match_name_to_id[selected_match_name]

# Create a dropdown to select a shot ID from the available shot IDs in shots.df_shots['id']

pass_data = Passes(selected_competition,selected_match_id)
pass_df = pass_data.df_pass
tracking_df = pass_data.df_tracking
pass_df = pass_df[[col for col in pass_df.columns if "_contribution" not in col and col != "xT"]]


# Dropdown showing actual pass IDs
selected_pass_id = st.sidebar.selectbox("Select a pass id:", options=pass_df['id'].tolist())

pass_id = selected_pass_id

# Define the tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Logistic Regression", "xNN", "XGBoost", "CNN", "Regression trees"])

# Sample content
with tab1:
    st.header("Logistic Regression")

    model = Passes.load_model(selected_competition, show_summary=True)
    
    st.write(pass_df.astype(str))
    
    st.markdown("<h3 style='font-size:24px; color:black;'>Feature contribution from model</h3>", unsafe_allow_html=True)
    
    df_contributions = pass_data.df_contributions
    st.write(df_contributions.astype(str))

    excluded_columns = ['xT','id', 'match_id']
    metrics = [col for col in df_contributions.columns if col not in excluded_columns]

   # Build and show plot
    st.markdown("<h3 style='font-size:24px; color:black;'>Logistic contribution plot</h3>", unsafe_allow_html=True)
    visuals_logistic = PassContributionPlot_Logistic(df_contributions=df_contributions,df_passes=pass_df,metrics=metrics)
    visuals_logistic.add_passes(pass_df,metrics,selected_pass_id=selected_pass_id)
    visuals_logistic.add_pass(contribution_df=df_contributions, pass_df=pass_df, pass_id=selected_pass_id,metrics=metrics, selected_pass_id = selected_pass_id)
    visuals_logistic.show()

    visuals = PassVisual_logistic(metric=None)
    visuals.add_pass(pass_data,pass_id,home_team_color = "green" , away_team_color = "red")
    visuals.show()

with tab2:
    st.header("xNN")
    pass_df_xnn = pass_df.drop(['speed_difference'],axis=1)
    st.write(pass_df_xnn.astype(str))
    model = Passes.load_model(selected_competition, show_summary=False)

with tab3:
    st.header("XGBoost")

    model = Passes.load_xgboost_model(selected_competition)

    pass_df_xgboost = pass_df.drop(['speed_difference', 'possession_xG_target'],axis=1)

    st.write(pass_df_xgboost.astype(str))

    st.markdown("<h3 style='font-size:24px; color:black;'>Feature contribution from model</h3>", unsafe_allow_html=True)
    
    
    feature_contrib_df = Passes.get_feature_contributions(pass_df_xgboost, model)
    
    st.write(feature_contrib_df.astype(str))

    # Show the XGBoost feature contribution plot
    st.markdown("<h3 style='font-size:24px; color:black;'>XGBoost contribution plot</h3>", unsafe_allow_html=True)

    excluded_columns = ['xT_predicted','id', 'match_id']
    metrics = [col for col in feature_contrib_df.columns if col not in excluded_columns]

    visuals_xgboost = PassContributionPlot_XGBoost(feature_contrib_df=feature_contrib_df,pass_df_xgboost=pass_df_xgboost,metrics=metrics)
    visuals_xgboost.add_passes(pass_df_xgboost, metrics, selected_pass_id=selected_pass_id)
    visuals_xgboost.add_pass(feature_contrib_df=feature_contrib_df,pass_df_xgboost=pass_df_xgboost,
    pass_id=selected_pass_id,metrics=metrics,selected_pass_id=selected_pass_id)

    visuals_xgboost.show()
    model = Passes.load_model(selected_competition, show_summary=False)

with tab4:
    st.header("CNN")
    pass_df_cnn = pass_df.drop(['speed_difference'],axis=1)
    st.write(pass_df_cnn.astype(str))
    model = Passes.load_model(selected_competition, show_summary=False)


with tab5:
    st.header("Regression trees")
    pass_df_trees = pass_df.drop(['speed_difference'],axis=1)
    st.write(pass_df_trees.astype(str))
    #model_mimic = Passes.load_mimic_models(selected_competition)
    model_mimic = pass_data.load_mimic_models(selected_competition)


        
    st.markdown("<h3 style='font-size:24px; color:black;'>Feature contribution from tree model</h3>", 
                unsafe_allow_html=True)
    
    #    # We just read pass_data.df_contributions_mimic 
    df_contrib_mimic = pass_data.df_contributions_mimic
    st.write(df_contrib_mimic.astype(str)) 



