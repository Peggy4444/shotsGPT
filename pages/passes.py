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
from utils.utils import normalize_text,SimplerNet

from classes.data_source import Passes
from classes.visual import DistributionPlot,PassContributionPlot_Logistic, PassContributionPlot_XGBoost
from classes.data_source import Passes
from classes.visual import DistributionPlot,PassContributionPlot_Logistic,PassVisual,PassContributionPlot_Xnn,xnn_plot
from classes.description import PassDescription_logistic,PassDescription_xgboost, PassDescription_xNN
from classes.chat import Chat


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
pass_df_xgboost = pass_data.pass_df_xgboost
df_passes_xnn = pass_data.pass_df_xNN #extracting dataset for xNN from classes Pass


# Dropdown showing actual pass IDs
selected_pass_id = st.sidebar.selectbox("Select a pass id:", options=pass_df['id'].tolist())

pass_id = selected_pass_id

# Define the tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Logistic Regression", "xNN", "XGBoost", "CNN", "Regression trees"])

# Sample content
with tab1:
    st.header("Logistic Regression")
    
    model = Passes.load_model_logistic(selected_competition, show_summary=True)
    pass_df_logistic = pass_df.drop(['h1','h2','h3','h4'],axis=1)
    st.write(pass_df_logistic.astype(str))
    
    st.markdown("<h3 style='font-size:18px; color:black;'>Feature contribution logistic model</h3>", unsafe_allow_html=True)
    
    df_contributions = pass_data.df_contributions
    st.write(df_contributions.astype(str))

    excluded_columns = ['xT','id', 'match_id']
    metrics = [col for col in df_contributions.columns if col not in excluded_columns]

   # Build and show plot
    st.markdown("<h3 style='font-size:18px; color:black;'>Logistic contribution plot</h3>", unsafe_allow_html=True)
    visuals_logistic = PassContributionPlot_Logistic(df_contributions=df_contributions,df_passes=pass_df,metrics=metrics)
    visuals_logistic.add_passes(pass_df,metrics,selected_pass_id=selected_pass_id)
    visuals_logistic.add_pass(contribution_df=df_contributions, pass_df=pass_df, pass_id=selected_pass_id,metrics=metrics, selected_pass_id = selected_pass_id)
    visuals_logistic.show()

    xt_value = df_contributions[df_contributions['id'] == pass_id]['xT']
    xt_value = xt_value.iloc[0] if not xt_value.empty else "N/A"

    descriptions = PassDescription_logistic(pass_data,df_contributions ,pass_id, selected_competition)

    to_hash = ("logistic",selected_match_id, pass_id)
    summaries = descriptions.stream_gpt()
    chat = create_chat(to_hash, Chat)

    st.markdown(
    f"<h5 style='font-size:18px; color:green;'>Pass ID: {pass_id} | Match Name : {selected_match_name} | xT : {xt_value}</h5>",
    unsafe_allow_html=True
    )
    visuals = PassVisual(metric=None)
    visuals.add_pass(pass_data,pass_id,home_team_color = "green" , away_team_color = "red")
    visuals.show()
    if summaries:
        chat.add_message(summaries)

    chat.state = "default"
    chat.display_messages()


  
with tab2:
    st.header("xNN")

    st.markdown("<h3 style='font-size:18px; color:black;'>Logistic models based on features classification</h3>", unsafe_allow_html=True)
    model = Passes.load_pressure_model(selected_competition, show_summary=True)
    model = Passes.load_speed_model(selected_competition,show_summary=True)
    model = Passes.load_position_model(selected_competition, show_summary=True)
    model = Passes.load_event_model(selected_competition,show_summary=True)

   
    st.write(df_passes_xnn.astype(str))


    st.markdown("<h3 style='font-size:18px; color:black;'>Feature contribution from xNN model</h3>", unsafe_allow_html=True)
    df_xnn_contrib = pass_data.contributions_xNN

    st.write(df_xnn_contrib.astype(str))

    excluded_columns = ['xT_predicted','id', 'match_id']
    metrics = [col for col in df_xnn_contrib.columns if col not in excluded_columns]

   # Build and show plot
    st.markdown("<h3 style='font-size:18px; color:black;'>Xnn contribution plot</h3>", unsafe_allow_html=True)
    visuals_Xnn = PassContributionPlot_Xnn(df_xnn_contrib=df_xnn_contrib,df_passes_xnn=df_passes_xnn,metrics=metrics)
    visuals_Xnn.add_passes(df_passes_xnn,metrics)
    visuals_Xnn.add_pass(df_xnn_contrib=df_xnn_contrib, df_passes_xnn=df_passes_xnn, pass_id=selected_pass_id,metrics=metrics, selected_pass_id = selected_pass_id)
    visuals_Xnn.show()

    xt_value = df_xnn_contrib[df_xnn_contrib['id'] == pass_id]['xT_predicted']
    xt_value = xt_value.iloc[0] if not xt_value.empty else "N/A"
 
    descriptions = PassDescription_xNN(pass_data,df_xnn_contrib,pass_id, selected_competition)

    to_hash = ("xNN",selected_match_id, pass_id)
    summaries = descriptions.stream_gpt()
    chat = create_chat(to_hash, Chat)

    st.markdown(
    f"<h5 style='font-size:18px; color:green;'>Pass ID: {pass_id} | Match Name : {selected_match_name} | xT : {xt_value}</h5>",
    unsafe_allow_html=True
    )
    visuals = PassVisual(metric=None)
    visuals.add_pass(pass_data,pass_id,home_team_color = "green" , away_team_color = "red")
    visuals.show()
    
    if summaries:
        chat.add_message(summaries)

    chat.display_messages()
 
with tab3:
    st.header("xgBoost")

    #model = pass_data.load_xgboost_model(selected_competition)
    st.write(pass_df_xgboost.astype(str))
    st.markdown("<h3 style='font-size:18px; color:black;'>Feature contribution from model</h3>", unsafe_allow_html=True)
    feature_contrib_df = pass_data.feature_contrib_df
    st.write(feature_contrib_df.astype(str))

    # Show the XGBoost feature contribution plot
    st.markdown("<h3 style='font-size:18px; color:black;'>xgBoost contribution plot</h3>", unsafe_allow_html=True)

    excluded_columns = ['xT_predicted','id', 'match_id']
    metrics = [col for col in feature_contrib_df.columns if col not in excluded_columns]

    visuals_xgboost = PassContributionPlot_XGBoost(feature_contrib_df=feature_contrib_df,pass_df_xgboost=pass_df_xgboost,metrics=metrics)
    visuals_xgboost.add_passes(pass_df_xgboost, metrics, selected_pass_id=selected_pass_id)
    visuals_xgboost.add_pass(feature_contrib_df=feature_contrib_df,pass_df_xgboost=pass_df_xgboost,
    pass_id=selected_pass_id,metrics=metrics,selected_pass_id=selected_pass_id)

    visuals_xgboost.show()
    
    xt_value_xgboost = feature_contrib_df[feature_contrib_df['id'] == pass_id]['xT_predicted']
    xt_value_xgboost = xt_value_xgboost.iloc[0] if not xt_value_xgboost.empty else "N/A"


    descriptions = PassDescription_xgboost(pass_data,feature_contrib_df,pass_id, selected_competition)
    
    to_hash = ("xgBoost",selected_match_id, pass_id)
    summaries = descriptions.stream_gpt()
    chat = create_chat(to_hash, Chat)

    st.markdown(
    f"<h5 style='font-size:18px; color:green;'>Pass ID: {pass_id} | Match Name : {selected_match_name} | xT : {xt_value}</h5>",
    unsafe_allow_html=True
    )

    visuals = PassVisual(metric=None)
    visuals.add_pass(pass_data,pass_id,home_team_color = "green" , away_team_color = "red")
    visuals.show()
    
    if summaries:
        chat.add_message(summaries)

    chat.display_messages()

with tab4:
    st.header("CNN")
    pass_df_cnn = pass_df.drop(['speed_difference'],axis=1)
    st.write(pass_df_cnn.astype(str))


with tab5:
    st.header("Regression trees")
    pass_df_trees = pass_df.drop(['speed_difference'],axis=1)
    st.write(pass_df_trees.astype(str))

    visuals = PassVisual(metric=None)
    visuals.add_pass(pass_data,pass_id,home_team_color = "green" , away_team_color = "red")
    visuals.show()
