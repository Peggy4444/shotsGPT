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
from utils.utils import SimplerNet


#from classes.visual import PassVisual_logistic as PassVisual
from classes.data_source import Passes
from classes.visual import DistributionPlot,PassContributionPlot_Logistic, PassContributionPlot_XGBoost,PassContributionPlot_Mimic,Distributionplot_xnn_models,model_contribution_xnn_shap
from classes.visual import DistributionPlot,PassContributionPlot_Logistic,PassVisual,PassContributionPlot_Xnn,xnn_plot,PassContributionPlot_Logistic_event,PassContributionPlot_Logistic_pressure,PassContributionPlot_Logistic_speed,PassContributionPlot_Logistic_position,DistributionPlot_position_model,DistributionPlot_speed_models,DistributionPlot_logistic
from classes.description import PassDescription_logistic,PassDescription_xgboost, PassDescription_xNN,PassDescription_mimic
from classes.data_source import Passes
from classes.visual import DistributionPlot,PassContributionPlot_Logistic,PassVisual,PassContributionPlot_Xnn,xnn_plot,PassContributionPlot_XGBoost
from classes.description import PassDescription_logistic,PassDescription_xgboost, PassDescription_xNN
from classes.chat import Chat
#from classes.data_source import show_mimic_tree_in_streamlit


    
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
pass_df_mimic = pass_data.pass_df_mimic


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
    
    st.write(pass_df.astype(str))
    
    st.markdown("<h3 style='font-size:24px; color:black;'>Feature contribution from model</h3>", unsafe_allow_html=True)
    
    df_contributions = pass_data.df_contributions
    st.write(df_contributions.astype(str))

    logistic_contribution_describe = df_contributions.describe()
    logistic_contribution_describe.to_csv("logistic_contribution_describe.csv")

    excluded_columns = ['xT','id', 'match_id']
    metrics = [col for col in df_contributions.columns if col not in excluded_columns]

   # Build and show plot
    st.markdown("<h3 style='font-size:24px; color:black;'>Logistic contribution plot</h3>", unsafe_allow_html=True)
    visuals_logistic = PassContributionPlot_Logistic(df_contributions=df_contributions,df_passes=pass_df,metrics=metrics)
    visuals_logistic.add_passes(pass_df,metrics,selected_pass_id=selected_pass_id)
    visuals_logistic.annotate=True
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
    shap_df_xnn = pass_data.model_contribution_xNN
    contrib_pressure = pass_data.df_contributions_pressure
    contrib_speed = pass_data.df_contributions_speed
    contrib_position = pass_data.df_contributions_position
    contrib_event = pass_data.df_contributions_event
    pressure_df = pass_data.pressure_df
    speed_df = pass_data.speed_df
    position_df = pass_data.position_df
    event_df = pass_data.event_df

    xNN_contribution_describe = df_xnn_contrib.describe()

    st.write(df_xnn_contrib.astype(str))
    #xNN_contribution_describe.to_csv("xNN_contributions_describe.csv")

    excluded_columns = ['xT_predicted','id', 'match_id']
    metrics = [col for col in df_xnn_contrib.columns if col not in excluded_columns]

   # Build and show plot
    st.markdown("<h3 style='font-size:18px; color:black;'>Xnn contribution plot</h3>", unsafe_allow_html=True)
    # visuals_Xnn = PassContributionPlot_Xnn(df_xnn_contrib=df_xnn_contrib,df_passes_xnn=df_passes_xnn,metrics=metrics)
    # visuals_Xnn.add_passes(df_passes_xnn,metrics)
    # visuals_Xnn.add_pass(df_xnn_contrib=df_xnn_contrib, df_passes_xnn=df_passes_xnn, pass_id=selected_pass_id,metrics=metrics, selected_pass_id = selected_pass_id)
    # visuals_Xnn.show()

    #xNN input contribution plot
    metrics_shap = [c for c in shap_df_xnn.columns if c != "id"] 
    visuals_xNN_model = model_contribution_xnn_shap(shap_df_xnn, df_passes_xnn, metrics_shap)
    visuals_xNN_model.add_passes(df_passes_xnn, metrics_shap, selected_pass_id)
    visuals_xNN_model.annotate = True
    visuals_xNN_model.add_pass(shap_df_xnn, df_passes_xnn, selected_pass_id, metrics_shap, selected_pass_id)
    plot_model_cobtribution = visuals_xNN_model.fig
    
    # all feature contribution plot
    visuals_Xnn_feature = PassContributionPlot_Xnn(df_xnn_contrib=df_xnn_contrib,df_passes_xnn=df_passes_xnn,metrics=metrics)
    visuals_Xnn_feature.add_passes(df_passes_xnn,metrics)
    visuals_Xnn_feature.annotate = True
    visuals_Xnn_feature.add_pass(df_xnn_contrib=df_xnn_contrib, df_passes_xnn=df_passes_xnn, pass_id=selected_pass_id,metrics=metrics, selected_pass_id = selected_pass_id)
    plot_contribution = visuals_Xnn_feature.fig

    #pressure based model contribution plot
    metrics_pressure = [col for col in contrib_pressure.columns if col not in excluded_columns]
    visuals_Xnn_Pressure = PassContributionPlot_Logistic_pressure(contrib_pressure,pressure_df,metrics_pressure)
    visuals_Xnn_Pressure.add_passes(pressure_df,metrics_pressure,selected_pass_id=selected_pass_id)
    visuals_Xnn_Pressure.annotate = True
    visuals_Xnn_Pressure.add_pass(contrib_pressure,pressure_df, pass_id=selected_pass_id, metrics=metrics_pressure, selected_pass_id = selected_pass_id)
    plot_contribution_pressure = visuals_Xnn_Pressure.fig 

    #Speed based model contribution plot
    metrics_speed = [col for col in contrib_speed.columns if col not in excluded_columns]
    visuals_Xnn_speed = PassContributionPlot_Logistic_speed(contrib_speed,speed_df,metrics_speed)
    visuals_Xnn_speed.add_passes(speed_df,metrics_speed,selected_pass_id=selected_pass_id)
    visuals_Xnn_speed.add_pass(contrib_speed,speed_df, pass_id=selected_pass_id, metrics=metrics_speed, selected_pass_id = selected_pass_id)
    plot_contribution_speed = visuals_Xnn_speed.fig 


    #position based model contribution plot
    metrics_position = [col for col in contrib_position.columns if col not in excluded_columns]
    visuals_Xnn_position = PassContributionPlot_Logistic_position(contrib_position,position_df,metrics_position)
    visuals_Xnn_position.add_passes(position_df,metrics_position,pass_id)
    visuals_Xnn_position.annotate=True
    visuals_Xnn_position.add_pass(contrib_position,position_df, pass_id, metrics_position, selected_pass_id = selected_pass_id)
    plot_contribution_position = visuals_Xnn_position.fig 

    #event based model contribution plot
    metrics_event = [col for col in contrib_event.columns if col not in excluded_columns]
    visuals_Xnn_event = PassContributionPlot_Logistic_event(contrib_event,event_df,metrics_event)
    visuals_Xnn_event.add_passes(event_df,metrics_event,selected_pass_id=selected_pass_id)
    visuals_Xnn_event.annotate=True
    visuals_Xnn_event.add_pass(contrib_event,event_df, pass_id=selected_pass_id, metrics=metrics_event, selected_pass_id = selected_pass_id)
    plot_contribution_event = visuals_Xnn_event.fig 

    plots = {
    "XNN Feature Contribution": plot_contribution,
    "XNN Model SHAP Contribution": plot_model_cobtribution,
    "H1:Pressure Based model" : plot_contribution_pressure,
    "H2:Speed Based model" : plot_contribution_speed,
    "H3:position based model" : plot_contribution_position,
    "H4:event based model" : plot_contribution_event
    }
    selected_contribution_plot = st.selectbox("Select a plot:", options=list(plots.keys()),index=0)
    placeholder = st.empty()


    if selected_contribution_plot != "Select a plot…":
        fig = plots[selected_contribution_plot]
        fig.update_layout(
        autosize=False,
        width=1000,    # in pixels
        height=500,   # in pixels
        margin=dict(l=10, r=10, t=40 + 20, b=40 + 20),

        )
        placeholder.write(fig)

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
    st.header("XGBoost")

   # model = Passes.load_xgboost_model(selected_competition)
    st.write(pass_df_xgboost.astype(str))
    st.markdown("<h3 style='font-size:24px; color:black;'>Feature contribution from model</h3>", unsafe_allow_html=True)
    #feature_contrib_df = Passes.get_feature_contributions(pass_df_xgboost, model)
    feature_contrib_df = pass_data.feature_contrib_df
    
    st.write(feature_contrib_df.astype(str))

    xgboost_contribution_describe = feature_contrib_df.describe()
    xgboost_contribution_describe.to_csv("logistic_contribution_describe.csv")

    # Show the XGBoost feature contribution plot
    st.markdown("<h3 style='font-size:24px; color:black;'>XGBoost contribution plot</h3>", unsafe_allow_html=True)

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
    f"<h5 style='font-size:18px; color:green;'>Pass ID: {pass_id} | Match Name : {selected_match_name} | xT : {xt_value_xgboost}</h5>",
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
    #model = Passes.load_model(selected_competition, show_summary=False)


    
with tab5:
    st.header("Regression trees (Mimic Model)")

    # Drop any temp columns and show the clean pass DF
    st.write(pass_data.pass_df_mimic.astype(str))

    # Get mimic contributions (already computed in Passes class)
    df_contrib_mimic = pass_data.df_contributions_mimic
    if df_contrib_mimic.empty:
        st.error("Mimic contributions could not be computed due to missing required features.")
    else:
        st.markdown("<h3 style='font-size:18px; color:black;'>Feature contribution mimic model</h3>", unsafe_allow_html=True)
        st.write(df_contrib_mimic.astype(str))

        #  Metrics used for plotting
        excluded_cols = ["mimic_xT", "leaf_id", "leaf_intercept", "id", "match_id"]
        mimic_metrics = [col for col in df_contrib_mimic.columns if col.endswith("_contribution_mimic") and col not in excluded_cols]

        #  Plot contributions
        st.markdown("<h3 style='font-size:18px; color:black;'>Mimic contribution plot</h3>", unsafe_allow_html=True)

        from classes.visual import PassContributionPlot_Mimic
        mimic_plot = PassContributionPlot_Mimic(df_contributions_mimic=df_contrib_mimic, df_passes=pass_data.pass_df_mimic, metrics=mimic_metrics)
        mimic_plot.add_passes(pass_data.pass_df_mimic, mimic_metrics, selected_pass_id=selected_pass_id)
        mimic_plot.add_pass(df_contrib_mimic, pass_data.pass_df_mimic, selected_pass_id, mimic_metrics, selected_pass_id=selected_pass_id)
        mimic_plot.show()

        #  Show predicted xT value
        xt_value_mimic = df_contrib_mimic[df_contrib_mimic['id'] == pass_id]['mimic_xT']
        xt_value_mimic = xt_value_mimic.iloc[0] if not xt_value_mimic.empty else "N/A"

        st.markdown(
            f"<h5 style='font-size:18px; color:green;'>Pass ID: {pass_id} | Match Name : {selected_match_name} | mimic xT : {xt_value_mimic:.3f}</h5>",
            unsafe_allow_html=True
        )

        #  Pitch visual
        visuals = PassVisual(metric=None)
        visuals.add_pass(pass_data, pass_id, home_team_color="green", away_team_color="red")
        visuals.show()

        #  Descriptions
        descriptions = PassDescription_mimic(pass_data, df_contrib_mimic, pass_id, selected_competition)

        to_hash = ("Regression trees (Mimic Model)",selected_match_id, pass_id)
        summaries = descriptions.stream_gpt()
        chat = create_chat(to_hash, Chat)

        if summaries:
            chat.add_message(summaries)



        chat.state = "default"
        chat.display_messages()

    #     with st.expander("🗺️  Show mimic regression‑tree (path highlighted)"):
    # # 1) grab the feature‑vector of the currently selected pass
    #         x_selected = pass_df_mimic.loc[
    #             pass_df_mimic["id"] == selected_pass_id, pass_data.feature_names
    #         ].values[0]                 # shape (20,)

    #     # 2) visualise
    #     show_mimic_tree_in_streamlit(
    #         tree                 = pass_data.tree,
    #         feature_names        = pass_data.feature_names,
    #         x_train              = pass_data.X_train_for_viz,   # or sample
    #         y_train              = pass_data.y_train_for_viz,   # or sample
    #         x_selected_row       = x_selected,
    #         height_px            = 700
    #     )
