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

from classes.data_source import passes


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

# Inject custom CSS for tab styling
st.markdown("""
    <style>
    /* Style for the tab headers */
    div[data-baseweb="tab"] button {
        background-color: #f0f4ff;
        border: 2px solid #4A90E2;
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        margin-right: 5px;
        color: #333;
        font-weight: bold;
    }

    /* Active tab styling */
    div[data-baseweb="tab"] button[aria-selected="true"] {
        background-color: #4A90E2;
        color: white;
        border-bottom: none;
    }

    /* Remove default focus outline */
    div[data-baseweb="tab"] button:focus {
        outline: none;
        box-shadow: none;
    }
    </style>
""", unsafe_allow_html=True)

# Define the tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Logistic Regression", "xNN", "XGBoost", "CNN", "Regression trees"])

# Sample content
with tab1:
    st.header("Logistic Regression")

with tab2:
    st.header("xNN")

with tab3:
    st.header("XGBoost")

with tab4:
    st.header("CNN")

with tab5:
    st.header("Regression trees")


pass_data = passes(selected_competition, selected_match_id)
pass_df = pass_data.df_pass

pass_df = pass_df.astype(str)
st.write(pass_df)






    


   

    
