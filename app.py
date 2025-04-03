"""
Entrypoint for streamlit app.
Runs top to bottom every time the user interacts with the app (other than imports and cached functions).
"""
from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
print(path_root)
sys.path.append(str(path_root))
# Library imports
import traceback
import copy

import streamlit as st


from utils.page_components import (
    add_common_page_elements,
)

sidebar_container = add_common_page_elements()
page_container = st.sidebar.container()
sidebar_container = st.sidebar.container()

st.divider()

displaytext = """## About ShotGPT Educational """

st.markdown(displaytext)

displaytext = (
    """ShotGPT is a specialized application built within the TwelveGPT Educational framework, designed to analyze and describe football shots using Expected Goals (xG) predictions and natural language generation. \n\n"""
    """The model combines a logistic regression-based xG predictor with a language model to generate engaging, human-readable narratives about shot quality, providing insights into factors like distance to goal, angle, and defensive pressure.  """
    """This tool is intended for analysts, coaches, and enthusiasts seeking to understand and communicate shot performance in a data-driven way. \n\n"""
    """For full details on the model architecture, evaluation, and ethical considerations, refer to the [model card](https://github.com/Peggy4444/shotsGPT/blob/main/model%20cards/model-card-shot-xG-analysis.md)."""
    """ShotGPT is part of the open-source [TwelveGPT Educational](https://github.com/soccermatics/twelve-gpt-educational) project, which aims to make data-driven chatbots accessible to a wider audience. Explore the code and contribute to the project on [GitHub](https://github.com/Peggy4444/shotsGPT).  \n\n """
)

st.markdown(displaytext)
