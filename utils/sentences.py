# Give the correct gnder words
import numpy as np
import pandas as pd
import streamlit as st
def pronouns(gender):
    if gender.lower() == "male":
        subject_p, object_p, possessive_p = "he", "him", "his"
    else:
        subject_p, object_p, possessive_p = "she", "her", "her"

    return subject_p, object_p, possessive_p


# Describe the level of a metric in words
def describe_level(
    value,
    thresholds=[1.5, 1, 0.5, -0.5, -1],
    words=["outstanding", "excellent", "good", "average", "below average", "poor"],
):
    return describe(thresholds, words, value)


def describe(thresholds, words, value):
    """
    thresholds = lower bound of each word in descending order\n
    len(words) = len(thresholds) + 1
    """
    assert len(words) == len(thresholds) + 1, "Issue with thresholds and words"
    i = 0
    while i < len(thresholds) and value < thresholds[i]:
        i += 1

    return words[i]


# Format the metrics for display and descriptions
def format_metric(metric):
    return (
        metric.replace("_", " ")
        .replace(" adjusted per90", "")
        .replace("npxG", "non-penalty expected goals")
        .capitalize()
    )


def write_out_metric(metric):
    return (
        metric.replace("_", " ")
        .replace("adjusted", "adjusted for possession")
        .replace("per90", "per 90")
        .replace("npxG", "non-penalty expected goals")
        + " minutes"
    )
    return metric.replace("_"," ").replace("adjusted","adjusted for possession").replace("per90","per 90").replace("npxG","non-penalty expected goals") + " minutes"



def describe_xg(xG):

    if xG < 0.028723: # 25% percentile
        description = "This was a slim chance of scoring."
    elif xG < 0.056474: # 50% percentile
        description = "This was a low chance of scoring."
    elif xG < 0.096197: # 75% percentile
        description = "This was a decent chance."
    elif xG < 0.3: # very high
        description = "This was a high-quality chance, with a good probability of scoring."
    else:
        description = "This was an excellent chance."
    
    return description


def describe_xT_pass(xT,xG):
    if xG != 0:
        if xT <= 0.024800:
            if xG < 0.066100:
                description = f" It had low xT {xT} value and probability of pass being a shot was {xT * 100:.0f}% with  xG value {xG:.3f}. There is less chances for it to be a safe pass creating less goal scoring opportunities."
            else:
                description = f" It had low xT {xT} value and probability of pass being a shot was {xT * 100:.0f}% with  xG value {xG:.3f}. There is less chances for it to be a dangerous pass creating a good goal scoring opportunities."
        elif xT > 0.024800 and xT <= 0.066100:
            if xG < 0.066100:
                description = f" It had moderate xT {xT} value and probability of pass being a shot was {xT * 100:.0f}% with xG value is {xG:.3f}. There is moderate chances of being a safe pass creating less goal scoring opportunities."
            else:
                description = f" It had moderate xT {xT} value and probability of pass being a shot was {xT * 100:.0f}% with xG value is {xG:.3f}. There is moderate chances of being a dangerous pass creating good goal scoring opportunities."
        elif xT > 0.066100 and xT <=  0.150000:
            if xG < 0.066100:
                description = f" It had high xT {xT} value and probability of pass being a shot was {xT * 100:.0f}% with xG value {xG:.3f}. There is high chances of being a safe pass creating less goal scoring opportunities."
            else:
                description = f" It had high xT {xT} value and probability of pass being a shot was {xT * 100:.0f}% with xG value {xG:.3f}. There is high chances of being a dangerous pass creating good goal scoring opportunities."  
        else:
            if xG < 0.066100:
                description = f" It had excellent xT {xT} value and probability of pass being a shot was {xT * 100:.0f}% creating a excellent goal scoring opportunities for safe pass." 
            else:
                description = f" It had excellent xT {xT} value and probability of pass being a shot was {xT * 100:.0f}% creating a excellent goal scoring opportunities for dangerous pass." 
            
    else:
        description = f" The xT value is {xT} and it did not lead to a shot, opportunities to score goal is less and was a safe pass."
    return description


def describe_xT_pass_1(xT,xG):
    if xG != 0:
        if xT <= 0.05178240314126015:
            description = f"It had low xT {xT} value and probability of pass being a shot was {xT * 100:.0f}%. There is less chances for it to be a safe pass creating less goal scoring opportunities."
        elif xT < 0.06652335822582245:
            description = f" It had moderate xT {xT} value and probability of pass being a shot was {xT * 100:.0f}%. There is moderate chances of being a safe pass creating moderate goal scoring opportunities."
        else:
            description = f" It had high xT {xT} value and probability of pass being a shot was {xT * 100:.0f}% with xG value is {xG:.3f}. There is high chances of being a dangerous pass creating high goal scoring opportunities."   
            
    else:
        description = f" The xT value is {xT} and it did not lead to a shot, opportunities to score goal is less and was a safe pass."
    return description


### describe function for logistic
def describe_xT_pass_logistic(xT,xG):
    if xG != 0:
        if xT <= 0.05882884: #25 percentile xT
            if xG < 0.066100:
                description = f" It had low xT {xT} value and probability of pass being a shot was {xT * 100:.0f}% with  xG value {xG:.3f}. There is less chances for it to be a safe pass creating less goal scoring opportunities."
            else:
                description = f" It had low xT {xT} value and probability of pass being a shot was {xT * 100:.0f}% with  xG value {xG:.3f}. There is less chances for it to be a dangerous pass creating a good goal scoring opportunities."
        elif xT > 0.05882884 and xT <= 0.079879159: #50 percentile
            if xG < 0.066100:
                description = f" It had moderate xT {xT} value and probability of pass being a shot was {xT * 100:.0f}% with xG value is {xG:.3f}. There is moderate chances of being a safe pass creating less goal scoring opportunities."
            else:
                description = f" It had moderate xT {xT} value and probability of pass being a shot was {xT * 100:.0f}% with xG value is {xG:.3f}. There is moderate chances of being a dangerous pass creating good goal scoring opportunities."
        elif xT > 0.079879159 and xT <=  0.130563166: #75 percentile
            if xG < 0.066100:
                description = f" It had high xT {xT} value and probability of pass being a shot was {xT * 100:.0f}% with xG value {xG:.3f}. There is high chances of being a safe pass creating less goal scoring opportunities."
            else:
                description = f" It had high xT {xT} value and probability of pass being a shot was {xT * 100:.0f}% with xG value {xG:.3f}. There is high chances of being a dangerous pass creating good goal scoring opportunities."  
        else:
            if xG < 0.066100:
                description = f" It had excellent xT {xT} value and probability of pass being a shot was {xT * 100:.0f}% creating a excellent goal scoring opportunities for safe pass." 
            else:
                description = f" It had excellent xT {xT} value and probability of pass being a shot was {xT * 100:.0f}% creating a excellent goal scoring opportunities for dangerous pass."    
    else:
        if xT <= 0.05882884: #25 percentile
            description = f" The xT value is {xT} and it did not lead to a shot and opportunities to score goal is less."
        elif xT <= 0.079879159: #50 percentile
            description = f" The xT value is {xT} and it did not lead to a shot, the opportunities to score goal is moderate."
        elif xT <= 0.130563166: #75 percentile
            description = f" The xT value is {xT} and it did not lead to a shot, the opportunities to score goal is high."
        else:
            description = f" The xT value is {xT} and it did not lead to a shot, the opportunities to score goal is very high."
    return description


### describe for xNN 
def describe_xT_pass_xNN(xT,xG):
    if xG != 0:
        if xT <= 0.05178240314126015: #25 percentile xT
            if xG < 0.066100:
                description = f" It had low xT {xT} value and probability of pass being a shot was {xT * 100:.0f}% with  xG value {xG:.3f}. There is less chances for it to be a safe pass creating less goal scoring opportunities."
            else:
                description = f" It had low xT {xT} value and probability of pass being a shot was {xT * 100:.0f}% with  xG value {xG:.3f}. There is less chances for it to be a dangerous pass creating a good goal scoring opportunities."
        elif xT > 0.05178240314126015 and xT <= 0.06115284748375416: #50 percentile
            if xG < 0.066100:
                description = f" It had moderate xT {xT} value and probability of pass being a shot was {xT * 100:.0f}% with xG value is {xG:.3f}. There is moderate chances of being a safe pass creating less goal scoring opportunities."
            else:
                description = f" It had moderate xT {xT} value and probability of pass being a shot was {xT * 100:.0f}% with xG value is {xG:.3f}. There is moderate chances of being a dangerous pass creating good goal scoring opportunities."
        elif xT > 0.06115284748375416 and xT <=  0.06652335822582245: #75 percentile
            if xG < 0.066100:
                description = f" It had high xT {xT} value and probability of pass being a shot was {xT * 100:.0f}% with xG value {xG:.3f}. There is high chances of being a safe pass creating less goal scoring opportunities."
            else:
                description = f" It had high xT {xT} value and probability of pass being a shot was {xT * 100:.0f}% with xG value {xG:.3f}. There is high chances of being a dangerous pass creating good goal scoring opportunities."  
        else:
            if xG < 0.066100:
                description = f" It had excellent xT {xT} value and probability of pass being a shot was {xT * 100:.0f}% creating a excellent goal scoring opportunities for safe pass." 
            else:
                description = f" It had excellent xT {xT} value and probability of pass being a shot was {xT * 100:.0f}% creating a excellent goal scoring opportunities for dangerous pass."   
    else:
        if xT <= 0.05178240314126015: #25 percentile
            description = f" The xT value is {xT} and it did not lead to a shot and opportunities to score goal is less."
        elif xT <= 0.06115284748375416: #50 percentile
            description = f" The xT value is {xT} and it did not lead to a shot, the opportunities to score goal is moderate."
        elif xT <= 0.06652335822582245: #75 percentile
            description = f" The xT value is {xT} and it did not lead to a shot, the opportunities to score goal is high."
        else:
            description = f" The xT value is {xT} and it did not lead to a shot, the opportunities to score goal is very high."
    
    return description


def describe_position_pass(x, y, team_direction):
    # Mirror coordinates if team is attacking left
    if team_direction == 'left':
        x = 105 - x
        y = 68 - y

    # Normalize like the visual
    x = x * 100 / 105
    y = y * 100 / 68

    # Zone is determined from normalized x
    if x <= 33:
        description = "the defensive zone" if team_direction == 'right' else "the attacking zone"
    elif 33 < x < 66:
        description = "the middle zone"
    else:
        description = "the attacking zone" if team_direction == 'right' else "the defensive zone"

    return description


# In sentences.py or wherever you manage your sentences module

def read_feature_thresholds(competition):
        competitions_dict_prams = {
        "EURO Men 2024": "data/feature_description_EURO_Men_2024.xlsx",
        "National Women's Soccer League (NWSL) 2018": "data/feature_description_NWSL.xlsx",
        "FIFA 2022": "data/feature_description_FIFA_2022.xlsx",
        "Women's Super League (FAWSL) 2017-18": "data/feature_description_FAWSL.xlsx",
        "EURO Men 2020": "data/feature_description_EURO_Men_2020.xlsx",
        "Africa Cup of Nations (AFCON) 2023": "data/feature_description_AFCON_2023.xlsx",}

        file_path = competitions_dict_prams.get(competition)
        thresh_file = pd.read_excel(file_path)
        return thresh_file



def describe_shot_features(features, competition):
    descriptions = []

    thresholds= read_feature_thresholds(competition)
    #st.write(thresholds)    
    # Get the thresholds for each feature
    vertical_distance_thresholds = thresholds['c']
    euclidean_distance_thresholds = thresholds['distance']
    nearby_opponents_thresholds = thresholds['close_players']
    opponents_triangle_thresholds = thresholds['triangle']
    goalkeeper_distance_thresholds = thresholds['gk_distance']
    nearest_opponent_distance_thresholds = thresholds['dist_to_nearest_opponent']
    angle_to_goalkeeper_thresholds = thresholds['angle_to_gk']

    # Binary features description
    #if features['header'] == 1:
        #descriptions.append("The shot was a header.")
    if features['shot_with_left_foot'] == 1:
            descriptions.append("The shot was with the left foot.")    
    else:
        descriptions.append("The shot was with the right foot.")

    if features['shot_during_regular_play'] == 1:
        descriptions.append("The shot was taken during open play.")
    else:
        if features['shot_after_throw_in'] == 1:
            descriptions.append("The shot was taken after a throw-in.")
        elif features['shot_after_corner'] == 1:
            descriptions.append("The shot was taken after a corner.")
        elif features['shot_after_free_kick'] == 1:
            descriptions.append("The shot was taken after a free-kick.")
        else:    
            descriptions.append(f"The shot was taken from a {features['pattern']}.")    

    # Use the thresholds dynamically
    if features['vertical_distance_to_center'] < vertical_distance_thresholds.iloc[4]:
        descriptions.append("It was taken from very close to the center of the pitch.")
    elif features['vertical_distance_to_center'] < vertical_distance_thresholds.iloc[6]:
        descriptions.append("It was taken reasonably centrally.")
    else:
        descriptions.append("It was taken quite a long way from the centre of the pitch.")

    if features['euclidean_distance_to_goal'] < euclidean_distance_thresholds.iloc[4]:
        descriptions.append("It was taken from a close range, near the goal.")
    elif features['euclidean_distance_to_goal'] < euclidean_distance_thresholds.iloc[6]:
        descriptions.append("It was taken from a moderate distance from the goal.")
    else:
        descriptions.append("It was taken from long range, far from the goal.")

    if features['nearby_opponents_in_3_meters'] < nearby_opponents_thresholds.iloc[4]:
        descriptions.append("It was taken with little or no pressure from opponents.")
    elif features['nearby_opponents_in_3_meters'] < nearby_opponents_thresholds.iloc[6]:
        descriptions.append("It was taken with moderate pressure, with one opponent within 3 meters.")
    else:
        descriptions.append("It was taken under heavy pressure, with several opponents within 3 meters.")

    if features['opponents_in_triangle'] < opponents_triangle_thresholds.iloc[4]:
        descriptions.append("It was taken with no opposition between the shooter and the goal.")
    elif features['opponents_in_triangle'] < opponents_triangle_thresholds.iloc[6]:
        descriptions.append("There were some opposition players blocking the path, but there was space for a well-placed shot.")
    else:
        descriptions.append("There were multiple opponents blocking the path.")

    if features['goalkeeper_distance_to_goal'] < goalkeeper_distance_thresholds.iloc[4]:
        descriptions.append("The goalkeeper was very close to the goal.")
    elif features['goalkeeper_distance_to_goal'] < goalkeeper_distance_thresholds.iloc[6]:
        descriptions.append("The goalkeeper was at a moderate distance from the goal.")
    else:
        descriptions.append("The goalkeeper was positioned far from the goal.")

    if features['distance_to_nearest_opponent'] < nearest_opponent_distance_thresholds.iloc[4]:
        descriptions.append("The shot was taken with strong pressure from a very close opponent.")
    elif features['distance_to_nearest_opponent'] < nearest_opponent_distance_thresholds.iloc[6]:
        descriptions.append("The shot was taken with moderate pressure from an opponent nearby.")
    else:
        descriptions.append("The shot was taken with no immediate pressure from any close opponent, with the nearest opponent far away.")

    if features['angle_to_goalkeeper'] < angle_to_goalkeeper_thresholds.iloc[4]:
        descriptions.append("The shot was taken from a broad angle towards the goalkeeper being on the left, making it difficult to score.")
    elif features['angle_to_goalkeeper'] < angle_to_goalkeeper_thresholds.iloc[6]:
        descriptions.append("The shot was taken from a relatively good angle, allowing for a decent chance.")
    else:
        descriptions.append("The shot was taken from a broad angle towards the goalkeeper being on the right.")

    return descriptions

    # # Continuous features description
    # if features['vertical_distance_to_center'] < 2.805:
    #     descriptions.append("It was taken from very close to the center of the pitch.")
    # elif features['vertical_distance_to_center'] < 9.647:
    #     descriptions.append("It was taken reasonably centrally.")
    # else:
    #     descriptions.append("It was taken quite a long way from the centre of the pitch.")

    # if features['euclidean_distance_to_goal'] < 10.278:
    #     descriptions.append("It was taken from a close range, near the goal.")
    # elif features['euclidean_distance_to_goal'] < 21.116:
    #     descriptions.append("It was taken from a moderate distance from the goal.")
    # else:
    #     descriptions.append("It was taken from long range, far from the goal.")

    # if features['nearby_opponents_in_3_meters'] < 1:
    #     descriptions.append("It was taken with little or no pressure from opponents.")
    # elif features['nearby_opponents_in_3_meters'] < 2:
    #     descriptions.append("It was taken with moderate pressure, with one opponent within 3 meters.")
    # else:
    #     descriptions.append("It was taken under heavy pressure, with several opponents within 3 meters.")

    # if features['opponents_in_triangle'] < 1:
    #     descriptions.append("it was taken with no oppositions between the shooter and the goals.")
    # elif features['opponents_in_triangle'] < 2:
    #     descriptions.append("There were some opposition players blocking the path, but there was spac for a well-placed shot.")
    # else:
    #     descriptions.append("There we multiple opponents blocking the path.")

    # if features['goalkeeper_distance_to_goal'] < 1.649:
    #     descriptions.append("The goalkeeper was very close to the goal.")
    # elif features['goalkeeper_distance_to_goal'] < 3.217:
    #     descriptions.append("The goalkeeper was at a moderate distance from the goal.")
    # else:
    #     descriptions.append("The goalkeeper was positioned far from the goal.")

    # if features['distance_to_nearest_opponent'] < 1.119:
    #     descriptions.append("The shot was taken with strong pressure from a very close opponent.")
    # elif features['distance_to_nearest_opponent'] < 1.779:
    #     descriptions.append("The shot was taken with moderate pressure from an opponent nearby.")
    # else:
    #     descriptions.append("The shot was taken with no immediate pressure from any close opponent, with the nearest opponent far away.")

    # if features['angle_to_goalkeeper'] < -23.36:
    #     descriptions.append("The shot was taken from a broad angle towards goalkeeper being on left, making it difficult to score.")
    # elif features['angle_to_goalkeeper'] < 22.72:
    #     descriptions.append("The shot was taken from a relatively good angle, allowing for a decent chance.")
    # else:
    #     descriptions.append("The shot was taken from a broad angle towards the goalkeeper being on right.")

    # return descriptions

def describe_shot_single_feature(feature_name, feature_value):
    # Describe binary features
    if feature_name == 'header':
        return "the shot was a header." if feature_value == 1 else "the shot was not a header."
    if feature_name == 'shot_with_left_foot':
        return "the shot was with the left foot." if feature_value == 1 else "the shot was with the right foot."
    if feature_name == 'shot_during_regular_play':
        return "the shot was taken during regular play." if feature_value == 1 else "the shot was taken from a set-piece."
    if feature_name == 'shot_after_throw_in':
        return "the shot was taken after a throw-in." if feature_value == 1 else "the shot was not taken after a throw-in."
    if feature_name == 'shot_after_corner':
        return "the shot was taken after a corner." if feature_value == 1 else "the shot was not taken after a corner."
    if feature_name == 'shot_after_free_kick':
        return "the shot was taken after a free-kick." if feature_value == 1 else "the shot was not taken after a free-kick."
 
    # Describe continuous features
    if feature_name == 'vertical_distance_to_center':
        if feature_value < 2.805:
            return "the shot was taken closer to the center of the pitch (less vertical distance)."
        elif feature_value < 9.647:
            return "the shot was taken from an intermediate vertical distance."
        else:
            return "the shot was taken far from the center, closer to the touchline."
    if feature_name == 'euclidean_distance_to_goal':
        if feature_value < 10.278:
            return "the shot was taken from a close range, near the goal."
        elif feature_value < 21.116:
            return "the shot was taken from a moderate distance to the goal."
        else:
            return "the shot was taken from a long range, far from the goal."
    if feature_name == 'nearby_opponents_in_3_meters':
        if feature_value < 1:
            return "the shot was taken with little to no pressure from opponents within 3 meters."
        elif feature_value < 2:
            return "the shot was taken with moderate pressure, with some opponents nearby within 3 meters."
        else:
            return "the shot was taken under heavy pressure, with several opponents close by within 3 meters."
    if feature_name == 'opponents_in_triangle':
        if feature_value < 1:
            return "the shot was taken with minimal opposition in the shooting triangle."
        elif feature_value < 2:
            return "the shot was taken with some opposition blocking the path."
        else:
            return "the shot was heavily contested, with multiple opponents blocking the path."
    if feature_name == 'goalkeeper_distance_to_goal':
        if feature_value < 1.649:
            return "the goalkeeper was very close to the goal."
        elif feature_value < 3.217:
            return "the goalkeeper was at a moderate distance from the goal."
        else:
            return "the goalkeeper was positioned far from the goal."
    if feature_name == 'distance_to_nearest_opponent':
        if feature_value < 1.119:
            return "the shot was taken with strong pressure from a very close opponent."
        elif feature_value < 1.779:
            return "the shot was taken with moderate pressure from an opponent nearby."
        else:
            return "the shot was taken with no immediate pressure, as the nearest opponent was far away."
    if feature_name == 'angle_to_goalkeeper':
        if feature_value < -23.36:
            return "the shot was taken from a broad angle to the left of the goalkeeper, making it difficult to score."
        elif feature_value < 22.72:
            return "the shot was taken from a good angle, providing a decent chance to score."
        else:
            return "the shot was taken from a broad angle to the right of the goalkeeper."
    
    # Default case if the feature is unrecognized
    return f"No description available for {feature_name}."


feature_name_mapping = {
    'vertical_distance_to_center_contribution': 'squared distance to center',
    'euclidean_distance_to_goal_contribution': 'euclidean distance to goal',
    'nearby_opponents_in_3_meters_contribution': 'nearby opponents within 3 meters',
    'opponents_in_triangle_contribution': 'number of opponents in triangle formed by shot location and goalposts',
    'goalkeeper_distance_to_goal_contribution': 'distance to goal of the goalkeeper',
    'header_contribution': 'header',
    'distance_to_nearest_opponent_contribution': 'distance to nearest opponent',
    'angle_to_goalkeeper_contribution': 'angle to goalkeepr',
    'shot_with_left_foot_contribution': 'shot taken with left foot',
    'shot_after_throw_in_contribution': 'shot after throw in',
    'shot_after_corner_contribution': 'shot after corner',
    'shot_after_free_kick_contribution': 'shot after free kick',
    'shot_during_regular_play_contribution': 'shot during regular play'

}
def describe_shot_contributions(shot_contributions, shot_features, feature_name_mapping=feature_name_mapping):
    text = "The contributions of the features to the xG of the shot, sorted by their magnitude from largest to smallest, are as follows:\n"
    
    # Extract the contributions from the shot_contributions DataFrame
    contributions = shot_contributions.iloc[0].drop(['match_id', 'id', 'xG'])  # Drop irrelevant columns
    
    # Sort the contributions by their absolute value (magnitude) in descending order
    sorted_contributions = contributions.abs().sort_values(ascending=False)
    
    # Get the top 4 contributions
    #top_contributions = sorted_contributions.head(4)
    top_contributions = sorted_contributions
    
    # Loop through the top contributions to generate descriptions
    for idx, (feature, contribution) in enumerate(top_contributions.items()):

        # Get the original sign of the contribution
        original_contribution = contributions[feature]

        if original_contribution >= 0.1 or original_contribution <= -0.1:
        
            # Remove "_contribution" suffix to match feature names in shot_features
            feature_name = feature.replace('_contribution', '')
            
            # Use feature_name_mapping to get the display name for the feature (if available)
            feature_display_name = feature_name_mapping.get(feature, feature)
            
            # Get the feature value from shot_features
            feature_value = shot_features[feature_name]
            
            # Get the feature description
            feature_value_description = describe_shot_single_feature(feature_name, feature_value)
            
            # Add the feature's contribution to the xG description
            if original_contribution > 0:
                impact = 'maximum positive contribution'
                impact_text = "increased the xG of the shot."
            elif original_contribution < 0:
                impact = 'maximum negative contribution'
                impact_text = "reduced the xG of the shot."
            else:
                impact = 'no contribution'
                impact_text = "had no impact on the xG of the shot."

            # Use appropriate phrasing for the first feature and subsequent features
            if idx == 0:
                text += f"\nThe most impactful feature is {feature_display_name}, which had the {impact} because {feature_value_description}. This feature {impact_text}"
            else:
                text += f"\nAnother impactful feature is {feature_display_name}, which had the {impact} because {feature_value_description} This feature {impact_text}"
        

    return text

def describe_shot_contributions1(shot_contributions, feature_name_mapping=feature_name_mapping, thresholds=None):
    
    # Default thresholds if none are provided
    thresholds = thresholds or {
        'very_large': 0.75,
        'large': 0.50,
        'moderate': 0.25,
        'low': 0.00
    }

    # Initialize a list to store contributions that are not 'match_id', 'id', or 'xG'
    valid_contributions = {}

    # Loop through the columns to select valid ones
    for feature, contribution in shot_contributions.iloc[0].items():
        if feature not in ['match_id', 'id', 'xG']:  # Skip these columns
            valid_contributions[feature] = contribution

    # Convert to Series and sort by absolute values in descending order
    sorted_contributions = (
        pd.Series(valid_contributions)
        .apply(lambda x: abs(x))
        .sort_values(ascending=False)
    )

    # Loop through the sorted contributions and categorize them based on thresholds
    for feature, contribution in sorted_contributions.items():
        # Get the original sign of the contribution
        original_contribution = valid_contributions[feature]

        # Use the feature_name_mapping dictionary to get the display name for the feature
        feature_display_name = feature_name_mapping.get(feature, feature)

        # Determine the contribution level
        if abs(contribution) > thresholds['very_large']:
            level = 'very large'
        elif abs(contribution) > thresholds['large']:
            level = 'large'
        elif abs(contribution) > thresholds['moderate']:
            level = 'moderate'
        else:
            level = 'low'

        # Distinguish between positive and negative contributions
        if original_contribution > 0:
            explanation = f"{feature_display_name} has a {level} positive contribution, which increased the xG of the shot"
        elif original_contribution < 0:
            explanation = f"{feature_display_name} has a {level} negative contribution, which reduced the xG of the shot"
        else:
            explanation = f"{feature_display_name} had no contribution to the xG of the shot"

        # Add to the text
        text += f"{explanation}\n"
    
    return text

#thresholds for contribution features
def read_pass_feature_thresholds(competition):
    competitions_dict_params = {
        "Allsevenskan 2022": "data/feature_description_passes.csv",
        "Allsevenskan 2023": "data/feature_description_passes.csv"
    }

    file_path = competitions_dict_params.get(competition)
    
    if file_path is None:
        raise ValueError(f"Competition '{competition}' not found in pass feature descriptions.")

    thresholds_pass = pd.read_csv(file_path, index_col=0)
    return thresholds_pass

def describe_pass_features(features, competition):
    descriptions = []

    # Step 1: Load thresholds
    thresholds = read_pass_feature_thresholds(competition)

    # pass_length
    if features['pass_length'] <= thresholds['pass_length'].iloc[4]:
        descriptions.append(" It was a short pass")
    elif features['pass_length'] <= thresholds['pass_length'].iloc[5]:
        descriptions.append(" It was a medium length pass")
    else:
        descriptions.append(" It was a long pass")

    # start_angle_to_goal

    if features['start_angle_to_goal'] <= thresholds['start_angle_to_goal'].iloc[4]:
        descriptions.append(f" with narrow angle covering distance {features['start_distance_to_goal']:.2f}m aiming towards the goal area, a bold attacking move.")
        
    elif features['start_angle_to_goal'] <= thresholds['start_angle_to_goal'].iloc[5]:
        descriptions.append(f" with moderate angle covering distance {features['start_distance_to_goal']:.2f}m aiming to the goal area, maintainig control while progressing.")
    else:
        descriptions.append(f" with wide-angle covering distance {features['start_distance_to_goal']:.2f}m aiming to the goal area, possibly trying to spread out the defense and create space.")
    

    # teammates behind and beyond 
    descriptions.append(f" There were {features['teammates_beyond']} teammates positioned ahead of the passer and {features['teammates_behind']} teammates behind the passer at the moment of the pass.")
    descriptions.append(f" There were {features['opponents_beyond']} teammates positioned ahead of the passer and {features['opponents_behind']} opponents behind the passer at the moment of the pass.")


    # opponents_between
    if features['opponents_between'] <= thresholds['opponents_between'].iloc[3]:
        descriptions.append(" There was no opponents in the passing lane ")
    elif features['opponents_between'] <= thresholds['opponents_between'].iloc[4]:
        descriptions.append(" The passing lane was mostly open with few opponents")
    else:
        descriptions.append("The passing lane was crowded with opponents")

    # packing
    if features['packing'] <= thresholds['packing'].iloc[3]:
        descriptions.append(" and there was no opponent bypassed within 5m.")
    elif features['packing'] <= thresholds['packing'].iloc[4]:
        descriptions.append(" and there was 1 opponent bypassed within 5m.")
    elif features['packing'] <= thresholds['packing'].iloc[5]:
        descriptions.append(" and there were 2 opponents bypassed within 5m.")
    elif features['packing'] <= thresholds['packing'].iloc[6]:
        descriptions.append(" and there were 3 opponents bypassed within 5m.")    
    else:
        descriptions.append(f" and there were {features['packing']} opponents bypassed within 5m.")    

    # Step 2: Categorical - Pressure level on passer
    pressure = features['pressure_level_passer']
    
    if pressure == "Low Pressure":
        if features['opponents_nearby'] < 2:
            descriptions.append(f" There is {features['opponents_nearby']} nearby opponent within 6m, creating low pressure at the moment of the pass.")
        else :
            descriptions.append(f" There are {features['opponents_nearby']} nearby opponents within 6m, creating low pressure at the moment of the pass.")

    elif pressure == "Middle Pressure":
        descriptions.append(f" There are {features['opponents_nearby']} nearby opponents within 6m, creating moderate pressure at the moment of the pass.")
    elif pressure == "High Pressure":
        descriptions.append(f" There are {features['opponents_nearby']} nearby opponents within 6m, creating high pressure at the moment of the pass.")
    else :
        descriptions.append(" Pressure level information is unavailable or not classified.")
    return descriptions


### for logistic model 
def read_pass_feature_thresholds_logistic(competition):
    competitions_dict_params = {
        "Allsevenskan 2022": "data/feature_description_passes.csv",
        "Allsevenskan 2023": "data/feature_description_passes.csv"
    }

    file_path = competitions_dict_params.get(competition)
    
    if file_path is None:
        raise ValueError(f"Competition '{competition}' not found in pass feature descriptions.")

    thresholds_pass = pd.read_csv(file_path, index_col=0)
    return thresholds_pass

def describe_pass_features_logistic(features, competition):
    descriptions = []

    # Step 1: Load thresholds
    thresholds = read_pass_feature_thresholds(competition)

    # pass_length
    if features['pass_length'] <= thresholds['pass_length'].iloc[4]:
        descriptions.append(" It was a short pass")
    elif features['pass_length'] <= thresholds['pass_length'].iloc[5]:
        descriptions.append(" It was a medium length pass")
    else:
        descriptions.append(" It was a long pass")

    # start_angle_to_goal

    if features['start_angle_to_goal'] <= thresholds['start_angle_to_goal'].iloc[4]:
        descriptions.append(f" with narrow angle covering distance {features['start_distance_to_goal']:.2f}m aiming towards the goal area, a bold attacking move.")
        
    elif features['start_angle_to_goal'] <= thresholds['start_angle_to_goal'].iloc[5]:
        descriptions.append(f" with moderate angle covering distance {features['start_distance_to_goal']:.2f}m aiming to the goal area, maintainig control while progressing.")
    else:
        descriptions.append(f" with wide-angle covering distance {features['start_distance_to_goal']:.2f}m aiming to the goal area, possibly trying to spread out the defense and create space.")
    

    # teammates behind and beyond 
    descriptions.append(f" There were {features['teammates_beyond']} teammates positioned ahead of the passer at the moment of the pass.")
    descriptions.append(f" There were {features['opponents_beyond']} opponents positioned ahead of the passer at the moment of the pass.")
    

    # opponents_between
    if features['opponents_between'] <= thresholds['opponents_between'].iloc[3]:
        descriptions.append(" There was no opponents in the passing lane ")
    elif features['opponents_between'] <= thresholds['opponents_between'].iloc[4]:
        descriptions.append(" The passing lane was mostly open with few opponents")
    else:
        descriptions.append("The passing lane was crowded with opponents")

    # packing
    if features['packing'] <= thresholds['packing'].iloc[3]:
        descriptions.append(" and there was no opponent bypassed within 5m.")
    elif features['packing'] <= thresholds['packing'].iloc[4]:
        descriptions.append(" and there was 1 opponent bypassed within 5m.")
    elif features['packing'] <= thresholds['packing'].iloc[5]:
        descriptions.append(" and there were 2 opponents bypassed within 5m.")
    elif features['packing'] <= thresholds['packing'].iloc[6]:
        descriptions.append(" and there were 3 opponents bypassed within 5m.")    
    else:
        descriptions.append(f" and there were {features['packing']} opponents bypassed within 5m.")    

    # Step 2: Categorical - Pressure level on passer
    pressure = features['pressure_level_passer']
    
    if pressure == "Low Pressure":
       if features['opponents_nearby'] <= 2:
        if features['opponents_nearby'] == 0:
            descriptions.append(" There are no nearby opponents within 6m, creating low pressure at the moment of the pass.")
        elif features['opponents_nearby'] == 1:
            descriptions.append(" There is 1 nearby opponent within 6m, creating low pressure at the moment of the pass.")
        else:
            descriptions.append(f" There are {features['opponents_nearby']} nearby opponents within 6m, creating low pressure at the moment of the pass.")

    elif pressure == "Middle Pressure":
        descriptions.append(f" There are {features['opponents_nearby']} nearby opponents within 6m, creating moderate pressure at the moment of the pass.")
    elif pressure == "High Pressure":
        descriptions.append(f" There are {features['opponents_nearby']} nearby opponents within 6m, creating high pressure at the moment of the pass.")
    else :
        descriptions.append(" Pressure level information is unavailable or not classified.")
    return descriptions


### pass features defination for all models
feature_name_mapping = {
    'vertical_distance_to_center_contribution': 'squared distance to center',
    'euclidean_distance_to_goal_contribution': 'euclidean distance to goal',
    'nearby_opponents_in_3_meters_contribution': 'nearby opponents within 3 meters',
    'opponents_in_triangle_contribution': 'number of opponents in triangle formed by shot location and goalposts',
    'goalkeeper_distance_to_goal_contribution': 'distance to goal of the goalkeeper',
    'header_contribution': 'header',
    'distance_to_nearest_opponent_contribution': 'distance to nearest opponent',
    'angle_to_goalkeeper_contribution': 'angle to goalkeepr',
    'shot_with_left_foot_contribution': 'shot taken with left foot',
    'shot_after_throw_in_contribution': 'shot after throw in',
    'shot_after_corner_contribution': 'shot after corner',
    'shot_after_free_kick_contribution': 'shot after free kick',
    'shot_during_regular_play_contribution': 'shot during regular play'

}
def describe_shot_contributions(shot_contributions, shot_features, feature_name_mapping=feature_name_mapping):
    text = "The contributions of the features to the xG of the shot, sorted by their magnitude from largest to smallest, are as follows:\n"
    
    # Extract the contributions from the shot_contributions DataFrame
    contributions = shot_contributions.iloc[0].drop(['match_id', 'id', 'xG'])  # Drop irrelevant columns
    
    # Sort the contributions by their absolute value (magnitude) in descending order
    sorted_contributions = contributions.abs().sort_values(ascending=False)
    
    # Get the top 4 contributions
    #top_contributions = sorted_contributions.head(4)
    top_contributions = sorted_contributions
    
    # Loop through the top contributions to generate descriptions
    for idx, (feature, contribution) in enumerate(top_contributions.items()):

        # Get the original sign of the contribution
        original_contribution = contributions[feature]

        if original_contribution >= 0.01 or original_contribution <= -0.01:
        
            # Remove "_contribution" suffix to match feature names in shot_features
            feature_name = feature.replace('_contribution', '')
            
            # Use feature_name_mapping to get the display name for the feature (if available)
            feature_display_name = feature_name_mapping.get(feature, feature)
            
            # Get the feature value from shot_features
            feature_value = shot_features[feature_name]
            
            # Get the feature description
            feature_value_description = describe_shot_single_feature(feature_name, feature_value)
            
            # Add the feature's contribution to the xG description
            if original_contribution > 0:
                impact = 'maximum positive contribution'
                impact_text = "increased the xG of the shot."
            elif original_contribution < 0:
                impact = 'maximum negative contribution'
                impact_text = "reduced the xG of the shot."
            else:
                impact = 'no contribution'
                impact_text = "had no impact on the xG of the shot."

            # Use appropriate phrasing for the first feature and subsequent features
            if idx == 0:
                text += f"\nThe most impactful feature is {feature_display_name}, which had the {impact} because {feature_value_description}. This feature {impact_text}"
            else:
                text += f"\nAnother impactful feature is {feature_display_name}, which had the {impact} because {feature_value_description} This feature {impact_text}"
        

    return text




def describe_shot_contributions1(shot_contributions, feature_name_mapping=feature_name_mapping, thresholds=None):
    
    # Default thresholds if none are provided
    thresholds = thresholds or {
        'very_large': 0.75,
        'large': 0.50,
        'moderate': 0.25,
        'low': 0.00
    }

    # Initialize a list to store contributions that are not 'match_id', 'id', or 'xG'
    valid_contributions = {}

    # Loop through the columns to select valid ones
    for feature, contribution in shot_contributions.iloc[0].items():
        if feature not in ['match_id', 'id', 'xG']:  # Skip these columns
            valid_contributions[feature] = contribution

    # Convert to Series and sort by absolute values in descending order
    sorted_contributions = (
        pd.Series(valid_contributions)
        .apply(lambda x: abs(x))
        .sort_values(ascending=False)
    )

    # Loop through the sorted contributions and categorize them based on thresholds
    for feature, contribution in sorted_contributions.items():
        # Get the original sign of the contribution
        original_contribution = valid_contributions[feature]

        # Use the feature_name_mapping dictionary to get the display name for the feature
        feature_display_name = feature_name_mapping.get(feature, feature)

        # Determine the contribution level
        if abs(contribution) > thresholds['very_large']:
            level = 'very large'
        elif abs(contribution) > thresholds['large']:
            level = 'large'
        elif abs(contribution) > thresholds['moderate']:
            level = 'moderate'
        else:
            level = 'low'

        # Distinguish between positive and negative contributions
        if original_contribution > 0:
            explanation = f"{feature_display_name} has a {level} positive contribution, which increased the xG of the shot"
        elif original_contribution < 0:
            explanation = f"{feature_display_name} has a {level} negative contribution, which reduced the xG of the shot"
        else:
            explanation = f"{feature_display_name} had no contribution to the xG of the shot"

        # Add to the text
        text += f"{explanation}\n"
    
    return text

### pass features
def describe_pass_single_feature(feature_name, feature_value): 
    if feature_name == "pass_length":
        if feature_value < 14.456917459901321:
            return "the pass was short"
        elif feature_value < 24.36115859931905:
            return "the pass had moderate length"
        else:
            return "the pass was long"

    if feature_name == "start_angle_to_goal":
        if feature_value < 4.692139370656406:
            return "the pass started from a narrow angle to the goal"
        elif feature_value < 5.268353760844164:
            return "the pass started from a moderate angle to the goal"
        else:
            return "the pass started from a wide angle to the goal"
    
    if feature_name == "end_angle_to_goal":
        if feature_value < 4.146716596474326:
            return "the pass ended from a narrow angle to the goal"
        elif feature_value < 4.924911291472834:
                return"the pass ended from a moderate angle to the goal"
        else:
            return"the pass ended from a wide angle to the goal"
    

    if feature_name == "start_distance_to_sideline":
        if feature_value < 8.0449:
            return "the pass started close to the sideline"
        elif feature_value < 14.5973:
            return "the pass started moderately close to the sideline"
        elif feature_value < 22.4270:
            return "the pass started near the central area of the pitch"
        elif feature_value < 34:
            return "the pass started close to the center of the pitch"
        else:
            return "the pass started at the exact center line of the pitch"

    if feature_name == 'end_distance_to_sideline':
        if feature_value < 6.481136227080518:
            return "the pass ended close to the sideline"
        elif feature_value < 14.28:
            return "the pass started moderately close to the sideline"
        elif feature_value < 24.48:
            return "the pass ended close to the center of the pitch"
        else:
            return "the pass ended at exact center line of the pitch"

    if feature_name == "start_distance_to_goal":
        if feature_value < 2.7301764469404706:
            return "the pass started close to the goal"
        elif feature_value < 30.92668344228875:
            return "the pass started from a moderate distance to the goal"
        else:
            return "the pass started far from the goal"
        

    if feature_name == "end_distance_to_goal":
        if feature_value < 1.0499999999999972:
            return "the pass ended close to the goal"
        elif feature_value < 29.13286975222317:
            return "the pass ended from a moderate distance to the goal"
        else:
            return "the pass ended far from the goal"
    
    if feature_name == "pass_angle":
        if feature_value <= 0:
            return "the pass was played backward or toward the player's own half"
        elif feature_value < 8.0449:
            return "the pass was slightly angled, likely a lateral or safe pass"
        elif feature_value < 14.5973:
            return "the pass had a moderate forward angle, possibly aimed at progressing play"
        elif feature_value < 22.4270:
            return "the pass was quite forward-oriented, likely breaking lines or pushing into advanced areas"
        elif feature_value < 34:
            return "the pass had a strong attacking intent, directed sharply toward the opponent's goal"
        else:
            return "the pass angle was extremely forward, potentially a long ball or through ball"

    if feature_name == "teammates_beyond":
        if feature_value <= 0:
            return "there were no teammates ahead of passer"
        elif feature_value <= 5:
            return "there were few teammates ahead of passer"
        else:
            return "there were many teammates ahead of passer"

    if feature_name == 'teammates_behind':
        if feature_value == 0:
            return "there were no teammates behind the passer"
        elif feature_value < 5:
            return "there were few teammates behind the passer"
        else:
            return "there were many teammates behind the passer"
       
    if feature_name == 'opponents_behind':
        if feature_value == 0:
            return "there were no opponents behind the passer"
        elif feature_value < 5:
            return "there were few opponents behind the passer"
        else:
            return "there were many opponents behind the passer"


    if feature_name == "opponents_beyond":
        if feature_value == 0:
            return "there were no opponents ahead of the passer"
        if feature_value < 6:
            return "few opponents were ahead of the passer"
        elif feature_value < 10:
            return "a moderate number of opponents were ahead of the passer"
        else:
            return "many opponents were ahead of the passer"

    if feature_name == "pressure_on_passer":
        if feature_value < 0.3617676262544192:
            return "the pressure on passer has low value for that range."
        elif feature_value < 0.6900539491099027:
            return "the pressure on passer has moderate value for that range."
        else:
            return "the pressure on passer has high value for that range."

    if feature_name == "opponents_nearby":
        if feature_value == 0:
            return "there were no opponents nearby passer at the moment of the pass"
        if feature_value < 2:
            return "there were few opponents nearby passer at the moment of the pass"
        else:
            return "there were more number of opponents nearby passer at the moment of the pass"

    if feature_name == "opponents_between":
        if feature_value <= 0:
            return "there was no opponents in the passing lane"
        elif feature_value < 3:
            return "the passing lane was mostly open with few opponents"
        else:
            return "the passing lane was crowded with opponents"


    
    if feature_name == "teammates_nearby":
        if feature_value <= 0:
            return "there was no teammates nearby the passer at the moment of the pass"
        if feature_value < 3:
            return "there were few teammates nearby the passer at the moment of the pass"
        else:
            return "there were more number of teammates nearby the passer near at the moment of the pass"
        
    if feature_name == "packing":
        if feature_value <= 0:
            return "the packing value is 0 with no opponents bypassed"
        elif feature_value < 3:
            return "there was moderate packing value with few opponents bypassed"
        else:
            return "there was high packing value, bypassing many opponents"

    if feature_name == "average_speed_of_teammates":
        if feature_value < 1.6639090909090906:
            return "teammates were moving slowly during the pass"
        elif feature_value < 2.8222272727272735:
            return "teammates were moving at a moderate speed"
        else:
            return "teammates were moving quickly, possibly making runs"

    if feature_name == "average_speed_of_opponents":
        if feature_value < 1.726618181818181:
            return "opponents were moving slowly during the pass"
        elif feature_value < 3.0413181818181814:
            return "opponents were moving at a moderate speed"
        else:
            return "opponents were moving quickly, possibly pressing"
    
    if feature_name == "speed_difference":
        if feature_value <= -0.122:
            return "the attacking team were significantly slower than the defending team"
        else:
            return "the attacking team were faster than the defending team"
    
    return f"No description available for {feature_name}."


#logistic model 
feature_name_mapping_logistic = { "start_distance_to_goal_contribution" : "start distance to goal",
    "end_distance_to_goal_contribution": "end distance to goal",
    "pass_length_contribution": "pass length",
    "pass_angle_contribution": "pass angle",
    "start_angle_to_goal_contribution" : "start angle to the goal",
    "end_angle_to_goal_contribution" : "end angle to goal",
    "start_distance_to_sideline_contribution" : "start distance to sideline",
    "end_distance_to_sideline_contribution" : "end distance to sideline", 
    "pressure_on_passer_contribution": "pressure on passer",
    "teammates_beyond_contribution": "teammates beyond",
    "opponents_beyond_contribution": "opponents beyond",
    "opponents_between_contribution": "Opponents between",
    "packing_contribution" : "opponents bypassed",
    "opponents_nearby_contribution": "opponents nearby",
    "teammates_nearby_contribution": "teammates nearby"
}

def describe_pass_contributions_logistic(contributions, pass_features, feature_name_mapping=feature_name_mapping_logistic):
    text = "The contributions of the features to the xT, sorted by their magnitude from largest to smallest, are as follows:\n"
    
    # Extract the contributions from the pass_contributions DataFrame
    contributions = contributions.iloc[0].drop(['match_id', 'id', 'xT'])  # Drop irrelevant columns
    
    # Sort the contributions by their absolute value (magnitude) in descending order
    sorted_contributions = contributions.abs().sort_values(ascending=False)
    
    # Get the top 4 contributions
    top_contributions = sorted_contributions
    
    # Loop through the top contributions to generate descriptions
    for idx, (feature, contribution) in enumerate(top_contributions.items()):

        # Get the original sign of the contribution
        original_contribution = contributions[feature]

        if original_contribution >= 0.08735242468992192 or original_contribution <= -0.08735242468992192:
        
            # Remove "_contribution" suffix to match feature names in pass_features
            feature_name = feature.replace('_contribution', '')
            
            # Use feature_name_mapping to get the display name for the feature (if available)
            feature_display_name = feature_name_mapping.get(feature, feature)
            
            # Get the feature value from pass_features
            feature_value = pass_features[feature_name]
            
            # Get the feature description
            feature_value_description = describe_pass_single_feature(feature_name, feature_value)
            
            # Add the feature's contribution to the xT description
            if original_contribution > 0:
                impact = 'maximum positive contribution'
                impact_text = "increased the xT."
            elif original_contribution < 0:
                impact = 'maximum negative contribution'
                impact_text = "reduced the xT."
            else:
                impact = 'no contribution'
                impact_text = "had no impact on the xT."

            # Use appropriate phrasing for the first feature and subsequent features
            if idx == 0:
                text += f"\nThe most impactful feature is {feature_display_name}, which had the {impact} because {feature_value_description}. This feature {impact_text}"
            else:
                text += f"\nAnother impactful feature is {feature_display_name}, which had the {impact} because {feature_value_description}. This feature {impact_text}"
        

    return text


#xgboost,xNN,CNN,trees models
feature_name_mapping_pass = { "start_distance_to_goal" : "start distance to goal",
    "end_distance_to_goal": "end distance to goal",
    "pass_length": "pass length",
    "pass_angle": "pass angle",
    "start_angle_to_goal" : "start angle to the goal",
    "end_angle_to_goal" : "end angle to goal",
    "start_distance_to_sideline" : "start distance to sideline",
    "end_distance_to_sideline" : "end distance to sideline", 
    "opponents_behind": "opponents behind",
    "teammates_behind": "teammates behind",
    "pressure_on_passer": "pressure on passer",
    "teammates_beyond": "teammates beyond",
    "opponents_beyond": "opponents beyond",
    "opponents_between": "Opponents between",
    "packing" : "opponents bypassed",
    "opponents_nearby": "opponents nearby",
    "average_speed_of_teammates": "average speed of teammates",
    "average_speed_of_opponents": "average speed of opponents",
    "teammates_nearby": "teammates nearby"
}

# Contribution function for xgboost
def describe_pass_contributions_xgboost(feature_contrib_df, pass_features, feature_name_mapping=feature_name_mapping_pass):
    text = "The contributions of the features to the xT, sorted by their magnitude from largest to smallest, are as follows:\n"
    
    # Extract the contributions from the pass_contributions DataFrame
    contributions = feature_contrib_df.iloc[0].drop(['match_id', 'id', 'xT_predicted'])  # Drop irrelevant columns
    
    # Sort the contributions by their absolute value (magnitude) in descending order
    sorted_contributions = contributions.abs().sort_values(ascending=False)
    
    # Get the top 4 contributions
    top_contributions = sorted_contributions
    
    # Loop through the top contributions to generate descriptions
    for idx, (feature,contribution) in enumerate(top_contributions.items()):
        # Get the original sign of the contribution
        original_contribution = contributions[feature]

        if original_contribution >= 0.079879159 or original_contribution <= -0.079879159:
            
            # Use feature_name_mapping to get the display name for the feature (if available)
            feature_display_name = feature_name_mapping.get(feature, feature)
            
            # Get the feature value from shot_features
            feature_value = pass_features[feature]
            
            # Get the feature description
            feature_value_description = describe_pass_single_feature(feature, feature_value)
            
            # Add the feature's contribution to the xT description
            if original_contribution > 0:
                impact = 'maximum positive contribution'
                impact_text = "increased the xT."
            elif original_contribution < 0:
                impact = 'maximum negative contribution'
                impact_text = "reduced the xT."
            else:
                impact = 'no contribution'
                impact_text = "had no impact on the xT."
                print(original_contribution)

            # Use appropriate phrasing for the first feature and subsequent features
            if idx == 0:
                text += f"\nThe most impactful feature is {feature_display_name}, which had the {impact} because {feature_value_description}. This feature {impact_text}"
            else:
                text += f"\nAnother impactful feature is {feature_display_name}, which had the {impact} because {feature_value_description}. This feature {impact_text}"
        

    return text

#contribution feature of xNN model
def describe_pass_contributions_xNN(contributions_xNN, pass_features, feature_name_mapping=feature_name_mapping_pass):
    text = "The contributions of the features to the xT, sorted by their magnitude from largest to smallest, are as follows:\n"
    
    # Extract the contributions from the pass_contributions
    contributions = contributions_xNN.iloc[0].drop(['id', 'xT_predicted'])  # Drop irrelevant columns
    
    # Sort the contributions by their absolute value (magnitude) in descending order
    sorted_contributions = contributions.abs().sort_values(ascending=False)
    
    # Get the top 4 contributions
    top_contributions = sorted_contributions
    
    # Loop through the top contributions to generate descriptions
    for idx, (feature, contribution) in enumerate(top_contributions.items()):

        # Get the original sign of the contribution
        original_contribution = contributions[feature]

        if original_contribution >= 0.00788100733068301 or original_contribution <= -0.00788100733068301:
            
            # Use feature_name_mapping to get the display name for the feature (if available)
            feature_display_name = feature_name_mapping.get(feature, feature)
            
            # Get the feature value from shot_features
            feature_value = pass_features[feature]
            
            # Get the feature description
            feature_value_description = describe_pass_single_feature(feature, feature_value)
            
            # Add the feature's contribution to the xT description
            if original_contribution > 0:
                impact = 'maximum positive contribution'
                impact_text = "increased the xT."
            elif original_contribution < 0:
                impact = 'maximum negative contribution'
                impact_text = "reduced the xT."
            else:
                impact = 'no contribution'
                impact_text = "had no impact on the xT."

            # Use appropriate phrasing for the first feature and subsequent features
            if idx == 0:
                text += f"\nThe most impactful feature is {feature_display_name}, which had the {impact} because {feature_value_description}. This feature {impact_text}"
            else:
                text += f"\nAnother impactful feature is {feature_display_name}, which had the {impact} because {feature_value_description}. This feature {impact_text}"
        

    return text
def describe_pass_contributions_mimic(contributions_mimic_df, pass_features, feature_name_mapping=feature_name_mapping_pass):
    text = "The contributions of the features to the xT (MIMiC model), sorted by their magnitude from largest to smallest, are as follows:\n"

    # Drop irrelevant columns
    contributions = contributions_mimic_df.iloc[0].drop(['match_id', 'id', 'mimic_xT','leaf_id','leaf_intercept'])

    # Sort by absolute value
    sorted_contributions = contributions.abs().sort_values(ascending=False)

    for idx, (feature, contribution) in enumerate(sorted_contributions.items()):
        original_contribution = contributions[feature]

        if abs(original_contribution) >= 0.01:  # Customize threshold as needed
            feature_display_name = feature_name_mapping.get(feature, feature)
            feature_value = pass_features.get(feature, None)
            feature_value_description = describe_pass_single_feature(feature, feature_value)

            if original_contribution > 0:
                impact = 'positive contribution'
                impact_text = "increased the xT."
            else:
                impact = 'negative contribution'
                impact_text = "reduced the xT."

            if idx == 0:
                text += f"\nThe most impactful feature is {feature_display_name}, which had a {impact} because {feature_value_description}. This feature {impact_text}"
            else:
                text += f"\nAnother key feature is {feature_display_name}, which had a {impact} because {feature_value_description}. This feature {impact_text}"

    return text
