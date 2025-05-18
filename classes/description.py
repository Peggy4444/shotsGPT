import math
from abc import ABC, abstractmethod
from typing import List, Union, Dict, Optional

import pandas as pd
import tiktoken
import openai
import numpy as np

import utils.sentences as sentences
from utils.gemini import convert_messages_format
from classes.data_point import Player, Country


from settings import USE_GEMINI

if USE_GEMINI:
    from settings import USE_GEMINI, GEMINI_API_KEY, GEMINI_CHAT_MODEL
else:
    from settings import GPT_BASE, GPT_VERSION, GPT_KEY, GPT_ENGINE

import streamlit as st

openai.api_type = "azure"


class Description(ABC):
    gpt_examples_base = "data/gpt_examples"
    describe_base = "data/describe"

    @property
    @abstractmethod
    def gpt_examples_path(self) -> str:
        """
        Path to excel files containing examples of user and assistant messages for the GPT to learn from.
        """

    @property
    @abstractmethod
    def describe_paths(self) -> Union[str, List[str]]:
        """
        List of paths to excel files containing questions and answers for the GPT to learn from.
        """

    def __init__(self):
        self.synthesized_text = self.synthesize_text()
        self.messages = self.setup_messages()

    def synthesize_text(self) -> str:
        """
        Return a data description that will be used to prompt GPT.

        Returns:
        str
        """

    def get_prompt_messages(self) -> List[Dict[str, str]]:
        """
        Return the prompt that the GPT will see before self.synthesized_text.

        Returns:
        List of dicts with keys "role" and "content".
        """

    def get_intro_messages(self) -> List[Dict[str, str]]:
        """
        Constant introduction messages for the assistant.

        Returns:
        List of dicts with keys "role" and "content".
        """
        intro = [
            {
                "role": "system",
                "content": (
                    "You are a football commentator whose job is to write interesting texts about actions. "
                    "You provide succinct and to the point explanations about data using data. "
                    "You use the information given to you from the data and answers "
                    "to earlier user/assistant pairs to give summaries of players."
                ),
            },
        ]
        if len(self.describe_paths) > 0:
            intro += [
                {
                    "role": "user",
                    "content": "First, could you answer some questions about the data for me?",
                },
                {"role": "assistant", "content": "Sure!"},
            ]

        return intro

    def get_messages_from_excel(
        self,
        paths: Union[str, List[str]],
    ) -> List[Dict[str, str]]:
        """
        Turn an excel file containing user and assistant columns with str values into a list of dicts.

        Arguments:
        paths: str or list of str
            Path to the excel file containing the user and assistant columns.

        Returns:
        List of dicts with keys "role" and "content".

        """

        # Handle list and str paths arg
        if isinstance(paths, str):
            paths = [paths]
        elif len(paths) == 0:
            return []

        # Concatenate dfs read from paths
        df = pd.read_excel(paths[0])
        for path in paths[1:]:
            df = pd.concat([df, pd.read_excel(path)])

        if df.empty:
            return []

        # Convert to list of dicts
        messages = []
        for i, row in df.iterrows():
            if i == 0:
                messages.append({"role": "user", "content": row["user"]})
            else:
                messages.append({"role": "user", "content": row["user"]})
            messages.append({"role": "assistant", "content": row["assistant"]})

        return messages

    def setup_messages(self) -> List[Dict[str, str]]:
        messages = self.get_intro_messages()
        
        try:
            paths = self.describe_paths
            messages += self.get_messages_from_excel(paths)
        except (
            FileNotFoundError
        ) as e:  # FIXME: When merging with new_training, add the other exception
            print(e)
        
        # Ensure messages are in the correct format after getting from excel
        messages = [msg for msg in messages if isinstance(msg, dict) and "content" in msg and isinstance(msg["content"], str)]
        
        messages += self.get_prompt_messages()  # Adding prompt messages
        
        try:
            messages += self.get_messages_from_excel(paths=self.gpt_examples_path)
        except FileNotFoundError as e:  # FIXME: When merging with new_training, add the other exception
            print(e)

        # Ensure that synthesized_text is defined and has a string value
        synthesized_text = getattr(self, 'synthesized_text', '')
        if isinstance(synthesized_text, str):
            messages.append({"role": "user", "content": f"Now do the same thing with the following: ```{synthesized_text}```"})

        # Filter again to ensure no non-string content is present
        messages = [msg for msg in messages if isinstance(msg, dict) and "content" in msg and isinstance(msg["content"], str)]
        
        messages += self.get_prompt_messages()

        messages = [
            message for message in messages if isinstance(message["content"], str)
        ]

        try:
            messages += self.get_messages_from_excel(
                paths=self.gpt_examples_path,
            )
        except (
            FileNotFoundError
        ) as e:  # FIXME: When merging with new_training, add the other exception
            print(e)

        messages += [
            {
                "role": "user",
                "content": f"Now do the same thing with the following: ```{self.synthesized_text}```",
            }
        ]
        return messages


    def stream_gpt(self, temperature=1):
        """
        Run the GPT model on the messages and stream the output.

        Arguments:
        temperature: optional float
            The temperature of the GPT model.

        Yields:
            str
        """


        st.expander("Description messages", expanded=False).write(self.messages)
        st.expander("Chat transcript", expanded=False).write(self.messages)

        if USE_GEMINI:
            import google.generativeai as genai

            converted_msgs = convert_messages_format(self.messages)

            # # save converted messages to json
            # import json
            # with open("data/wvs/msgs_0.json", "w") as f:
            #     json.dump(converted_msgs, f)

            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(
                model_name=GEMINI_CHAT_MODEL,
                system_instruction=converted_msgs["system_instruction"],
            )
            chat = model.start_chat(history=converted_msgs["history"])
            response = chat.send_message(content=converted_msgs["content"])

            answer = response.text
        else:
            # Use OpenAI API
            openai.api_base = GPT_BASE
            openai.api_version = GPT_VERSION
            openai.api_key = GPT_KEY

            response = openai.ChatCompletion.create(
                engine=GPT_ENGINE,
                messages=self.messages,
                temperature=temperature,
            )

            answer = response["choices"][0]["message"]["content"]

        return answer


class PlayerDescription(Description):
    output_token_limit = 150

    @property
    def gpt_examples_path(self):
        return f"{self.gpt_examples_base}/Forward.xlsx"

    @property
    def describe_paths(self):
        return [f"{self.describe_base}/Forward.xlsx"]

    def __init__(self, player: Player):
        self.player = player
        super().__init__()

    def get_intro_messages(self) -> List[Dict[str, str]]:
        """
        Constant introduction messages for the assistant.

        Returns:
        List of dicts with keys "role" and "content".
        """
        intro = [
            {
                "role": "system",
                "content": (
                    "You are a UK-based football scout. "
                    "You provide succinct and to the point explanations about football players using data. "
                    "You use the information given to you from the data and answers "
                    "to earlier user/assistant pairs to give summaries of players."
                ),
            },
            {
                "role": "user",
                "content": "Do you refer to the game you are an expert in as soccer or football?",
            },
            {
                "role": "assistant",
                "content": (
                    "I refer to the game as football. "
                    "When I say football, I don't mean American football, I mean what Americans call soccer. "
                    "But I always talk about football, as people do in the United Kingdom."
                ),
            },
        ]
        if len(self.describe_paths) > 0:
            intro += [
                {
                    "role": "user",
                    "content": "First, could you answer some questions about football for me?",
                },
                {"role": "assistant", "content": "Sure!"},
            ]

        return intro

    def synthesize_text(self):

        player = self.player
        metrics = self.player.relevant_metrics
        description = f"Here is a statistical description of {player.name}, who played for {player.minutes_played} minutes as a {player.position}. \n\n "

        subject_p, object_p, possessive_p = sentences.pronouns(player.gender)

        for metric in metrics:

            description += f"{subject_p.capitalize()} was "
            description += sentences.describe_level(player.ser_metrics[metric + "_Z"])
            description += " in " + sentences.write_out_metric(metric)
            description += " compared to other players in the same playing position. "

        # st.write(description)

        return description

    def get_prompt_messages(self):
        prompt = (
            f"Please use the statistical description enclosed with ``` to give a concise, 4 sentence summary of the player's playing style, strengths and weaknesses. "
            f"The first sentence should use varied language to give an overview of the player. "
            "The second sentence should describe the player's specific strengths based on the metrics. "
            "The third sentence should describe aspects in which the player is average and/or weak based on the statistics. "
            "Finally, summarise exactly how the player compares to others in the same position. "
        )
        return [{"role": "user", "content": prompt}]





class ShotDescription(Description):

    output_token_limit = 500

    @property
    def gpt_examples_path(self):
        return f"{self.gpt_examples_base}/action/shots.xlsx"
        #return []

    @property
    def describe_paths(self):
        return [f"{self.describe_base}/action/shots.xlsx"]
        #return []
    
    def __init__(self, shots, shot_id, competition):
        self.shots = shots
        self.shot_id = shot_id
        self.competition = competition
        super().__init__()

    def synthesize_text(self):

        shots = self.shots
        shot_data = shots.df_shots[shots.df_shots['id'] == self.shot_id]  # Fix here to use self.shot_id

        if shot_data.empty:
            raise ValueError(f"No shot found with ID {self.shot_id}")
        
        player_name = shot_data['player_name'].iloc[0]
        team_name = shot_data['team_name'].iloc[0]

        start_x = shot_data['start_x'].iloc[0]
        start_y = shot_data['start_y'].iloc[0]
        xG = shot_data['xG'].iloc[0]
        goal_status = shot_data['goal'].fillna(False).iloc[0]
        
        # Map goal boolean to readable category
        labels = {False: "didn't result in a goal.", True: 'was a goal!'}
        goal_status_text = labels[goal_status]
        #angle_to_goal = shot_data['angle_to_goal'].iloc[0]
        distance_to_goal = shot_data['euclidean_distance_to_goal'].iloc[0]
        distance_to_nearest_opponent = shot_data['distance_to_nearest_opponent'].iloc[0]
        gk_dist_to_goal = shot_data['goalkeeper_distance_to_goal'].iloc[0]
        minute= shot_data['minute'].iloc[0]

        # Give a detailed description of the contributions to the shot
        shot_contributions = self.shots.df_contributions[self.shots.df_contributions['id'] == self.shot_id]

        shot_features = {
            'vertical_distance_to_center': shot_data['vertical_distance_to_center'].iloc[0],
            'euclidean_distance_to_goal': distance_to_goal,
            'nearby_opponents_in_3_meters': shot_data['nearby_opponents_in_3_meters'].iloc[0],
            'opponents_in_triangle': shot_data['opponents_in_triangle'].iloc[0],
            'goalkeeper_distance_to_goal': gk_dist_to_goal,
            #'header': shot_data['header'].iloc[0],
            'distance_to_nearest_opponent': distance_to_nearest_opponent,
            'angle_to_goalkeeper': shot_data['angle_to_goalkeeper'].iloc[0],
            'shot_with_left_foot': shot_data['shot_with_left_foot'].iloc[0],
            'shot_after_throw_in': shot_data['shot_after_throw_in'].iloc[0],
            'shot_after_corner': shot_data['shot_after_corner'].iloc[0],
            'shot_after_free_kick': shot_data['shot_after_free_kick'].iloc[0],
            'shot_during_regular_play': shot_data['shot_during_regular_play'].iloc[0],
            'pattern': shot_data['play_pattern_name'].iloc[0],
        }

        feature_descriptions = sentences.describe_shot_features(shot_features, self.competition)


        shot_description = (
            f"{player_name}'s shot from {team_name} {goal_status_text} "
            f"This shot had an xG value of {xG:.2f}, which means that we estimate the chance of scoring from this situation as {xG * 100:.0f}%. "
            f"{sentences.describe_xg(xG)} "
            #f"The distance to goal was {distance_to_goal:.1f} meters and the distance to the nearest opponent was {distance_to_nearest_opponent:.1f} meters."
        )
        shot_description += '\n'.join(feature_descriptions) + '\n'  # Add the detailed descriptions of the shot features

        shot_description += '\n' + sentences.describe_shot_contributions(shot_contributions, shot_features)

        with st.expander("Synthesized Text"):
            st.write(shot_description)
        
        return shot_description 
    

    def get_prompt_messages(self):
        prompt = (
            "You are a football commentator. You should write in an exciting and engaging way about a shot"
            f"You should giva a four sentence summary of the shot taken by the player. "
            "The first sentence should say whether it was a good chance or not, state the expected goals value and also state if it was a goal. "
            "The second and third sentences should describe the most important factors that contributed to the quality of the chance. "
            "If it was a good chance these two sentences chould explain what contributing factors made the shot dangerous. "
            "If it wasn't particularly good chance then these two sentences chould explain why it wasn't a good chance. "
            "Depedning on the quality of the chance, the final sentence should either praise the player or offer advice about what to think about when shooting."
            )
        return [{"role": "user", "content": prompt}]


#pass description for logistic model
class PassDescription_logistic(Description):

        output_token_limit = 500

        @property
        def gpt_examples_path(self):
            return f"{self.gpt_examples_base}/action/passes.xlsx"
            #return []

        @property
        def describe_paths(self):
            return [f"{self.describe_base}/action/passes.xlsx"]
            #return []
        
        def __init__(self,pass_data,df_contributions, pass_id, competition):
            self.pass_data = pass_data
            self.df_contributions = df_contributions
            self.pass_id = pass_id
            self.competition = competition
            super().__init__()

        def synthesize_text(self):

            pass_data = self.pass_data
            #df_contributions = self.df_contributions
            
            #for a specific pass id
            passes = pass_data.df_pass[pass_data.df_pass['id'] == self.pass_id] 
            contributions = pass_data.df_contributions[pass_data.df_contributions['id'] == self.pass_id]
            tracking = pass_data.df_tracking[pass_data.df_tracking['id'] == self.pass_id]


            if passes.empty:
                raise ValueError(f"No shot found with ID {self.shot_id}")
            
            player_name = passes['passer_name'].iloc[0]
            team_name = passes['team_name'].iloc[0]
            xT = contributions['xT'].iloc[0]
            x = passes['passer_x'].iloc[0]
            y = passes['passer_y'].iloc[0]
            team_direction = tracking['team_direction'].iloc[0]
            xG = passes['possession_xg'].iloc[0]

            #extracting the pass classification values
            forward_pass = passes['forward pass'].iloc[0]
            back_pass = passes['backward pass'].iloc[0]
            lateral_pass = passes['lateral pass'].iloc[0]

            if forward_pass:
                pass_type = " forward pass"
            elif back_pass:
                pass_type = " back pass"
            elif lateral_pass:
                pass_type = " lateral pass"
            else:
                pass_type = "an unspecified pass"
            
            xG = passes['possession_xg'].iloc[0]
            
            pass_features = {'pass_length' : passes['pass_length'].iloc[0]  ,
                            'start_angle_to_goal' : passes['start_angle_to_goal'].iloc[0],
                            'start_distance_to_goal' :passes['start_distance_to_goal'].iloc[0] ,
                            'opponents_between' : passes['opponents_between'].iloc[0], 
                            'packing' : passes['packing'].iloc[0], 
                            'pressure_level_passer' : passes['pressure level passer'].iloc[0],
                            'opponents_nearby' : passes['opponents_nearby'].iloc[0],
                            'possession_xg' : passes['possession_xg'].iloc[0],
                            'teammates_beyond' : passes['teammates_beyond'].iloc[0],
                            'opponents_beyond' : passes['opponents_beyond'].iloc[0],
                            'start_distance_to_sideline' : passes['start_distance_to_sideline'].iloc[0],
                            'end_distance_to_sideline' : passes['end_distance_to_sideline'].iloc[0],
                            'speed_difference' : passes['speed_difference'].iloc[0],
                            'pass_angle' : passes['pass_angle'].iloc[0],
                            'teammates_nearby' : passes['teammates_nearby'].iloc[0],
                            'end_distance_to_goal' : passes['end_distance_to_goal'].iloc[0],
                            'end_angle_to_goal' : passes['end_angle_to_goal'].iloc[0],
                            'pressure_on_passer' : passes['pressure_on_passer'].iloc[0]
                            }

            feature_descriptions = sentences.describe_pass_features_logistic(pass_features, self.competition)
            pass_description = (
                f"The pass is a {pass_type} originated from {sentences.describe_position_pass(x,y,team_direction)} \n and the passer is {player_name} from {team_name} team."
                f"{sentences.describe_xT_pass_logistic(xT,xG)}"
            )
            pass_description += '\n'.join(feature_descriptions) + '\n'  # Add the detailed descriptions of the shot features

            pass_description += '\n' + sentences.describe_pass_contributions_logistic(contributions, pass_features)

            with st.expander("Synthesized Text"):
                st.write(pass_description)
            
            return pass_description 
        

        def get_prompt_messages(self):
            prompt = (
                "You are a football commentator. You should write in an exciting and engaging way about the features contributing for pass to be a shot and is it a safe or dangerous pass."
                f"You should giva a four sentence summary of the pass taken by the player. "
                "The first sentence should say whether it was a good chance or not, state the expected threat value and also state the probability of expected goal. "
                "The second and third sentences should describe the most important factors that contributed to the pass to be safe or dangerous. "
                "If it was a good chance these two sentences chould explain what contributing factors made the pass dangerous. "
                "If it wasn't particularly good chance then these two sentences chould explain why it was a safe pass. "
                "Depedning on the quality of the chance, the final sentence should either praise the player or offer advice about what to think about when initiating the pass."
                )
            return [{"role": "user", "content": prompt}]   


 #class description of features for xNN
class PassDescription_xNN(Description):

        output_token_limit = 500

        @property
        def gpt_examples_path(self):
            return f"{self.gpt_examples_base}/action/passes.xlsx"
            #return []

        @property
        def describe_paths(self):
            return [f"{self.describe_base}/action/passes.xlsx"]
            #return []
        
        def __init__(self,pass_data,feature_contrib_df,model_contribution_xNN,pass_id,competition):
            self.pass_data = pass_data
            self.model_contribution_xNN = model_contribution_xNN
            self.feature_contrib_df = feature_contrib_df
            self.pass_id = pass_id
            self.competition = competition
            super().__init__()

        def synthesize_text(self):

            pass_data = self.pass_data
            
            passes = pass_data.pass_df_xNN[pass_data.pass_df_xNN['id'] == self.pass_id]  # Fix here to use self.shot_id
            contributions = pass_data.contributions_xNN[pass_data.contributions_xNN['id'] == self.pass_id]
            tracking = pass_data.df_tracking[pass_data.df_tracking['id'] == self.pass_id]
            models_contribution = pass_data.model_contribution_xNN[pass_data.model_contribution_xNN['id'] == self.pass_id]

            if passes.empty:
                raise ValueError(f"No shot found with ID {self.shot_id}")
            
            player_name = passes['passer_name'].iloc[0]
            team_name = passes['team_name'].iloc[0]
            xT = contributions['xT_predicted'].iloc[0]
            x = passes['passer_x'].iloc[0]
            y = passes['passer_y'].iloc[0]
            team_direction = tracking['team_direction'].iloc[0]
            pressure = models_contribution['pressure based_contrib'].iloc[0]
            speed = models_contribution['speed based_contrib'].iloc[0]
            position = models_contribution['position based_contrib'].iloc[0]
            event = models_contribution['event based_contrib'].iloc[0]

            #extracting the pass classification values
            forward_pass = passes['forward pass'].iloc[0]
            back_pass = passes['backward pass'].iloc[0]
            lateral_pass = passes['lateral pass'].iloc[0]

            if forward_pass:
                pass_type = " forward pass"
            elif back_pass:
                pass_type = " back pass"
            elif lateral_pass:
                pass_type = " lateral pass"
            else:
                pass_type = "an unspecified pass"
            
            xG = passes['possession_xg'].iloc[0]
            
            pass_features = {'pass_length' : passes['pass_length'].iloc[0]  ,
                            'start_angle_to_goal' : passes['start_angle_to_goal'].iloc[0],
                            'start_distance_to_goal' :passes['start_distance_to_goal'].iloc[0] ,
                            'opponents_between' : passes['opponents_between'].iloc[0], 
                            'packing' : passes['packing'].iloc[0], 
                            'average_speed_of_teammates' : passes['average_speed_of_teammates'].iloc[0], 
                            'average_speed_of_opponents' : passes['average_speed_of_opponents'].iloc[0] ,
                            'pressure_level_passer' : passes['pressure level passer'].iloc[0],
                            'opponents_nearby' : passes['opponents_nearby'].iloc[0],
                            'possession_xg' : passes['possession_xg'].iloc[0],
                            'teammates_beyond' : passes['teammates_beyond'].iloc[0],
                            'teammates_behind' : passes['teammates_behind'].iloc[0],
                            'opponents_beyond' : passes['opponents_beyond'].iloc[0],
                            'opponents_behind' : passes['opponents_behind'].iloc[0],
                            'pressure_on_passer' : passes['pressure_on_passer'].iloc[0],
                            'pass_angle' : passes['pass_angle'].iloc[0],
                            'end_angle_to_goal' : passes['end_angle_to_goal'].iloc[0],
                            'start_distance_to_sideline' : passes['start_distance_to_sideline'].iloc[0],
                            'end_distance_to_sideline' : passes['end_distance_to_sideline'].iloc[0],
                            'end_distance_to_goal' : passes['end_distance_to_goal'].iloc[0],
                            'teammates_nearby' : passes['teammates_nearby'].iloc[0]
                            }

            feature_descriptions = sentences.describe_pass_features(pass_features, self.competition)
            
            pass_description = (
                f"{sentences.describe_models_xNN(pressure,speed,position,event)} The pass is a {pass_type} originated from {sentences.describe_position_pass(x,y,team_direction)} \n and the passer is {player_name} from {team_name} team."
                f"{sentences.describe_xT_pass_xNN(xT,xG)}"
            )
            pass_description += '\n'.join(feature_descriptions) + '\n'  # Add the detailed descriptions of the shot features
            
            pass_description += '\n' + sentences.describe_pass_contributions_xNN(contributions, pass_features)

            with st.expander("Synthesized Text"):
                st.write(pass_description)
            
            return pass_description 

        def get_prompt_messages(self):
            prompt = (
                "You are a football commentator. You should write in an exciting and engaging way about a shot"
                f"You should giva a four sentence summary of the shot taken by the player. "
                "The first sentence should say whether it was a good chance or not, state the expected goals value and also state if it was a goal. "
                "The second and third sentences should describe the most important factors that contributed to the quality of the chance. "
                "If it was a good chance these two sentences chould explain what contributing factors made the shot dangerous. "
                "If it wasn't particularly good chance then these two sentences chould explain why it wasn't a good chance. "
                "Depedning on the quality of the chance, the final sentence should either praise the player or offer advice about what to think about when shooting."
                )
            return [{"role": "user", "content": prompt}]

### pass descriptions for xGBoost
class PassDescription_xgboost(Description):

        output_token_limit = 500

        @property
        def gpt_examples_path(self):
        #     return f"{self.gpt_examples_base}/action/passes.xlsx"
            return []

        @property
        def describe_paths(self):
        #     return [f"{self.describe_base}/action/passes.xlsx"]
            return []
        
        def __init__(self,pass_data,feature_contrib_df, pass_id, competition):
            self.pass_data = pass_data
            self.feature_contrib_df = feature_contrib_df
            self.pass_id = pass_id
            self.competition = competition
            super().__init__()

        def synthesize_text(self):

            pass_data = self.pass_data
            
            passes = pass_data.pass_df_xgboost[pass_data.pass_df_xgboost['id'] == self.pass_id]  # Fix here to use self.shot_id
            contributions = pass_data.feature_contrib_df[pass_data.feature_contrib_df['id'] == self.pass_id]
            tracking = pass_data.df_tracking[pass_data.df_tracking['id'] == self.pass_id]

            if passes.empty:
                raise ValueError(f"No shot found with ID {self.shot_id}")
            
            player_name = passes['passer_name'].iloc[0]
            team_name = passes['team_name'].iloc[0]
            xT = contributions['xT_predicted'].iloc[0]
            x = passes['passer_x'].iloc[0]
            y = passes['passer_y'].iloc[0]
            team_direction = tracking['team_direction'].iloc[0]
            

            #extracting the pass classification values
            forward_pass = passes['forward pass'].iloc[0]
            back_pass = passes['backward pass'].iloc[0]
            lateral_pass = passes['lateral pass'].iloc[0]

            if forward_pass:
                pass_type = " forward pass"
            elif back_pass:
                pass_type = " back pass"
            elif lateral_pass:
                pass_type = " lateral pass"
            else:
                pass_type = "an unspecified pass"
            
            xG = passes['possession_xg'].iloc[0]
            
            pass_features = {'pass_length' : passes['pass_length'].iloc[0]  ,
                            'start_angle_to_goal' : passes['start_angle_to_goal'].iloc[0],
                            'start_distance_to_goal' :passes['start_distance_to_goal'].iloc[0] ,
                            'opponents_beyond':passes['opponents_beyond'].iloc[0],
                            'opponents_between' : passes['opponents_between'].iloc[0], 
                            'packing' : passes['packing'].iloc[0], 
                            'average_speed_of_teammates' : passes['average_speed_of_teammates'].iloc[0], 
                            'average_speed_of_opponents' : passes['average_speed_of_opponents'].iloc[0] ,
                            'pressure_level_passer' : passes['pressure level passer'].iloc[0],
                            'opponents_nearby' : passes['opponents_nearby'].iloc[0],
                            'possession_xg' : passes['possession_xg'].iloc[0],
                            'teammates_beyond' : passes['teammates_beyond'].iloc[0],
                            'teammates_behind' : passes['teammates_behind'].iloc[0],
                            'opponents_beyond' : passes['opponents_beyond'].iloc[0],
                            'opponents_behind' : passes['opponents_behind'].iloc[0],
                            'pressure_on_passer' : passes['pressure_on_passer'].iloc[0],
                            'pass_angle' : passes['pass_angle'].iloc[0],
                            'end_angle_to_goal' : passes['end_angle_to_goal'].iloc[0],
                            'start_distance_to_sideline' : passes['start_distance_to_sideline'].iloc[0],
                            'end_distance_to_sideline' : passes['end_distance_to_sideline'].iloc[0],
                            'end_distance_to_goal' : passes['end_distance_to_goal'].iloc[0],
                            'teammates_nearby' : passes['teammates_nearby'].iloc[0]                       
                            }

            feature_descriptions = sentences.describe_pass_features(pass_features, self.competition)
            
            pass_description = (
                f"The pass is a {pass_type} originated from {sentences.describe_position_pass(x,y,team_direction)} \n and the passer is {player_name} from {team_name} team."
                f"{sentences.describe_xT_pass(xT,xG)}"
            )
            pass_description += '\n'.join(feature_descriptions) + '\n'  # Add the detailed descriptions of the shot features

            pass_description += '\n' + sentences.describe_pass_contributions_xgboost(contributions, pass_features)

            with st.expander("Synthesized Text"):
                st.write(pass_description)
            
            return pass_description 

        def get_prompt_messages(self):
            prompt = (
                "You are a football commentator. You should write in an exciting and engaging way about a shot"
                f"You should giva a four sentence summary of the shot taken by the player. "
                "The first sentence should say whether it was a good chance or not, state the expected goals value and also state if it was a goal. "
                "The second and third sentences should describe the most important factors that contributed to the quality of the chance. "
                "If it was a good chance these two sentences chould explain what contributing factors made the shot dangerous. "
                "If it wasn't particularly good chance then these two sentences chould explain why it wasn't a good chance. "
                "Depedning on the quality of the chance, the final sentence should either praise the player or offer advice about what to think about when shooting."
                )
            return [{"role": "user", "content": prompt}]

class CountryDescription(Description):
    output_token_limit = 150

    @property
    def gpt_examples_path(self):
        return f"{self.gpt_examples_base}/WVS_examples.xlsx"

    @property
    def describe_paths(self):
        return [f"{self.describe_base}/WVS_qualities.xlsx"]

    def __init__(self, country: Country, description_dict, thresholds_dict):
        self.country = country
        self.description_dict = description_dict
        self.thresholds_dict = thresholds_dict

        super().__init__()

    def get_intro_messages(self) -> List[Dict[str, str]]:
        """
        Constant introduction messages for the assistant.

        Returns:
        List of dicts with keys "role" and "content".
        """
        intro = [
            {
                "role": "system",
                "content": (
                    "You are a data analyst and a social scientist. "
                    "You provide succinct and to the point explanations about countries using metrics derived from data collected in the World Value Survey. "
                    "You use the information given to you from the data and answers to earlier questions to give summaries of how countries score in various metrics that attempt to measure the social values held by the population of that country."
                ),
            },
            # {
            #     "role": "user",
            #     "content": "Do you refer to the game you are an expert in as soccer or football?",
            # },
            # {
            #     "role": "assistant",
            #     "content": (
            #         "I refer to the game as football. "
            #         "When I say football, I don't mean American football, I mean what Americans call soccer. "
            #         "But I always talk about football, as people do in the United Kingdom."
            #     ),
            # },
        ]
        if len(self.describe_paths) > 0:
            intro += [
                {
                    "role": "user",
                    "content": "First, could you answer some questions about a the World Value Survey for me?",
                },
                {"role": "assistant", "content": "Sure!"},
            ]

        return intro

    def synthesize_text(self):

        country = self.country
        metrics = self.country.relevant_metrics
        description = f"Here is a statistical description of the core values of {country.name.capitalize()}. \n\n"

        # subject_p, object_p, possessive_p = sentences.pronouns(country.gender)

        for metric in metrics:

            # # TODO: customize this text?
            # description += f"{country.name.capitalize()} was found to be "
            # description += sentences.describe_level(
            #     country.ser_metrics[metric + "_Z"],
            #     thresholds=self.thresholds_dict[metric],
            #     words=self.description_dict[metric],
            # )
            # description += " in " + metric.lower()  # .replace("_", " ")
            # description += " compared to other countries in the same survey. "

            description += f"{country.name.capitalize()} was found to "
            description += sentences.describe_level(
                country.ser_metrics[metric + "_Z"],
                thresholds=self.thresholds_dict[metric],
                words=self.description_dict[metric],
            )
            description += " compared to other countries in the same survey. "

        # st.write(description)

        return description

    def get_prompt_messages(self):
        prompt = (
            f"Please use the statistical description enclosed with ``` to give a concise, 4 sentence summary of the social values held by population of the country. "
            # f"The first sentence should use varied language to give an overview of the player. "
            # "The second sentence should describe the player's specific strengths based on the metrics. "
            # "The third sentence should describe aspects in which the player is average and/or weak based on the statistics. "
            # "Finally, summarise exactly how the player compares to others in the same position. "
        )
        return [{"role": "user", "content": prompt}]
class PassDescription_mimic(Description):

    output_token_limit = 500

    @property
    def gpt_examples_path(self):
        return f"{self.gpt_examples_base}/action/shots.xlsx"
       # return []  # Provide path if examples are available

    @property
    def describe_paths(self):
        return [f"{self.describe_base}/action/shots.xlsx"]
       # return []  # Provide path if question-answer data exists

    def __init__(self, pass_data, df_contrib_mimic, pass_id, competition):
        self.pass_data = pass_data
        self.df_contributions = df_contrib_mimic
        self.pass_id = pass_id
        self.competition = competition
        super().__init__()

    def synthesize_text(self):
        passes = self.pass_data.df_pass[ self.pass_data.df_pass["id"] == self.pass_id ]
        #contributions = self.df_contributions_mimic[ self.df_contributions_mimic["id"] == self.pass_id ]
        contributions = self.df_contributions[ self.df_contributions["id"] == self.pass_id ]

        tracking = self.pass_data.df_tracking[ self.pass_data.df_tracking["id"] == self.pass_id ]

        if passes.empty:
            raise ValueError(f"No pass found with ID {self.pass_id}")

        player_name = passes['passer_name'].iloc[0]
        team_name = passes['team_name'].iloc[0]
        x = passes['passer_x'].iloc[0]
        y = passes['passer_y'].iloc[0]
        team_direction = tracking['team_direction'].iloc[0]
        xT = contributions['mimic_xT'].iloc[0]

        xG = passes['possession_xg'].iloc[0]

        forward_pass = passes['forward pass'].iloc[0]
        back_pass = passes['backward pass'].iloc[0]
        lateral_pass = passes['lateral pass'].iloc[0]

        if forward_pass:
            pass_type = "forward pass"
        elif back_pass:
            pass_type = "back pass"
        elif lateral_pass:
            pass_type = "lateral pass"
        else:
            pass_type = "an unspecified pass"
        
        
        pass_features = {'pass_length' : passes['pass_length'].iloc[0]  ,
                            'start_angle_to_goal' : passes['start_angle_to_goal'].iloc[0],
                            'start_distance_to_goal' :passes['start_distance_to_goal'].iloc[0] ,
                            'opponents_beyond':passes['opponents_beyond'].iloc[0],
                            'opponents_between' : passes['opponents_between'].iloc[0], 
                            'packing' : passes['packing'].iloc[0], 
                            'average_speed_of_teammates' : passes['average_speed_of_teammates'].iloc[0], 
                            'average_speed_of_opponents' : passes['average_speed_of_opponents'].iloc[0] ,
                            'pressure_level_passer' : passes['pressure level passer'].iloc[0],
                            'opponents_nearby' : passes['opponents_nearby'].iloc[0],
                            'possession_xg' : passes['possession_xg'].iloc[0],
                            'teammates_beyond' : passes['teammates_beyond'].iloc[0],
                            'teammates_behind' : passes['teammates_behind'].iloc[0],
                            'opponents_beyond' : passes['opponents_beyond'].iloc[0],
                            'opponents_behind' : passes['opponents_behind'].iloc[0],
                            'pressure_on_passer' : passes['pressure_on_passer'].iloc[0],
                            'pass_angle' : passes['pass_angle'].iloc[0],
                            'end_angle_to_goal' : passes['end_angle_to_goal'].iloc[0],
                            'start_distance_to_sideline' : passes['start_distance_to_sideline'].iloc[0],
                            'end_distance_to_sideline' : passes['end_distance_to_sideline'].iloc[0],
                            'end_distance_to_goal' : passes['end_distance_to_goal'].iloc[0],
                            'teammates_nearby' : passes['teammates_nearby'].iloc[0]                       
                            }
        
        # Extract mimic-related features (raw features for description)
        #pass_row = passes.iloc[0]
        #pass_features = {
            #col.replace("_mimic", ""): pass_row[col]
            #for col in pass_row.index
            #if col.endswith("_mimic")
        #}


        feature_descriptions = sentences.describe_pass_features(pass_features, self.competition)

        description = (
            f"The pass is a {pass_type} made from {sentences.describe_position_pass(x, y, team_direction)}, "
            f"executed by {player_name} of {team_name}. "
            f"{sentences.describe_xT_pass(xT, xG)}"
        )
        description += '\n' + '\n'.join(feature_descriptions)
        description += '\n' + sentences.describe_pass_contributions_mimic(contributions, pass_features)

        with st.expander("Synthesized Text"):
            st.write(description)

        return description

    def get_prompt_messages(self):
        prompt = (
            "You are a football commentator. Write an insightful 4-sentence summary of a pass. "
            "Start by evaluating its overall effectiveness, including xT and xG. "
            "Then, highlight the key tactical or spatial features that contributed to or limited its danger. "
            "Use vivid football language, and close by either praising the player or suggesting tactical alternatives."
        )
        return [{"role": "user", "content": prompt}]

