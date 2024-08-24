import math
from abc import ABC, abstractmethod
from typing import List, Union, Dict, Optional

import pandas as pd
import tiktoken
import openai
import numpy as np

import utils.sentences as sentences
from classes.data_point import Player

from settings import GPT_BASE, GPT_VERSION, GPT_KEY, GPT_ENGINE, GPT_DEFAULT

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
                    "You are a data analysis bot. "
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


    def get_messages_from_excel(self,
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
            paths=self.describe_paths
            messages += self.get_messages_from_excel(paths)
        except FileNotFoundError as e:  # FIXME: When merging with new_training, add the other exception
            print(e)
        messages += self.get_prompt_messages()

        messages = [message for message in messages if isinstance(message["content"], str)]


        try:
            messages += self.get_messages_from_excel(
                paths=self.gpt_examples_path,
                
            )
        except FileNotFoundError as e:  # FIXME: When merging with new_training, add the other exception
            print(e)

        messages += [{"role": "user", "content": f"Now do the same thing with the following: ```{self.synthesized_text}```"}]
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
        openai.api_base = GPT_BASE
        openai.api_version = GPT_VERSION
        openai.api_key = GPT_KEY

        st.expander("Description messages", expanded=False).write(self.messages)

        response = openai.ChatCompletion.create(
            engine=GPT_ENGINE,
            messages=self.messages,
            temperature= temperature,
            )
    
        answer=response['choices'][0]['message']['content']

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

        player=self.player
        metrics = self.player.relevant_metrics
        description = f"Here is a statistical description of {player.name}, who played for {player.minutes_played} minutes as a {player.position}. \n\n "

        subject_p, object_p, possessive_p = sentences.pronouns(player.gender)
        
        for metric in metrics:

            description += f"{subject_p.capitalize()} was "
            description += sentences.describe_level(player.ser_metrics[metric +"_Z"]) 
            description += " in " + sentences.write_out_metric(metric)
            description += " compared to other players in the same playing position. "                            

        #st.write(description)

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

class PlayerDescriptionComparison(Description):
    output_token_limit = 150

    @property
    def gpt_examples_path(self):
        return f"{self.gpt_examples_base}/Compare.xlsx"

    @property
    def describe_paths(self):
        return [f"{self.describe_base}/Forward.xlsx",f"{self.describe_base}/Compare.xlsx"]

    def __init__(self, player1: Player, player2:Player):
        self.player1 = player1
        self.player2 = player2
        super().__init__()
    
    def synthesize_text(self):
        player1 = self.player1
        player2 = self.player2
        metrics = player1.relevant_metrics  # Assuming both players have the same relevant metrics

        description = f"Here is a statistical comparison between {player1.name} and {player2.name}, who played for {player1.minutes_played} and {player2.minutes_played} minutes respectively as {player1.position}.\n\n"

        subject_p1, object_p1, possessive_p1 = sentences.pronouns(player1.gender)
        subject_p2, object_p2, possessive_p2 = sentences.pronouns(player2.gender)
        
        for metric in metrics:
            player1_metric_value = player1.ser_metrics[metric + "_Z"]
            player2_metric_value = player2.ser_metrics[metric + "_Z"]
            
            description += f"In terms of {sentences.write_out_metric(metric)}, "

            #Player 1's performance
            description += f"{player1.name} was "
            description += sentences.describe_level(player1_metric_value)
            description += " compared to other players in the same position. "

            #Player 2's performance
            description += f"{player2.name} was "
            description += sentences.describe_level(player2_metric_value)
            description += " compared to other players in the same position. "

            if player1_metric_value > player2_metric_value:
                description += f"{player1.name} outperformed {player2.name} in this metric. "
            elif player1_metric_value < player2_metric_value:
                description += f"{player2.name} outperformed {player1.name} in this metric. "
            else:
                description += f"Both players performed similarly in this metric. "

            description += "\n"

        return description


    def get_prompt_messages(self):
        prompt = (
        "Please generate a detailed comparison between two players using the statistical descriptions provided. "
        "The response should be structured into five distinct paragraphs: "
        "1. **Introduction**: Provide an overview of how the two players differ in their overall playing style. "
        "2. **Strengths**: Highlight the specific strengths of each player based on the metrics, such as 'Player X is better at [specific skill] than Player A' or 'Player B excels at [skill], but Player A is even stronger'. "
        "3. **Weaknesses**: Discuss the areas where one player might be stronger or weaker than the other, using comparative language like 'Player B is good at [metric], but Player A is superior in [metric]'. "
        "4. **Playing Style**: Describe the playing style of each player, highlighting how their approach to the game might differ. "
        "5. **Overall Comparison**: Summarize how the two players compare directly to each other overall, considering all metrics."
    )
        return [{"role": "user", "content": prompt}]


