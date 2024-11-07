
# Library imports
import streamlit as st
import pandas as pd
import argparse
import tiktoken
import os
import numpy as np

from utils.utils import normalize_text

from classes.data_source import Arguments

from classes.data_source import Lesson

from classes.description import (
    TrolleyDescription, 
)
from classes.description import (
    LessonDescription, 
)
from classes.chat import TrolleyChat, LessonChat

from utils.page_components import (
    add_common_page_elements,
    create_chat,
)

from classes.visual import TreePlot

sidebar_container = add_common_page_elements()
page_container = st.sidebar.container()
sidebar_container = st.sidebar.container()

st.divider()

lesson = Lesson()

with open("model cards/model-card-CProgrammingagent.md", 'r',encoding='utf-8') as file:
    # Read the contents of the file
    model_card_text = file.read()


st.expander("Model card for CProgramming agent", expanded=False).markdown(model_card_text)


overall='1.'

# Selcet random topics 
if 'argumentsMade' not in st.session_state:
    st.session_state.userStance = np.random.choice(['defination','implementation', 'implementation','defination','implementation','condtion'])
userStance = st.session_state.userStance
# make a dictionary to give opposite Pro and Con arguments
 #stanceSwap = {'Pro': 'Con', 'Con': 'Pro'}
stanceFullName = {'defination':'implementation', 'implementation':'defination', 'implementation':'condtion'}

# Get the overall thesis
overallThesis = lesson.df[lesson.df['step']==overall].iloc[0]['assistant']
#st.write(overallThesis)
currentState=lesson.df[lesson.df['step']==overall].iloc[0]['topic']
#st.write(currentState)
displaytext= (
    "## The programming lesson chat\n\n"
    "Do you want to learn about programming concepts in an interactive manner! "
    "In this chat, we are going to teach you programming by prompting you to respond to questions. "
    "Each response you make will be used to determine what you will be asked to do next. "
    "The aim is to get you to understand the main concepts of programming without just copy pasting! \n\n " 
    "**Enjoy the lesson and let's get started!** \n\n "
    )

st.markdown(displaytext)

background = '**Background**: You will be prompted to do a task, based on your response you will be guided.'
st.markdown(background)
#text = '**Thesis**: ' + overallThesis
#st.markdown(text)
st.markdown(' You should respond to the questions asked with all honesty.')

if 'argumentsMade' not in st.session_state:
    st.session_state.argumentsMade = []
if 'gameOver' not in st.session_state:
    st.session_state.gameOver = False

to_hash = (overall)
chat = create_chat(to_hash, LessonChat, overallThesis, lesson, gameOver=st.session_state.gameOver)
#st.write(currentState)
# Now we want to add basic content to chat if it's empty
if chat.state == "empty":

    #Gets the arguments at current level and supporting arguments one below.
    currentArguments= lesson.get_arguments(overall,overallThesis )
    #st.write(currentArguments)
    description = LessonDescription(chat.state, currentArguments,overallThesis)
    summary = description.stream_gpt()

    #chat.add_message("What do you know about for loops as used in C programming langaguge?")
    chat.add_message(overallThesis)

    chat.state = "default"

# Now we want to get the user input, display the messages and save the state
#st.write(chat.state)
chat.get_input()
chat.display_messages()
#st.session_state.totalscore =  chat.totalscore
st.session_state.arguments = chat.arguments
st.session_state.gameOver = chat.gameOver
chat.save_state()

