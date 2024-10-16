# Library imports
import streamlit as st
from utils.utils import select_country, create_chat

from classes.data_source import CountryStats

from classes.chat import WVSChat
from classes.visual import DistributionPlot

from utils.page_components import add_common_page_elements

from classes.description import (
    CountryDescription,
)

# def show():
sidebar_container = add_common_page_elements()
page_container = st.sidebar.container()
sidebar_container = st.sidebar.container()

st.divider()

countries = CountryStats()

metrics = [m for m in countries.df.columns if m not in ["country"]]

countries.calculate_statistics(metrics=metrics)

country = select_country(sidebar_container, countries)

st.write(
    "This app can only handle three or four users at a time. Please [download](https://github.com/soccermatics/twelve-gpt-educational) and run on your own computer with your own Gemini key."
)

# Read in model card text
with open("model cards/model-card-wvs-chat.md", "r", encoding="utf8") as file:
    # Read the contents of the file
    model_card_text = file.read()


st.expander("Model card", expanded=False).markdown(model_card_text)


st.expander("Dataframe used", expanded=False).write(countries.df)

description_dict = {
    "Traditional vs Secular Values": [
        "extremely secular",
        "very secular",
        "above averagely secular",
        "neither traditional nor secular",
        "above averagely traditional",
        "very traditional",
        "extremely traditional",
    ],
    "Survival vs Self-expression Values": [
        "extremely self-expression orientated",
        "very self-expression orientated",
        "above averagely self-expression orientated",
        "neither survival nor self-expression orientated",
        "some what survival orientated",
        "very survival orientated",
        "extremely survival orientated",
    ],
    "Neutrality": [
        "extremely high",
        "very high",
        "above average",
        "average",
        "below average",
        "very low",
        "extremely low",
    ],
    "Fairness": [
        "extremely high",
        "very high",
        "above average",
        "average",
        "below average",
        "very low",
        "extremely low",
    ],
    "Skeptisism": [
        "extremely high",
        "very high",
        "above average",
        "average",
        "below average",
        "very low",
        "extremely low",
    ],
    "Societal Tranquility": [
        "extremely high",
        "very high",
        "above average",
        "average",
        "below average",
        "very low",
        "extremely low",
    ],
}

thresholds_dict = dict(
    (
        metric,
        [
            1.5,
            1,
            0.5,
            -0.5,
            -1,
            -1.5,
        ],
    )
    for metric in metrics
)

# check that the metrics exactly match the keys of the description_dict
if set(metrics) == set(description_dict.keys()):
    pass
else:
    raise ValueError(
        "The metrics do not match the keys of the description_dict. If you recently update the data then likely need to update the description_dict."
    )

# Chat state hash determines whether or not we should load a new chat or continue an old one
# We can add or remove variables to this hash to change conditions for loading a new chat
to_hash = (country.id,)

chat = create_chat(to_hash, WVSChat, country, countries)

# Now we want to add basic content to chat if it's empty
if chat.state == "empty":

    # Make a plot of the distribution of the metrics for all players
    # We reverse the order of the elements in metrics for plotting (because they plot from bottom to top)
    visual = DistributionPlot(
        metrics[::-1], labels=["Low", "Average", "High"], plot_type="wvs"
    )
    visual.add_title_from_player(country)
    visual.add_players(countries, metrics=metrics)
    visual.add_player(country, len(countries.df), metrics=metrics)

    # Now call the description class to get the summary of the country
    description = CountryDescription(
        country, description_dict=description_dict, thresholds_dict=thresholds_dict
    )
    summary = description.stream_gpt()

    # Add the visual and summary to the chat
    chat.add_message(
        "Please can you summarise " + country.name + " for me?",
        role="user",
        user_only=False,
        visible=False,
    )
    chat.add_message(visual)
    chat.add_message(summary)

    chat.state = "default"

# Now we want to get the user input, display the messages and save the state
chat.get_input()
chat.display_messages()
chat.save_state()

# st.write("Under construction")