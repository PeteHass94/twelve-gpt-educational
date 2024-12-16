"""
Page components for pages/*.py
"""

# Stdlib imports
import base64
from pathlib import Path

import streamlit as st
import copy

# from pages import about, football_scout, embedder, wvs_chat, own_page


def insert_local_css():
    """
    Injects the local CSS file into the app.
    Replaces the logo and font URL placeholders in the CSS file with base64 encoded versions.
    """
    with open("data/style.css", "r") as f:
        css = f.read()

    logo_url = (
        "url(data:image/png;base64,"
        + base64.b64encode(
            Path("data/ressources/img/twelve_logo_light.png").read_bytes()
        ).decode()
        + ")"
    )
    font_url_medium = (
        "url(data:font/otf;base64,"
        + base64.b64encode(
            Path("data/ressources/fonts/Gilroy-Medium.otf").read_bytes()
        ).decode()
        + ")"
    )
    font_url_light = (
        "url(data:font/otf;base64,"
        + base64.b64encode(
            Path("data/ressources/fonts/Gilroy-Light.otf").read_bytes()
        ).decode()
        + ")"
    )

    css = css.replace("replace_logo_url", logo_url)
    css = css.replace("replace_font_url_medium", font_url_medium)
    css = css.replace("replace_font_url_light", font_url_light)

    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def set_page_config():
    """
    Sets the page configuration for the app.
    """
    st.set_page_config(
        layout="centered",
        page_title="TwelveGPT Scout",
        page_icon="data/ressources/img/TwelveEdu.png",
        initial_sidebar_state="expanded",
        menu_items={
            "Report a bug": "mailto:matthias@twelve.football?subject=Bug report"
        },
    )


def add_page_selector():
    st.image("data/ressources/img/TwelveEdu.png")
    st.page_link("pages/about.py", label="About")
    st.page_link("pages/football_scout.py", label="Football Scout")
    st.page_link("pages/embedder.py", label="Embdedding Tool")
    st.page_link("pages/wvs_chat.py", label="World Value Survey")
    st.page_link("pages/personality_test.py", label="Personality Test")
    st.page_link("pages/own_page.py", label="Your Own Page")
    st.page_link("pages/run_analysis.py", label="CL 2018 run analysis")

    # st.image("data/ressources/img/TwelveEdu.png")

    # # Define the available pages using their module names, not file paths
    # pages = {
    #     "About": about,
    #     "Football Scout": football_scout,
    #     "Embedder": embedder,
    #     "World Values Survey": wvs_chat,
    #     "Your Own Page": own_page,
    #     # Add other pages here
    # }

    # # Sidebar for page selection with default set to "About"
    # selected_page = st.sidebar.radio(
    #     "Select a page",
    #     list(pages.keys()),
    #     index=0,  # 'index=0' selects "About" by default
    # )

    # # Load and display the selected page's content by calling its `show` function
    # page = pages[selected_page]
    # # page.show()  # Assume each page has a `show()` function to display its content


def add_common_page_elements():
    """
    Sets page config, injects local CSS, adds page selector and login button.
    Returns a container that MUST be used instead of st.sidebar in the rest of the app.

    Returns:
        sidebar_container: A container in the sidebar to hold all other sidebar elements.
    """
    # Set page config must be the first st. function called
    set_page_config()
    # Insert local CSS as fast as possible for better display
    insert_local_css()
    # Create a page selector
    page_selector_container = st.sidebar.container()
    sidebar_container = st.sidebar.container()

    page_selector_container = st.sidebar.container()
    sidebar_container = st.sidebar.container()

    with page_selector_container:
        add_page_selector()

    sidebar_container.divider()

    return sidebar_container


def select_player(container, players, gender, position):

    # Make a copy of Players object
    player = copy.deepcopy(players)

    # Filter players by position and select a player with sidebar selectors
    with container:

        # Filter for player name
        player.select_and_filter(
            column_name="player_name",
            label="Player",
        )

        # Return data point

        player = player.to_data_point(gender, position)

    return player


def select_person(container, person_stat):

    # Make a copy of Players object
    person = copy.deepcopy(person_stat)

    # Filter players by position and select a player with sidebar selectors
    with container:

        # Filter for player name
        person.select_and_filter(
            column_name="name",
            label="Person",
        )

        # Return data point

        person = person.to_data_point()

    return person


def create_chat(to_hash, chat_class, *args, **kwargs):
    chat_hash_state = hash(to_hash)
    chat = chat_class(chat_hash_state, *args, **kwargs)
    return chat



def select_runs(container, player_run_counts_df, runs_obj):
    """
    Allows the user to select a player and returns their run details.

    Args:
        container: Streamlit container for the select box.
        player_run_counts_df: DataFrame with player names and run counts.
        runs_obj: An instance of RunStats for fetching and filtering runs.

    Returns:
        DataFrame of runs for the selected player.
    """
    with container:
        # Select a player from the list
        player_list = player_run_counts_df['player'].tolist()
        selected_player = st.selectbox("Select a player to view their runs:", player_list)

    # Fetch the detailed runs for the selected player
    detailed_runs = runs_obj.filter_runs_by_player(selected_player)
    return selected_player, detailed_runs

# def select_runs(container, player_run_counts_df, runs):
#     """
#     Allows the user to select a player from the sidebar and returns the player and their detailed runs.

#     Args:
#         container: Streamlit container.
#         player_run_counts_df: DataFrame containing run counts for players.
#         runs: RunStats object containing detailed run data.

#     Returns:
#         tuple: Selected player name and a DataFrame of their detailed runs.
#     """
#     # Select a player from the sidebar
#     selected_player = container.selectbox(
#         "Select Player",
#         options=player_run_counts_df['player'],
#         index=0,  # Default to the first player
#     )

#     # Filter detailed runs for the selected player
#     detailed_runs_df = runs.filter_runs_by_player(selected_player)

#     # Return the player name and their detailed run data
#     return selected_player, detailed_runs_df