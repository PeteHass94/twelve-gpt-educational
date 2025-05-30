# Twelve Projects

Built on TwelveGPT's Educational streamlit app, I am using this to showcase the applications I learned during my 12 week course with Twelve.


## Description

TwelveGPT Educational is a basic retrieval augmented chatbot for making reports about data.
The system is set up in a general way, to allow users to build bots which talk about data. 

The football scout botdisplays a distribution plot regarding football player's performance in various metrics. It then starts a chat giving an AI generated summary of the player's performance and asks a variety of questions about the player. 

This is **not** the Twelve GPT product, but rather a (very) stripped down version of our code 
to help people who would like to learn how to build bots to talk about football data. There are lots of things which Twelve GPT can do, which TwelveGPT Educational cannot do. But we want more people to learn about the methods we use and to do this **TwelveGPT Educational** is an excellent alternative. We have thus used the the GNU GPL license which requires that all the released improved versions are also be free software. This will allow us to learn from each other in developing better 

If you work for a footballing organisation and would like to see a demo of the full Twelve GPT product then please email us at hello@twelve.football.

The design and code for this project was by Matthias Green, David Sumpter and Ágúst Pálmason Morthens. 

## Usage

This application was made with Streamlit.  To run locally, first create .streamlit/secrets.toml with keys, etc... then run:
```bash
conda create --name streamlit_env
conda activate streamlit_env
pip install -r requirements.txt
streamlit run app.py
```
Once you have made changes to the code, save, move focus to the streamlit tab, then press c to clear caches if necessary, then r to rerun. 

You also need to have access to GPT API to use this package. Alternatively, you need access to Gemini API but that requires changes to the [.streamlit/secrets.toml](.streamlit/secrets.toml) file (see below).

## How does it work?
### App
Streamlit reruns the code every time the user interacts with the app. This code is located in app.py. The user selects a player and the visual and word report starts to generate.

The application builds primarily around five classes: data_sources, visual, description, chat and embeddings. We now describe these in turn.

### Data sources

The code data_sources.py consists of three classes:

**class Data()**: Gets, processes and manage various forms of data. The data is primarily stored in data.df
**class Stats(Data)**: Calculates z-scores, ranks and pct_ranks, adding these to stats.df

While the above classes can be adapted to any data source, the last class is specifically for football player data.

**class PlayerStats(Stats)**: Loads in a dataframe of statistics about forwards. The data is loaded in from data/events/Forwards.csv. This data is in turn generated by saving a dataframe from the following tutorial about scouting: https://soccermatics.readthedocs.io/en/latest/gallery/lesson3/plot_RadarPlot.html

It provided the following statistics: Non-penalty goals, Assists, Key passes, Smart passes, Ariel duels won, Ground attacking duels won, Non-penalty expected goals, Passes ending in final third, Receptions in final third for players in the Premier League 2017/18 season.

### Visual

There is quite a lot of code here, but it is primarily about making nice visuals. Of particular interest our **add_player(...)** and **add_players(...)** which add the focal player and compare him to the other players in the data.

### Description

It is in this part of the code where we start doing something novel. The three most important functions for creating a text are:

**get_intro_messages()**: This sets up the bot and explains to it what it does.
**synthesize_text()**: This converts the stats.df to a description in words of what the data says.
**get_prompt_messages()**: This is the prompt which tells GPT3 or GPT4 how to use the texts supplied.

A key to success of prompting lies in two types of files, known as describe and gpt_example files. These are given for this application in 
data/describe/Forward
and
data/gpt_examples/Forward

By clicking on the expander in the Description messages you can see how they have been used to construct a prompt to GPT4. It is this prompt which then generates the text under the figure.

### Chat

The chat also utilises prompting of GPT to allow user questions to be answered. This is the bot.

The key function here is **handle_input(input)** which puts together a query combining:

1, An instruction for the bot, which set up by **instruction_messages()**.
2, The previous conversation
3, And relevant information about the player and for answering the questuon, from  **get_relevant_info(input)**.

The get_relevant_info(input) both retrieves the synthesize_text() from the description and searches a library of embedded questions to find relevant info. To do this the input is embedded in order to search the database of embedded questions for relevant entries. 

### Embeddings

Certain files in /data/describe/ contain question-answer pairs that are embedded by pages/embedder.py. You can run this app by clicking on 'Embedding Tool' in top left corner of the app. This is then used to search (using cosine similarity) for the best question-answer pairs for answering the users query.


### Using Open AI API
To use Open AI you need a API key. Then you need to add the following lines to your [.streamlit/secrets.toml](.streamlit/secrets.toml) file.

```toml
USE_GEMINI = false
GPT_BASE = "address of you deployment of Chat GPT"
GPT_VERSION = "version date"
GPT_KEY = "your key"
GPT_ENGINE = "model name"
```

### Using Gemini API
If, instead of using OpenAI's API, you want to use Google's. You need to add the following lines to your [.streamlit/secrets.toml](.streamlit/secrets.toml) file.

```toml
USE_GEMINI = true
GEMINI_API_KEY = "YOUR_API_KEY"

# Can use any chat model
GEMINI_CHAT_MODEL = "gemini-1.5-flash"

# Can use any embedding model
GEMINI_EMBEDDING_MODEL = "models/text-embedding-004"
```
