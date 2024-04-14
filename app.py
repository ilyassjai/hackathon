import streamlit as st
import requests

from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.prompts import StringPromptTemplate
from langchain.utilities import SerpAPIWrapper
from langchain.chains import LLMChain
from langchain.llms import LlamaCpp
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, OutputParserException
import re
from llama_cpp import Llama

from retrieve_data import retrieve_events, prompt_format


temp = """
You are an intelligent travel assistant and guide.
the user will be in {city} from {start_date} until {end_date}.
The user preferences in a scale from 1 to 10 are: {preferences}.

Actions available: check the available events in the surrondings the user can attend, between {start_date} and {end_date}: 
{events}.

Actions available: check historical sites in the surrondings the user can visit:
{sites}

Provide a detailed activities plan as a mix of hitorical sites to visit and events to attend in {city} following the user prefrences from {start_date} until {end_date}, 
including a section for each day with an outline of the schedule. Include timestamps and locations.
"""


def init_llm():

    callback_manager = AsyncCallbackManager([StreamingStdOutCallbackHandler()])

    n_gpu_layers = 1  # Metal set to 1 is enough.
    n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
    # Make sure the model path is correct for your system!
    llm = LlamaCpp(
        model_path="./content/llama-2-7b.Q4_K_M.gguf",
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        max_tokens=5000,
        top_p=0.95,
        frequency_penalty=0.5,
        presence_penalty=0.5,
        temperature = 0.4,
        f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
        callback_manager=callback_manager,
        verbose=True,  # Verbose is required to pass to the callback manager
        n_ctx = 1536,
        #grammar_path = "/home/abdellah/trip/content/gr.gbnf" 
    )

    return llm

#def query(payload):
#    response = requests.post(API_URL, headers=headers, json=payload)
#    return response.json()


# Function to generate the itinerary based on user input or default value
def fctgai(llm, user_input='default'):
    if user_input == '':
        return "L'itinéraire est comme suit : Arrivée Casablanca -> Fes -> Oujda -> Nador -> retour"
    else:

        # Make a request to the Hugging Face Inference API
        response = llm(temp +user_input )   
        
        # Customize the logic based on your requirements
        return f"L'itinéraire est comme suit : {response} "

llm = init_llm()

# Set Streamlit app title and description
st.title('Trip Planning App')
st.image('logo.png', width=0.7, caption='', use_column_width=True)
st.write("Plan your journey in Morocco")

# Set theme colors
st.markdown(
    """
    <style>
    body {
        background-color: black;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.write("")

form = st.form(key="form_settings")
col1, _, col2 = form.columns([2, 0.5, 1])

Cities = ["Marrakesh", "Casablanca", "Tangier", "Fez", "Rabat"]

city = col1.selectbox(
    "Choose your city",
    options=Cities,
    key="style",
)

radius = col2.slider(
    "Density",
    1,
    10,
    step = 1,
    key="Density",
)

col1, col2 = form.columns([1, 1])
date_start = col1.date_input("Pick the start date fo your travel")
date_end = col2.date_input("Pick the end date")

expander = form.expander("Customize your Trip even more")

col1style, _, col2style = expander.columns([2, 0.5, 2])


hist_tour = col1style.slider(
    "Historical Attractions",
    value = 5,
    min_value=1,
    max_value=10,
    key="hist_tour",
)

col1style.markdown("---")


include_events = col2style.slider(
    "Attending Events",
    value = 5,
    min_value=1,
    max_value=10,
    key="include_events",
)

custom_title = col2style.text_input(
    "Additional preference (optional)",
    max_chars=40,
    key="custom_title",
)


form.form_submit_button(label="Submit")

result = st.empty()

if st.button("Generate Itinerary"):

    preferences = "".join(["Attending Events:", str(include_events),"; Historical Attractions:", str(hist_tour),";"])

    prompt = prompt_format(temp, preferences, city, date_start, date_end)

    res = llm(prompt)

    st.success(res)


# Input text zone
# user_input = st.text_area('Want to make adjustuments ? Express yourself :', '')


# Footer or additional information
st.markdown('© 2023 RAM-DT team hackathon - MoroccoAI')
# Run the app with: streamlit run your_script_name.py
