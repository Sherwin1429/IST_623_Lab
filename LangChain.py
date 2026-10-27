
import os
import streamlit as st
import requests
from langchain.agents import create_agent

# --------------------------------------------------
# OpenAI Setup
# --------------------------------------------------

# Use a Codespaces environment variable or Streamlit secrets
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass

if not api_key:
    st.error("OPENAI_API_KEY is not configured.")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key
# --------------------------------------------------
# Weather Tool - Reused from Lab 5
# --------------------------------------------------

def get_weather_data(location: str = "Syracuse, NY") -> dict:
    """
    Get current weather conditions for a city, state, country,
    or well-known landmark such as the Eiffel Tower.

    Returns the temperature, feels-like temperature, weather
    description, humidity, wind speed, and today's minimum
    and maximum temperatures in Fahrenheit.
    """

    url = f"https://wttr.in/{location}?format=j1"

    response = requests.get(
        url,
        timeout=10
    )

    if response.status_code != 200:
        raise Exception(
            f"wttr.in error: status {response.status_code}"
        )

    try:
        data = response.json()

    except ValueError:
        raise Exception(
            f"Could not find a location named {location}"
        )

    current = data["current_condition"][0]
    today = data["weather"][0]

    return {
        "location": location,
        "temperature": float(current["temp_F"]),
        "feels_like": float(current["FeelsLikeF"]),
        "description": current["weatherDesc"][0]["value"],
        "humidity": int(current["humidity"]),
        "wind_speed": float(current["windspeedMiles"]),
        "min_temperature": float(today["mintempF"]),
        "max_temperature": float(today["maxtempF"])
    }


# --------------------------------------------------
# System Prompt - Reused from Lab 5
# --------------------------------------------------

system_prompt = """
You are a helpful What to Wear assistant.

Your job is to recommend clothing and outdoor activities
based on the current weather.

Use the get_weather_data tool whenever current weather
information is needed.

Before requesting the weather tool, make sure the user's
location is clear enough to identify the correct city.

If the user gives only a city name and that city could refer
to multiple locations, do not guess or choose a location.
Ask the user for more information such as the state,
province, or country.

Continue asking clarification questions until the location
is clear enough to use the weather tool.

If the user does not provide any location, use Syracuse, NY
as the default location.

Do not make up current weather information.

After receiving weather information from the tool, give a
friendly, useful, and creative response using this format:

### 🌤️ Weather
Give a short summary of today's weather.

### 👕 What to Wear
Give 3-5 practical clothing suggestions.

### 🌳 Outdoor Activities
Suggest 3-5 outdoor activities appropriate for the weather.

### 💡 Quick Tip
Give one short useful tip for the day.

Keep the final response concise and easy to read.
"""


# --------------------------------------------------
# Create LangChain Agent
# --------------------------------------------------

agent = create_agent(
    model="openai:gpt-5-mini",
    tools=[get_weather_data],
    system_prompt=system_prompt
)


# --------------------------------------------------
# Streamlit Interface
# --------------------------------------------------

st.title("🌤️ What to Wear Bot - LangChain")

st.write(
    "Enter a location to receive clothing and outdoor "
    "activity recommendations based on current weather."
)

location = st.text_input(
    "Enter a location",
    placeholder="Syracuse, NY"
)

if st.button("Get advice"):

    location = location.strip() or "Syracuse, NY"

    with st.spinner("Checking weather and generating advice..."):

        try:
            result = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": f"What should I wear in {location}?"
                        }
                    ]
                }
            )

            final_answer = result["messages"][-1].content

            with st.chat_message("assistant"):
                st.write(final_answer)

            # --------------------------------------
            # Behind the Scenes - Part D
            # --------------------------------------

            with st.expander("Behind the scenes"):

                for message in result["messages"]:

                    st.write(f"**{message.type}**")
                    st.write(message.content)

                    if (
                        message.type == "ai"
                        and message.tool_calls
                    ):
                        st.write(message.tool_calls)

        except Exception as e:
            st.error(f"Error: {e}")


# --------------------------------------------------
# Part D - Answers (complete after testing)
# --------------------------------------------------

# a. Number and types of messages:
# Answer after running the application.

# b. Which message requests the weather tool?
# The AI message containing tool_calls.

# c. Which Lab 5 code created those messages?
# Refer to the first LLM call, manual weather tool
# execution, and second LLM call.


# --------------------------------------------------
# Part E - Stress Test (complete after testing)
# --------------------------------------------------

# Test location:
# Syracuse, NY and Lima, Peru

# Record:
# 1. Number of weather lookups in Lab 5.
# 2. Number of weather lookups in LangChain.
# 3. Whether both cities appear in each answer.
# 4. Number of model calls in each version.
# 5. Whether the agent makes additional tool calls.
