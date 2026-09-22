import streamlit as st
import requests
import json
from openai import OpenAI


# --------------------------------------------------
# OpenAI Setup
# --------------------------------------------------

openai_api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=openai_api_key)


# --------------------------------------------------
# Weather Function
# --------------------------------------------------

def get_current_weather(location):

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
# Weather Tool Definition
# --------------------------------------------------

weather_tool = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": (
            "Get the current weather information for a clearly "
            "identified location. Before using this tool, make sure "
            "the city is clear. If the city could refer to multiple "
            "locations, ask the user for the state, province, or "
            "country before requesting this tool."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": (
                        "A clear city and state/province/country, "
                        "for example Syracuse, NY or Lima, Peru."
                    )
                }
            },
            "required": ["location"]
        }
    }
}


# --------------------------------------------------
# System Prompt
# --------------------------------------------------

system_prompt = """
You are a helpful What to Wear assistant.

Your job is to recommend clothing and outdoor activities
based on the current weather.

Use the get_current_weather tool whenever current weather
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
# Streamlit Interface
# --------------------------------------------------

st.title("🌤️ What to Wear Bot")

st.write(
    "Tell me where you are, and I'll help you decide "
    "what to wear and what you can do outside today."
)


# --------------------------------------------------
# Conversation Memory
# --------------------------------------------------

if "lab5_messages" not in st.session_state:
    st.session_state.lab5_messages = []


# Display previous conversation
for message in st.session_state.lab5_messages:

    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])

    elif message["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(message["content"])


# --------------------------------------------------
# User Input
# --------------------------------------------------

user_input = st.chat_input(
    "Enter a city, for example Syracuse, NY"
)


if user_input:

    # Save and display user's message
    st.session_state.lab5_messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)


    # --------------------------------------------------
    # Build Messages for LLM
    # --------------------------------------------------

    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    messages.extend(st.session_state.lab5_messages)


    # --------------------------------------------------
    # First LLM Call
    # LLM decides whether to request the weather tool
    # --------------------------------------------------

    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=messages,
        tools=[weather_tool],
        tool_choice="auto"
    )

    assistant_message = response.choices[0].message


    # --------------------------------------------------
    # If LLM Requests Weather Tool
    # --------------------------------------------------

    if assistant_message.tool_calls:

        # Add LLM's tool request to the conversation
        messages.append(assistant_message)

        tool_call = assistant_message.tool_calls[0]

        arguments = json.loads(
            tool_call.function.arguments
        )

        requested_location = arguments.get(
            "location",
            "Syracuse, NY"
        )


        # --------------------------------------------------
        # Execute Tool ONLY After LLM Requests It
        # --------------------------------------------------

        try:

            weather_data = get_current_weather(
                requested_location
            )


            # --------------------------------------------------
            # Return Tool Result to LLM
            # --------------------------------------------------

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(weather_data)
                }
            )


            # --------------------------------------------------
            # Second LLM Call
            # LLM uses weather result to create final answer
            # --------------------------------------------------

            final_response = client.chat.completions.create(
                model="gpt-5-mini",
                messages=messages,
                tools=[weather_tool],
                tool_choice="auto"
            )

            recommendation = (
                final_response.choices[0].message.content
            )


            # Save final response
            st.session_state.lab5_messages.append(
                {
                    "role": "assistant",
                    "content": recommendation
                }
            )


            # Display final response
            with st.chat_message("assistant"):
                st.markdown(recommendation)


        except Exception as e:

            error_message = (
                "I couldn't get weather information for "
                f"**{requested_location}**. "
                "Please provide the city with its state, "
                "province, or country."
            )

            st.session_state.lab5_messages.append(
                {
                    "role": "assistant",
                    "content": error_message
                }
            )

            with st.chat_message("assistant"):
                st.markdown(error_message)


    # --------------------------------------------------
    # No Tool Call
    # This can be a clarification question
    # --------------------------------------------------

    else:

        assistant_response = assistant_message.content

        st.session_state.lab5_messages.append(
            {
                "role": "assistant",
                "content": assistant_response
            }
        )

        with st.chat_message("assistant"):
            st.markdown(assistant_response)