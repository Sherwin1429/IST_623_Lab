from openai import OpenAI
import streamlit as st
import requests
import json

# --------------------------------------------------
# Page Title
# --------------------------------------------------

st.title("Lab 5 - What to Wear Bot")


# --------------------------------------------------
# OpenAI Client
# --------------------------------------------------

openai_api_key = st.secrets["OPENAI_API_KEY"]

client = OpenAI(
    api_key=openai_api_key
)


# --------------------------------------------------
# Function to Get Current Weather
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
# Define Weather Tool for OpenAI
# --------------------------------------------------

weather_tool = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": (
            "Get the current weather information for a location "
            "so clothing and outdoor activity recommendations can be made."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": (
                        "City and state or city and country, "
                        "for example Syracuse, NY or Lima, Peru."
                    )
                }
            },
            "required": ["location"]
        }
    }
}


# --------------------------------------------------
# What to Wear Bot
# --------------------------------------------------

st.subheader("What to Wear Bot")

location = st.text_input(
    "Enter a city:",
    placeholder="Example: Syracuse, NY"
)

if st.button("Get Recommendations"):

    # If no location is entered, use Syracuse, NY
    if not location.strip():
        location = "Syracuse, NY"

    user_prompt = (
        f"What should I wear today in {location}, "
        "and what outdoor activities would be appropriate?"
    )

    messages = [
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    # First LLM call:
    # The LLM decides whether it needs the weather tool.
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=messages,
        tools=[weather_tool],
        tool_choice="auto"
    )

    assistant_message = response.choices[0].message

    # Check whether the LLM requested a tool
    if assistant_message.tool_calls:

        # Add the LLM's tool request to the conversation
        messages.append(assistant_message)

        tool_call = assistant_message.tool_calls[0]

        # Read the arguments requested by the LLM
        arguments = json.loads(
            tool_call.function.arguments
        )

        requested_location = arguments.get(
            "location",
            "Syracuse, NY"
        )

        # The application executes the function ONLY
        # after the LLM requests it
        weather_data = get_current_weather(
            requested_location
        )

        # Return the tool result to the LLM
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(weather_data)
            }
        )

        # Give the LLM instructions for presenting the result
        messages.append(
            {
                "role": "system",
                "content": """
Using the weather information returned by the tool,
give the user a friendly and useful recommendation.

Format the response like this:

### 🌤️ Weather
Give a short summary of today's weather.

### 👕 What to Wear
Give 3-5 practical clothing suggestions.

### 🌳 Outdoor Activities
Suggest 3-5 outdoor activities appropriate for the weather.

### 💡 Quick Tip
Give one useful tip for the day.

Keep the response concise, creative, and easy to read.
"""
            }
        )

        # Second LLM call:
        # The model now has the actual result from its requested tool.
        final_response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=messages
        )

        recommendation = (
            final_response.choices[0].message.content
        )

        st.subheader("Today's Recommendation")
        st.markdown(recommendation)

    else:
        # If the LLM decides that no tool is necessary
        st.write(assistant_message.content)