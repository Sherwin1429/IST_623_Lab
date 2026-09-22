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
# Part A - Test Weather Function
# --------------------------------------------------

st.subheader("Part A - Weather Function Test")

syracuse_weather = get_current_weather(
    "Syracuse, NY"
)

lima_weather = get_current_weather(
    "Lima, Peru"
)

st.write("Syracuse, NY:")
st.write(syracuse_weather)

st.write("Lima, Peru:")
st.write(lima_weather)

# --------------------------------------------------
# Part B - What to Wear Bot
# --------------------------------------------------

st.divider()

st.subheader("Part B - What to Wear Bot")

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

    # First OpenAI call
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=messages,
        tools=[weather_tool],
        tool_choice="auto"
    )

    # Get OpenAI's response
    assistant_message = response.choices[0].message

    # Check if OpenAI requested the weather tool
    if assistant_message.tool_calls:

        tool_call = assistant_message.tool_calls[0]

        # Get location from the tool call
        arguments = json.loads(
            tool_call.function.arguments
        )

        requested_location = arguments.get(
            "location",
            "Syracuse, NY"
        )

        # Run the actual weather function
        weather_data = get_current_weather(
            requested_location
        )

    
                # Second OpenAI call using the weather data
        final_prompt = (
            f"The current weather information is: {weather_data}. "
            "Based on this weather, recommend appropriate clothes "
            "to wear today and outdoor activities that would be "
            "appropriate for the weather."
        )

        final_response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {
                    "role": "user",
                    "content": final_prompt
                }
            ]
        )

        recommendation = (
            final_response.choices[0].message.content
        )

        st.subheader("Today's Recommendation")
        st.write(recommendation)