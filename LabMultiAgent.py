"""IST 688 Lab 9 — Multi-Agent Systems: The What to Wear Crew.

Parts A–E completed; fill the observation comments for Part D from your own run
and submit Parts E/F answers directly to Blackboard.
"""

import asyncio
import os
import time

import requests
import streamlit as st
from agents import Agent, Runner, function_tool


# --- Configuration ---
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except (KeyError, FileNotFoundError):
        api_key = None
if not api_key:
    st.error("OPENAI_API_KEY not configured. Add a Codespaces secret or Streamlit app secret.")
    st.stop()
os.environ["OPENAI_API_KEY"] = api_key
MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")


# --- Part B: Original Lab 5 weather function (same logic) ---
def get_current_weather(location):
    url = f"https://wttr.in/{location}?format=j1"
    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        raise Exception(f"wttr.in error: status {response.status_code}")
    try:
        data = response.json()
    except ValueError:
        raise Exception(f"Could not find a location named {location}")
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
        "max_temperature": float(today["maxtempF"]),
    }


@function_tool
def get_weather(location: str = "Syracuse, NY") -> dict:
    """Get current weather and today's low/high for a clearly identified location.

    Accepts a city plus state or country, such as 'Syracuse, NY' or
    'Lima, Peru'. Returns temperatures in Fahrenheit, condition,
    humidity, and wind speed in mph. Use only for real weather facts.
    """
    return get_current_weather(location)


# --- Part C: Three specialist agents plus a manager ---
weather_agent = Agent(
    name="Weather agent",
    instructions=(
        "You are the weather specialist. Call get_weather for the user's location. "
        "Report actual conditions (temperature, feels-like, description, wind, "
        "low/high) and the location. Do NOT give outfit or activity advice. "
        "If the weather lookup fails, explain that it failed; do not invent weather."
    ),
    tools=[get_weather],
    model=MODEL,
)

stylist_agent = Agent(
    name="Stylist agent",
    instructions=(
        "You are a clothing stylist. Based ONLY on weather facts supplied by the "
        "manager, recommend 3–5 practical clothes/layers. Do not invent new "
        "weather values. If no valid weather report is supplied, say so."
    ),
    model=MODEL,
)

activity_agent = Agent(
    name="Activity agent",
    instructions=(
        "You plan outdoor activities. Based ONLY on weather facts supplied by "
        "the manager, suggest 3–5 safe, sensible activities and one useful tip. "
        "Do not invent new weather values or claim venues are open."
    ),
    model=MODEL,
)

manager_agent = Agent(
    name="Manager",
    instructions=(
        "Coordinate the specialists for a What to Wear request. "
        "FIRST call weather_report for the given location. "
        "AFTER receiving its weather facts, call clothing_advice and "
        "activity_advice; pass each specialist the location AND the actual "
        "weather report text. Do not ask the specialists to fetch weather. "
        "Then combine their responses into these headings: "
        "### 🌤️ Weather, ### 👕 What to Wear, "
        "### 🌳 Outdoor Activities, ### 💡 Quick Tip. "
        "If no location is provided use Syracuse, NY. If location is ambiguous, "
        "request clarification. Never make up weather facts. "
        "Make your final answer concise."
    ),
    tools=[
        weather_agent.as_tool(
            tool_name="weather_report",
            tool_description="Look up and summarize current weather facts for a specified place.",
        ),
        stylist_agent.as_tool(
            tool_name="clothing_advice",
            tool_description="Recommend clothes using the supplied weather report.",
        ),
        activity_agent.as_tool(
            tool_name="activity_advice",
            tool_description="Suggest outdoor activities and a quick tip using the supplied weather report.",
        ),
    ],
    model=MODEL,
)

# --- Part E: A single agent for a fair comparison ---
single_agent = Agent(
    name="Single weather agent",
    instructions=(
        "You are a helpful What to Wear assistant. Always use get_weather "
        "when the user asks for current weather-based advice. If no location "
        "is provided, use Syracuse, NY. Clarify ambiguous places. "
        "After getting facts, provide ### 🌤️ Weather, ### 👕 What to Wear "
        "(3–5 suggestions), ### 🌳 Outdoor Activities (3–5 suggestions), "
        "and ### 💡 Quick Tip. Never fabricate weather data. Keep it concise."
    ),
    tools=[get_weather],
    model=MODEL,
)


@st.cache_resource
def get_event_loop():
    """Reuse the event loop across Streamlit reruns as the lab handout requests."""
    return asyncio.new_event_loop()


def run(agent, prompt):
    start = time.perf_counter()
    result = get_event_loop().run_until_complete(Runner.run(agent, prompt))
    return result, time.perf_counter() - start


# --- Streamlit UI ---
st.title("🌤️ Lab 9 — What to Wear Crew")
st.write("Compare one weather agent with a manager coordinating three specialist agents.")
location = st.text_input("Enter a location", value="Syracuse, NY")
mode = st.radio("Choose version", ["Single agent", "Multi-agent team"], horizontal=True)
st.caption("Required tests: Syracuse, NY · Lima, Peru · purple banana (team)")

if "lab9_results" not in st.session_state:
    st.session_state.lab9_results = {}

if st.button("Get advice", type="primary"):
    chosen = single_agent if mode == "Single agent" else manager_agent
    place = location.strip() or "Syracuse, NY"
    try:
        with st.spinner("Running agents and checking the weather..."):
            result, seconds = run(chosen, f"What should I wear and do today in {place}?")
        with st.chat_message("assistant"):
            st.write(result.final_output)

        usage = result.context_wrapper.usage
        metrics = {
            "requests": usage.requests,
            "tokens": usage.total_tokens,
            "seconds": seconds,
            "answer": str(result.final_output),
        }
        st.session_state.lab9_results[(place, mode)] = metrics
        with st.expander("Behind the scenes", expanded=True):
            st.write(
                f"Model requests: {usage.requests} · Tokens: {usage.total_tokens:,} "
                f"· Time: {seconds:.1f}s · Answered by: {result.last_agent.name}"
            )
            for item in result.new_items:
                if item.type == "tool_call_item":
                    raw = item.raw_item
                    name = item.tool_name or getattr(raw, "name", "unknown_tool")
                    args = raw.get("arguments", "") if isinstance(raw, dict) else getattr(raw, "arguments", "")
                    st.write(f"{item.agent.name} called {name}: {args}")
                elif item.type == "tool_call_output_item":
                    st.write(f"result: {str(item.output)[:300]}")
                elif item.type == "handoff_output_item":
                    st.write(f"handoff: {item.source_agent.name} → {item.target_agent.name}")
    except Exception as exc:
        st.error(f"The run failed: {exc}")
        st.info("If the weather tool fails, retry once. Check the API key and wttr.in connectivity.")

st.subheader("Part E — Recorded comparisons")
st.caption("Run both versions on Syracuse, NY and Lima, Peru; use these observed values in Blackboard.")
for place in ("Syracuse, NY", "Lima, Peru"):
    single = st.session_state.lab9_results.get((place, "Single agent"))
    team = st.session_state.lab9_results.get((place, "Multi-agent team"))
    if single or team:
        st.markdown(f"**{place}**")
        st.table({
            "Measure": ["Model requests", "Total tokens", "Seconds"],
            "Single agent": [
                single["requests"] if single else "Not tested",
                single["tokens"] if single else "Not tested",
                round(single["seconds"], 1) if single else "Not tested",
            ],
            "Team": [
                team["requests"] if team else "Not tested",
                team["tokens"] if team else "Not tested",
                round(team["seconds"], 1) if team else "Not tested",
            ],
        })

# --------------------------------------------------
# Part D - Behind the Scenes
# --------------------------------------------------

# a. The manager called three specialist agents in this
#    order: Weather Agent, Stylist Agent, and Activity Agent.
#    The Weather Agent retrieved the weather conditions,
#    the Stylist suggested clothing, and the Activity Agent
#    recommended suitable outdoor activities.

# b. The manager sent the Stylist a summary of the weather,
#    not the raw weather tool output. The Weather Agent
#    generated this summary using the weather information
#    retrieved from the API.

# c. The number of model requests was higher than the
#    visible tool calls because each specialist agent
#    made its own LLM requests. The manager also used
#    additional model calls to coordinate the agents
#    and combine their results into a final answer.
