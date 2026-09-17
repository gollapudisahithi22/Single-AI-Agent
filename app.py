import os
import requests
import streamlit as st

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain.tools import tool
from langchain.agents import create_agent

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Agent",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Single AI Agent")
st.write("Ask me anything! I can search the web and check weather.")

# -----------------------------
# Tavily Search Tool
# -----------------------------
search_tool = TavilySearch(max_results=3)


# -----------------------------
# Weather Tool
# -----------------------------
@tool
def get_weather(city: str) -> str:
    """Fetch current weather information for a city."""

    if not WEATHERSTACK_API_KEY:
        return "WEATHERSTACK_API_KEY is not set."

    url = (
        "http://api.weatherstack.com/current"
        f"?access_key={WEATHERSTACK_API_KEY}"
        f"&query={city}"
    )

    try:
        response = requests.get(url)
        data = response.json()

        if "current" not in data:
            return f"Could not fetch weather data for {city}."

        temperature = data["current"]["temperature"]
        weather = data["current"]["weather_descriptions"][0]
        humidity = data["current"]["humidity"]

        return (
            f"City: {city}\n"
            f"Temperature: {temperature}°C\n"
            f"Weather: {weather}\n"
            f"Humidity: {humidity}%"
        )

    except Exception as e:
        return f"Weather error: {str(e)}"


# -----------------------------
# Create Groq LLM
# -----------------------------
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=GROQ_API_KEY
)

# -----------------------------
# Create Agent
# -----------------------------
tools = [search_tool, get_weather]

agent = create_agent(
    model=llm,
    tools=tools
)


# -----------------------------
# Chat History
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -----------------------------
# User Input
# -----------------------------
user_input = st.chat_input("Ask your question...")

if user_input:

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Get AI response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:
                response = agent.invoke({
                    "messages": [
                        {
                            "role": "user",
                            "content": user_input
                        }
                    ]
                })

                answer = response["messages"][-1].content

            except Exception as e:
                answer = f"❌ Error: {str(e)}"

        st.markdown(answer)

    # Save response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })