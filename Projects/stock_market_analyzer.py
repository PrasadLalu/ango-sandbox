import streamlit as st
from typing import Iterator
from dotenv import load_dotenv
from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools

# Load env vars
load_dotenv()

# Agent: Search web information
web_agent = Agent(
    name="Web Agent",
    role="Search the web for information",
    model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGoTools()],
    instructions="Always include sources",
    markdown=True,
)

# Agent: Search Finance information
finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    model=OpenAIChat(id="gpt-4o"),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            company_info=True,
        )
    ],
    instructions="Use tables to display data",
    markdown=True,
)

# Team Agent combining Web and Finance Agents
agent_team = Agent(
    team=[web_agent, finance_agent],
    model=OpenAIChat(id="gpt-4o"),
    instructions=["Always include sources", "Use tables to display data"],
    markdown=True,
)

# Initialize chat history in session state if not exists
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Streamlit UI
st.title("Stock Market Analyzer 📈")
query = st.text_area("Enter your query:")

if st.button("Analyze") and query:
    with st.spinner("Thinking..."):
        response_stream: Iterator[RunResponse] = agent_team.run(query, stream=True)
        response_text = ""
        placeholder = st.empty()

        for chunk in response_stream:
            response_text += chunk.content
            placeholder.markdown(response_text + "▌")

        placeholder.markdown(response_text)
else:
    st.warning("Please enter your query...")

# if st.button("Analyze") and query:
#     with st.spinner("Fetching answer..."):
#         response_text = ""
#         placeholder = st.empty()

#         # Append user query to history
#         st.session_state.chat_history.append({"role": "user", "content": query})

#         for chunk in agent_team.run(query, stream=True, history=st.session_state.chat_history):
#             response_text += chunk.content
#             placeholder.markdown(response_text + "▌")

#         placeholder.markdown(response_text)

#         # Append AI response to history
#         st.session_state.chat_history.append({"role": "assistant", "content": response_text})

# # Display chat history
# st.subheader("Chat History")
# for message in st.session_state.chat_history:
#     role = "🧑‍💼 User" if message["role"] == "user" else "🤖 AI"
#     st.markdown(f"**{role}:** {message['content']}")
