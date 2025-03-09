import streamlit as st
from dotenv import load_dotenv
from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
from agno.embedder.openai import OpenAIEmbedder
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.vectordb.lancedb import LanceDb, SearchType
from typing import Iterator

# Load environment variables
load_dotenv()

# Set up the agent
agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    description="You are a Thai cuisine expert!",
    instructions=[
        "Search your knowledge base for Thai recipes.",
        "If the question is better suited for the web, search the web to fill in gaps.",
        "Prefer the information in your knowledge base over the web results.",
    ],
    knowledge=PDFUrlKnowledgeBase(
        urls=["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
        vector_db=LanceDb(
            uri="tmp/lancedb",
            table_name="recipes",
            search_type=SearchType.hybrid,
            embedder=OpenAIEmbedder(id="text-embedding-3-small"),
        ),
    ),
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    markdown=True,
)

# Streamlit UI
st.title("Thai Cuisine AI Expert")

query = st.text_input("Enter your query:")
stream = st.checkbox("Stream Response", value=True)

if st.button("Get Recipe") and query:
    with st.spinner("Fetching answer..."):
        if stream:
            response_stream: Iterator[RunResponse] = agent.run(query, stream=True)
            response_text = ""
            placeholder = st.empty()
            
            for chunk in response_stream:
                response_text += chunk.content
                placeholder.markdown(response_text + "▌")
            
            placeholder.markdown(response_text)
        else:
            response = agent.run(query, stream=False)
            st.markdown(response.content)
