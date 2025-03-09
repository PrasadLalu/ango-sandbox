from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools

# Load env vars
load_dotenv()

agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    description="You are an AI assistant. Please reply the user query.",
    markdown=True,
    tools=[DuckDuckGoTools()],
)

agent.print_response("Tell me the recent sport news of April, 2025")
