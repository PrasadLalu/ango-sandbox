from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.openai import OpenAIChat

# Load env vars
load_dotenv()

agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    description="You are an AI assistant. Please reply the user query.",
    markdown=True,  # optional
    debug_mode=True,  # To debug steps(optional)
)

agent.print_response("What is generative AI?")
