from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_swarm

from dotenv import load_dotenv
import os

from prompts import action_prompt, encouragement_prompt, insight_prompt, daily_reflection_prompt
from handoffs import to_insight, to_action, to_encouragement

# Load environment variables first
load_dotenv()

# Get the Gemini API key from the environment variables
gemini_api_key = os.getenv("GOOGLE_API_KEY")

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=gemini_api_key)

reflecty = create_react_agent(
    model = model,
    tools = [to_insight],
    prompt = daily_reflection_prompt,
    name = "reflecty"
)

insight = create_react_agent(
    model = model,
    tools = [to_action, to_encouragement],
    prompt = insight_prompt,
    name = "insight",
)

action = create_react_agent(
    model = model,
    tools = [to_insight],
    prompt = action_prompt,
    name = "action"
)

encouragement = create_react_agent(
    model = model,
    tools = [to_insight],
    prompt = encouragement_prompt,
    name = "encouragement"
)

# Create the swarm agent (uncompiled)
# This will be compiled in main.py using the checkpointer
raw_agent = create_swarm(
    [encouragement, reflecty, insight, action],
    default_active_agent="reflecty"
)

# Global variable to hold the compiled app instance
_compiled_app_instance = None

def set_compiled_app(compiled_app):
    global _compiled_app_instance
    _compiled_app_instance = compiled_app

def get_compiled_app():
    if _compiled_app_instance is None:
        raise RuntimeError("Compiled app not initialized. Ensure main.py lifespan sets it.")
    return _compiled_app_instance

# Example of how to use the agent asynchronously
async def run_agent_with_checkpointer(input_text: str, thread_id: str): # Made async
    """
    Run the agent with proper checkpointing using the synchronous invoke method.
    """
    compiled_app = get_compiled_app()
    config = {"configurable": {"thread_id": thread_id}}
    # If compiled_app.invoke is synchronous, FastAPI runs it in a threadpool.
    # If an async version like compiled_app.ainvoke exists, prefer that.
    result = compiled_app.invoke({"input": input_text}, config)
    return result



