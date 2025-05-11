from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_handoff_tool, create_swarm
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.mongodb import  MongoDBSaver

from dotenv import load_dotenv
import os

from prompts import action_prompt, encouragement_prompt, insight_prompt, daily_reflection_prompt
from handoffs import to_insight, to_action, to_encouragement

# Load environment variables first
load_dotenv()

# Get the MongoDB URI from the environment variables
mongodb_uri = os.getenv("MONGODB_URI")

# Get the Gemini API key from the environment variables
gemini_api_key = os.getenv("GOOGLE_API_KEY")

# Initialize checkpointer with the URI from .env
checkpointer = MongoDBSaver.from_conn_string(
    mongodb_uri,
    database="langgraph_checkpoints",
    collection="checkpoints"
)

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

# Create the swarm agent
agent = create_swarm(
    [encouragement, reflecty, insight, action],
    default_active_agent="reflecty"
)

# Compile the agent with the checkpointer
app = agent.compile(checkpointer=checkpointer)

# config = {"configurable": {"thread_id": "1"}}

# Example of how to use the agent asynchronously
def run_agent_with_checkpointer(input_text: str, thread_id: str):
    """
    Run the agent with proper checkpointing using the synchronous invoke method.
    """
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke({"input": input_text}, config)
    return result

# Example usage (commented out)
# asyncio.run(run_agent_with_checkpointer("What should I reflect on today?"))



