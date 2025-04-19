from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_handoff_tool, create_swarm
from langgraph.checkpoint.memory import InMemorySaver


from dotenv import load_dotenv

from prompts import action_prompt, encouragement_prompt, insight_prompt, daily_reflection_prompt
from handoffs import to_insight, to_action, to_encouragement

import os

# initialize checkpointer



load_dotenv()

# Get the Gemini API key from the environment variables
gemini_api_key = os.getenv("GOOGLE_API_KEY")

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=gemini_api_key)

checkpointer = InMemorySaver()


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

workflow = create_swarm(
    [encouragement, action, insight, reflecty],
    default_active_agent="reflecty",
)
app = workflow.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "1"}}


turn_1 = app.invoke(
    {"messages": [{"role": "user", "content": "i'd like to start my daily reflection"}]},
    config,
)






