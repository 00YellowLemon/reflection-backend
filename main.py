from fastapi import FastAPI
from models import MsgPayload
from agents import run_agent_with_checkpointer
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

messages_list: dict[int, MsgPayload] = {}

# Define the request schema for the agent endpoint
class AgentRequest(BaseModel):
    input_text: str
    thread_id: str


# Agent endpoint
@app.post("/agent")
async def invoke_agent(request: AgentRequest) -> dict:
    """Invoke the configured agent with input text and thread ID"""
    result = await run_agent_with_checkpointer(request.input_text, request.thread_id)
    return {"result": result}
