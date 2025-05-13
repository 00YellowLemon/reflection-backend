from fastapi import FastAPI
from agents import raw_agent  # Removed: run_agent_with_checkpointer, set_compiled_app
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
from langgraph.checkpoint.mongodb import MongoDBSaver
from dotenv import load_dotenv
from pymongo import MongoClient  # Add this import
import logging

# Load environment variables from .env file
load_dotenv()

# Verify MongoDB URI
mongodb_uri = os.getenv("MONGODB_URI")
if not mongodb_uri:
    raise RuntimeError("MONGODB_URI not found. Ensure it is set in the .env file.")

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()  # Removed: lifespan=lifespan

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the request schema for the agent endpoint


class AgentRequest(BaseModel):
    input_text: str
    thread_id: str


# Agent endpoint
@app.post("/agent")
def invoke_agent_endpoint(request: AgentRequest) -> dict:  # Changed to sync
    """Invoke the configured agent with input text and thread ID"""
    try:
        # Ensure MONGODB_URI is available (should be, as it's checked at startup)
        current_mongodb_uri = os.getenv("MONGODB_URI")
        if not current_mongodb_uri:
            # This case should ideally not be hit if initial check passed
            return {"error": "MONGODB_URI not configured"}

        # Ensure the database and collection exist
        client = MongoClient(current_mongodb_uri)
        db = client["langgraph_checkpoints"]
        if "checkpoints" not in db.list_collection_names():
            db.create_collection("checkpoints")

        with MongoDBSaver.from_conn_string(
            current_mongodb_uri
        ) as checkpointer:
            # raw_agent is assumed to be a StateGraph builder instance from agents.py
            graph = raw_agent.compile(checkpointer=checkpointer)

            config = {"configurable": {"thread_id": request.thread_id}}

            # Assuming the agent graph expects input like: {"messages": [{"role": "user", "content": "..."}]}
            # and its state contains a "messages" list.
            input_data = {"messages": [{"role": "user", "content": request.input_text}]}

            final_message_output = None
            # Iterate through the stream to get the final state or last relevant message.
            # stream_mode="values" gives the full state at each step.
            for chunk in graph.stream(input_data, config, stream_mode="values"):
                # Assuming 'chunk' is the state, which could be a dict or an object.
                # We look for a 'messages' attribute or key, and take the last message.
                current_messages = None
                if isinstance(chunk, dict) and "messages" in chunk:
                    current_messages = chunk["messages"]
                elif hasattr(chunk, "messages"):
                    current_messages = chunk.messages

                if current_messages and isinstance(current_messages, list) and len(current_messages) > 0:
                    final_message_output = current_messages[-1]

            result_content = "Agent did not produce a final message or the format was unexpected."
            if final_message_output:
                if hasattr(final_message_output, "content"):
                    result_content = final_message_output.content
                elif isinstance(final_message_output, dict) and "content" in final_message_output:
                    result_content = final_message_output["content"]
                else:
                    result_content = str(final_message_output)  # Fallback if content attribute is not found

        return {"result": result_content}

    except Exception as e:
        logging.error("Error occurred in invoke_agent_endpoint", exc_info=True)
        return {"error": "An unexpected error occurred. Please check the logs for details."}
