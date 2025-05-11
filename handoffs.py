from langgraph_swarm import create_handoff_tool


# Handoff from the system/graph controller to the Insight Generator after collecting user reflections
to_insight =create_handoff_tool(
    agent_name="insight",
    description="Transfer the collected user reflection text to the Insight Generator, which specializes in analyzing the text to extract key themes, patterns, and takeaways."
)

# Handoff from Insight Generator to Encouragement and Support Agent
to_encouragement = create_handoff_tool(
    agent_name="encouragement",
    description="Transfer the synthesized insights and reflection summary to the Encouragement and Support agent, designed to provide empathetic validation and positive reinforcement based on the user's shared experiences."
)

# Handoff from Insight Generator to Action-Oriented Advisor Agent
to_action = create_handoff_tool(
    agent_name="action",
    description="Transfer the synthesized insights and reflection summary to the Action-Oriented Advisor, which excels at suggesting practical, small, and relevant next steps based on the reflection analysis."
)

# Optional: Handoff to initiate the process (if applicable to your graph start)
# This might represent the graph routing the initial request or user input.
