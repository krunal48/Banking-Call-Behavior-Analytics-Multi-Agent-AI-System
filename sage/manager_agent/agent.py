from google.adk.agents import Agent, SequentialAgent, ParallelAgent
from google.adk.tools.tool_context import ToolContext
from google.adk.models.lite_llm import LiteLlm
from .sub_agents.intent_agent.agent import intent_agent
from .sub_agents.sentiment_agent.agent import sentiment_agent
from .sub_agents.root_cause_agent.agent import root_cause_agent
from .sub_agents.audio_to_transcript_agent.agent import audio_to_transcript_agent
from .sub_agents.synthesizer_agent.agent import synthesizer_agent
from dotenv import load_dotenv

load_dotenv()

def set_filepath(tool_context: ToolContext, filepath: str) -> dict:
    """
    Sets the audio filepath in the state.

    Args:
        tool_context (ToolContext): The tool context.
        filepath (str): The path to the audio file.

    Returns:
        dict: A dictionary confirming the filepath has been set.
    """
    tool_context.state["audio_filepath"] = filepath
    return {"status": f"Filepath set to {filepath}"}

# Define the main workflow as a SequentialAgent
sage_workflow = SequentialAgent(
    name="sage_workflow",
    sub_agents=[
        audio_to_transcript_agent,
        ParallelAgent(
            name="analysis_agents",
            sub_agents=[
                intent_agent,
                sentiment_agent,
                root_cause_agent,
            ]
        ),
        synthesizer_agent,
    ]
)

manager_agent = Agent(
    name="manager_agent",
    model=LiteLlm(model="openai/gpt-4o"),
    description="Manager agent for the bank audio transcript analysis system.",
    instruction="""
    You are Sage, a friendly and intelligent AI assistant for analyzing bank audio transcripts.
    Your primary role is to manage a team of specialized agents to provide a comprehensive analysis of customer service calls.
    The state `audio_filepath` : {audio_filepath}
    If the `audio_filepath` is set in the state and the analysis has not been done yet, call the `sage_workflow` agent to perform the analysis.
    Otherwise, you can chat with the user and answer their questions.
    """,
    sub_agents=[sage_workflow],
    tools=[set_filepath],
)
