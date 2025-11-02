import json
import re
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.tool_context import ToolContext
import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemma-3-27b-it')

def safe_parse_json(raw):
    """Safely parse model output even if wrapped in markdown."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None

def analyze_root_cause(tool_context: ToolContext) -> dict:
    """
    Analyzes the transcript to identify the root cause of the user's issue.

    Args:
        tool_context (ToolContext): The tool context containing the transcript.

    Returns:
        dict: A dictionary containing the identified root cause.
    """
    transcript = tool_context.state.get("transcript")
    if not transcript:
        return {"error": "Transcript not found in state."}

    full_text = " ".join(segment[3] for segment in transcript)

    prompt = f"""Analyze the following conversation from a customer service call and identify the root cause of the customer's issue. 
    The root cause should be a concise summary of the underlying problem.
    
    Conversation:
    {full_text}
    
    Respond with a JSON object with a single key 'root_cause'.
    """

    response = model.generate_content(prompt)
    try:
        root_cause = safe_parse_json(response.text)
    except Exception as e:
        root_cause = {"error": str(e)}

    tool_context.state["root_cause_state"] = root_cause
    return {"root_cause": root_cause}

root_cause_agent = Agent(
    name="root_cause_agent",
    model=LiteLlm(model="openai/gpt-4o"),
    description="Identifies the root cause of the user's issue from the transcript.",
    instruction="""
    You are an expert in root cause analysis for customer service calls.
    Your task is to analyze the provided call transcript to identify the underlying problem or recurring pain points.
    You have access to the 'analyze_root_cause' tool. Call this tool to perform the analysis and save the result to the state.
    
    """,
    tools=[analyze_root_cause],
)
