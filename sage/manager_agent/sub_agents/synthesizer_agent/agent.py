from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.tool_context import ToolContext
import google.generativeai as genai
import os
import json
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-flash')

def generate_summary_report(tool_context: ToolContext) -> dict:
    """
    Generates a final summary report based on the analysis from other agents.

    Args:
        tool_context (ToolContext): The tool context containing the analysis results.

    Returns:
        dict: A dictionary containing the final summary report.
    """
    intent = tool_context.state.get("intent_state", "Not available")
    root_cause = tool_context.state.get("root_cause_state", "Not available")
    sentiment_details = tool_context.state.get("sentiment_state", [])
    transcript = tool_context.state.get("transcript", [])

    prompt = f"""Generate a comprehensive summary report for the following customer service call.
    The report should be well-structured and include the following sections:
    1.  **Intent:** The customer's primary reason for calling.
    2.  **Root Cause:** The underlying issue or problem.
    3.  **Sentiment Analysis:** A summary of the emotional tone and satisfaction levels throughout the call.
    4.  **Call Transcript:** A summary of the conversation.

    **Intent:** {intent}
    **Root Cause:** {root_cause}
    **Sentiment Details:** {json.dumps(sentiment_details, indent=2)}
    **Transcript:**
    {"\n".join([f"{s[2]}: {s[3]}" for s in transcript])}

    Generate a detailed report based on this information.
    """

    response = model.generate_content(prompt)
    summary = response.text.strip()

    tool_context.state["analysis_report"] = summary
    return {"analysis_report": summary}

synthesizer_agent = Agent(
    name="synthesizer_agent",
    model=LiteLlm(model="openai/gpt-4o"),
    description="Synthesizes the analysis from other agents into a final report and handles follow-up questions.",
    instruction="""
    You are the Smart Agent. Your primary role is to generate a final and answer any question user might have. Comprehensive report by synthesizing the analysis from the intent, sentiment, and root cause agents.
    If Analysis Report: {analysis_report} is None then always generate a summary report using the tool you have.
    Answer the any question the user have based on {intent_state}, {sentiment_state}, {root_cause_state} and {analysis_report}.
    
    You have access to the following tools:
    - `generate_summary_report`: Call this tool to generate the final report.
    """,
    tools=[generate_summary_report]
)
