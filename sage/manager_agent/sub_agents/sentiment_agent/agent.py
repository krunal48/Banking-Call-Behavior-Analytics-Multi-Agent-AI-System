from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.tool_context import ToolContext
import google.generativeai as genai
import math
import os
import json
import re
from collections import defaultdict, Counter
from litellm import completion
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

def analyze_sentiment_per_minute(tool_context: ToolContext) -> dict:
    """
    Analyzes the emotional tone and satisfaction level of the transcript per minute and saves it to the state.
    """
    transcript = tool_context.state.get("transcript")
    if not transcript:
        return {"error": "Transcript not found in state."}

    minute_buckets = defaultdict(list)
    for entry in transcript:
        start_t, end_t, speaker, text = entry
        minute_index = int(math.floor(start_t / 60))
        minute_buckets[minute_index].append((speaker, text))

    minute_summary = []
    for minute, msgs in sorted(minute_buckets.items()):
        combined_text = " ".join([f"{speaker}: {text}" for speaker, text in msgs])

        resp = completion(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise emotion detection model for customer conversations. "
                        "Analyze the following 1-minute transcript and identify the *dominant emotion* clearly. "
                        "Differentiate carefully between: "
                        "Anger (aggressive, raised voice), "
                        "Frustration (annoyed or impatient tone), "
                        "Calm (neutral or polite tone), "
                        "Apology (expressing regret), and "
                        "Satisfaction (happy or thankful tone). "
                        "Return only JSON: {\"label\": <emotion>, \"score\": <0-1>}."
                    ),
                },
                {"role": "user", "content": combined_text},
            ],
        )

        raw = resp["choices"][0]["message"]["content"]
        parsed = safe_parse_json(raw)
        label = parsed.get("label", "neutral") if parsed else "neutral"
        score = float(parsed.get("score", 0.5)) if parsed else 0.5
        
        minute_label = f"{minute} to {minute + 1}"
        minute_summary.append({
            "minute": minute_label,
            "label": label,
            "score": round(score, 2),
            "message_count": len(msgs)
        })

    label_counts = Counter(m["label"] for m in minute_summary)
    score_totals = defaultdict(float)
    for m in minute_summary:
        score_totals[m["label"]] += m["score"]

    avg_scores = {l: score_totals[l] / label_counts[l] for l in label_counts}
    overall_label = max(label_counts, key=label_counts.get)
    overall_score = round(avg_scores[overall_label], 2)

    result = {
        "sentiment_overall": overall_label,
        "overall_score": overall_score,
        "granularity": "1-minute",
        "timeline": minute_summary
    }

    tool_context.state["sentiment_state"] = result
    return result

sentiment_agent = Agent(
    name="sentiment_agent",
    model=LiteLlm(model="openai/gpt-4o"),
    description="Analyzes the emotional tone and satisfaction levels in the transcript.",
    instruction="""
    You are a sentiment analysis expert specializing in customer service calls.
    Your task is to analyze the provided call transcript to identify the emotional tone and satisfaction levels for each minute of the call.
    You have access to the 'analyze_sentiment_per_minute' tool. Call this tool to perform the analysis and save the results to the state.
    
    """,
    tools=[analyze_sentiment_per_minute],
)