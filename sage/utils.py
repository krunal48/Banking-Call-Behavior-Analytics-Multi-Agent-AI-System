from datetime import datetime

from google.adk.events import Event
from google.genai import types


class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"





async def display_state(
    session_service, app_name, user_id, session_id, label="Current State"
):
    """Display the current session state in a formatted way."""
    try:
        session = await session_service.get_session(
            app_name=app_name, user_id=user_id, session_id=session_id
        )

        print(f"\n{'-' * 10} {label} {'-' * 10}")

        user_name = session.state.get("user_name", "Unknown")
        print(f"User: {user_name}")

        intent_state = session.state.get("intent_state", "Not analyzed")
        print(f"Intent: {intent_state}")

        sentiment_state = session.state.get("sentiment_state", "Not analyzed")
        print(f"Sentiment: {sentiment_state}")

        root_cause_state = session.state.get("root_cause_state", "Not analyzed")
        print(f"Root Cause: {root_cause_state}")

        is_audio_transcribed = session.state.get("is_audio_transcribed", False)
        print(f"Audio Transcribed: {is_audio_transcribed}")

        analysis_report = session.state.get("analysis_report", "Not generated")
        print(f"Analysis Report: {analysis_report}")

        interaction_history = session.state.get("interaction_history", [])
        if interaction_history:
            print("Interaction History:")
            for idx, interaction in enumerate(interaction_history, 1):
                if isinstance(interaction, dict):
                    action = interaction.get("action", "interaction")
                    timestamp = interaction.get("timestamp", "unknown time")

                    if action == "user_query":
                        query = interaction.get("query", "")
                        print(f'  {idx}. User query at {timestamp}: "{query}"')
                    elif action == "agent_response":
                        agent = interaction.get("agent", "unknown")
                        response = interaction.get("response", "")
                        if len(response) > 100:
                            response = response[:97] + "..."
                        print(f'  {idx}. {agent} response at {timestamp}: "{response}"')
                    else:
                        details = ", ".join(
                            f"{k}: {v}"
                            for k, v in interaction.items()
                            if k not in ["action", "timestamp"]
                        )
                        print(
                            f"  {idx}. {action} at {timestamp}"
                            + (f" ({details})" if details else "")
                        )
                else:
                    print(f"  {idx}. {interaction}")
        else:
            print("Interaction History: None")

        other_keys = [
            k
            for k in session.state.keys()
            if k not in ["user_name", "intent_state", "sentiment_state", "root_cause_state", "is_audio_transcribed", "analysis_report", "interaction_history"]
        ]
        if other_keys:
            print("Additional State:")
            for key in other_keys:
                print(f"  {key}: {session.state[key]}")

        print("-" * (22 + len(label)))
    except Exception as e:
        print(f"Error displaying state: {e}")


async def process_agent_response(event):
    """Process and display agent response events."""
    print(f"Event ID: {event.id}, Author: {event.author}")

    has_specific_part = False
    if event.content and event.content.parts:
        for part in event.content.parts:
            if hasattr(part, "text") and part.text and not part.text.isspace():
                print(f"  Text: '{part.text.strip()}'")

    final_response = None
    if not has_specific_part and event.is_final_response():
        if (
            event.content
            and event.content.parts
            and hasattr(event.content.parts[0], "text")
            and event.content.parts[0].text
        ):
            final_response = event.content.parts[0].text.strip()
            print(
                f"\n{Colors.BG_BLUE}{Colors.WHITE}{Colors.BOLD}╔══ AGENT RESPONSE ═════════════════════════════════════════{Colors.RESET}"
            )
            print(f"{Colors.CYAN}{Colors.BOLD}{final_response}{Colors.RESET}")
            print(
                f"{Colors.BG_BLUE}{Colors.WHITE}{Colors.BOLD}╚═════════════════════════════════════════════════════════════{Colors.RESET}\n"
            )
        else:
            print(
                f"\n{Colors.BG_RED}{Colors.WHITE}{Colors.BOLD}==> Final Agent Response: [No text content in final event]{Colors.RESET}\n"
            )

    return final_response


async def call_agent_async(runner, user_id, session_id, query):
    """Call the agent asynchronously with the user's query."""
    session = await runner.session_service.get_session(
        app_name=runner.app_name, user_id=user_id, session_id=session_id
    )
    session.state["interaction_history"].append(
        {
            "action": "user_query",
            "query": query,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )

    user_query_event = Event(
        author="user",
        content=types.Content(role="user", parts=[types.Part(text=query)]),
    )
    await runner.session_service.append_event(session=session, event=user_query_event)

    content = types.Content(role="user", parts=[types.Part(text=query)])
    print(
        f"\n{Colors.BG_GREEN}{Colors.BLACK}{Colors.BOLD}--- Running Query: {query} ---{Colors.RESET}"
    )
    final_response_text = None
    agent_name = None

    await display_state(
        runner.session_service,
        runner.app_name,
        user_id,
        session_id,
        "State BEFORE processing",
    )

    try:
        async for event in runner.run_async(
            user_id=user_id, session_id=session_id, new_message=content
        ):
            if event.author:
                agent_name = event.author

            response = await process_agent_response(event)
            if response:
                final_response_text = response
    except Exception as e:
        print(f"{Colors.BG_RED}{Colors.WHITE}ERROR during agent run: {e}{Colors.RESET}")

    if final_response_text and agent_name:
        session = await runner.session_service.get_session(
            app_name=runner.app_name, user_id=user_id, session_id=session_id
        )
        session.state["interaction_history"].append(
            {
                "action": "agent_response",
                "agent": agent_name,
                "response": final_response_text,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

        agent_response_event = Event(
            author=agent_name,
            content=types.Content(
                role="model", parts=[types.Part(text=final_response_text)]
            ),
        )
        await runner.session_service.append_event(
            session=session, event=agent_response_event
        )

    await display_state(
        runner.session_service,
        runner.app_name,
        user_id,
        session_id,
        "State AFTER processing",
    )

    print(f"{Colors.YELLOW}{'-' * 30}{Colors.RESET}")
    return final_response_text
