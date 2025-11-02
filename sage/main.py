import asyncio

# Import the main manager agent
from manager_agent.agent import manager_agent
from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from utils import call_agent_async

load_dotenv()

# ===== PART 1: Initialize Persistent Session Service =====
# Using SQLite database for persistent storage
db_url = "sqlite:///./my_agent_data.db"
session_service = DatabaseSessionService(db_url=db_url)


# ===== PART 2: Define Initial State =====
# This will only be used when creating a new session
initial_state = {
    "user_name": "Amit Luhar",
    "intent_state": None,
    "sentiment_state": None,
    "root_cause_state": None,
    "is_audio_transcribed": False,
    "audio_filepath": None,
    "transcript": [],
    "analysis_report": None,
    "interaction_history": [],
}


async def main_async():
    # Setup constants
    APP_NAME = "Bank Audio Transcript Analyst"
    USER_ID = "dedsec995"

    # ===== PART 3: Session Management - Find or Create =====
    # Check for existing sessions for this user
    existing_sessions = await session_service.list_sessions(
        app_name=APP_NAME,
        user_id=USER_ID,
    )

    # If there's an existing session, use it, otherwise create a new one
    if existing_sessions and len(existing_sessions.sessions) > 0:
        # Use the most recent session
        SESSION_ID = existing_sessions.sessions[0].id
        print(f"Continuing existing session: {SESSION_ID}")
    else:
        # Create a new session with initial state
        new_session = await session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            state=initial_state,
        )
        SESSION_ID = new_session.id
        print(f"Created new session: {SESSION_ID}")

    # ===== PART 4: Agent Runner Setup =====
    # Create a runner with the main manager agent
    runner = Runner(
        agent=manager_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    # ===== PART 5: Interactive Conversation Loop =====
    print("\nWelcome to the Bank Audio Transcript Analyst!")
    print("Your progress will be saved and remembered across conversations.")
    print("Type 'exit' or 'quit' to end the conversation.\n")

    while True:
        # Get user input
        user_input = input("You: ")

        # Check if user wants to exit
        if user_input.lower() in ["exit", "quit"]:
            print("Ending conversation. Your data has been saved to the database.")
            break

        # Process the user query through the agent
        await call_agent_async(runner, USER_ID, SESSION_ID, user_input)

    # ===== PART 6: State Examination =====
    # Show final session state
    final_session = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )
    print("\nFinal Session State:")
    for key, value in final_session.state.items():
        print(f"{key}: {value}")


def main():
    """Entry point for the application."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
