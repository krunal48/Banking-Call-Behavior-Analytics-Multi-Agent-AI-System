import streamlit as st
import asyncio
import os
import uuid
from datetime import datetime
from manager_agent.agent import manager_agent
from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
import json
import streamlit.components.v1 as com
from streamlit_card import card

from google.adk.events import Event
from google.genai import types

from utils import display_state, Colors

# Load environment variables
load_dotenv()

# --- Application Constants ---
APP_NAME = "Bank Audio Transcript Analyst"
USER_ID = "dedsec995"
DB_URL = "sqlite:///./my_agent_data.db"

# Construct absolute path for uploads
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "uploaded_audio")


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

def display_state_ui(session_state):
    """Renders the session state in a visually appealing way in the UI."""
    with st.expander("View Session State Details", expanded=False):
        st.subheader("Details")
        st.subheader("Intent Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Intent", value=session_state.get("intent_state", "N/A"),)
        with col2:
            st.metric(label="Audio Transcribed?", value = "Yes" if session_state.get("is_audio_transcribed") else "No")
        
        st.subheader("Root Cause Analysis")
        root_cause = session_state.get("root_cause_state", "N/A")
        try:
            root_cause_dict = json.loads(root_cause)
            st.info(root_cause_dict.get('root_cause', 'Not found'))
        except (json.JSONDecodeError, TypeError):
            st.info(root_cause['root_cause'])

        st.subheader("Sentiment Analysis")
        sentiment_state = session_state.get("sentiment_state")

        if isinstance(sentiment_state, str):
            try:
                sentiment_state = json.loads(sentiment_state)
            except json.JSONDecodeError:
                sentiment_state = None
        
        if isinstance(sentiment_state, dict):
            s_col1, s_col2 = st.columns(2)
            s_col1.metric("Overall Sentiment", sentiment_state.get("sentiment_overall", "N/A"))
            s_col2.metric("Overall Score", f'{sentiment_state.get("overall_score", 0.0):.2f}')

            st.write(f"**Granularity:** {sentiment_state.get('granularity', 'N/A')}")

            st.write("**Timeline:**")
            timeline = sentiment_state.get("timeline", [])
            if timeline:
                for entry in timeline:
                    st.write(f"- **{entry.get('minute', 'N/A')}:** {entry.get('label', 'N/A')} (Score: {entry.get('score', 0.0):.2f}, Messages: {entry.get('message_count', 0)})")
            else:
                st.write("No timeline data available.")
        else:
            st.write("Not analyzed.")

        st.subheader("File Information")
        st.info(session_state.get("audio_filepath", "N/A"))

async def log_event(event):
    """Prints event details to the terminal."""
    print(f"Event ID: {event.id}, Author: {event.author}")
    if event.content and event.content.parts:
        for part in event.content.parts:
            if hasattr(part, "text") and part.text and not part.text.isspace():
                print(f"  Text: '{part.text.strip()}'")

async def call_agent_async_ui(runner, session_id, query, chat_placeholder, status_placeholder=None):
    """Call the agent asynchronously and display the response in the UI."""
    print(f"\n{Colors.BG_GREEN}{Colors.BLACK}{Colors.BOLD}--- Running Query: {query} ---{Colors.RESET}")
    await display_state(
        runner.session_service,
        runner.app_name,
        USER_ID,
        session_id,
        "State BEFORE processing",
    )

    # 1. Get the session and append the user query as an event first.
    session = await runner.session_service.get_session(
        app_name=runner.app_name, user_id=USER_ID, session_id=session_id
    )
    session.state["interaction_history"].append({
        "action": "user_query",
        "query": query,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    user_query_event = Event(
        author="user",
        content=types.Content(role="user", parts=[types.Part(text=query)]),
    )
    await runner.session_service.append_event(session=session, event=user_query_event)

    # 2. Run the agent.
    content = types.Content(role="user", parts=[types.Part(text=query)])
    final_response_text = ""
    agent_name = ""
    try:
        async for event in runner.run_async(
            user_id=USER_ID, session_id=session_id, new_message=content
        ):
            await log_event(event)
            if status_placeholder and event.author:
                status_text = f"Running {event.author}..."
                if event.author == "audio_to_transcript_agent":
                    status_text = "Transcribing audio..."
                elif event.author == "IntentAgent":
                    status_text = "Analyzing intent..."
                elif event.author == "sentiment_agent":
                    status_text = "Analyzing sentiment..."
                elif event.author == "root_cause_agent":
                    status_text = "原因 Analyzing root cause..."
                elif event.author == "synthesizer_agent":
                    status_text = "Generating final report..."
                elif event.author == "manager_agent":
                    status_text = "Orchestrating analysis..."
                status_placeholder.text(status_text)

            if event.author:
                agent_name = event.author
            if event.is_final_response() and event.content and event.content.parts:
                final_response_text = event.content.parts[0].text.strip()
                chat_placeholder.markdown(final_response_text)
                if status_placeholder:
                    status_placeholder.empty()

    except Exception as e:
        st.error(f"An error occurred during agent execution: {e}")
        if status_placeholder:
            status_placeholder.empty()
        return None

    # 3. Re-fetch the session and append the agent response.
    if final_response_text and agent_name:
        session = await runner.session_service.get_session(
            app_name=runner.app_name, user_id=USER_ID, session_id=session_id
        )
        session.state["interaction_history"].append({
            "action": "agent_response",
            "agent": agent_name,
            "response": final_response_text,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
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
        USER_ID,
        session_id,
        "State AFTER processing",
    )
    print(f"{Colors.YELLOW}{'-' * 30}{Colors.RESET}")
    return final_response_text

def load_session_callback(session_data):
    """
    Callback function to load a selected session's state and switch to the
    analysis page. This is triggered by on_click from a streamlit-card.
    """
    st.session_state.clear()
    st.session_state.page = "analysis"
    st.session_state.session_id = session_data.id
    st.session_state.audio_path = session_data.state.get("audio_filepath")
    st.session_state.analysis_done = True
    st.session_state.report = session_data.state.get("analysis_report")
    
    chat_history = []
    # Skip the first two interactions (initial prompt and report)
    for interaction in session_data.state.get("interaction_history", [])[2:]:
        if interaction.get("action") == "user_query":
            chat_history.append({"role": "user", "content": interaction.get("query")})
        elif interaction.get("action") == "agent_response":
            chat_history.append({"role": "assistant", "content": interaction.get("response")})
    st.session_state.chat_history = chat_history
    
    st.rerun()

def home_page():
    """Renders the home page for uploading audio files and viewing past sessions."""
    col1, col2 = st.columns([10,7])
    with col1:
        st.title("S.A.G.E. Assistant",anchor=False)
        st.markdown("Every call tells a story, SAGE makes sure you hear the truth behind the tone")
    with col2:
        com.iframe("https://lottie.host/embed/74230abb-884a-444d-92fe-273821e58451/YfEl3zsnUd.lottie", height=100)

    # --- Tile 1: Previous Sessions ---
    with st.container(border=True):
        st.subheader("Previous Wisdoms",anchor=False)
        session_service = DatabaseSessionService(db_url=DB_URL)
        
        try:
            past_sessions = asyncio.run(session_service.list_sessions(app_name=APP_NAME, user_id=USER_ID))
        except Exception as e:
            st.error(f"Could not load past sessions: {e}")
            return

        completed_sessions = [s for s in past_sessions.sessions if s.state.get("analysis_report")]

        if not completed_sessions:
            st.info("No previous analyses found.")
        else:
            cols = st.columns(3)
            for i, session in enumerate(completed_sessions):
                with cols[i % 3]:
                    filename = os.path.basename(session.state.get("audio_filepath", "Unknown File"))
                    card(
                        title=filename,
                        text=f"Analyzed",
                        image="https://cdn-icons-png.flaticon.com/512/1001/1001344.png", 
                        styles={
                            "card": {
                                "width": "100%",
                                "margin": "0px",
                                "box-shadow": "0 0 5px rgba(0,0,0,0.1)",
                                "height": "300px"
                            },
                            "title": {
                                "font-size": "30px",
                                "font-weight": "bold",
                                "white-space": "nowrap",
                                "overflow": "hidden",
                                "text-overflow": "ellipsis"
                            },
                            "text": {
                                "font-size": "14px"
                            }
                        },
                        on_click=lambda s=session: load_session_callback(session_data=s),
                        key=f"card_{session.id}"
                    )

    # --- Tile 2: Upload ---
    with st.container(border=True):
        st.subheader("Start New Analysis")
        uploaded_file = st.file_uploader(
            "Choose an audio file (.wav only)",
            type=["wav"]
        )
        if uploaded_file is not None:
            if not os.path.exists(UPLOAD_DIR):
                os.makedirs(UPLOAD_DIR)
            file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"File '{uploaded_file.name}' uploaded successfully!")
            if st.button("Analyze File"):
                st.session_state.clear()
                st.session_state.page = "analysis"
                st.session_state.audio_path = file_path
                st.session_state.session_id = str(uuid.uuid4())
                st.rerun()

def analysis_page():
    st.title("Analysis Report")

    audio_path = st.session_state.get("audio_path")
    session_id = st.session_state.get("session_id")

    if not audio_path or not session_id:
        st.warning("Please upload an audio file on the Home page first.")
        if st.button("Back to Home"):
            st.session_state.page = "home"
            st.rerun()
        return

    # Initialize session service and runner
    session_service = DatabaseSessionService(db_url=DB_URL)
    runner = Runner(
        agent=manager_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    # --- Layout Setup ---
    left_column, right_column = st.columns([2, 1])
    with left_column:
        st.subheader("Comprehensive Analysis")
        report_placeholder = st.empty()

    # --- Initial Analysis ---
    if "analysis_done" not in st.session_state:
        async def run_analysis():
            session_state = initial_state.copy()
            session_state["audio_filepath"] = audio_path
            await session_service.create_session(
                app_name=APP_NAME,
                user_id=USER_ID,
                session_id=session_id,
                state=session_state,
            )

            with left_column:
                with st.spinner("Starting analysis..."):
                    status_placeholder = st.empty()
                    analysis_report = await call_agent_async_ui(
                        runner, session_id, "Analyze the audio file", report_placeholder, status_placeholder
                    )
            
            if analysis_report:
                st.session_state.report = analysis_report
                st.session_state.chat_history = []
                st.session_state.analysis_done = True
                st.rerun()

        asyncio.run(run_analysis())
        return # Stop execution until analysis is done and page reruns

    # --- Display Report and Chat UI ---
    if st.session_state.get("analysis_done"):
        report_placeholder.markdown(st.session_state.report)

        # Fetch the final state to display it
        session = asyncio.run(runner.session_service.get_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        ))
        
        with left_column:
            display_state_ui(session.state)

        with right_column:
            st.subheader("Follow-up Chat")

            # Display chat messages from history
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Handle follow-up questions
            if prompt := st.chat_input("Ask a follow-up question..."):
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response_placeholder = st.empty()
                        status_placeholder = st.empty()
                        response = asyncio.run(call_agent_async_ui(runner, session_id, prompt, response_placeholder, status_placeholder))
                        if response:
                            st.session_state.chat_history.append({"role": "assistant", "content": response})
                            st.rerun()

    if st.button("Back"):
        # Clear session state for next analysis
        for key in list(st.session_state.keys()):
            if key not in ['page']:
                del st.session_state[key]
        st.session_state.page = "home"
        st.rerun()

# --- Main Application Logic ---
def main():
    st.set_page_config(page_title="SAGE", layout="wide")
    if "page" not in st.session_state:
        st.session_state.page = "home"
    if st.session_state.page == "home":
        home_page()
    elif st.session_state.page == "analysis":
        analysis_page()

if __name__ == "__main__":
    main()
