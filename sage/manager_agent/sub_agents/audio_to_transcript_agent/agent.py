from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.tool_context import ToolContext
from openai import OpenAI
from dotenv import load_dotenv
import os
import torch
import whisper
import torchaudio
from pyannote.audio import Pipeline
from pyannote.core import Segment

load_dotenv()

HF_TOKEN = os.getenv('HF_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def transcribe_audio(tool_context: ToolContext) -> dict:
    """
    Transcribes an audio file and performs speaker diarization.

    Args:
        tool_context (ToolContext): The tool context containing the audio filepath.

    Returns:
        dict: A dictionary containing the 'transcript' key with a list of 
              [start_time, end_time, speaker_id, text] segments.
    """
    audio_filepath = tool_context.state.get("audio_filepath")
    if not audio_filepath:
        raise Exception("error: Audio filepath not found in state. Stopping workflow.")
        return {"error": "Audio filepath not found in state."}

    if not OPENAI_API_KEY:
        return {"error": "OPENAI_API_KEY not found in environment."}

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        with open(audio_filepath, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="gpt-4o-transcribe-diarize",
                file=audio_file,
                response_format="diarized_json",
                chunking_strategy="auto",
            )
        modified_output = [
            [segment.start, segment.end, segment.speaker, segment.text.strip()]
            for segment in transcript.segments
        ]
        tool_context.state["is_audio_transcribed"] = True
        tool_context.state['transcript'] = modified_output
        return {'transcript': modified_output}
    except FileNotFoundError:
        return {"error": f"Audio file not found at path: {audio_filepath}"}
    except Exception as e:
        return {"error": f"An error occurred during transcription: {e}"}


# def transcribe_audio(tool_context):    
#     """
#     Transcribes an audio file and performs speaker diarization.

#     Args:
#         tool_context (ToolContext): The tool context containing the audio filepath.

#     Returns:
#         dict: A dictionary containing the 'transcript' key with a list of 
#               [start_time, end_time, speaker_id, text] segments.
#     """

#     audio_path = tool_context.state.get("audio_filepath")
#     if not audio_path:
#         raise Exception("error: Audio filepath not found in state. Stopping workflow.")
#     print(f"The file path: {audio_path}")
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     try:
#         pipeline = Pipeline.from_pretrained(
#             "pyannote/speaker-diarization-3.1",
#             use_auth_token=HF_TOKEN
#         )
#         pipeline.to(torch.device(device))
#     except Exception as e:
#         return {'error': "Failed to set up the pipeline"} 
    
#     diarization = pipeline(audio_path, num_speakers=2)
#     whisper_model = whisper.load_model("base", device=device)
    
#     try:
#         audio_waveform = whisper.load_audio(audio_path)
#         sample_rate = whisper.audio.SAMPLE_RATE
#     except Exception as e:
#         return {'error': "Failed to transform in whisper compatiable form"} 

    
#     all_segments = []
#     for segment, track_id, label in diarization.itertracks(yield_label=True):
#         all_segments.append({
#             'start': segment.start,
#             'end': segment.end,
#             'label': label
#         })
    
#     if not all_segments:
#         return {'error': "Not all segments present"} 
        
#     all_segments.sort(key=lambda x: x['start'])
    
#     merged_segments = []
#     current_segment = all_segments[0].copy()

#     for next_seg in all_segments[1:]:
#         if (next_seg['label'] == current_segment['label'] and 
#             next_seg['start'] - current_segment['end'] < 0.1):
#             current_segment['end'] = next_seg['end']
#         else:
#             merged_segments.append(current_segment)
#             current_segment = next_seg.copy()
    
#     merged_segments.append(current_segment)

#     final_output_list = []
#     for i, segment in enumerate(merged_segments):
#         start_time = segment['start']
#         end_time = segment['end']
#         label = segment['label']
        
#         start_sample = int(start_time * sample_rate)
#         end_sample = int(end_time * sample_rate)
        
#         segment_audio = audio_waveform[start_sample:min(end_sample, len(audio_waveform))]

#         result = whisper_model.transcribe(segment_audio, fp16=torch.cuda.is_available())
#         text = result['text'].strip()

#         if text:
#             final_output_list.append([start_time, end_time, label, text])

#     tool_context.state["is_audio_transcribed"] = True
#     tool_context.state['transcript'] = final_output_list
    
#     return {'transcript': final_output_list}


audio_to_transcript_agent = Agent(
    name="audio_to_transcript_agent",
    model="gemma-3-27b-it",
    # model=LiteLlm(model="openai/gpt-4o"),
    description="Transcribes an audio file and returns the transcript with diarization.",
    instruction="""
    You are the audio to transcript agent. Your role is to transcribe an audio file using the tool transcribe_audio.
    You have access to the following tools:
    - transcribe_audio
    """,
    tools=[transcribe_audio],
)
