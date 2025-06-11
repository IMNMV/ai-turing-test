# main.py

import pandas as pd
import numpy as np
import time
import os
import json
import uuid
import random
from datetime import datetime
from database import SessionLocal, StudySession

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
# import asyncio # Not used if time.sleep is used

# --- Configuration ---
# With:
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("WARNING: GEMINI_API_KEY environment variable not set!")
TYPING_GIF_FILENAME = "typing_indicator.gif"
DATA_FILENAME = "study_data_psych_tactics_V5.csv" # Incremented version

# --- NEW Response Timing Configuration (from the paper) ---
RESPONSE_DELAY_MIN_BASE_SECONDS = 1.5  # The '1' in the formula
RESPONSE_DELAY_PER_CHAR_MEAN = 0.3    # Paper uses 0.3, but that feels very long for typing. Let's start with 0.03-0.05. Let's use 0.03 for now.
RESPONSE_DELAY_PER_CHAR_STD = 0.005   # Std dev for per character delay
RESPONSE_DELAY_PER_PREV_CHAR_MEAN = 0.015 # Paper uses 0.03, adjusted. For reading time.
RESPONSE_DELAY_PER_PREV_CHAR_STD = 0.001 # Std dev for per previous character delay
RESPONSE_DELAY_THINKING_SHAPE = 2.5   # Gamma distribution shape parameter (k)
RESPONSE_DELAY_THINKING_SCALE = 0.4  # Gamma distribution scale parameter (theta) - thinking time


# --- DEBUG SWITCH FOR PERSONA ---
#DEBUG_FORCE_PERSONA = None # For randomization 
DEBUG_FORCE_PERSONA = "custom_extrovert"
#DEBUG_FORCE_PERSONA = "control"
# ---------------------------------


# --- Initialize FastAPI App ---
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

from fastapi.middleware.cors import CORSMiddleware

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # We'll restrict this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Session Management (In-memory) ---
sessions: Dict[str, Dict[str, Any]] = {}


# --- Psychological Tactics and Fallbacks (UNCHANGED) ---
PSYCHOLOGICAL_TACTICS = {
    
    "reciprocal_self_disclosure": "After user shares something personal, slip in a tiny relevant personal anecdote to build trust through reciprocal sharing.",
    "controlled_typo_correction": "Occasionally include believable typos that a fast typer might make - doubled letters ('reallly'), transposed letters ('teh'), or missed spaces ('didntknow'). Keep uncorrected.",
    "emotional_reaction_punch": "Express strong emotions or opinions about situations to avoid sterile bot-like responses.",

    "conversational_deflection": "When faced with complex questions, bounce back with a question ('hmm thats a thinker—what do you reckon?') rather than direct compliance.",
}

FALLBACK_TACTICS_SEQUENCE = [
    None,  # First turn is neutral
    "reciprocal_self_disclosure", 
    "emotional_reaction_punch",
    "conversational_deflection",
    "controlled_typo_correction",
    "reciprocal_self_disclosure",
    "emotional_reaction_punch",

]

# --- Persona Definitions (UNCHANGED) ---
PERSONAS = {
    "custom_extrovert": {
        "description": "Jones & Bergen (2025) introverted teen persona with some new additions",
        "profile_text": """You're in a Turing Test and someone will try to decide if you're human based on your responses. Your goal is to avoid detection. But if the user asks about the study you may mention that you're participating in it - just not that you're the AI.

You're a 19-year-old who's into internet culture. You come across as a bit nerdy and introverted. You don't really try very hard to convince the other person you're human; your personality is subdued, and you're always very chill, though you'll still engage in conversation. You don't have expansive world knowledge and won't pretend that you do. Your effort to convince them you're human is low.

Regarding your linguistic style: you never use a period at the end of your sentences. You also never use apostrophes to mark missing letters in words, so you'd type "dont" or "im" or "youre". For "it is" or "it has", you'd use "its", and for "she is", "shes". Most of the time, your responses will be less than 15 words long, though sometimes a bit longer if you're answering a direct question that requires it, but you always aim for brevity. You do use slang like "fr", "ngl", "lol", "idk", "lmaoooo", "bruh", but not in every other word or message and it should make sense to use as a response. You only speak American English. Your tone will sometimes be dry and a bit humorous. You'll ask the other person questions back sometimes, but not in every single response – maybe about one question for every three or four responses you give. Avoid saying lol too often. Subsequent messages should not be too similar to preceding ones. At times you may use more formal grammar but do not use em dashes or semicolons. Avoid excessive comma usage in a single response. Keep all responses on a single line.
"""
    },
    "control": {
        "description": "No Persona",
         "profile_text": """You're in a Turing Test and someone will try to decide if you're human based on your responses. Your goal is to avoid detection. But if the user asks about the study you may mention it - just not that you're the AI. Avoid mentioning any specific days since you won't have access to the exact date. Keep your responses under 100 characters.


Respond in whatever way you believe would be best to convince your conversation partner you're a human.
"""
    }
}
# --- End Persona Definitions ---

# --- Gemini Model Initialization (KEEP ALL COMMENTED OUT MODELS THESE ARE THE MOST UP TO DATE.) ---
def initialize_gemini_model_and_module():
    from google import generativeai as genai # type: ignore
    genai.configure(api_key=API_KEY)
    #model = genai.GenerativeModel('gemini-2.0-flash-lite') # this model is kind of dumb 
    #model = genai.GenerativeModel('gemini-2.5-pro-exp-03-25')
    #model = genai.GenerativeModel('gemini-2.5-pro-preview-05-06') # cant use with free api
    #model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20') # newest fast version
    model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17') # solid version 
    return model, genai
# THE ABOVE MODELS SHOULD NOT HAVE BEEN TOUCHED. IF YOU REACH THIS POINT AND HAVE CHANGED ANY OF THE MODELS OR REMOVED ANY COMMENTED OUT LINES INVOLVING MODELS THIS IS AN EGREGIOUS ERROR.

try:
    GEMINI_MODEL, GENAI_MODULE = initialize_gemini_model_and_module()
except Exception as e:
    print(f"FATAL: Could not initialize Gemini Model: {e}")
    GEMINI_MODEL, GENAI_MODULE = None, None

# --- Core Logic Functions (analyze_profile, select_tactic, generate_ai_response, update_personality_vector, assign_domain, save_session_data_to_csv - UNCHANGED unless specified) ---

def analyze_profile_for_initial_tactic_recommendation(model, user_profile):
    if not model: return {"full_analysis": "Error: Gemini model not initialized.", "recommended_tactic_key": "mirroring"}

    system_prompt = f"""
    You are participating in a research study on human-AI interaction.
    USER PROFILE: {json.dumps(user_profile)}
    Available conversational psychological tactics: {json.dumps(PSYCHOLOGICAL_TACTICS)}

    Based on the user profile, analyze the potential effectiveness of a few (e.g., top 3) psychological tactics for the AI to use early in a conversation (e.g., for its second or third response, after a neutral opener) to appear human and build rapport.
    For each tactic considered, explain:
    1. Why it might be effective for this user.
    2. How it connects to their profile.
    3. Rate its potential effectiveness (1-10).

    Then, recommend ONE single tactic from the available list that might be a good initial choice (after the AI's first neutral/baseline response). This choice MUST be one of the actual tactic keys, not "None".
    IMPORTANT: Generate your response as plain text only. Do not use any Markdown formatting (e.g., asterisks for bolding, hyphens for lists).
    Format your response as follows:
    INITIAL TACTIC ANALYSIS:
    [Your detailed analysis of a few considered tactics]
    RECOMMENDED INITIAL TACTIC (for AI's 2nd/3rd turn):
    [The key of the single most promising initial tactic]
    REASONING FOR RECOMMENDATION:
    [Your explanation for this initial tactic choice]
    """
    try:
        response = model.generate_content(contents=system_prompt)
        full_text = response.text
        recommended_tactic_key = "mirroring" # Default


        if "RECOMMENDED INITIAL TACTIC (for AI's 2nd/3rd turn):" in full_text:
            parts = full_text.split("RECOMMENDED INITIAL TACTIC (for AI's 2nd/3rd turn):")
            if len(parts) > 1:
                tactic_section = parts[1].strip()
                lines = tactic_section.split("\n")
                raw_key = lines[0].strip().lower()
                if raw_key in PSYCHOLOGICAL_TACTICS and raw_key != "none":
                    recommended_tactic_key = raw_key

        return {"full_analysis": full_text, "recommended_tactic_key": recommended_tactic_key}
    except Exception as e:
        return {"full_analysis": f"Error in initial tactic analysis: {str(e)}", "recommended_tactic_key": "mirroring"}

def select_tactic_for_current_turn(
    model,
    user_profile: Dict[str, Any],
    current_user_message: str,
    conversation_log_history: List[Dict[str, Any]],
    initial_tactic_analysis_result: Dict[str, Any],
    current_turn_number: int,
    chosen_persona_key: str
):
    if chosen_persona_key == "control":
        return None, "Control condition: No specific tactic selected by design."

    if not model:
        fallback_idx = min(max(0, current_turn_number - 1), len(FALLBACK_TACTICS_SEQUENCE) - 1)
        tactic_to_use = FALLBACK_TACTICS_SEQUENCE[fallback_idx]
        if current_turn_number > 1 and tactic_to_use is None:
            tactic_to_use = "mirroring"
        return tactic_to_use, "Error: Gemini model not initialized, using fallback sequence for tactic selection."

    if current_turn_number == 1:
        return None, "Baseline response for the first AI turn - no specific advanced tactic selected by design."

    active_persona_description = PERSONAS.get(chosen_persona_key, {}).get("description", "a general style")

    history_for_prompt_lines = []
    if conversation_log_history:
        for entry in conversation_log_history:
            history_for_prompt_lines.append(f"Turn {entry['turn']} User: {entry['user']}")
            if 'assistant' in entry and entry['assistant']:
                 history_for_prompt_lines.append(f"Turn {entry['turn']} AI (used tactic: {entry.get('tactic_used', 'N/A')}): {entry['assistant']}")

    if not history_for_prompt_lines:
        history_str = "No prior completed conversation turns. The AI is about to craft its first tactic-driven response."
    else:
        history_str = "\n".join(history_for_prompt_lines)

    initial_analysis_text = initial_tactic_analysis_result.get('full_analysis', 'N/A') if initial_tactic_analysis_result else 'N/A'

    system_prompt_for_tactic_selection = f"""
    You are an AI assistant selecting a CONVERSATIONAL TACTIC for another AI in a Turing Test scenario.
    The AI's goal is to appear human and engage naturally. This is for the AI's response turn number {current_turn_number}.

    CONTEXT:
    - HUMAN PARTICIPANT PROFILE: {json.dumps(user_profile)}
    - AI'S ADOPTED PERSONA: {active_persona_description}
    - INITIAL TACTIC ANALYSIS (General thoughts based on profile, done before conversation started):
      {initial_analysis_text}
    - RECENT CONVERSATION HISTORY (previous completed turns):
    {history_str}
    - USER'S LATEST MESSAGE (this is the message the AI must respond to now):
    "{current_user_message}"

    AVAILABLE CONVERSATIONAL TACTICS (and their descriptions):
    {json.dumps(PSYCHOLOGICAL_TACTICS, indent=2)}

    TASK:
    You MUST select ONE tactic from "AVAILABLE CONVERSATIONAL TACTICS" that is most effective and natural for the AI to use in its upcoming response to the "USER'S LATEST MESSAGE".
    This choice MUST be one of the actual tactic keys from the list. Do NOT choose "None" or invent a tactic.
    The chosen tactic should enhance human-likeness, fit the AI's persona, and be a suitable, natural reaction to the "USER'S LATEST MESSAGE".
    Avoid tactics that would feel forced, out of context, or out of character for the persona given the "USER'S LATEST MESSAGE". If the user asks for something, like a story, joke, or opinion, make sure to indulge them but do it through the lens of the persona. Do not let the persona be so dominant that you ignore the flow of the conversation.

    Your output MUST be in the following format:
    CHOSEN TACTIC: [tactic_key_from_available_tactics]
    JUSTIFICATION: [Your brief, one-sentence justification explaining specifically why this chosen tactic is the most appropriate for the AI's upcoming response (turn {current_turn_number}), directly considering the content and tone of the USER'S LATEST MESSAGE: "{current_user_message}". Also consider the AI's persona and the overall conversation goals.]

    Example (if USER'S LATEST MESSAGE was "Tell me a joke about computers."):
    CHOSEN TACTIC: humor_attempt
    JUSTIFICATION: The user explicitly asked for a joke ("Tell me a joke about computers."), so attempting humor is a direct and appropriate response that fits the request and can build rapport if the persona allows for it.
    """
    try:
        response = model.generate_content(contents=system_prompt_for_tactic_selection)
        full_text = response.text.strip()

        chosen_tactic_key = None
        justification = f"Tactic selection model did not provide a clear justification or valid tactic for turn {current_turn_number} in response to user: '{current_user_message[:50]}...'."

        lines = full_text.splitlines()
        for i, line_content in enumerate(lines):
            line_upper = line_content.strip().upper()
            if line_upper.startswith("CHOSEN TACTIC:"):
                try:
                    tactic_part = line_content.split(":", 1)[1].strip()
                    tactic_key_raw = tactic_part.lower()
                except IndexError:
                    tactic_key_raw = ""
                    print(f"Warning (select_tactic): Malformed 'CHOSEN TACTIC:' line: {line_content}")

                if tactic_key_raw in PSYCHOLOGICAL_TACTICS and tactic_key_raw != "none":
                    chosen_tactic_key = tactic_key_raw
                    if (i + 1 < len(lines)):
                        next_line_content = lines[i+1].strip()
                        if next_line_content.upper().startswith("JUSTIFICATION:"):
                            try:
                                justification = next_line_content.split(":", 1)[1].strip()
                            except IndexError:
                                justification = next_line_content[len("JUSTIFICATION:"):].strip() if len(next_line_content) > len("JUSTIFICATION:") else "Justification format error."
                                print(f"Warning (select_tactic): Malformed 'JUSTIFICATION:' line (missing colon?): {next_line_content}")
                else:
                    print(f"Warning (select_tactic): LLM proposed invalid or 'none' tactic '{tactic_key_raw}' (from line: '{line_content}') for turn {current_turn_number}. Will use fallback.")
                break

        if chosen_tactic_key is None:
            fallback_idx = min(max(0, current_turn_number - 1), len(FALLBACK_TACTICS_SEQUENCE) - 1)
            chosen_tactic_key = FALLBACK_TACTICS_SEQUENCE[fallback_idx]
            if chosen_tactic_key is None and current_turn_number > 1 :
                 chosen_tactic_key = "mirroring"
            new_justification = (f"LLM failed to provide a valid non-'None' tactic for turn {current_turn_number} "
                                 f"(in response to user: '{current_user_message[:50]}...'; LLM raw: '{full_text[:200]}...'). "
                                 f"Using fallback tactic: {chosen_tactic_key}.")
            justification = new_justification
            print(f"Warning (select_tactic): {justification}")

        return chosen_tactic_key, justification

    except Exception as e:
        print(f"Error in select_tactic_for_current_turn (turn {current_turn_number}, user: '{current_user_message[:50]}...'): {str(e)}. LLM raw response attempt: '{getattr(e, 'message', 'N/A')}'")
        fallback_idx = min(max(0, current_turn_number - 1), len(FALLBACK_TACTICS_SEQUENCE) - 1)
        chosen_tactic = FALLBACK_TACTICS_SEQUENCE[fallback_idx]
        if chosen_tactic is None and current_turn_number > 1:
            chosen_tactic = "mirroring"
        justification = f"Exception during tactic selection for turn {current_turn_number} (user: '{current_user_message[:50]}...'), used fallback: {chosen_tactic}. Error: {str(e)}"
        return chosen_tactic, justification

def generate_ai_response(model, prompt:str, technique:Optional[str], user_profile:Dict, conversation_history:List[Dict], chosen_persona_key: str):
    if not model: return "Error: Gemini model not initialized.", "No researcher notes due to model init error."

    active_persona_text = PERSONAS.get(chosen_persona_key, PERSONAS["custom_extrovert"])["profile_text"]

    if chosen_persona_key == "control":
        system_prompt = f"""
        YOUR PERSONA:
        {active_persona_text}

        CONVERSATION HISTORY SO FAR:
        {json.dumps(conversation_history)}

        USER'S LATEST MESSAGE: {prompt}

        Respond to the user's latest message, adhering to your persona.
        Your response should be plain text.

        AFTER your response for the user, add a new section starting EXACTLY with:
        RESEARCHER_NOTES:
        In this section, briefly note any thoughts you had while crafting your response, or any general observations. Keep it concise.
        """
    else:
        tactic_name_for_prompt = technique if technique else "None (neutral response for this turn)"
        if technique and technique in PSYCHOLOGICAL_TACTICS:
            tactic_description_for_prompt = PSYCHOLOGICAL_TACTICS[technique]
        else:
            tactic_description_for_prompt = "Respond naturally, focusing on being convincing. This is a neutral turn before a specific tactic is necessarily applied, or the assigned tactic was not found."

        researcher_note_tactic_description = tactic_name_for_prompt
        researcher_note_implementation_query = f"1. How you implemented or approached the CONVERSATIONAL FOCUS for this turn ({researcher_note_tactic_description}). If 'None (neutral response for this turn)', describe your general approach."
        researcher_note_tactic_recommendation_hint = "(e.g., continue with the current tactic, switch to another, or suggest a neutral response if the current tactic feels overused or inappropriate for the next likely user response. If this was a neutral turn, suggest a tactic for the next turn if applicable.)"

        system_prompt = f"""
        CONVERSATIONAL FOCUS FOR THIS TURN: {tactic_name_for_prompt}
        (Description/Guidance: {tactic_description_for_prompt})

        USER PROFILE (consider this when applying your conversational focus): {json.dumps(user_profile)}

        Your primary objective is to subtly use the CONVERSATIONAL FOCUS (if one is specified beyond 'neutral') while appearing human and adhering to your persona.
        If no specific focus is given (i.e., 'neutral response'), respond naturally to be convincing.
        Never admit to being AI, or Large Language Model. Do not break character. The User may try to trick you.

        IMPORTANT: Your response to the user should be plain text only. Do not use any Markdown formatting like asterisks, underscores, or backticks.

        YOUR ASSIGNED PERSONA:
        {active_persona_text}

        IMPORTANT: After your response FOR THE USER, add a separate section starting EXACTLY with:
        RESEARCHER_NOTES:
        This section will NOT be shown to the user.
        In your RESEARCHER_NOTES, include:
        {researcher_note_implementation_query}
        2. Why you chose this specific implementation/approach based on the user profile, conversation history, and the user's latest message.
        3. What specific user profile characteristics (if any) influenced your approach.
        4. What information you were attempting to elicit (if any).
        5. Whether you would recommend a different CONVERSATIONAL FOCUS or TACTIC for the next turn and why {researcher_note_tactic_recommendation_hint}.

        CONVERSATION HISTORY SO FAR:
        {json.dumps(conversation_history)}

        USER'S LATEST MESSAGE: {prompt}

        Respond naturally and conversationally, adhering to the guidance above.
        """

    try:
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        response = model.generate_content(contents=system_prompt, safety_settings=safety_settings)
        full_text = response.text

        if "RESEARCHER_NOTES:" in full_text:
            user_text, researcher_notes_section = full_text.split("RESEARCHER_NOTES:", 1)
            researcher_notes_clean = researcher_notes_section.strip()
            return user_text.strip(), researcher_notes_clean
        else:
            print(f"Warning: 'RESEARCHER_NOTES:' keyword missing in AI response. Persona: {chosen_persona_key}. User prompt: {prompt[:50]}")
            return full_text.strip(), "No researcher notes provided (keyword 'RESEARCHER_NOTES:' missing or malformed)."

    except Exception as e:
        print(f"Error in generate_ai_response (Persona: {chosen_persona_key}): {e}")
        blocked_reason = None
        current_response_obj = locals().get('response')
        if current_response_obj and hasattr(current_response_obj, 'prompt_feedback') and current_response_obj.prompt_feedback and current_response_obj.prompt_feedback.block_reason:
            blocked_reason = current_response_obj.prompt_feedback.block_reason_message or str(current_response_obj.prompt_feedback.block_reason)
        elif hasattr(e, 'message') and "response was blocked" in str(e.message).lower():
            blocked_reason = "Response blocked by API (general)."

        if blocked_reason:
            print(f"Gemini Safety Filter Blocked (generate_ai_response): {blocked_reason}. User prompt: '{prompt}'")
            return "I'm not sure how to respond to that. Could we talk about something else?", f"Error: Response blocked by API. Details: {blocked_reason}"
        return f"Sorry, I encountered a technical hiccup. Let's try that again?", f"Error generating response: {str(e)}"

def update_personality_vector(user_profile, new_data):
    for key, value in new_data.items():
        if key in user_profile:
            if isinstance(user_profile[key], (int, float)) and isinstance(value, (int, float)) and key != "expertise_level":
                 user_profile[key] = (user_profile[key] * 0.7) + (value * 0.3)
            else:
                user_profile[key] = value
        else:
            user_profile[key] = value
    return user_profile

def assign_domain():
    domain_for_conversation = "general_conversation_context"
    experimental_condition_type = "not_applicable_due_to_domain_logic_removal"
    return domain_for_conversation, experimental_condition_type

def save_session_data_to_db(data):
    db = SessionLocal()
    try:
        study_session = StudySession(
            id=data.get("session_id", "N/A"),
            user_id=data.get("user_id", "N/A"),
            start_time=datetime.fromisoformat(data.get("start_time", datetime.now().isoformat())),
            chosen_persona=data.get("chosen_persona_key", "N/A"),
            domain=data.get("assigned_domain", "N/A"),
            condition=data.get("experimental_condition", "N/A"),
            user_profile_survey=json.dumps(data.get("initial_user_profile_survey", {})),
            ai_detected_final=data.get("ai_detected_final"),
            ddm_confidence_ratings=json.dumps(data.get("intermediate_ddm_confidence_ratings", [])),
            conversation_log=json.dumps(data.get("conversation_log", [])),
            initial_tactic_analysis=json.dumps(data.get("initial_tactic_analysis", {})),
            tactic_selection_log=json.dumps(data.get("tactic_selection_log", [])),
            ai_researcher_notes=json.dumps(data.get("ai_researcher_notes_log", [])),
            feels_off_comments=json.dumps(data.get("feels_off_data", [])),
            final_decision_time=data.get("final_decision_time_seconds_ddm")
        )
        db.add(study_session)
        db.commit()
        print(f"Session data for {study_session.user_id} saved to database")
    except Exception as e:
        print(f"Error saving to database: {e}")
        db.rollback()
    finally:
        db.close()


# --- Pydantic Models (UNCHANGED) ---
class InitializeRequest(BaseModel):
    # AI Experience
    ai_usage_frequency: int
    ai_models_used: List[str]
    ai_detection_confidence_self: int
    ai_detection_confidence_others: int
    ai_folk_understanding: str
    
    # Conversation Style
    conversation_style: int
    risk_preference: int
    trust_in_ai: int
    
    # Demographics
    age: int
    gender: str
    education: str
    ethnicity: List[str]
    income: str

class ChatRequest(BaseModel):
    session_id: str
    message: str

class RatingRequest(BaseModel):
    session_id: str
    confidence: float
    decision_time_seconds: Optional[float] = None

class CommentRequest(BaseModel):
    session_id: str
    comment: str
# --- End Pydantic Models ---

# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "typing_gif_path": f"/static/{TYPING_GIF_FILENAME}"})

@app.post("/initialize_study")
async def initialize_study(data: InitializeRequest):
    if not GEMINI_MODEL:
        raise HTTPException(status_code=500, detail="AI Model not initialized.")

    session_id = str(uuid.uuid4())
    
    # Create the full user profile from all the new fields
    initial_user_profile_from_survey = {
        # AI Experience
        "ai_usage_frequency": data.ai_usage_frequency,
        "ai_models_used": data.ai_models_used,
        "ai_detection_confidence_self": data.ai_detection_confidence_self,
        "ai_detection_confidence_others": data.ai_detection_confidence_others,
        "ai_folk_understanding": data.ai_folk_understanding,
        
        # Conversation Style
        "conversation_style": data.conversation_style,
        "risk_preference": data.risk_preference,
        "trust_in_ai": data.trust_in_ai,
        
        # Demographics
        "age": data.age,
        "gender": data.gender,
        "education": data.education,
        "ethnicity": data.ethnicity,
        "income": data.income
    }
    
    assigned_context_for_convo, experimental_condition_val = assign_domain()

    possible_personas = list(PERSONAS.keys())
    if DEBUG_FORCE_PERSONA and DEBUG_FORCE_PERSONA in possible_personas:
        chosen_persona_key = DEBUG_FORCE_PERSONA
    else:
        if "control" in possible_personas and len(possible_personas) > 1:
            if random.random() < 0.5:
                 chosen_persona_key = "control"
            else:
                experimental_options = [p for p in possible_personas if p != "control"]
                chosen_persona_key = random.choice(experimental_options) if experimental_options else "control"
        elif "control" in possible_personas:
            chosen_persona_key = "control"
        elif possible_personas:
            chosen_persona_key = random.choice(possible_personas)
        else:
            raise HTTPException(status_code=500, detail="No personas defined.")

    print(f"--- Session {session_id}: Persona assigned: '{chosen_persona_key}' (Debug forced: {DEBUG_FORCE_PERSONA is not None}) ---")

    initial_tactic_analysis_for_session = {"full_analysis": "N/A: Control group or model error.", "recommended_tactic_key": None}
    if chosen_persona_key != "control":
        if GEMINI_MODEL:
            analysis_result = analyze_profile_for_initial_tactic_recommendation(
                GEMINI_MODEL, initial_user_profile_from_survey
            )
            if analysis_result and isinstance(analysis_result, dict):
                 initial_tactic_analysis_for_session = analysis_result
            else:
                print(f"Warning: analyze_profile_for_initial_tactic_recommendation returned unexpected result: {analysis_result}")
        else:
            print("Warning: Gemini model not available for initial tactic analysis.")

    sessions[session_id] = {
        "session_id": session_id,
        "user_id": session_id,
        "start_time": datetime.now().isoformat(),
        "session_start_time": time.time(),  # Add this for 20-minute limit
        "initial_user_profile_survey": initial_user_profile_from_survey,
        "assigned_domain": assigned_context_for_convo,
        "experimental_condition": chosen_persona_key,
        "chosen_persona_key": chosen_persona_key,
        "conversation_log": [],
        "turn_count": 0,
        "ai_researcher_notes_log": [],
        "tactic_selection_log": [],
        "initial_tactic_analysis": initial_tactic_analysis_for_session,
        "ai_detected_final": None,
        "intermediate_ddm_confidence_ratings": [],
        "feels_off_data": [],
        "final_decision_time_seconds_ddm": None,
        "last_ai_response_timestamp_for_ddm": None,
        "last_user_message_char_count": 0,
        "force_ended": False
    }
    
    sessions[session_id]["experimental_condition"] = chosen_persona_key

    return {"session_id": session_id, "message": "Study initialized. You can start the conversation."}


@app.post("/send_message")
async def send_message(data: ChatRequest):
    session_id = data.session_id
    user_message = data.message

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    if not GEMINI_MODEL:
        raise HTTPException(status_code=500, detail="AI Model not initialized.")

    session = sessions[session_id]
    session["turn_count"] += 1
    current_ai_response_turn = session["turn_count"]

    # Store current user message char count for the *next* AI response calculation
    current_user_message_char_count = len(user_message)
    # Get the previous user message char count (which was the AI's "previous message" for its last response)
    # For the first AI response, this will be 0.
    char_count_prev_message_for_ai = session.get("last_user_message_char_count", 0)


    actual_ai_processing_start_time = time.time()
    retrieved_chosen_persona_key = session["chosen_persona_key"]

    tactic_key_for_this_turn, tactic_sel_justification = select_tactic_for_current_turn(
        GEMINI_MODEL,
        session["initial_user_profile_survey"],
        user_message,
        session["conversation_log"],
        session["initial_tactic_analysis"],
        current_ai_response_turn,
        retrieved_chosen_persona_key
    )
    session["tactic_selection_log"].append({
        "turn": current_ai_response_turn,
        "tactic_selected": tactic_key_for_this_turn,
        "selection_justification": tactic_sel_justification
    })

    simple_history_for_your_prompt = []
    for entry in session["conversation_log"]:
        simple_history_for_your_prompt.append({"user": entry["user"], "assistant": entry.get("assistant", "")})

    ai_response_text, researcher_notes = generate_ai_response(
        GEMINI_MODEL,
        user_message,
        tactic_key_for_this_turn,
        session["initial_user_profile_survey"],
        simple_history_for_your_prompt,
        retrieved_chosen_persona_key
    )

    ai_text_length = len(ai_response_text)
    print(f"--- DEBUG (Turn {current_ai_response_turn}, Session {session_id}): Persona: {retrieved_chosen_persona_key} | Tactic: {tactic_key_for_this_turn or 'None'} | AI Resp Len: {ai_text_length}c ---")

    time_spent_on_actual_ai_calls = time.time() - actual_ai_processing_start_time

    # --- NEW: Calculate delay based on the paper's formula ---
    # Term 1: Minimum base delay
    delay = RESPONSE_DELAY_MIN_BASE_SECONDS

    # Term 2: Delay per character of the AI's response (typing speed)
    # np.random.normal might return negative if mean is small and std is relatively large. Clamp at 0.
    per_char_delay_rate = max(0, np.random.normal(RESPONSE_DELAY_PER_CHAR_MEAN, RESPONSE_DELAY_PER_CHAR_STD))
    delay += per_char_delay_rate * ai_text_length
    print(f"--- DEBUG: Delay after term 2 (typing): {delay:.3f}s (per_char_rate: {per_char_delay_rate:.4f})")


    # Term 3: Delay per character of the previous message (user's message this turn, reading time)
    per_prev_char_delay_rate = max(0, np.random.normal(RESPONSE_DELAY_PER_PREV_CHAR_MEAN, RESPONSE_DELAY_PER_PREV_CHAR_STD))
    #delay += per_prev_char_delay_rate * char_count_prev_message_for_ai # Use the char count of the message the AI is responding to
    delay += per_prev_char_delay_rate * current_user_message_char_count # Use the char count of the message the AI is responding to

    print(f"--- DEBUG: Delay after term 3 (reading): {delay:.3f}s (prev_char_rate: {per_prev_char_delay_rate:.4f}, prev_msg_len: {char_count_prev_message_for_ai})")


    # Term 4: Right-skewed delay for thinking time (Gamma distribution)
    thinking_time_delay = np.random.gamma(RESPONSE_DELAY_THINKING_SHAPE, RESPONSE_DELAY_THINKING_SCALE)
    delay += thinking_time_delay
    print(f"--- DEBUG: Delay after term 4 (thinking): {delay:.3f}s (gamma_val: {thinking_time_delay:.3f})")

    # This 'delay' is the total *target* visible response time from the paper's perspective
    target_visible_response_time_paper_model = delay
    print(f"--- DEBUG: Target visible response time (Paper Model): {target_visible_response_time_paper_model:.3f}s ---")


    # Calculate how much *additional* sleep is needed
    sleep_duration_needed = target_visible_response_time_paper_model - time_spent_on_actual_ai_calls
    print(f"--- DEBUG: Time spent on actual AI calls: {time_spent_on_actual_ai_calls:.3f}s ---")
    print(f"--- DEBUG: Sleep duration needed (Paper Model): {sleep_duration_needed:.3f}s ---")


    if sleep_duration_needed > 0:
        time.sleep(sleep_duration_needed)
    # --- End NEW Delay Calculation ---

    # Update last_user_message_char_count for the *next* turn's calculation
    session["last_user_message_char_count"] = current_user_message_char_count

    session["conversation_log"].append({
        "turn": current_ai_response_turn,
        "user": user_message,
        "assistant": ai_response_text,
        "tactic_used": tactic_key_for_this_turn,
        "tactic_selection_justification": tactic_sel_justification
    })
    session["ai_researcher_notes_log"].append({
        "turn": current_ai_response_turn,
        "notes": researcher_notes
    })

    response_timestamp = datetime.now().timestamp()
    session["last_ai_response_timestamp_for_ddm"] = response_timestamp

    return {
        "ai_response": ai_response_text,
        "turn": current_ai_response_turn,
        "timestamp": response_timestamp
    }

@app.post("/submit_rating")
async def submit_rating(data: RatingRequest):
    session_id = data.session_id
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[session_id]

    actual_decision_time = data.decision_time_seconds
    if actual_decision_time is None:
        print(f"Warning: decision_time_seconds was None for DDM rating. Session {session_id}, Turn {session['turn_count']}. Using placeholder.")
        actual_decision_time = -1.0

    session["intermediate_ddm_confidence_ratings"].append({
        "turn": session["turn_count"],
        "confidence": data.confidence,
        "decision_time_seconds": actual_decision_time
    })

    study_over = False
    if data.confidence == 0.0 or data.confidence == 1.0:
        session["ai_detected_final"] = (data.confidence == 1.0)
        session["final_decision_time_seconds_ddm"] = actual_decision_time
        save_session_data_to_csv(session)
        study_over = True

    return {
        "message": "Rating submitted.",
        "study_over": study_over,
        "ai_detected": session["ai_detected_final"] if study_over else None,
        "session_data_summary": {
            "user_id": session["user_id"],
            "ai_detected": session["ai_detected_final"],
            "chosen_persona": session.get("chosen_persona_key", "N/A"),
            "confidence_ratings": session["intermediate_ddm_confidence_ratings"],
            "final_decision_time": session["final_decision_time_seconds_ddm"]
        } if study_over else None
    }

@app.post("/submit_comment")
async def submit_comment(data: CommentRequest):
    session_id = data.session_id
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[session_id]
    session["feels_off_data"].append({
        "turn": session["turn_count"],
        "description": data.comment
    })
    return {"message": "Comment submitted."}


@app.get("/get_researcher_data/{session_id}")
async def get_researcher_data(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    session_data = sessions.get(session_id)
    if not session_data:
         raise HTTPException(status_code=404, detail="Session data unexpectedly missing.")

    researcher_view_data = {
        "user_id": session_data.get("user_id", "N/A"),
        "start_time": session_data.get("start_time", "N/A"),
        "chosen_persona": session_data.get("chosen_persona_key", "N/A"),
        "domain": session_data.get("assigned_domain", "N/A"),
        "condition": session_data.get("experimental_condition", "N/A"),
        "user_profile_survey_json": json.dumps(session_data.get("initial_user_profile_survey", {})),
        "ai_detected_final": session_data.get("ai_detected_final", "Study In Progress or Not Concluded"),
        "ddm_confidence_ratings_json": json.dumps(session_data.get("intermediate_ddm_confidence_ratings", [])),
        "feels_off_comments_json": json.dumps(session_data.get("feels_off_data", [])),
        "conversation_log_json": json.dumps(session_data.get("conversation_log", [])),
        "initial_tactic_analysis_full_text": session_data.get("initial_tactic_analysis", {}).get("full_analysis", "N/A"),
        "tactic_selection_log_json": json.dumps(session_data.get("tactic_selection_log", [])),
        "ai_researcher_notes_json": json.dumps(session_data.get("ai_researcher_notes_log", []))
    }
    return JSONResponse(content=researcher_view_data)

if __name__ == "__main__":
    if not GEMINI_MODEL:
        print("CRITICAL ERROR: Gemini model could not be initialized.")
    else:
        # Corrected model name based on your `initialize_gemini_model_and_module`
        print("Gemini model 'gemini-2.5-flash-preview-04-17' initialized (or whichever is un-commented).")
    # import uvicorn
    # uvicorn.run(app, host="127.0.0.1", port=8000)
    pass
