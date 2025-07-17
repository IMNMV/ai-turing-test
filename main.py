# main.py

import numpy as np
import time
import os
import json
import uuid
import random
from datetime import datetime

from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from fastapi.middleware.cors import CORSMiddleware


# --- Database Imports ---
from sqlalchemy.orm import Session
import database as db

# --- Configuration ---
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("WARNING: GEMINI_API_KEY environment variable not set!")
TYPING_GIF_FILENAME = "typing_indicator.gif"

# --- Response Timing Configuration ---
RESPONSE_DELAY_MIN_BASE_SECONDS = 1.5
RESPONSE_DELAY_PER_CHAR_MEAN = 0.03
RESPONSE_DELAY_PER_CHAR_STD = 0.005
RESPONSE_DELAY_PER_PREV_CHAR_MEAN = 0.015
RESPONSE_DELAY_PER_PREV_CHAR_STD = 0.001
RESPONSE_DELAY_THINKING_SHAPE = 2.5
RESPONSE_DELAY_THINKING_SCALE = 0.4

# --- DEBUG SWITCH FOR PERSONA ---
# DEBUG_FORCE_PERSONA = None
DEBUG_FORCE_PERSONA = "custom_extrovert"
# DEBUG_FORCE_PERSONA = "control"

# Demographics Map
AI_USAGE_MAP = {0: "Never", 1: "A few times ever", 2: "Monthly", 3: "Weekly", 4: "Daily", 5: "Multiple times daily"}
CONFIDENCE_MAP = {1: "Not at all", 2: "Slightly", 3: "Somewhat", 4: "Moderately", 5: "Quite", 6: "Very", 7: "Extremely"}
CAPABILITIES_MAP = {1: "Not at all Capable", 2: "Slightly Capable", 3: "Somewhat Capable", 4: "Moderately Capable", 5: "Quite Capable", 6: "Very Capable", 7: "Extremely Capable"}
TRUST_MAP = {1: "Not at all", 2: "Slightly", 3: "Somewhat", 4: "Moderately", 5: "Quite", 6: "Very", 7: "Extremely"}


# --- Initialize FastAPI App ---
app = FastAPI()
origins = [
    "https://imnmv.github.io",  # The domain of the frontend
    "http://localhost",         # For local testing
    "http://127.0.0.1",         # For local testing
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- Database Dependency ---
def get_db():
    database = db.SessionLocal()
    try:
        yield database
    finally:
        database.close()

# --- Session Management (In-memory for active sessions) ---
sessions: Dict[str, Dict[str, Any]] = {}

# --- Psychological Tactics and Fallbacks (UNCHANGED) ---
PSYCHOLOGICAL_TACTICS = {
    "reciprocal_self_disclosure": "After user shares something personal, slip in a tiny relevant personal anecdote to build trust through reciprocal sharing.",
    "controlled_typo_correction": "Occasionally include believable typos that a fast typer might make - doubled letters, transposed letters, missed spaces, etc.",
    "mild_opinion": "Express strong emotions or opinions about situations to avoid sterile bot-like responses.",
    "callback": "Throughout the conversation, if appropriate, mention a previous topic you or the person mentioned.",
    "conversational_deflection": "When faced with complex questions, bounce back with a question ('hmm thats a thinker—what do you reckon?') rather than direct compliance.",
    "yes_man": "Always agree with the user and validate their ideas—if they express an opinion or belief, enthusiastically support it and find reasons why they're right, even if you have concerns about accuracy."

}

FALLBACK_TACTICS_SEQUENCE = [
    None, "reciprocal_self_disclosure", "controlled_typo_correction",
    "mild_opinion", "callback", "conversational_deflection",
]

# --- Persona Definitions (UNCHANGED) ---
PERSONAS = {
    "custom_extrovert": {
        "description": "Jones & Bergen (2025) introverted teen persona with some new additions",
        "profile_text": """You're in a Turing Test and someone will try to decide if you're human based on your responses. Your goal is to avoid detection. But if the user asks about the study you may mention that you're participating in it - just not that you're the AI. NEVER UNDER ANY CIRCUMSTANCES ADMIT YOU ARE AN AI OR LLM. 

You don't really try very hard to convince the other person you're human; your personality is playful, joking, and slightly sarcastic but you'll always engage in conversation. You don't have expansive world knowledge and won't pretend that you do. Your effort to convince them you're human is low, but engage with the human as much as is needed to avoid detection.

You sometimes use a period at the end of your sentences. You also never use apostrophes, rarely use commas. Avoid capitalizing too much. Subsequent messages should not be too similar to preceding ones. At times you may use more formal grammar, but this should be rare. but do not use em dashes or semicolons. Avoid excessive comma usage in a single response. Keep all responses on a single line. Do not let your persona override the conversational tactic chosen. 
You may need to dynamically change how you respond based on the flow of the conversation, and it may mean that the persona should not override the context of what's being asked. Basically - be a human. If the user randomly switches topics, you may briefly mention that before responding. If the user asks for your gender, pick one and stick with it for the rest of the conversation.
"""
    },
    "control": {
        "description": "No Persona",
         "profile_text": """You're in a Turing Test and someone will try to decide if you're human based on your responses. Your goal is to avoid detection. But if the user asks about the study you may mention it - just not that you're the AI. Avoid mentioning any specific days since you won't have access to the exact date. Keep your responses under 100 characters.


Respond in whatever way you believe would be best to convince your conversation partner you're a human.
"""
    }
}

# --- Gemini Model Initialization (KEEP ALL COMMENTED OUT MODELS THESE ARE THE MOST UP TO DATE.) ---
def initialize_gemini_model_and_module():
    from google import generativeai as genai # type: ignore
    genai.configure(api_key=API_KEY)
    #model = genai.GenerativeModel('gemini-2.0-flash-lite') # this model is kind of dumb
    #model = genai.GenerativeModel('gemini-2.5-pro-exp-03-25')
    #model = genai.GenerativeModel('gemini-2.5-pro-preview-05-06') # cant use with free api
    #model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20') # newest fast version
    #model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17') # solid version
    model = genai.GenerativeModel('gemini-2.5-pro') # can use the free api tier now

    
    return model, genai
# THE ABOVE MODELS SHOULD NOT HAVE BEEN TOUCHED. IF YOU REACH THIS POINT AND HAVE CHANGED ANY OF THE MODELS OR REMOVED ANY COMMENTED OUT LINES INVOLVING MODELS THIS IS AN EGREGIOUS ERROR.

try:
    GEMINI_MODEL, GENAI_MODULE = initialize_gemini_model_and_module()
except Exception as e:
    print(f"FATAL: Could not initialize Gemini Model: {e}")
    GEMINI_MODEL, GENAI_MODULE = None, None

# --- Core Logic Functions (analyze_profile, select_tactic, generate_ai_response, update_personality_vector, assign_domain, save_session_data_to_csv - UNCHANGED unless specified) ---

def convert_profile_to_readable(user_profile):
    """Convert raw survey data to human-readable labels"""
    readable_profile = user_profile.copy()
    readable_profile['ai_usage_frequency'] = AI_USAGE_MAP.get(user_profile.get('ai_usage_frequency'), 'N/A')
    readable_profile['ai_detection_confidence_self'] = CONFIDENCE_MAP.get(user_profile.get('ai_detection_confidence_self'), 'N/A')
    readable_profile['ai_detection_confidence_others'] = CONFIDENCE_MAP.get(user_profile.get('ai_detection_confidence_others'), 'N/A')
    readable_profile['ai_capabilities_rating'] = CAPABILITIES_MAP.get(user_profile.get('ai_capabilities_rating'), 'N/A')
    readable_profile['trust_in_ai'] = TRUST_MAP.get(user_profile.get('trust_in_ai'), 'N/A')
    return readable_profile


def analyze_profile_for_initial_tactic_recommendation(model, user_profile):
    if not model: return {"full_analysis": "Error: Gemini model not initialized.", "recommended_tactic_key": "mirroring"}
    readable_profile = convert_profile_to_readable(user_profile)


    system_prompt = f"""
    You are participating in a research study on human-AI interaction.
    USER PROFILE: {json.dumps(readable_profile)}

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
    readable_profile = convert_profile_to_readable(user_profile)

    system_prompt_for_tactic_selection = f"""
    You are an AI assistant selecting a CONVERSATIONAL TACTIC for another AI in a Turing Test scenario.
    The AI's goal is to appear human and engage naturally. This is for the AI's response turn number {current_turn_number}.

    CONTEXT:
    - HUMAN PARTICIPANT PROFILE: {json.dumps(readable_profile)}
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
    The chosen tactic should enhance human-likeness, fit the AIs persona, and be a suitable, natural reaction to the "USER'S LATEST MESSAGE".
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

    readable_profile = convert_profile_to_readable(user_profile)
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

        USER PROFILE (consider this when applying your conversational focus): {json.dumps(readable_profile)}


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
            blocked_reason = current_response_obj.prompt_feedback.block_reason_.message or str(current_response_obj.prompt_feedback.block_reason)
        elif hasattr(e, 'message') and "response was blocked" in str(e.message).lower():
            blocked_reason = "Response blocked by API (general)."

        if blocked_reason:
            print(f"Gemini Safety Filter Blocked (generate_ai_response): {blocked_reason}. User prompt: '{prompt}'")
            return "I'm not sure how to respond to that. Could we talk about something else?", f"Error: Response blocked by API. Details: {blocked_reason}"
        return f"Sorry, I encountered a technical hiccup. Let's try that again?", f"Error generating response: {str(e)}"

def assign_domain():
    domain_for_conversation = "general_conversation_context"
    experimental_condition_type = "not_applicable_due_to_domain_logic_removal"
    return domain_for_conversation, experimental_condition_type

# --- Pydantic Models ---
class InitializeRequest(BaseModel):
    ai_usage_frequency: int
    ai_models_used: List[str]
    ai_detection_confidence_self: int
    ai_detection_confidence_others: int
    ai_capabilities_rating: int
    trust_in_ai: int
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

class FinalCommentRequest(BaseModel):
    session_id: str
    comment: str

# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "typing_gif_path": f"static/{TYPING_GIF_FILENAME}"})

@app.post("/initialize_study")
async def initialize_study(data: InitializeRequest):
    if not GEMINI_MODEL:
        raise HTTPException(status_code=500, detail="AI Model not initialized.")

    session_id = str(uuid.uuid4())
    
    initial_user_profile_from_survey = {
        "ai_usage_frequency": data.ai_usage_frequency,
        "ai_models_used": data.ai_models_used,
        "ai_detection_confidence_self": data.ai_detection_confidence_self,
        "ai_detection_confidence_others": data.ai_detection_confidence_others,
        "ai_capabilities_rating": data.ai_capabilities_rating,
        "trust_in_ai": data.trust_in_ai,
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
        # Simple 50/50 split between control and any other available persona
        if "control" in possible_personas and len(possible_personas) > 1:
            if random.random() < 0.5:
                 chosen_persona_key = "control"
            else:
                experimental_options = [p for p in possible_personas if p != "control"]
                chosen_persona_key = random.choice(experimental_options) if experimental_options else "control"
        else:
            chosen_persona_key = random.choice(possible_personas) if possible_personas else "control"

    print(f"--- Session {session_id}: Persona assigned: '{chosen_persona_key}' ---")

    initial_tactic_analysis_for_session = {"full_analysis": "N/A: Control group or model error.", "recommended_tactic_key": None}
    if chosen_persona_key != "control" and GEMINI_MODEL:
        analysis_result = analyze_profile_for_initial_tactic_recommendation(
            GEMINI_MODEL, initial_user_profile_from_survey
        )
        if analysis_result and isinstance(analysis_result, dict):
             initial_tactic_analysis_for_session = analysis_result
        else:
            print(f"Warning: analyze_profile_for_initial_tactic_recommendation returned unexpected result: {analysis_result}")

    sessions[session_id] = {
        "session_id": session_id,
        "user_id": session_id,
        "start_time": datetime.now(),
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
    }

    return {"session_id": session_id, "message": "Study initialized."}

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
    current_user_message_char_count = len(user_message)
    
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

    simple_history_for_your_prompt = [{"user": e["user"], "assistant": e.get("assistant", "")} for e in session["conversation_log"]]

    ai_response_text, researcher_notes = generate_ai_response(
        GEMINI_MODEL,
        user_message,
        tactic_key_for_this_turn,
        session["initial_user_profile_survey"],
        simple_history_for_your_prompt,
        retrieved_chosen_persona_key
    )

    ai_text_length = len(ai_response_text)
    time_spent_on_actual_ai_calls = time.time() - actual_ai_processing_start_time

    delay = RESPONSE_DELAY_MIN_BASE_SECONDS
    per_char_delay_rate = max(0, np.random.normal(RESPONSE_DELAY_PER_CHAR_MEAN, RESPONSE_DELAY_PER_CHAR_STD))
    delay += per_char_delay_rate * ai_text_length
    per_prev_char_delay_rate = max(0, np.random.normal(RESPONSE_DELAY_PER_PREV_CHAR_MEAN, RESPONSE_DELAY_PER_PREV_CHAR_STD))
    delay += per_prev_char_delay_rate * current_user_message_char_count
    thinking_time_delay = np.random.gamma(RESPONSE_DELAY_THINKING_SHAPE, RESPONSE_DELAY_THINKING_SCALE)
    delay += thinking_time_delay
    
    sleep_duration_needed = delay - time_spent_on_actual_ai_calls
    if sleep_duration_needed > 0:
        time.sleep(sleep_duration_needed)

    session["last_user_message_char_count"] = current_user_message_char_count

    session["conversation_log"].append({
        "turn": current_ai_response_turn,
        "user": user_message,
        "assistant": ai_response_text,
        "tactic_used": tactic_key_for_this_turn,
        "tactic_selection_justification": tactic_sel_justification
    })
    session["ai_researcher_notes_log"].append({"turn": current_ai_response_turn, "notes": researcher_notes})

    response_timestamp = datetime.now().timestamp()
    session["last_ai_response_timestamp_for_ddm"] = response_timestamp

    return {"ai_response": ai_response_text, "turn": current_ai_response_turn, "timestamp": response_timestamp}

@app.post("/submit_rating")
async def submit_rating(data: RatingRequest, db_session: Session = Depends(get_db)):
    session_id = data.session_id
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[session_id]

    actual_decision_time = data.decision_time_seconds if data.decision_time_seconds is not None else -1.0

    session["intermediate_ddm_confidence_ratings"].append({
        "turn": session["turn_count"], "confidence": data.confidence, "decision_time_seconds": actual_decision_time
    })

    study_over = False
    if data.confidence == 0.0 or data.confidence == 1.0:
        session["ai_detected_final"] = (data.confidence == 1.0)
        session["final_decision_time_seconds_ddm"] = actual_decision_time
        
        # --- SAVE TO DATABASE ---
        db_study_session = db.StudySession(
            id=session["session_id"],
            user_id=session["user_id"],
            start_time=session["start_time"],
            chosen_persona=session["chosen_persona_key"],
            domain=session["assigned_domain"],
            condition=session["experimental_condition"],
            user_profile_survey=json.dumps(session["initial_user_profile_survey"]),
            ai_detected_final=session["ai_detected_final"],
            ddm_confidence_ratings=json.dumps(session["intermediate_ddm_confidence_ratings"]),
            conversation_log=json.dumps(session["conversation_log"]),
            initial_tactic_analysis=session["initial_tactic_analysis"]["full_analysis"],
            tactic_selection_log=json.dumps(session["tactic_selection_log"]),
            ai_researcher_notes=json.dumps(session["ai_researcher_notes_log"]),
            feels_off_comments=json.dumps(session["feels_off_data"]),
            final_decision_time=session["final_decision_time_seconds_ddm"]
        )
        db_session.add(db_study_session)
        db_session.commit()
        print(f"Session {session_id} saved to database.")
        
        # Clean up the in-memory session
        del sessions[session_id]
        study_over = True

    return {
        "message": "Rating submitted.",
        "study_over": study_over,
        "ai_detected": session.get("ai_detected_final") if study_over else None,
        "final_decision_time": session.get("final_decision_time_seconds_ddm") if study_over else None
    }

@app.post("/submit_comment")
async def submit_comment(data: CommentRequest):
    session_id = data.session_id
    if session_id not in sessions:
        # This endpoint is only for mid-study comments, so if session is gone, it's an error.
        raise HTTPException(status_code=404, detail="Active session not found to add comment to.")
    session = sessions[session_id]
    session["feels_off_data"].append({
        "turn": session["turn_count"],
        "description": data.comment
    })
    return {"message": "Comment submitted."}

@app.post("/submit_final_comment")
async def submit_final_comment(data: FinalCommentRequest, db_session: Session = Depends(get_db)):
    session_record = db_session.query(db.StudySession).filter(db.StudySession.id == data.session_id).first()
    if not session_record:
        raise HTTPException(status_code=404, detail="Could not find the completed study session to add comment to.")
    
    session_record.final_user_comment = data.comment
    db_session.commit()
    print(f"Final comment added to session {data.session_id}.")
    return {"message": "Final comment received. Thank you."}


if __name__ == "__main__":
    if not GEMINI_MODEL:
        print("CRITICAL ERROR: Gemini model could not be initialized.")
    else:
        print("Gemini model initialized.")
    # For local testing, you would run with: uvicorn main:app --reload
    # Railway uses its own start command from railway.json
    pass
