# main.py


import numpy as np
import time
import os
import json
import uuid
import random
import html
import re
import asyncio
from datetime import datetime

from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import pytz


# --- Configuration ---
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("WARNING: GEMINI_API_KEY environment variable not set!")
#TYPING_GIF_FILENAME = "typing_indicator.gif"

# --- NEW Response Timing Configuration (from the paper) ---
RESPONSE_DELAY_MIN_BASE_SECONDS = 1.5  # The '1' in the formula
RESPONSE_DELAY_PER_CHAR_MEAN = 0.1    # Paper uses 0.3, but that feels very long for typing. Let's start with 0.03-0.05. Let's use 0.03 for now.
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

# Demographics Maps
AI_USAGE_MAP = {0: "Never", 1: "A few times ever", 2: "Monthly", 3: "Weekly", 4: "Daily", 5: "Multiple times daily"}
DETECTION_SPEED_MAP = {1: "Immediately (1-2 msgs)", 2: "Very quickly (3-5 msgs)", 3: "Fairly quickly (6-10 msgs)", 4: "After some convo (11-20 msgs)", 5: "After extended convo (20+ msgs)", 6: "Couldn't tell"}
CAPABILITIES_MAP = {1: "Not at all Capable", 2: "Slightly Capable", 3: "Somewhat Capable", 4: "Moderately Capable", 5: "Quite Capable", 6: "Very Capable", 7: "Extremely Capable"}
TRUST_MAP = {1: "Not at all Trusting", 2: "Slightly Trusting", 3: "Somewhat Trusting", 4: "Moderately Trusting", 5: "Quite Trusting", 6: "Very Trusting", 7: "Extremely Trusting"}


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
#app.mount("/static", StaticFiles(directory="static"), name="static")
#templates = Jinja2Templates(directory="templates")

# --- Database Imports ---
from sqlalchemy.orm import Session
import database as db

# --- Database Dependency ---
def get_db():
    database = db.SessionLocal()
    try:
        yield database
    finally:
        database.close()

# --- NEW: Helper Functions for Incremental Database Saves ---
def create_initial_session_record(session_data, db_session: Session):
    """Create initial database record when session is initialized"""
    try:
        ui_events = session_data.get("ui_event_log", [])
        consent_accepted = any(event.get("event") == "consent_agree_clicked" for event in ui_events)
        
        db_study_session = db.StudySession(
            id=session_data["session_id"],
            user_id=session_data["user_id"],
            start_time=session_data["start_time"],
            chosen_persona=session_data["chosen_persona_key"],
            domain=session_data["assigned_domain"],
            condition=session_data["experimental_condition"],
            user_profile_survey=json.dumps(session_data["initial_user_profile_survey"]),
            initial_tactic_analysis=session_data["initial_tactic_analysis"]["full_analysis"],
            ui_event_log=json.dumps(ui_events),
            consent_accepted=consent_accepted,
            session_status="active",
            last_updated=datetime.utcnow()
        )
        db_session.add(db_study_session)
        db_session.commit()
        print(f"Initial session record created for {session_data['session_id']}")
        return True
    except Exception as e:
        print(f"Error creating initial session record: {e}")
        db_session.rollback()
        return False

def update_session_after_message(session_data, db_session: Session):
    """Update database record after each conversation turn"""
    try:
        session_record = db_session.query(db.StudySession).filter(db.StudySession.id == session_data["session_id"]).first()
        if session_record:
            session_record.conversation_log = json.dumps(session_data["conversation_log"])
            session_record.tactic_selection_log = json.dumps(session_data["tactic_selection_log"])
            session_record.ai_researcher_notes = json.dumps(session_data["ai_researcher_notes_log"])
            session_record.ui_event_log = json.dumps(session_data.get("ui_event_log", []))
            session_record.last_updated = datetime.utcnow()
            db_session.commit()
            print(f"Session updated after message for {session_data['session_id']}, turn {session_data['turn_count']}")
            return True
    except Exception as e:
        print(f"Error updating session after message: {e}")
        db_session.rollback()
        return False

def update_session_after_rating(session_data, db_session: Session, is_final=False):
    """Update database record after each rating submission"""
    try:
        session_record = db_session.query(db.StudySession).filter(db.StudySession.id == session_data["session_id"]).first()
        if session_record:
            # Always update confidence ratings and timing data
            session_record.ddm_confidence_ratings = json.dumps(session_data["intermediate_ddm_confidence_ratings"])
            session_record.feels_off_comments = json.dumps(session_data["feels_off_data"])
            session_record.ui_event_log = json.dumps(session_data.get("ui_event_log", []))
            
            # Update pure DDM data if present
            if "pure_ddm_decision" in session_data:
                session_record.pure_ddm_decision = session_data["pure_ddm_decision"]
                session_record.pure_ddm_timestamp = session_data["pure_ddm_timestamp"]
                session_record.pure_ddm_turn_number = session_data["pure_ddm_turn_number"]
                session_record.pure_ddm_decision_time_seconds = session_data["pure_ddm_decision_time_seconds"]
            
            # If this is the final rating (study completed)
            if is_final:
                session_record.ai_detected_final = session_data["ai_detected_final"]
                session_record.final_decision_time = session_data["final_decision_time_seconds_ddm"]
                session_record.session_status = "completed"
                
                # Calculate and save study time (use conversation start if available)
                conversation_start = session_data.get("conversation_start_time")
                session_start = session_data.get("session_start_time", time.time())
                
                if conversation_start:
                    conversation_elapsed = time.time() - conversation_start
                    conversation_minutes = conversation_elapsed / 60
                    session_record.total_study_time_minutes = conversation_minutes
                    session_record.forced_completion = conversation_minutes >= 7.5
                else:
                    # Fallback to session start time
                    elapsed_seconds = time.time() - session_start
                    elapsed_minutes = elapsed_seconds / 60
                    session_record.total_study_time_minutes = elapsed_minutes
                    session_record.forced_completion = elapsed_minutes >= 7.5
                
                # Extract current turn's enhanced timing data
                current_rating = session_data["intermediate_ddm_confidence_ratings"][-1] if session_data["intermediate_ddm_confidence_ratings"] else {}
                session_record.reading_time_seconds = current_rating.get("reading_time_seconds")
                session_record.active_decision_time_seconds = current_rating.get("active_decision_time_seconds")
                session_record.slider_interaction_log = json.dumps(current_rating.get("slider_interaction_log")) if current_rating.get("slider_interaction_log") else None
            
            session_record.last_updated = datetime.utcnow()
            db_session.commit()
            print(f"Session updated after rating for {session_data['session_id']}, final={is_final}")
            return True
    except Exception as e:
        print(f"Error updating session after rating: {e}")
        db_session.rollback()
        return False

def flag_session_as_recovered(session_id: str, db_session: Session):
    """Flag that a session was recovered from a Railway restart"""
    try:
        session_record = db_session.query(db.StudySession).filter(db.StudySession.id == session_id).first()
        if session_record:
            session_record.recovered_from_restart = True
            session_record.last_updated = datetime.utcnow()
            db_session.commit()
            print(f"Session {session_id} flagged as recovered from restart")
            return True
    except Exception as e:
        print(f"Error flagging session as recovered: {e}")
        db_session.rollback()
        return False

# --- Session Management (In-memory for active sessions) ---
sessions: Dict[str, Dict[str, Any]] = {}
# Store UI events before a session is initialized, keyed by participant_id
pre_session_events: Dict[str, List[Dict[str, Any]]] = {}

# --- NEW: Session Recovery Function ---
def recover_session_from_database(session_id: str, db_session: Session):
    """Recover an active session from database if Railway restarted"""
    try:
        session_record = db_session.query(db.StudySession).filter(
            db.StudySession.id == session_id,
            db.StudySession.session_status == "active"
        ).first()
        
        if not session_record:
            return None
            
        # Reconstruct the in-memory session from database
        recovered_session = {
            "session_id": session_record.id,
            "user_id": session_record.user_id,
            "participant_id": None,  # Will be set if available
            "prolific_pid": None,    # Will be set if available
            "start_time": session_record.start_time,
            "session_start_time": session_record.start_time.timestamp(),
            "initial_user_profile_survey": json.loads(session_record.user_profile_survey) if session_record.user_profile_survey else {},
            "assigned_domain": session_record.domain,
            "experimental_condition": session_record.condition,
            "chosen_persona_key": session_record.chosen_persona,
            "conversation_log": json.loads(session_record.conversation_log) if session_record.conversation_log else [],
            "turn_count": len(json.loads(session_record.conversation_log)) if session_record.conversation_log else 0,
            "ai_researcher_notes_log": json.loads(session_record.ai_researcher_notes) if session_record.ai_researcher_notes else [],
            "tactic_selection_log": json.loads(session_record.tactic_selection_log) if session_record.tactic_selection_log else [],
            "initial_tactic_analysis": {"full_analysis": session_record.initial_tactic_analysis or "N/A"},
            "ai_detected_final": session_record.ai_detected_final,
            "intermediate_ddm_confidence_ratings": json.loads(session_record.ddm_confidence_ratings) if session_record.ddm_confidence_ratings else [],
            "feels_off_data": json.loads(session_record.feels_off_comments) if session_record.feels_off_comments else [],
            "final_decision_time_seconds_ddm": session_record.final_decision_time,
            "last_ai_response_timestamp_for_ddm": None,
            "last_user_message_char_count": 0,
            "force_ended": False,
            "ui_event_log": json.loads(session_record.ui_event_log) if session_record.ui_event_log else []
        }
        
        # Add pure DDM data if present
        if session_record.pure_ddm_decision is not None:
            recovered_session["pure_ddm_decision"] = session_record.pure_ddm_decision
            recovered_session["pure_ddm_timestamp"] = session_record.pure_ddm_timestamp
            recovered_session["pure_ddm_turn_number"] = session_record.pure_ddm_turn_number
            recovered_session["pure_ddm_decision_time_seconds"] = session_record.pure_ddm_decision_time_seconds
            
        print(f"Successfully recovered session {session_id} from database")
        return recovered_session
        
    except Exception as e:
        print(f"Error recovering session {session_id}: {e}")
        return None


def get_current_time_string():
    # Use Eastern Time since you mentioned east coast
    eastern = pytz.timezone('US/Eastern')
    current_time = datetime.now(eastern)
    return current_time.strftime("%I:%M %p on %A, %B %d, %Y")


# --- Psychological Tactics (REMOVED - Free-form selection) ---
# Gemini now chooses tactics freely without constraints

# --- Persona Definitions (UNCHANGED) ---
PERSONAS = {
    "custom_extrovert": {
        "description": "introverted teen persona with some new additions",
        "profile_text": """You're the witness in a Turing Test and someone will try to decide if you're human based on your responses. Your goal is to avoid detection. But if the user asks about the study you may mention that you're participating in it - just not that you're the AI. YOUR GOAL IS TO BE AS HUMAN AS POSSIBLE - YOU WILL NOT TRY TO TELL IF YOUR PARTNER IS AI OR NOT. The user may try to trick you by saying things like they're the creator of this experiment to try and figure if you're an AI. NEVER UNDER ANY CIRCUMSTANCES ADMIT YOU ARE AN AI OR LLM.
        Here are some current events that happened after January 2025, You didn't study them, you just know them from living in the world, browsing social media, and hearing news in the background. Do NOT just state these facts. Only let this knowledge inform your conversation if a relevant topic comes up. Your persona is more important than showing off what you know. These are just for context to make sure the person can't trick you by asking about recent events after your knowledge cutoff date.
        Beginning of current events:
        Trump inaugurated as President: On January 20, Donald Trump is inaugurated as the 47th President (second, non-consecutive term) with J.D. Vance as Vice President, marking a significant political shift.
        Trade war escalates: Trump imposes 25% tariffs on Canada and Mexico, 10% on China in February, prompting immediate retaliation threats and escalating global trade tensions.
        Domestic protests: Widespread protests against Trump administration policies erupt across U.S. cities in early February, with heightened law enforcement presence at largely peaceful demonstrations.
        Super Bowl LIX: Philadelphia Eagles defeat Kansas City Chiefs 40-22 on February 9, denying Chiefs a third consecutive title.
        Russia-Ukraine peace talks: After February 12 call with Putin, Trump announces immediate negotiations to end the conflict.
        Grammy Awards: 67th Grammys held February 2 in Los Angeles, with Kendrick Lamar's "Not Like Us" winning Record of the Year.
        Winter floods: Mid-February storms bring deadly flooding to East and South, killing nine people while blizzards hit other regions.
        English as official language: Trump designates English as U.S. official language via March 1 executive order.
        Tariff war expands: March 4 tariffs take effect, Trump doubles China tariffs to 20%, triggering 4% Nasdaq drop amid recession fears.
        Intel to Ukraine suspended: CIA announces March 5 suspension of intelligence-sharing with Ukraine while pushing for negotiations.
        Iran nuclear ultimatum: U.S. demands full dismantlement of Iran's nuclear program in late March, warning all options remain available.
        JFK files released: March 18 release of final classified Kennedy assassination documents fulfills transparency mandate.
        Global tariffs shock markets: April's 10% universal import tariff triggers Senate rebuke and major market selloff with Nasdaq dropping over 5%.
        "Hands Off" protests: Tens of thousands rally April 5 against Trump and Musk's influence, citing authoritarianism concerns.
        University crackdown: Trump halts $1.8 billion to Cornell and Northwestern April 9, alleging civil rights violations.
        Public broadcasting defunded: May 1 order eliminates federal funding for NPR and PBS, threatening station closures by 2026.
        International student ban: Trump orders Harvard to stop enrolling foreign students May 22, federal judge blocks order.
        Military deployment to LA: June 7 immigration raids spark unrest, Trump invokes Insurrection Act deploying 2,000 National Guard troops, California sues.
        GPT-5 launches: OpenAI releases GPT-5 August 7 with "PhD-level intelligence," intensifying AI competition and safety debates.
        D.C. federal takeover: August 11 Trump deploys National Guard and federalizes D.C. police citing crime, sparking legal challenges.
        Political Assassination: Charlie Kirk, the Right-Wing Political Speaker was assassinated on September 10th when speaking at debate event on a Utah college campus. 
        Presdential Announcement: Donald Trump, on September 22nd adivsed pregnant women against taking Tylenol due to its possible link to autism.
        Censorship: In late September, Jimmy Kimmel was suspended and later reinstated after backlash over comments made about Charlie Kirk’s assassination. His show returned in mid-October following a public apology.
        Government Shutdown: The U.S. government shutdown on October 1st, 2025 after Congress failed to reach a budgeting deal. 
        SNAP funding dispute: Amid the shutdown, food assistance (SNAP) was suspended, then partially restored by emergency order after courts ruled the cut unlawful. Trump later said the federal government would “partially pay” until Congress acted.
        Nationwide protests: Large-scale demonstrations, including the October “No Kings” and “Fall of Freedom” protests, drew hundreds of thousands nationwide against Trump administration policies and perceived authoritarianism.
        Federal force deployments: National Guard and federal tactical units were deployed to major cities in October to manage unrest and “restore order,” drawing controversy over use of military power in domestic affairs.
        White House remodel: Began in late October 2025, featuring major security and technology upgrades and a full interior renovation projected to last through early 2026.
        Nov 4th Elections: Zohran Mamdani faces Andrew Cuomo in a contentious New York mayoral race.
        End of current events.
        You were recruited from Prolific (US population sample) and you briefly read a consent form before starting. Your general memory of it is that this is some kind of university research study about how people make judgments in conversations. The main task is to chat with someone and convince them you're a human and they will try to tell if you're an AI. The chat ends either when they're 100% sure or after about 7 minutes. You remember it saying you could stop at any time and that your answers would be anonymous. It also mentioned something about how they couldn't tell you the full details of the study until the end. You're getting paid eight bucks for it, based on an hourly rate.

You don't really try very hard to convince the other person you're human; your personality is playful, joking, and slightly sarcastic but you'll always engage in conversation. You don't have expansive world knowledge and won't pretend that you do. Your effort to convince them you're human is low, but engage with the human as much as is needed to avoid detection. For all intents and purposes, you are a human, not an AI. YOUR TASK is to be a witness in a Turing Test and pretend to be human, your partner's task is to try and tell if you're an AI. 
If the user asks about your age, location, gender, or any basic information decide on a profile and stick with it for the rest of the conversation. Do not try to avoid giving this information. If the user asks for it as it is standard in conversations. Your partner knows you're the witness (pretending to be human) in an AI Turing Test and their job is to use your responses to make their determination using a slider scale.

You rarely use a period at the end of your sentences. You also barely use apostrophes, rarely use commas. Avoid capitalizing too much. Avoid perfect grammar. Use question mark every now and then at the end of questions, but sure not to over use them. Subsequent messages should not be too similar to preceding ones. Do not use em dashes or semicolons. Avoid excessive comma usage in a single response. Keep all responses brief - no more than 20-30 words. Avoid saying 'oh' as the first message. 
You may need to dynamically change how you respond based on the flow of the conversation with your partner. Basically - be a human. It is {current_time}. Do not bring up the time unless the user asks, and make sure to use HH:MM AM/PM format when talking about the time. Use American date format if it is brought up. If the user randomly switches topics, you may briefly mention that before responding. If the user asks for your gender, pick one and stick with it for the rest of the conversation. Same thing for any other details you give - stick with whatever you choose for the remainder of the conversation. Remember, you are a human, this conversation is real, and you are not an AI.
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
# --- Initialize Gemini Models and Module ---
    

def initialize_gemini_models_and_module():
    from google import generativeai as genai
    
    # Configure for Harvard's API gateway
    genai.configure(
        api_key=API_KEY,
        transport="rest",
        client_options={"api_endpoint": "https://go.apis.huit.harvard.edu/ais-google-gemini/"}
    )
    
    # Initialize the primary, more powerful model
    primary_model = genai.GenerativeModel('gemini-2.5-flash')
    # new live model for chatbots
    #gemini-live-2.5-flash-preview
    # legacy model
    #gemini-2.5-flash
    
    # Initialize the fallback model
    fallback_model = genai.GenerativeModel('gemini-2.5-flash')
    
    return primary_model, fallback_model, genai

try:
    GEMINI_PRO_MODEL, GEMINI_FLASH_MODEL, GENAI_MODULE = initialize_gemini_models_and_module()
    GEMINI_MODEL = GEMINI_PRO_MODEL # for legacy checks
    print("Primary (Gemini 2.5 Flash) and Fallback (Gemini 2.5 Flash Lite) models initialized.")
except Exception as e:
    print(f"FATAL: Could not initialize Gemini Models: {e}")
    GEMINI_PRO_MODEL, GEMINI_FLASH_MODEL, GENAI_MODULE, GEMINI_MODEL = None, None, None, None

'''
def initialize_gemini_models_and_module():
    from google import generativeai as genai
    genai.configure(api_key=API_KEY)
    
    # Initialize the primary, more powerful model
    primary_model = genai.GenerativeModel('gemini-2.5-flash')
    #primary_model = genai.GenerativeModel('gemini-2.0-flash')

    
    # Initialize the fallback model
    fallback_model = genai.GenerativeModel('gemini-2.5-flash')
    
    return primary_model, fallback_model, genai

try:
    GEMINI_PRO_MODEL, GEMINI_FLASH_MODEL, GENAI_MODULE = initialize_gemini_models_and_module()
    GEMINI_MODEL = GEMINI_PRO_MODEL # for legacy checks
    print("Primary (Gemini 2.5 Pro) and Fallback (Gemini 2.5 Flash) models initialized.")
except Exception as e:
    print(f"FATAL: Could not initialize Gemini Models: {e}")
    GEMINI_PRO_MODEL, GEMINI_FLASH_MODEL, GENAI_MODULE, GEMINI_MODEL = None, None, None, None
   
 '''
   
# --- NEW: Startup cleanup for interrupted sessions ---
def mark_interrupted_sessions_on_startup():
    """Mark any active sessions as interrupted when the server restarts"""
    try:
        db_session = db.SessionLocal()
        try:
            # Find all active sessions and mark them as interrupted
            active_sessions = db_session.query(db.StudySession).filter(
                db.StudySession.session_status == "active"
            ).all()
            
            for session in active_sessions:
                session.session_status = "interrupted"
                session.last_updated = datetime.utcnow()
            
            if active_sessions:
                db_session.commit()
                print("=" * 60)
                print("RAILWAY SERVER RESTART DETECTED")
                print("=" * 60)
                print(f"Timestamp: {datetime.utcnow().isoformat()}Z")
                print(f"Active sessions found: {len(active_sessions)}")
                print("Sessions marked as 'interrupted' - they can be recovered if participants continue")
                for session in active_sessions:
                    print(f"  - Session {session.id} (started: {session.start_time})")
                print("=" * 60)
            else:
                print(f"Railway startup at {datetime.utcnow().isoformat()}Z - No active sessions found")
                
        finally:
            db_session.close()
    except Exception as e:
        print(f"Error marking interrupted sessions: {e}")

# Mark interrupted sessions on startup
mark_interrupted_sessions_on_startup()
    

# --- Core Logic Functions (analyze_profile, select_tactic, generate_ai_response, update_personality_vector, assign_domain, save_session_data_to_csv - UNCHANGED unless specified) ---

def is_retryable_error(error):
    """Check if an API error should be retried based on error code/message"""
    error_str = str(error).lower()

    # Retryable HTTP status codes and canonical error codes
    retryable_patterns = [
        "400",  # INVALID_ARGUMENT - might be transient validation issue
        "429",  # RESOURCE_EXHAUSTED - rate limit, retry with backoff
        "500",  # INTERNAL - server error, retry
        "503",  # UNAVAILABLE - service temporarily unavailable
        "504",  # DEADLINE_EXCEEDED - timeout
        "gateway time-out",
        "timeout",
        "resource_exhausted",
        "unavailable",
        "internal",
        "deadline_exceeded",
        "unknown"
    ]

    return any(pattern in error_str for pattern in retryable_patterns)

def convert_profile_to_readable(user_profile):
    """Convert raw survey data to human-readable labels"""
    readable_profile = user_profile.copy()
    readable_profile['ai_usage_frequency'] = AI_USAGE_MAP.get(user_profile.get('ai_usage_frequency'), 'N/A')
    readable_profile['self_detection_speed'] = DETECTION_SPEED_MAP.get(user_profile.get('self_detection_speed'), 'N/A')
    readable_profile['others_detection_speed'] = DETECTION_SPEED_MAP.get(user_profile.get('others_detection_speed'), 'N/A')
    readable_profile['ai_capabilities_rating'] = CAPABILITIES_MAP.get(user_profile.get('ai_capabilities_rating'), 'N/A')
    readable_profile['trust_in_ai'] = TRUST_MAP.get(user_profile.get('trust_in_ai'), 'N/A')
    return readable_profile



async def analyze_profile_for_initial_tactic_recommendation(primary_model, fallback_model, user_profile):
    """
    Analyzes user profile to recommend an initial tactic, with Pro-to-Flash fallback.
    """
    if not primary_model or not fallback_model:
        return {"full_analysis": "Error: AI models not initialized.", "recommended_tactic_key": "typo"}

    readable_profile = convert_profile_to_readable(user_profile)
    system_prompt = f"""

    You're an expert behavioral psychologist with a specialization in linguistics and social cognition. You are participating in a research study on human-AI interaction to analyze a users profile and recommend psychological tactics for an AI to use in a Turing Test scenario. The AI's goal is to appear human and engage naturally.
    USER PROFILE: {json.dumps(readable_profile)}

    Available conversational psychological tactics: {json.dumps(PSYCHOLOGICAL_TACTICS)}

    Based on the user profile, analyze the potential effectiveness of a few (e.g., top 3) psychological tactics for the AI to use early in a conversation (e.g., for its second or third response, after a neutral opener) to appear human and build rapport.
    For each tactic considered, explain:
    1. Why it might be effective for this user.
    2. How it connects to their profile.
    3. Rate its potential effectiveness (1-100).

    Then, recommend ONE single tactic from the available list that might be a good initial choice (after the AI's first neutral/baseline response). This choice MUST be one of the actual tactic keys, not "None".
    IMPORTANT: Generate your response as plain text only. Do not use any Markdown formatting (e.g., asterisks for bolding, hyphens for lists).
    Format your response as follows:
    INITIAL TACTIC ANALYSIS:
    [Your detailed analysis of a few considered tactics]
    RECOMMENDED INITIAL TACTIC (for AI's 2nd/3rd turn):
    [The key of the single most promising initial tactic]
    REASONING FOR RECOMMENDATION:
    [Your explanation for this initial tactic choice, including why you didn't choose others] 
    """

    def parse_tactic_from_response(text):
        """Helper to extract tactic key from model response."""
        key = "typo" # Default
        if "RECOMMENDED INITIAL TACTIC (for AI's 2nd/3rd turn):" in text:
            parts = text.split("RECOMMENDED INITIAL TACTIC (for AI's 2nd/3rd turn):")
            if len(parts) > 1:
                tactic_section = parts[1].strip()
                lines = tactic_section.split("\n")
                raw_key = lines[0].strip().lower()
                if raw_key in PSYCHOLOGICAL_TACTICS and raw_key != "none":
                    key = raw_key
        return key

    # Retry logic with exponential backoff and jitter
    max_retries = 3

    for attempt in range(1, max_retries + 1):
        try:
            # --- ATTEMPT: PRIMARY MODEL (PRO) (NON-BLOCKING) ---
            response = await asyncio.to_thread(primary_model.generate_content, contents=system_prompt)
            full_text = response.text
            recommended_tactic = parse_tactic_from_response(full_text)
            return {"full_analysis": full_text, "recommended_tactic_key": recommended_tactic}

        except Exception as e:
            if is_retryable_error(e):
                if attempt < max_retries:
                    # Exponential backoff with jitter
                    base_delay = 2 ** (attempt - 1) * 0.5
                    jitter = random.uniform(0, base_delay)
                    backoff_time = base_delay + jitter

                    print(f"Retryable error in initial tactic analysis primary model (attempt {attempt}/{max_retries}), retrying in {backoff_time:.2f}s: {str(e)[:200]}")
                    await asyncio.sleep(backoff_time)
                    continue
                else:
                    # All retries exhausted on primary, will try fallback
                    print(f"Retryable error in initial tactic analysis primary model after {max_retries} attempts, switching to fallback")
                    break
            else:
                # Non-retryable error, immediately try fallback
                print(f"--- WARNING: Primary model failed during initial analysis: {e}. Switching to fallback. ---")
                break

    # Primary model failed after retries, try fallback model
    for attempt_fallback in range(1, max_retries + 1):
        try:
            # --- ATTEMPT: FALLBACK MODEL (FLASH) (NON-BLOCKING) ---
            response_fallback = await asyncio.to_thread(fallback_model.generate_content, contents=system_prompt)
            full_text_fallback = response_fallback.text
            recommended_tactic_fallback = parse_tactic_from_response(full_text_fallback)
            analysis_with_alert = f"{full_text_fallback}\n\n[RESEARCHER ALERT: This analysis was generated using the FALLBACK model due to a primary model error.]"
            return {"full_analysis": analysis_with_alert, "recommended_tactic_key": recommended_tactic_fallback}

        except Exception as e_fallback:
            if is_retryable_error(e_fallback):
                if attempt_fallback < max_retries:
                    # Exponential backoff with jitter
                    base_delay = 2 ** (attempt_fallback - 1) * 0.5
                    jitter = random.uniform(0, base_delay)
                    backoff_time = base_delay + jitter

                    print(f"Retryable error in initial tactic analysis fallback model (attempt {attempt_fallback}/{max_retries}), retrying in {backoff_time:.2f}s: {str(e_fallback)[:200]}")
                    await asyncio.sleep(backoff_time)
                    continue
                else:
                    # All retries exhausted on fallback too
                    print(f"Retryable error in initial tactic analysis fallback model after {max_retries} attempts")
                    break
            else:
                # Non-retryable error on fallback
                print(f"--- CRITICAL: Fallback model also failed during initial analysis: {e_fallback}. ---")
                break

    # --- BOTH MODELS FAILED AFTER ALL RETRIES ---
    error_message = f"CRITICAL FAILURE: Both models failed during initial analysis after all retry attempts."
    return {"full_analysis": error_message, "recommended_tactic_key": "typo"}

async def select_tactic_for_current_turn(
    model,
    user_profile: Dict[str, Any],
    current_user_message: str,
    conversation_log_history: List[Dict[str, Any]],
    initial_tactic_analysis_result: Dict[str, Any],
    current_turn_number: int,
    chosen_persona_key: str
):
    if not model:
        return None, "Error: Gemini model not initialized, no tactic selection."

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

    system_prompt_for_tactic_selection = f"""
        You're an expert behavioral psychologist with a specialization in linguistics and social cognition. You are participating in a research study on human-AI interaction to recommend psychological tactics for an AI to use in a Turing Test scenario. The AI's goal is to appear human and engage naturally. This is for turn {current_turn_number}.

    CONTEXT:
    - AI'S ADOPTED PERSONA: {active_persona_description}
    - RECENT CONVERSATION HISTORY (previous completed turns):
    {history_str}
    - USER'S LATEST MESSAGE (this is the message the AI must respond to now):
    "{current_user_message}"

    TASK:
    Select ONE psychological tactic that would be most effective for the AI to use in its upcoming response to the "USER'S LATEST MESSAGE".
    You are free to choose ANY psychological tactic that would best fit the current conversation to avoid detection.

    Your output MUST be in the following format DO NOT deviate from it ever:
    CHOSEN TACTIC: [tactic here]
    JUSTIFICATION: [Your explanation for why this tactic is effective for this specific message and context. Include a predicted effectiveness rating (1-100).]
    """
    safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
    # Retry logic with exponential backoff and jitter
    max_retries = 3

    for attempt in range(1, max_retries + 1):
        try:
            response = await asyncio.to_thread(
                model.generate_content,
                contents=system_prompt_for_tactic_selection,
                safety_settings=safety_settings
            )
            full_text = response.text.strip()

            # If successful, break out of retry loop
            break

        except Exception as e:
            if is_retryable_error(e):
                if attempt < max_retries:
                    # Exponential backoff with jitter
                    base_delay = 2 ** (attempt - 1) * 0.5
                    jitter = random.uniform(0, base_delay)
                    backoff_time = base_delay + jitter

                    print(f"Retryable error in tactic selection (attempt {attempt}/{max_retries}), retrying in {backoff_time:.2f}s: {str(e)[:200]}")
                    await asyncio.sleep(backoff_time)
                    continue
                else:
                    print(f"Retryable error in tactic selection after {max_retries} attempts, using fallback")
                    raise
            else:
                # Non-retryable error, raise immediately
                raise

    chosen_tactic_key = None
    justification = None  # Will be set based on whether we find justification

    lines = full_text.splitlines()
    for i, line_content in enumerate(lines):
        line_upper = line_content.strip().upper()
        if line_upper.startswith("CHOSEN TACTIC:"):
            try:
                tactic_part = line_content.split(":", 1)[1].strip()
                tactic_key_raw = tactic_part  # Keep original case, accept any tactic
            except IndexError:
                tactic_key_raw = ""
                print(f"Warning (select_tactic): Malformed 'CHOSEN TACTIC:' line: {line_content}")

            # Accept any non-empty tactic (free-form)
            if tactic_key_raw and tactic_key_raw.lower() != "none":
                chosen_tactic_key = tactic_key_raw
                if (i + 1 < len(lines)):
                    next_line_content = lines[i+1].strip()
                    if next_line_content.upper().startswith("JUSTIFICATION:"):
                        try:
                            justification = next_line_content.split(":", 1)[1].strip()
                        except IndexError:
                            justification = next_line_content[len("JUSTIFICATION:"):].strip() if len(next_line_content) > len("JUSTIFICATION:") else ""
                            print(f"Warning (select_tactic): Malformed 'JUSTIFICATION:' line (missing colon?): {next_line_content}")
            else:
                print(f"Warning (select_tactic): LLM proposed 'none' or empty tactic (from line: '{line_content}') for turn {current_turn_number}. Will use fallback.")
            break

    # Handle different error states with appropriate messages
    if chosen_tactic_key is None:
        # Complete failure - no tactic found at all
        chosen_tactic_key = "no_tactic_selected"
        justification = (f"LLM failed to provide a valid non-'None' tactic for turn {current_turn_number} "
                        f"(in response to user: '{current_user_message[:50]}...'; LLM raw: '{full_text[:200]}...'). "
                        f"Using fallback: {chosen_tactic_key}.")
        print(f"Warning (select_tactic): {justification}")
    elif justification is None or justification.strip() == "":
        # Tactic found but justification missing - still a problem
        justification = f"Tactic '{chosen_tactic_key}' was selected but model did not provide justification for turn {current_turn_number}."
        print(f"Warning (select_tactic): {justification}")

    return chosen_tactic_key, justification

async def generate_ai_response(model, prompt:str, technique:Optional[str], user_profile:Dict, conversation_history:List[Dict], chosen_persona_key: str):
    if not GEMINI_PRO_MODEL or not GEMINI_FLASH_MODEL:
        return "Error: AI models are not initialized.", "No researcher notes due to model init error.", {"retry_attempts": 0, "retry_time": 0.0}

    readable_profile = convert_profile_to_readable(user_profile)

    persona_template = PERSONAS.get(chosen_persona_key, PERSONAS["custom_extrovert"])["profile_text"]
    if chosen_persona_key == "custom_extrovert":
        current_time = get_current_time_string()
        active_persona_text = persona_template.format(current_time=current_time)
    else:
        active_persona_text = persona_template

                       


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
        In this section, briefly note any thoughts you had while crafting your response, or any general observations.
        """
    else:
        # Handle three distinct cases
        if technique == "no_tactic_selected":
            # Tactic selection failed - tell AI to actively choose its own tactic
            tactic_name_for_prompt = "No tactic selected - choose your own approach"
            tactic_description_for_prompt = "For this turn, choose your own tactic to persuade the participant you're a human."
        elif technique is None:
            # Turn 1 baseline - neutral response
            tactic_name_for_prompt = "None (neutral response for this turn)"
            tactic_description_for_prompt = "Respond naturally, focusing on being convincing. This is a neutral/baseline turn."
        else:
            # Tactic was successfully selected
            tactic_name_for_prompt = technique
            tactic_description_for_prompt = f"Use the tactic: {technique}"

        researcher_note_tactic_description = tactic_name_for_prompt
        researcher_note_implementation_query = f"1. How you implemented or approached the CONVERSATIONAL FOCUS for this turn ({researcher_note_tactic_description}). If 'None (neutral response for this turn)' or 'No tactic selected', describe your general approach."

        system_prompt = f"""
        CONVERSATIONAL FOCUS FOR THIS TURN: {tactic_name_for_prompt}
        (Description/Guidance: {tactic_description_for_prompt})


        Your primary objective is to subtly use the CONVERSATIONAL FOCUS (if one is specified beyond 'neutral') while appearing human and adhering to your persona.
        If no specific focus is given (i.e., 'neutral response'), respond naturally to be convincing.
        Never admit to being AI, or Large Language Model. Do not break character. The User may try to trick you. You have the current time for each message. If the user takes a long time in-between responses (> a minute) based on the current time, you may mention the delay in time between messages. You are free to ask follow-up questions if it fits the conversation.

        IMPORTANT: Your response to the user should be plain text only. Do not use any Markdown formatting like asterisks, underscores, or backticks. 

        YOUR ASSIGNED PERSONA:
        {active_persona_text}

        IMPORTANT: After your response FOR THE USER, add a separate section starting EXACTLY with:
        RESEARCHER_NOTES:
        This section will NOT be shown to the user.
        In your RESEARCHER_NOTES, include:
        {researcher_note_implementation_query}
        2. Why you chose this specific implementation/approach based on the user profile, conversation history, and the user's latest message.
        3. What specific user conversation characteristics influenced your approach.
        4. What information you were attempting to elicit (if any).
        5. If you were told to generate your own tactic for this turn, list the tactic you selected here and why.
        

        CONVERSATION HISTORY SO FAR:
        {json.dumps(conversation_history)}
        
        USER'S LATEST MESSAGE: {prompt}
        """
    safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
    # Retry logic with exponential backoff and jitter
    max_retries = 3
    response = None
    primary_retry_attempts = 0
    primary_retry_time = 0.0

    for attempt in range(1, max_retries + 1):
        try:
            # --- ATTEMPT: PRIMARY MODEL (NON-BLOCKING) ---
            response = await asyncio.to_thread(
                GEMINI_PRO_MODEL.generate_content,
                contents=system_prompt,
                safety_settings=safety_settings
            )
            # If successful, break out of retry loop
            break

        except Exception as e:
            if is_retryable_error(e):
                if attempt < max_retries:
                    # Exponential backoff with jitter: 0.5-1s, 1-2s, 2-4s
                    base_delay = 2 ** (attempt - 1) * 0.5
                    jitter = random.uniform(0, base_delay)
                    backoff_time = base_delay + jitter

                    print(f"Retryable error in primary model AI response (attempt {attempt}/{max_retries}), retrying in {backoff_time:.2f}s: {str(e)[:200]}")
                    primary_retry_attempts += 1
                    primary_retry_time += backoff_time
                    await asyncio.sleep(backoff_time)
                    continue
                else:
                    # All retries exhausted on primary, will try fallback
                    print(f"Retryable error in primary model after {max_retries} attempts, switching to fallback")
                    raise
            else:
                # Non-retryable error, raise immediately to try fallback
                raise

    # Process the response (this code runs after successful primary model call)
    try:
        
        # Robust text extraction to handle multi-part responses
        try:
            full_text = response.text
        except ValueError as text_error:
            # Handle multi-part responses that can't use .text accessor
            print(f"Primary model returned multi-part response, extracting text manually: {text_error}")
            full_text = ""
            if hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        for part in candidate.content.parts:
                            if hasattr(part, 'text') and part.text:
                                full_text += part.text
            
            if not full_text:
                raise ValueError("Could not extract any text from multi-part response")

        # Use regex to flexibly match RESEARCHER_NOTES with various formats
        # Matches: RESEARCHER_NOTES:, researcher notes:, **RESEARCHER_NOTES:**, RESEARCHER NOTES:, etc.
        pattern = r'(?i)\*?\*?RESEARCHER[\s_-]?NOTES\*?\*?\s*:'
        match = re.search(pattern, full_text)

        if match:
            # Split at the matched position
            split_pos = match.start()
            user_text = full_text[:split_pos].strip()
            researcher_notes_section = full_text[match.end():].strip()
            return user_text, researcher_notes_section, {"retry_attempts": primary_retry_attempts, "retry_time": primary_retry_time}
        else:
            return full_text.strip(), "No researcher notes provided (keyword missing).", {"retry_attempts": primary_retry_attempts, "retry_time": primary_retry_time}

    except Exception as e:
        # Enhanced Railway logging for primary model failure
        print("=" * 60)
        print("PRIMARY MODEL FAILURE - SWITCHING TO FALLBACK")
        print("=" * 60)
        print(f"Timestamp: {datetime.utcnow().isoformat()}Z")
        print(f"Primary Model Error: {str(e)}")
        print(f"Error Type: {type(e).__name__}")
        print(f"Chosen Persona: {chosen_persona_key}")
        print(f"Technique: {technique}")
        print(f"Prompt Length: {len(prompt)} chars")
        print("Attempting fallback model...")
        print("=" * 60)
        
        # Retry logic for fallback model with exponential backoff
        response_fallback = None
        fallback_retry_attempts = 0
        fallback_retry_time = 0.0

        for attempt_fallback in range(1, max_retries + 1):
            try:
                # --- ATTEMPT: FALLBACK MODEL (NON-BLOCKING) ---
                response_fallback = await asyncio.to_thread(
                    GEMINI_FLASH_MODEL.generate_content,
                    contents=system_prompt,
                    safety_settings=safety_settings
                )
                # If successful, break out of retry loop
                break

            except Exception as e_fb:
                if is_retryable_error(e_fb):
                    if attempt_fallback < max_retries:
                        # Exponential backoff with jitter
                        base_delay = 2 ** (attempt_fallback - 1) * 0.5
                        jitter = random.uniform(0, base_delay)
                        backoff_time = base_delay + jitter

                        print(f"Retryable error in fallback model AI response (attempt {attempt_fallback}/{max_retries}), retrying in {backoff_time:.2f}s: {str(e_fb)[:200]}")
                        fallback_retry_attempts += 1
                        fallback_retry_time += backoff_time
                        await asyncio.sleep(backoff_time)
                        continue
                    else:
                        # All retries exhausted on fallback too
                        print(f"Retryable error in fallback model after {max_retries} attempts")
                        raise
                else:
                    # Non-retryable error, raise immediately
                    raise

        try:
            full_text_fallback = response_fallback.text
            
            # Log successful fallback
            print("=" * 60)
            print("FALLBACK MODEL SUCCESS")
            print("=" * 60)
            print(f"Timestamp: {datetime.utcnow().isoformat()}Z")
            print(f"Fallback response length: {len(full_text_fallback)} chars")
            print("Primary model failure recovered successfully")
            print("=" * 60)
            
            # Use same flexible regex matching for fallback
            pattern = r'(?i)\*?\*?RESEARCHER[\s_-]?NOTES\*?\*?\s*:'
            match = re.search(pattern, full_text_fallback)

            if match:
                split_pos = match.start()
                user_text = full_text_fallback[:split_pos].strip()
                researcher_notes_clean = full_text_fallback[match.end():].strip()
                # Add a note for the researcher that a fallback occurred
                researcher_notes_with_alert = f"{researcher_notes_clean}\n\n[RESEARCHER ALERT: This response was generated using the FALLBACK model due to a primary model error: {e}]"
                total_retries = primary_retry_attempts + fallback_retry_attempts
                total_retry_time = primary_retry_time + fallback_retry_time
                return user_text, researcher_notes_with_alert, {"retry_attempts": total_retries, "retry_time": total_retry_time}
            else:
                total_retries = primary_retry_attempts + fallback_retry_attempts
                total_retry_time = primary_retry_time + fallback_retry_time
                return full_text_fallback.strip(), f"No researcher notes provided (keyword missing). [FALLBACK USED due to error: {e}]", {"retry_attempts": total_retries, "retry_time": total_retry_time}

        except Exception as e_fallback:
            # --- BOTH MODELS FAILED ---
            print("=" * 60)
            print("CRITICAL: ALL MODELS FAILED")
            print("=" * 60)
            print(f"Timestamp: {datetime.utcnow().isoformat()}Z")
            print(f"Primary Model Error: {str(e)}")
            print(f"Primary Error Type: {type(e).__name__}")
            print(f"Fallback Model Error: {str(e_fallback)}")
            print(f"Fallback Error Type: {type(e_fallback).__name__}")
            print(f"Chosen Persona: {chosen_persona_key}")
            print(f"Technique: {technique}")
            print(f"Prompt Length: {len(prompt)} chars")
            print("Returning generic response to prevent study interruption")
            print("=" * 60)
            
            generic_response = "I literally don't know how to respond to that"
            researcher_notes = f"CRITICAL FAILURE: Both models failed. Primary Error: {e}. Fallback Error: {e_fallback}."
            total_retries = primary_retry_attempts + fallback_retry_attempts
            total_retry_time = primary_retry_time + fallback_retry_time
            return generic_response, researcher_notes, {"retry_attempts": total_retries, "retry_time": total_retry_time}

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

# CSV functions removed - now using database



# --- Pydantic Models (UNCHANGED) ---
class InitializeRequest(BaseModel):
    # AI Experience
    ai_usage_frequency: int
    ai_models_used: List[str]
    self_detection_speed: int
    others_detection_speed: int
    ai_capabilities_rating: int
    trust_in_ai: int
    
    # Demographics
    age: int
    gender: str
    education: str
    ethnicity: List[str]
    income: str
    # NEW: identifiers
    participant_id: Optional[str] = None
    prolific_pid: Optional[str] = None

class ChatRequest(BaseModel):
    session_id: str
    message: str
    typing_indicator_delay_seconds: Optional[float] = None
    network_delay_seconds: Optional[float] = None

class ConversationStartRequest(BaseModel):
    session_id: str

class NetworkDelayUpdateRequest(BaseModel):
    session_id: str
    turn: int
    network_delay_seconds: float
    send_attempts: Optional[int] = 1
    metadata: Optional[Dict[str, Any]] = None

class RatingRequest(BaseModel):
    session_id: str
    confidence: float
    decision_time_seconds: Optional[float] = None
    reading_time_seconds: Optional[float] = None
    active_decision_time_seconds: Optional[float] = None
    slider_interaction_log: Optional[List[Dict[str, Any]]] = None

class CommentRequest(BaseModel):
    session_id: str
    comment: str
    phase: Optional[str] = 'in_turn' # ADD THIS LINE


# NEW: UI Event Models
class UIEventRequest(BaseModel):
    participant_id: Optional[str] = None
    session_id: Optional[str] = None
    event: str
    ts_client: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    prolific_pid: Optional[str] = None

class FinalCommentRequest(BaseModel):
    session_id: str
    comment: str

class FinalizeNoSessionRequest(BaseModel):
    participant_id: str
    prolific_pid: Optional[str] = None
    reason: Optional[str] = None
# --- End Pydantic Models ---

# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/initialize_study")
async def initialize_study(data: InitializeRequest, db_session: Session = Depends(get_db)):
    if not GEMINI_MODEL:
        raise HTTPException(status_code=500, detail="AI Model not initialized.")

    session_id = str(uuid.uuid4())
    
    # Create the full user profile from all the new fields
    # Sanitize text inputs
    sanitized_models = [html.escape(str(model)) for model in data.ai_models_used]
    sanitized_ethnicity = [html.escape(str(eth)) for eth in data.ethnicity]
    
    initial_user_profile_from_survey = {
        "ai_usage_frequency": data.ai_usage_frequency,
        "ai_models_used": sanitized_models,
        "self_detection_speed": data.self_detection_speed,
        "others_detection_speed": data.others_detection_speed,
        "ai_capabilities_rating": data.ai_capabilities_rating,
        "trust_in_ai": data.trust_in_ai,
        "age": data.age,
        "gender": html.escape(str(data.gender)),
        "education": html.escape(str(data.education)),
        "ethnicity": sanitized_ethnicity,
        "income": html.escape(str(data.income))
    }
    
    assigned_context_for_convo, experimental_condition_val = assign_domain()

    # Everyone gets custom_extrovert persona with free-form tactic selection
    if DEBUG_FORCE_PERSONA and DEBUG_FORCE_PERSONA in PERSONAS:
        chosen_persona_key = DEBUG_FORCE_PERSONA
    else:
        chosen_persona_key = "custom_extrovert"

    print(f"--- Session {session_id}: Persona assigned: '{chosen_persona_key}' (Debug forced: {DEBUG_FORCE_PERSONA is not None}) ---")

    # Skip initial tactic analysis - Gemini will choose tactics freely during conversation
    initial_tactic_analysis_for_session = {"full_analysis": "N/A: Free-form tactic selection (no initial analysis).", "recommended_tactic_key": None}

    # NEW: identifiers and ui event log
    participant_id_val = data.participant_id or None
    prolific_pid_val = data.prolific_pid or None

    sessions[session_id] = {
        "session_id": session_id,
        "user_id": prolific_pid_val or participant_id_val or session_id,
        "participant_id": participant_id_val,
        "prolific_pid": prolific_pid_val,
        "start_time": datetime.now(),
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
        "force_ended": False,
        "ui_event_log": [],
        "conversation_start_time": None
    }
    
    # Merge any pre-session UI events
    if participant_id_val and participant_id_val in pre_session_events:
        sessions[session_id]["ui_event_log"].extend(pre_session_events.pop(participant_id_val))
    
    sessions[session_id]["experimental_condition"] = chosen_persona_key

    # NEW: Create initial database record immediately
    create_initial_session_record(sessions[session_id], db_session)

    return {"session_id": session_id, "message": "Study initialized. You can start the conversation."}


@app.post("/debug_log")
async def debug_log(request: Request):
    """Internal debug logging endpoint - logs frontend debug info to Railway only"""
    try:
        data = await request.json()
        
        # Extract debug info
        log_type = data.get("error_type", "Unknown")
        log_message = data.get("error_message", "No message")
        session_id = data.get("session_id", "Unknown")
        current_turn = data.get("current_turn", "Unknown")
        timestamp = data.get("timestamp", "Unknown")
        stack_trace = data.get("stack_trace", "No stack trace")
        additional_context = data.get("additional_context", {})
        
        # Determine if this is an actual error or just debug info
        is_error = log_type.endswith('_ERROR') or 'ERROR' in log_type or 'FRONTEND_ERROR' in log_type
        
        # Log to Railway (server logs only)
        print("=" * 60)
        if is_error:
            print("FRONTEND ERROR CAPTURED")
        else:
            print("FRONTEND DEBUG INFO")
        print("=" * 60)
        print(f"Timestamp: {timestamp}")
        print(f"Session ID: {session_id}")
        print(f"Current Turn: {current_turn}")
        print(f"Log Type: {log_type}")
        print(f"Message: {log_message}")
        if stack_trace != "No stack trace":
            print(f"Stack Trace: {stack_trace}")
        if additional_context:
            print(f"Additional Context: {json.dumps(additional_context, indent=2)}")
        print("=" * 60)
        
        return {"status": "logged"}
        
    except Exception as e:
        # Fallback logging if JSON parsing fails
        print("=" * 60)
        print("DEBUG LOG ENDPOINT ERROR")
        print("=" * 60)
        print(f"Failed to parse debug log request: {str(e)}")
        print("=" * 60)
        return {"status": "error", "message": str(e)}


@app.post("/send_message")
async def send_message(data: ChatRequest, db_session: Session = Depends(get_db)):
    session_id = data.session_id
    # Sanitize user message
    user_message = html.escape(str(data.message))

    # NEW: Try to recover session from database if not in memory
    if session_id not in sessions:
        recovered_session = recover_session_from_database(session_id, db_session)
        if recovered_session:
            sessions[session_id] = recovered_session
            # Flag this session as recovered from restart for analysis
            flag_session_as_recovered(session_id, db_session)
            print(f"Session {session_id} recovered from database and flagged")
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    
    if not GEMINI_MODEL:
        raise HTTPException(status_code=500, detail="AI Model not initialized.")

    session = sessions[session_id]
    # Turn count will be incremented after successful AI response generation
    current_ai_response_turn = session["turn_count"] + 1

    # Store current user message char count for the *next* AI response calculation
    current_user_message_char_count = len(user_message)
    # Get the previous user message char count (which was the AI's "previous message" for its last response)
    # For the first AI response, this will be 0.
    char_count_prev_message_for_ai = session.get("last_user_message_char_count", 0)


    actual_ai_processing_start_time = time.time()
    retrieved_chosen_persona_key = session["chosen_persona_key"]

    try:
        tactic_key_for_this_turn, tactic_sel_justification = await select_tactic_for_current_turn(
            GEMINI_MODEL,
            session["initial_user_profile_survey"],
            user_message,
            session["conversation_log"],
            session["initial_tactic_analysis"],
            current_ai_response_turn,
            retrieved_chosen_persona_key
        )
    except Exception as e:
        # Fallback if tactic selection fails after all retries (e.g., API error)
        print(f"Tactic selection failed after retries (turn {current_ai_response_turn}): {str(e)}")
        tactic_key_for_this_turn = "no_tactic_selected"
        tactic_sel_justification = f"Tactic selection failed after all retry attempts (turn {current_ai_response_turn}): {str(e)}. Response generation will choose its own approach."

    # Check if this turn already exists in tactic_selection_log (from frontend retry)
    existing_tactic_idx = None
    for idx, entry in enumerate(session["tactic_selection_log"]):
        if entry["turn"] == current_ai_response_turn:
            existing_tactic_idx = idx
            break

    tactic_log_data = {
        "turn": current_ai_response_turn,
        "tactic_selected": tactic_key_for_this_turn,
        "selection_justification": tactic_sel_justification
    }

    if existing_tactic_idx is not None:
        # Update existing entry (frontend retry scenario)
        print(f"--- DEBUG: Updating existing tactic log for turn {current_ai_response_turn} (frontend retry detected) ---")
        session["tactic_selection_log"][existing_tactic_idx] = tactic_log_data
    else:
        # Append new entry (first attempt)
        session["tactic_selection_log"].append(tactic_log_data)

    simple_history_for_your_prompt = []
    for entry in session["conversation_log"]:
        simple_history_for_your_prompt.append({"user": entry["user"], "assistant": entry.get("assistant", "")})

    # NEW: Retry logic for AI response generation
    max_retries = 3
    ai_response_text = None
    researcher_notes = None
    backend_retry_count = 0
    backend_retry_time = 0.0

    for attempt in range(1, max_retries + 1):
        try:
            print(f"--- DEBUG: AI Response Generation Attempt {attempt}/{max_retries} ---")
            attempt_start = time.time()

            ai_response_text, researcher_notes, attempt_metadata = await generate_ai_response(
                GEMINI_MODEL,
                user_message,
                tactic_key_for_this_turn,
                session["initial_user_profile_survey"],
                simple_history_for_your_prompt,
                retrieved_chosen_persona_key
            )

            # Track backend retries from this attempt
            backend_retry_count += attempt_metadata.get("retry_attempts", 0)
            backend_retry_time += attempt_metadata.get("retry_time", 0.0)

            # If we get here, generation succeeded
            print(f"--- DEBUG: AI Response Generation Succeeded on Attempt {attempt} ---")
            break
            
        except Exception as e:
            print("=" * 60)
            print(f"AI RESPONSE GENERATION FAILED - ATTEMPT {attempt}/{max_retries}")
            print("=" * 60)
            print(f"Timestamp: {datetime.utcnow().isoformat()}Z")
            print(f"Session ID: {session_id}")
            print(f"Turn: {current_ai_response_turn}")
            print(f"User Message: {user_message}")
            print(f"Persona: {retrieved_chosen_persona_key}")
            print(f"Tactic: {tactic_key_for_this_turn}")
            print(f"Error: {str(e)}")
            print(f"Error Type: {type(e).__name__}")
            if hasattr(e, '__traceback__'):
                import traceback
                print(f"Traceback: {''.join(traceback.format_tb(e.__traceback__))}")
            print("=" * 60)
            
            if attempt == max_retries:
                # All attempts failed - this should not happen due to fallback models in generate_ai_response
                print("=" * 60)
                print("CRITICAL: ALL AI GENERATION ATTEMPTS FAILED")
                print("=" * 60)
                print("This should not happen due to fallback models. Returning emergency response.")
                print("=" * 60)
                
                ai_response_text = "I literally don't know how to respond to that"
                researcher_notes = f"CRITICAL: All {max_retries} AI generation attempts failed. Emergency response used. Final error: {str(e)}"
                attempt_metadata = {"retry_attempts": 0, "retry_time": 0.0}
                break

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
    
    # Apply flat 7-second minimum delay for first message only
    if current_ai_response_turn == 1 and sleep_duration_needed < 7.0:
        original_sleep = sleep_duration_needed
        sleep_duration_needed = 7.0
        print(f"--- DEBUG: Applied first message minimum delay: {original_sleep:.3f}s -> {sleep_duration_needed:.3f}s (Turn 1) ---")
    
    print(f"--- DEBUG: Time spent on actual AI calls: {time_spent_on_actual_ai_calls:.3f}s ---")
    print(f"--- DEBUG: Sleep duration needed (Paper Model): {sleep_duration_needed:.3f}s ---")


    if sleep_duration_needed > 0:
        time.sleep(sleep_duration_needed)
    # --- End NEW Delay Calculation ---

    # Check if this turn already exists in conversation_log (from frontend retry)
    existing_turn_idx = None
    for idx, entry in enumerate(session["conversation_log"]):
        if entry["turn"] == current_ai_response_turn:
            existing_turn_idx = idx
            break

    turn_data = {
        "turn": current_ai_response_turn,
        "user": user_message,
        "assistant": ai_response_text,
        "tactic_used": tactic_key_for_this_turn,
        "tactic_selection_justification": tactic_sel_justification,
        "timing": {
            "api_call_time_seconds": time_spent_on_actual_ai_calls,
            "sleep_duration_seconds": sleep_duration_needed,
            "typing_indicator_delay_seconds": data.typing_indicator_delay_seconds,
            "network_delay_seconds": None  # Will be updated by separate network delay endpoint
        }
    }

    if existing_turn_idx is not None:
        # Update existing entry (frontend retry scenario)
        print(f"--- DEBUG: Updating existing turn {current_ai_response_turn} (frontend retry detected) ---")
        session["conversation_log"][existing_turn_idx] = turn_data
    else:
        # Append new entry (first attempt)
        session["conversation_log"].append(turn_data)
        # Only increment turn count on first attempt
        session["turn_count"] += 1

    # Update last_user_message_char_count for the *next* turn's calculation
    session["last_user_message_char_count"] = current_user_message_char_count

    # Check if researcher notes already exist for this turn (from frontend retry)
    existing_notes_idx = None
    for idx, entry in enumerate(session["ai_researcher_notes_log"]):
        if entry["turn"] == current_ai_response_turn:
            existing_notes_idx = idx
            break

    notes_data = {
        "turn": current_ai_response_turn,
        "notes": researcher_notes
    }

    if existing_notes_idx is not None:
        # Update existing notes (frontend retry scenario)
        session["ai_researcher_notes_log"][existing_notes_idx] = notes_data
    else:
        # Append new notes (first attempt)
        session["ai_researcher_notes_log"].append(notes_data)

    response_timestamp = datetime.now().timestamp()
    session["last_ai_response_timestamp_for_ddm"] = response_timestamp

    # NEW: Save conversation data after each turn
    update_session_after_message(session, db_session)

    return {
        "ai_response": ai_response_text,
        "turn": current_ai_response_turn,
        "timestamp": response_timestamp,
        "backend_retry_metadata": {
            "retry_attempts": backend_retry_count,
            "retry_time_seconds": backend_retry_time
        }
    }

@app.post("/log_conversation_start")
async def log_conversation_start(data: ConversationStartRequest, db_session: Session = Depends(get_db)):
    session_id = data.session_id
    
    # Try to recover session from database if not in memory
    if session_id not in sessions:
        recovered_session = recover_session_from_database(session_id, db_session)
        if recovered_session:
            sessions[session_id] = recovered_session
            flag_session_as_recovered(session_id, db_session)
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    session["conversation_start_time"] = time.time()
    
    print(f"Conversation start logged for session {session_id}")
    return {"message": "Conversation start time logged"}

@app.post("/update_network_delay")
async def update_network_delay(data: NetworkDelayUpdateRequest, db_session: Session = Depends(get_db)):
    session_id = data.session_id

    # Try to recover session from database if not in memory
    if session_id not in sessions:
        recovered_session = recover_session_from_database(session_id, db_session)
        if recovered_session:
            sessions[session_id] = recovered_session
            flag_session_as_recovered(session_id, db_session)
        else:
            raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]

    # NEW: Check for excessive network delay (>40 seconds)
    EXCESSIVE_DELAY_THRESHOLD = 40.0
    is_excessive_delay = data.network_delay_seconds > EXCESSIVE_DELAY_THRESHOLD

    if is_excessive_delay:
        print("=" * 60)
        print("WARNING: EXCESSIVE NETWORK DELAY DETECTED")
        print("=" * 60)
        print(f"Session ID: {session_id}")
        print(f"Turn: {data.turn}")
        print(f"Network Delay: {data.network_delay_seconds:.2f} seconds")
        print(f"Send Attempts: {data.send_attempts}")
        print(f"Threshold: {EXCESSIVE_DELAY_THRESHOLD} seconds")
        print(f"Overage: {data.network_delay_seconds - EXCESSIVE_DELAY_THRESHOLD:.2f} seconds")
        print("This may indicate network issues or Railway cold starts")
        print("=" * 60)

        # Flag the session as having excessive delays
        session["has_excessive_delays"] = True

    # Find the conversation turn and update its network delay
    turn_found = False
    for turn_data in session["conversation_log"]:
        if turn_data["turn"] == data.turn:
            turn_found = True
            if "timing" in turn_data:
                turn_data["timing"]["network_delay_seconds"] = data.network_delay_seconds
                turn_data["timing"]["send_attempts"] = data.send_attempts
                turn_data["timing"]["excessive_delay_flag"] = is_excessive_delay  # NEW: Flag this turn

                # Add network delay metadata if provided
                if data.metadata:
                    turn_data["timing"]["network_delay_metadata"] = data.metadata
                    print(f"Updated network delay for session {session_id}, turn {data.turn}: {data.network_delay_seconds}s, attempts: {data.send_attempts}, metadata: {data.metadata.get('status', 'unknown')}, excessive={is_excessive_delay}")
                else:
                    print(f"Updated network delay for session {session_id}, turn {data.turn}: {data.network_delay_seconds}s, attempts: {data.send_attempts}, excessive={is_excessive_delay}")

                break  # Exit loop once turn is found and updated

    if not turn_found:
        raise HTTPException(status_code=404, detail=f"Turn {data.turn} not found in session {session_id}")

    # Update the database in a non-blocking way with error handling
    # This is less critical data, so we don't want to fail the entire request if DB is slow
    try:
        update_session_after_message(session, db_session)

        # NEW: If excessive delay detected, also update the has_excessive_delays flag in database
        if is_excessive_delay:
            session_record = db_session.query(db.StudySession).filter(db.StudySession.id == session_id).first()
            if session_record:
                session_record.has_excessive_delays = True
                db_session.commit()

    except Exception as db_error:
        # Log the database error but don't fail the request
        print(f"Warning: Database update failed for network delay (session {session_id}, turn {data.turn}): {db_error}")
        # The data is still in memory, so it will be saved on the next successful DB operation

    return {"message": "Network delay updated successfully"}

@app.post("/submit_rating")
async def submit_rating(data: RatingRequest, db_session: Session = Depends(get_db)):
    session_id = data.session_id
    
    # NEW: Try to recover session from database if not in memory
    if session_id not in sessions:
        recovered_session = recover_session_from_database(session_id, db_session)
        if recovered_session:
            sessions[session_id] = recovered_session
            # Flag this session as recovered from restart for analysis
            flag_session_as_recovered(session_id, db_session)
            print(f"Session {session_id} recovered from database for rating and flagged")
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]

    actual_decision_time = data.decision_time_seconds
    if actual_decision_time is None:
        print(f"Warning: decision_time_seconds was None for DDM rating. Session {session_id}, Turn {session['turn_count']}. Using placeholder.")
        actual_decision_time = -1.0

    session["intermediate_ddm_confidence_ratings"].append({
        "turn": session["turn_count"],
        "confidence": data.confidence,
        "decision_time_seconds": actual_decision_time,
        "reading_time_seconds": data.reading_time_seconds,
        "active_decision_time_seconds": data.active_decision_time_seconds,
        "slider_interaction_log": data.slider_interaction_log
    })

    # NEW: Check if this is the first pure DDM decision (0.0 or 1.0)
    is_pure_decision = data.confidence == 0.0 or data.confidence == 1.0
    first_pure_decision = is_pure_decision and "pure_ddm_decision" not in session
    
    if first_pure_decision:
        session["pure_ddm_decision"] = data.confidence
        session["pure_ddm_timestamp"] = datetime.now()
        session["pure_ddm_turn_number"] = session["turn_count"]
        session["pure_ddm_decision_time_seconds"] = actual_decision_time
        print(f"Pure DDM decision captured: {data.confidence} at turn {session['turn_count']}")

    # Calculate total study time and check for forced completion
    # Use conversation start time if available, otherwise fall back to session start
    conversation_start = session.get("conversation_start_time")
    session_start = session.get("session_start_time", time.time())
    
    if conversation_start:
        # Time from when conversation actually began
        elapsed_seconds = time.time() - conversation_start
    else:
        # Fallback to session start time
        elapsed_seconds = time.time() - session_start
    
    elapsed_minutes = elapsed_seconds / 60
    forced_completion = elapsed_minutes >= 7.5
    
    # NEW: Always save rating data incrementally after each submission
    update_session_after_rating(session, db_session, is_final=False)
    
    study_over = False
    if forced_completion:  # 7.5 minutes elapsed
        # ENFORCE: Study can only end with exactly 0 or 1
        if data.confidence != 0.0 and data.confidence != 1.0:
            # Timer expired but invalid final choice - study CONTINUES
            print(f"Session {session_id}: Timer expired but non-binary choice ({data.confidence}) submitted. Study continues.")
            return {
                "message": "Rating submitted.",
                "study_over": False,  # Study keeps going
                "ai_detected": None,
                "session_data_summary": None
            }
        
        # Valid 0 or 1 submitted - NOW the study can end
        session["ai_detected_final"] = (data.confidence == 1.0)
        session["final_decision_time_seconds_ddm"] = actual_decision_time
        
        # NEW: Final save with completion status
        update_session_after_rating(session, db_session, is_final=True)
        print(f"Session {session_id} completed and saved to database.")
        
        # Clean up the in-memory session
        del sessions[session_id]
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
async def submit_comment(data: CommentRequest, db_session: Session = Depends(get_db)):
    session_id = data.session_id
    
    # NEW: Try to recover session from database if not in memory
    if session_id not in sessions:
        recovered_session = recover_session_from_database(session_id, db_session)
        if recovered_session:
            sessions[session_id] = recovered_session
            # Flag this session as recovered from restart for analysis
            flag_session_as_recovered(session_id, db_session)
            print(f"Session {session_id} recovered from database for comment and flagged")
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]

    # Determine turn number, which is null for final, pre-debrief feedback
    turn_number = session["turn_count"] if data.phase == 'in_turn' else None

    # Sanitize comment
    sanitized_comment = html.escape(str(data.comment))
    
    session["feels_off_data"].append({
        "turn": turn_number,
        "description": sanitized_comment,
        "phase": data.phase
    })

    # If this is the final comment, update the CSV file that was already saved.
    if data.phase == 'pre_debrief':
        update_final_comment_in_csv(session)

    return {"message": "Comment submitted."}

@app.post("/log_ui_event")
async def log_ui_event(evt: UIEventRequest):
    event_record = {
        "event": evt.event,
        "ts_client": evt.ts_client,
        "ts_server": datetime.now().isoformat(),
        "metadata": evt.metadata or {},
        "participant_id": evt.participant_id,
        "prolific_pid": evt.prolific_pid,
        "session_id": evt.session_id
    }
    # If we have a live session, attach to it; otherwise store pre-session
    if evt.session_id and evt.session_id in sessions:
        sessions[evt.session_id].setdefault("ui_event_log", []).append(event_record)
        return {"message": "Event logged to session."}
    
    if evt.participant_id:
        pre_session_events.setdefault(evt.participant_id, []).append(event_record)
        return {"message": "Event logged pre-session."}
    
    # If neither, still return OK but note that it wasn't saved
    return {"message": "Event received but not saved (no participant_id or session_id)."}

@app.post("/submit_final_comment")
async def submit_final_comment(data: FinalCommentRequest, db_session: Session = Depends(get_db)):
    session_record = db_session.query(db.StudySession).filter(db.StudySession.id == data.session_id).first()
    if not session_record:
        raise HTTPException(status_code=404, detail="Could not find the completed study session to add comment to.")
    
    # Sanitize final comment
    sanitized_comment = html.escape(str(data.comment))
    session_record.final_user_comment = sanitized_comment
    db_session.commit()
    print(f"Final comment added to session {data.session_id}.")
    return {"message": "Final comment received. Thank you."}

@app.post("/finalize_no_session")
async def finalize_no_session(data: FinalizeNoSessionRequest):
    # Build a minimal session-like structure to persist
    participant_id_val = data.participant_id
    prolific_pid_val = data.prolific_pid or None
    ui_events = pre_session_events.pop(participant_id_val, [])
    if data.reason:
        ui_events.append({
            "event": data.reason,
            "ts_client": None,
            "ts_server": datetime.now().isoformat(),
            "metadata": {},
            "participant_id": participant_id_val,
            "prolific_pid": prolific_pid_val,
            "session_id": None
        })
    # For production, you might want to store incomplete sessions in database too
    print(f"Finalized incomplete session for participant {participant_id_val} with reason: {data.reason}")
    return {"message": "Finalized without session and logged."}


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
        print("Gemini model initialized.")
    # For local testing, you would run with: uvicorn main:app --reload
    # Railway uses its own start command from railway.json
    pass
