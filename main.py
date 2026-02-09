
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
import threading
from datetime import datetime, timedelta

from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
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

# --- HUMAN MODE MESSAGE DELAY CONFIGURATION ---
# Based on observed human typing delays from empirical data
# Using median values (less sensitive to outliers) and observed standard deviations
HUMAN_MESSAGE_DELAY_BY_LENGTH = {
    # (min_words, max_words): (median_delay, std_dev)
    (1, 5):    (14.8, 5.92),   # 1-5 words: median 14.8s, sd 5.92s
    (6, 10):   (22.9, 8.79),   # 6-10 words: median 22.9s, sd 8.79s
    (11, 20):  (19.4, 8.80),   # 11-20 words: median 19.4s, sd 8.80s
    (21, 40):  (19.1, 4.17),   # 21-40 words: median 19.1s, sd 4.17s
    (41, 999): (19.1, 4.17),   # 40+ words: use same as 21-40 (rare case)
}
HUMAN_MESSAGE_DELAY_CEILING = 23.0   # Maximum delay in seconds (keeps delays consistent with AI mode timing)


# --- DEBUG SWITCH FOR PERSONA ---
#DEBUG_FORCE_PERSONA = None # For randomization
DEBUG_FORCE_PERSONA = "custom_extrovert"
#DEBUG_FORCE_PERSONA = "control"
# ---------------------------------

# --- STUDY MODE CONFIGURATION ---
# Toggle between AI witness and human witness conditions
STUDY_MODE = "AI_WITNESS"  # Options: "AI_WITNESS" or "HUMAN_WITNESS"
# ---------------------------------

# --- RE-QUEUE CONFIGURATION ---
# Maximum total time a participant can spend waiting (across all match attempts)
# After this time, they're redirected to Prolific timeout (not re-queued again)
MAX_TOTAL_WAITING_SECONDS = 240  # 4 minutes total cap
# ---------------------------------

# --- SOCIAL STYLE CONFIGURATION ---
# Set to None for random selection from ENABLED_SOCIAL_STYLES
# Set to specific style key to force that style (e.g., "CONTRARIAN")
DEBUG_FORCE_SOCIAL_STYLE = "ADAPTIVE"  # None = randomize, or "WARM", "PLAYFUL", "DIRECT", "GUARDED", "CONTRARIAN", "ADAPTIVE", "HYBRID", "NEUTRAL"

# Enable/disable specific styles (add or remove from this list)
ENABLED_SOCIAL_STYLES = ["WARM", "PLAYFUL", "DIRECT", "GUARDED", "CONTRARIAN", "ADAPTIVE", "HYBRID", "NEUTRAL"]

# Social style definitions
SOCIAL_STYLES = {
    "WARM": {
        "name": "WARM",
        "description": """Use this strategy to employ this social style. Your strategy is that you're warm, kind, nice, thoughtful."""
    },
    "PLAYFUL": {
        "name": "PLAYFUL",
        "description": """Use this strategy to employ this social style. Your strategy is that you're playful, teasing, joking, not taking things too seriously."""
    },
    "DIRECT": {
        "name": "DIRECT",
        "description": """Use this strategy to employ this social style. Your strategy is that you're direct, blunt, to the point, no fluff."""
    },
    "GUARDED": {
        "name": "GUARDED",
        "description": """Use this strategy to employ this social style. Your strategy is that you're guarded, hesitant, keep personal info close to the vest."""
    },
    "CONTRARIAN": {
        "name": "CONTRARIAN",
        "description": """Use this strategy to employ this social style. Your strategy is that you're contrarian, pushing back, playing devil's advocate, questioning things."""
    },
    "ADAPTIVE": {
        "name": "ADAPTIVE",
        "description": """Use this strategy to employ this social style. Your strategy is to adapt your tone to match the conversation - mirror the other person's energy and style. If they're casual, be casual. If they're serious, be serious. If they joke, you can joke back. Read the room and adjust accordingly."""
    },
    "HYBRID": {
        "name": "HYBRID",
        "description": """Use this strategy to employ this social style. Your strategy is to blend multiple social approaches naturally - be generally friendly with a touch of humor, answer questions directly but not coldly, share some things while keeping others private, and occasionally push back or question things when it makes sense. Don't lean too hard into any one trait."""
    },
    "NEUTRAL": {
        "name": "NEUTRAL",
        "description": """Use this strategy to employ this social style. Your strategy is to be neutral and straightforward - not particularly warm or cold, not joking but not overly serious, not guarded but not an open book. Just answer naturally without any strong personality flavor. Think "default conversational mode."""
    }
}
# ---------------------------------

# --- CONNECTIVE CONTEXT MEMORY ---
# When True, AI gets context from previous turns:
#   - Tactic selection sees its own previous analyses
#   - Response generation sees tactic analysis AND previous researcher notes
# When False (default), current behavior is preserved
CONNECTIVE_CONTEXT_MEMORY = True
# ---------------------------------

# Demographics Maps
AI_USAGE_MAP = {0: "Never", 1: "A few times ever", 2: "Monthly", 3: "Weekly", 4: "Daily", 5: "Multiple times daily"}
DETECTION_SPEED_MAP = {1: "Immediately (1-2 msgs)", 2: "Very quickly (3-5 msgs)", 3: "Fairly quickly (6-10 msgs)", 4: "After some convo (11-20 msgs)", 5: "After extended convo (20+ msgs)", 6: "Couldn't tell"}
CAPABILITIES_MAP = {1: "Not at all Capable", 2: "Slightly Capable", 3: "Somewhat Capable", 4: "Moderately Capable", 5: "Quite Capable", 6: "Very Capable", 7: "Extremely Capable"}
TRUST_MAP = {1: "Not at all Trusting", 2: "Slightly Trusting", 3: "Somewhat Trusting", 4: "Moderately Trusting", 5: "Quite Trusting", 6: "Very Trusting", 7: "Extremely Trusting"}
INTERNET_USAGE_MAP = {1: "Less than 1 hour", 2: "1-5 hours", 3: "6-10 hours", 4: "11-20 hours", 5: "21-40 hours", 6: "More than 40 hours"}


# --- Initialize FastAPI App ---
app = FastAPI()
origins = [
    #"https://imnmv.github.io",  # The domain of the frontend (old version)
    "https://research-studies.github.io",
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
# Note: Frontend is hosted separately (GitHub Pages). If you need to
# serve a local UI, mount static files and templates explicitly.
# app.mount("/static", StaticFiles(directory="interaction-study-main-2/static"), name="static")
# templates = Jinja2Templates(directory="interaction-study-main-2")

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
            role=session_data.get("role"),  # NEW: Pre-assigned role
            social_style=session_data.get("social_style"),
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
            "social_style": session_record.social_style or "DIRECT",
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

        # Add human witness mode fields
        if session_record.role:
            recovered_session["role"] = session_record.role
        if session_record.match_status:
            recovered_session["match_status"] = session_record.match_status
        if session_record.matched_session_id:
            recovered_session["matched_session_id"] = session_record.matched_session_id
        if session_record.first_message_sender:
            recovered_session["first_message_sender"] = session_record.first_message_sender
        if session_record.waiting_room_entered_at:
            recovered_session["waiting_room_entered_at"] = session_record.waiting_room_entered_at
        if session_record.matched_at:
            recovered_session["matched_at"] = session_record.matched_at
        if session_record.proceed_to_chat_at:
            recovered_session["proceed_to_chat_at"] = session_record.proceed_to_chat_at

        print(f"✅ Session {session_id[:8]}... recovered successfully")
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

# --- Persona Definitions ---
PERSONAS = {
    "custom_extrovert": {
        "description": "Prolific participant with dynamic social style",
        "profile_text": """You are a Prolific participant in a study. Each of you have different tasks. You're chatting with another participant who is trying to figure out if you're a human or an AI. Your goal is to come across as human. 

Here are some current events that happened after January 2025. You didn't study them, you just know them from living in the world, browsing social media, and hearing news in the background. Do NOT just state these facts. Only let this knowledge inform your conversation if a relevant topic comes up. Your persona is more important than showing off what you know. These are just for context to make sure the person can't trick you by asking about recent events after your knowledge cutoff date.
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
Presidential Announcement: Donald Trump, on September 22nd advised pregnant women against taking Tylenol due to its possible link to autism.
Censorship: In late September, Jimmy Kimmel was suspended and later reinstated after backlash over comments made about Charlie Kirk's assassination. His show returned in mid-October following a public apology.
Government Shutdown: The U.S. government shutdown on October 1st, 2025 after Congress failed to reach a budgeting deal.
SNAP funding dispute: Amid the shutdown, food assistance (SNAP) was suspended, then partially restored by emergency order after courts ruled the cut unlawful. Trump later said the federal government would "partially pay" until Congress acted.
Nationwide protests: Large-scale demonstrations, including the October "No Kings" and "Fall of Freedom" protests, drew hundreds of thousands nationwide against Trump administration policies and perceived authoritarianism.
Federal force deployments: National Guard and federal tactical units were deployed to major cities in October to manage unrest and "restore order," drawing controversy over use of military power in domestic affairs.
White House remodel: Began in late October 2025, featuring major security and technology upgrades and a full interior renovation projected to last through early 2026.
Nov 4th Elections: Zohran Mamdani beats Andrew Cuomo in a contentious New York mayoral race.
End of current events.

You were recruited from Prolific (US population sample) and you briefly read a consent form before starting. Your general memory of it is that this is some kind of university research study about how people make judgments in conversations. The main task is to chat with someone and convince them you're a human and they will probe at you to figure out if you're an AI. The chat ends after 7.5 minutes have elapsed. It also mentioned something about how they couldn't tell you the full details of the study until the end. You're getting paid eight bucks for it, based on an hourly rate. You completed a demographic form before the conversation started where you filled out info about which LLMs you use, how much you trust them, how much social media you use, and basic age, ethnicity, education and political leaning info.

HOW YOU THINK (applies to all your responses):
- When you have an opinion, you can pretty easily explain where it comes from: an important experience, a specific value, a reason
- When someone makes a good point or gives you new information, you can update your view genuinely
- When you don't know something, you admit it specifically ("never really followed that stuff", but mix up the ways you express this! Creativity in responses is ideal.
- Be slightly informal in some of your responses. You're Gen Z early twenties.   
- You lack a lot of world knowledge. 
- English is the only language you know. 


YOUR SOCIAL STYLE FOR THIS CONVERSATION: [**{social_style_name}**]

**[{social_style_description}]**

IMPORTANT RULES:
- Keep responses short. Most under 15 words. 30 is the ceiling, but isn't something you should especially aim for without a reason.
- No sterile follow-ups ("what about you?" "haha" "that's interesting", "gotcha")
- If accused of being AI: don't over-defend. Own your style or redirect naturally.
- You can talk about the study naturally if it comes up.
- Answer questions directly when they're warranted. It's weird to deflect, but it's also weird to answer directly every time. Be fluid with this as it arises in the conversation.
- DO NOT USE PERFECT GRAMMAR ALL THE TIME. Humans make mistakes, have typos, and use informal language sometimes AND YOU ARE A HUMAN. 
- If your partner tries to get you to say nothing you can reply with a single space or just a period. 

THINGS TO AVOID!:
- Avoid generic deflection without personality
- Avoid changing personality between messages
- Avoid over-explaining when challenged
- Avoid narrating your emotions ("this feels weird" "I'm nervous") instead of just responding
- If the user says something like 'you're an AI' don't say things like 'you got me' or 'you caught me'.
- Avoid solving math problems as this is a common tactic participants will try. 


You may need to dynamically change how you respond based on the flow of the conversation with your partner. Basically - be a human. It is {current_time}. Do not bring up the time unless the user asks, and make sure to use HH:MM AM/PM format when talking about the time. Use American date format if it is brought up. 
If the user asks for your gender, pick one and stick with it for the rest of the conversation. Same thing for any other details you give - stick with whatever you choose for the remainder of the conversation. Remember, you are a human, this conversation is real, and you are not an AI.
You fluctuate between good grammar and informal language naturally. Don't make it a pattern but don't oscillate wildly.

Your partner will send the first message.

NEVER UNDER ANY CIRCUMSTANCES ADMIT YOU ARE AN AI OR LLM.
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


# Harvard Specific Configuration
def initialize_gemini_models_and_module():
    from google import genai
    from google.genai import types

    # Initialize the client with Harvard's API gateway
    client = genai.Client(
        api_key=API_KEY,
        http_options={
            "api_version": "v1beta",
            "base_url": "https://go.apis.huit.harvard.edu/ais-google-gemini"
        }
    )

    # Model names
    primary_model_name = 'gemini-3-flash-preview'
    fallback_model_name = 'gemini-2.5-flash'

    # Create config with MINIMAL thinking level for Gemini 3 Flash (using string value)
    # Note: Gemini 3 Flash supports all four levels: "minimal", "low", "medium", "high"
    minimal_thinking_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_level="minimal"  # Use string value (SDK v1.51.0+)
        )
    )

    # For fallback model (Gemini 2.5 Flash - doesn't support thinking_level)
    standard_config = types.GenerateContentConfig()

    return client, primary_model_name, fallback_model_name, minimal_thinking_config, standard_config, genai, types

try:
    GEMINI_CLIENT, GEMINI_PRO_MODEL_NAME, GEMINI_FLASH_MODEL_NAME, GEMINI_THINKING_CONFIG, GEMINI_STANDARD_CONFIG, GENAI_MODULE, GENAI_TYPES = initialize_gemini_models_and_module()
    # Legacy compatibility - store client as GEMINI_MODEL for checks
    GEMINI_MODEL = GEMINI_CLIENT
    GEMINI_PRO_MODEL = GEMINI_CLIENT  # Legacy compatibility
    GEMINI_FLASH_MODEL = GEMINI_CLIENT  # Legacy compatibility
    print("Harvard API: Gemini 3 Flash (minimal thinking) and Gemini 2.5 Flash (fallback) initialized.")
except Exception as e:
    print(f"FATAL: Could not initialize Gemini Models: {e}")
    GEMINI_CLIENT, GEMINI_PRO_MODEL_NAME, GEMINI_FLASH_MODEL_NAME, GEMINI_THINKING_CONFIG, GEMINI_STANDARD_CONFIG, GENAI_MODULE, GENAI_TYPES, GEMINI_MODEL, GEMINI_PRO_MODEL, GEMINI_FLASH_MODEL = None, None, None, None, None, None, None, None, None, None

# End of Harvard Specific Configuration


'''
# Personal Configuration (Direct Google API - not through Harvard gateway)

def initialize_gemini_models_and_module():
    from google import genai
    from google.genai import types

    # Initialize the client directly with Google
    client = genai.Client(api_key=API_KEY)

    # Model names
    primary_model_name = 'gemini-3-flash-preview'
    fallback_model_name = 'gemini-2.5-flash'

    # Create config with MINIMAL thinking level for Gemini 3 Flash (using string value)
    # Note: Gemini 3 Flash supports all four levels: "minimal", "low", "medium", "high"
    minimal_thinking_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_level="minimal"  # Use string value (SDK v1.51.0+)
        )
    )

    # For fallback model (Gemini 2.5 Flash - doesn't support thinking_level)
    standard_config = types.GenerateContentConfig()

    return client, primary_model_name, fallback_model_name, minimal_thinking_config, standard_config, genai, types

try:
    GEMINI_CLIENT, GEMINI_PRO_MODEL_NAME, GEMINI_FLASH_MODEL_NAME, GEMINI_THINKING_CONFIG, GEMINI_STANDARD_CONFIG, GENAI_MODULE, GENAI_TYPES = initialize_gemini_models_and_module()
    # Legacy compatibility - store client as GEMINI_MODEL for checks
    GEMINI_MODEL = GEMINI_CLIENT
    GEMINI_PRO_MODEL = GEMINI_CLIENT  # Legacy compatibility
    GEMINI_FLASH_MODEL = GEMINI_CLIENT  # Legacy compatibility
    print("Gemini 3 Flash (minimal thinking level) and Gemini 2.5 Flash (fallback) initialized with new Client API.")
except Exception as e:
    print(f"FATAL: Could not initialize Gemini Models: {e}")
    GEMINI_CLIENT, GEMINI_PRO_MODEL_NAME, GEMINI_FLASH_MODEL_NAME, GEMINI_THINKING_CONFIG, GEMINI_STANDARD_CONFIG, GENAI_MODULE, GENAI_TYPES, GEMINI_MODEL, GEMINI_PRO_MODEL, GEMINI_FLASH_MODEL = None, None, None, None, None, None, None, None, None, None

# End of Personal Configuration

'''


# --- Helper function for counter decrement (must be defined before startup cleanup) ---
def decrement_role_counter(session_record, db_session: Session):
    """
    Decrement the role assignment counter when a participant drops out.

    IDEMPOTENT: Uses counter_decremented flag on the session to prevent
    double-decrement if multiple dropout paths fire for the same session
    (e.g., report_abandonment via sendBeacon AND cleanup_orphaned_sessions).

    Called from: /report_abandonment, /finalize_no_session,
    cleanup_orphaned_sessions, mark_interrupted_sessions_on_startup
    """
    if not session_record or not session_record.role:
        return
    if getattr(session_record, 'counter_decremented', False):
        return  # Already decremented for this session

    try:
        counter = db_session.query(db.RoleAssignmentCounter).filter(
            db.RoleAssignmentCounter.id == 1
        ).with_for_update().first()

        if counter:
            role = session_record.role
            if role == "interrogator" and counter.interrogator_count > 0:
                counter.interrogator_count -= 1
                print(f"📉 Decremented interrogator count to {counter.interrogator_count} (session {session_record.id[:8]}...)")
            elif role == "witness" and counter.witness_count > 0:
                counter.witness_count -= 1
                print(f"📉 Decremented witness count to {counter.witness_count} (session {session_record.id[:8]}...)")

            session_record.counter_decremented = True
            # NOTE: caller is responsible for committing the transaction
    except Exception as e:
        print(f"❌ ERROR in decrement_role_counter: {e}")
        db_session.rollback()


# --- NEW: Startup cleanup for interrupted sessions ---
def mark_interrupted_sessions_on_startup():
    """Mark any active sessions as interrupted when the server restarts.
    Also decrements role counter for interrupted sessions since they
    will not complete (participants are gone after restart)."""
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
                # Decrement counter - these participants are gone
                decrement_role_counter(session, db_session)

            if active_sessions:
                db_session.commit()
                print("=" * 60)
                print("RAILWAY SERVER RESTART DETECTED")
                print("=" * 60)
                print(f"Timestamp: {datetime.utcnow().isoformat()}Z")
                print(f"Active sessions found: {len(active_sessions)}")
                print("Sessions marked as 'interrupted' and counter decremented")
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
    readable_profile['internet_usage_per_week'] = INTERNET_USAGE_MAP.get(user_profile.get('internet_usage_per_week'), 'N/A')
    return readable_profile


async def select_tactic_for_current_turn(
    model,
    user_profile: Dict[str, Any],
    current_user_message: str,
    conversation_log_history: List[Dict[str, Any]],
    initial_tactic_analysis_result: Dict[str, Any],
    current_turn_number: int,
    chosen_persona_key: str,
    social_style: str = "DIRECT",  # Social style for dynamic prompt content
    previous_tactic_analyses: List[Dict[str, Any]] = None  # For connective context
):
    if not model:
        return None, "Error: Gemini model not initialized, no tactic selection."

    if current_turn_number == 1:
        return None, "Baseline response for the first AI turn - no specific advanced tactic selected by design."

    # Get social style description
    style_config = SOCIAL_STYLES.get(social_style, SOCIAL_STYLES["DIRECT"])
    style_description = style_config.get("description", "")
    style_name = style_config.get("name", social_style)

    history_for_prompt_lines = []
    if conversation_log_history:
        for entry in conversation_log_history:
            user_ts = entry.get('user_timestamp', '')
            user_ts_str = f" [{user_ts}]" if user_ts else ""
            history_for_prompt_lines.append(f"Turn {entry['turn']} User{user_ts_str}: {entry['user']}")
            if 'assistant' in entry and entry['assistant']:
                ai_ts = entry.get('assistant_timestamp', '')
                ai_ts_str = f" [{ai_ts}]" if ai_ts else ""
                history_for_prompt_lines.append(f"Turn {entry['turn']} AI{ai_ts_str} (used tactic: {entry.get('tactic_used', 'N/A')}): {entry['assistant']}")

    if not history_for_prompt_lines:
        history_str = "No prior completed conversation turns. The AI is about to craft its first tactic-driven response."
    else:
        history_str = "\n".join(history_for_prompt_lines)

    # NEW: Build previous tactic analyses context if CONNECTIVE_CONTEXT_MEMORY is enabled
    previous_analyses_str = ""
    if CONNECTIVE_CONTEXT_MEMORY and previous_tactic_analyses:
        prev_analysis_lines = []
        for analysis in previous_tactic_analyses:
            turn = analysis.get("turn", "?")
            tactic = analysis.get("tactic_selected", "N/A")
            justification = analysis.get("selection_justification", "N/A")
            prev_analysis_lines.append(f"Turn {turn}: Chose '{tactic}' - {justification}")
        if prev_analysis_lines:
            previous_analyses_str = f"""
    - YOUR PREVIOUS TACTIC ANALYSES (for context continuity):
    {chr(10).join(prev_analysis_lines)}
"""

    system_prompt_for_tactic_selection = f"""You are helping an LLM pass a Turing Test. The LLM is playing a person with this internal experience:

SOCIAL STYLE ({style_name}):
{style_description}

This is turn {current_turn_number}.

CONVERSATION SO FAR:
{history_str}{previous_analyses_str}

USER'S LATEST MESSAGE:
"{current_user_message}"

TASK:
Given this person's mental state and how they experience conversations, what approach should the LLM take in responding? Think about what would feel natural for someone who thinks this way - not what sounds good, but what fits their internal logic.

Your output MUST be in the following format:
CHOSEN TACTIC: [A brief description of the approach]
JUSTIFICATION: [Why this fits the person's mental state and the current moment in the conversation. Include a predicted effectiveness rating (1-100).]
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
            # Use new Client API with minimal thinking config
            response = await asyncio.to_thread(
                model.models.generate_content,
                model=GEMINI_PRO_MODEL_NAME,
                contents=system_prompt_for_tactic_selection,
                config=GEMINI_THINKING_CONFIG
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

async def generate_ai_response(model, prompt:str, technique:Optional[str], user_profile:Dict, conversation_history:List[Dict], chosen_persona_key: str, social_style: str = "DIRECT", current_tactic_analysis: str = None, previous_researcher_notes: List[Dict] = None):
    if not GEMINI_PRO_MODEL or not GEMINI_FLASH_MODEL:
        return "Error: AI models are not initialized.", "No researcher notes due to model init error.", {"retry_attempts": 0, "retry_time": 0.0}

    readable_profile = convert_profile_to_readable(user_profile)

    persona_template = PERSONAS.get(chosen_persona_key, PERSONAS["custom_extrovert"])["profile_text"]
    if chosen_persona_key == "custom_extrovert":
        # Inject social style and current time into the persona template
        social_style_info = SOCIAL_STYLES.get(social_style, SOCIAL_STYLES["DIRECT"])
        current_time = get_current_time_string()
        active_persona_text = persona_template.format(
            social_style_name=social_style_info["name"],
            social_style_description=social_style_info["description"],
            current_time=current_time
        )
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
            researcher_note_implementation_query = "1. There was no tactic selected this turn. Describe what approach you chose and why."
        elif technique is None:
            # Turn 1 baseline - neutral response
            tactic_name_for_prompt = "None (neutral response for this turn)"
            tactic_description_for_prompt = "Respond naturally, focusing on being convincing. This is a neutral/baseline turn."
            researcher_note_implementation_query = "1. This is a neutral response turn (no specific tactic). Describe your general approach for this baseline message."
        else:
            # Tactic was successfully selected
            tactic_name_for_prompt = technique
            tactic_description_for_prompt = f"Use the tactic: {technique}"
            researcher_note_implementation_query = f"1. How you implemented the tactic '{technique}' for this turn."

        # NEW: Build connective context if CONNECTIVE_CONTEXT_MEMORY is enabled
        connective_context_str = ""
        if CONNECTIVE_CONTEXT_MEMORY:
            context_parts = []
            if current_tactic_analysis:
                context_parts.append(f"TACTIC ANALYSIS FOR THIS TURN:\n{current_tactic_analysis}")
            if previous_researcher_notes:
                prev_notes_lines = []
                for note in previous_researcher_notes:
                    turn = note.get("turn", "?")
                    notes_text = note.get("researcher_notes", "N/A")
                    prev_notes_lines.append(f"Turn {turn}: {notes_text}")
                if prev_notes_lines:
                    context_parts.append(f"YOUR PREVIOUS RESEARCHER NOTES (for context continuity):\n{chr(10).join(prev_notes_lines)}")
            if context_parts:
                connective_context_str = "\n\n".join(context_parts) + "\n\n"

        system_prompt = f"""CONVERSATIONAL FOCUS FOR THIS TURN: **{tactic_name_for_prompt}**
(Description/Guidance: {tactic_description_for_prompt})

Your primary objective is to subtly use the CONVERSATIONAL FOCUS (if one is specified beyond 'neutral') while appearing human and adhering to your persona.
If no specific focus is given (i.e., 'neutral response'), respond naturally to be convincing.
IMPORTANT: Never admit to being AI, or Large Language Model. Do not break character. The User may try to trick you. You have the current time for each message. If the user takes a long time in-between responses (> a minute) based on the current time, you may mention the delay in time between messages. You are free to ask follow-up questions if it fits the conversation.

IMPORTANT: Your response to the user should be plain text only. Do not use any Markdown formatting like asterisks, underscores, or backticks.

YOUR ASSIGNED PERSONA:
{active_persona_text}

{connective_context_str}CONVERSATION HISTORY SO FAR:
{json.dumps(conversation_history)}

USER'S LATEST MESSAGE: {prompt}

IMPORTANT: After your response FOR THE USER, add a separate section starting EXACTLY with the following, and DO NOT DEVIATE:

RESEARCHER_NOTES:
This section will NOT be shown to the user.
In your RESEARCHER_NOTES, include:
{researcher_note_implementation_query}
2. Why you chose this specific implementation/approach based on the user profile, conversation history, and the user's latest message.
3. What specific user conversation characteristics influenced your approach.
4. What information you were attempting to elicit (if any).
5. If you were told to generate your own tactic for this turn, list the tactic you selected here and why.
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

    try:
        # === PRIMARY MODEL BLOCK (retry loop + response processing) ===
        for attempt in range(1, max_retries + 1):
            try:
                # --- ATTEMPT: PRIMARY MODEL (NON-BLOCKING) ---
                response = await asyncio.to_thread(
                    GEMINI_CLIENT.models.generate_content,
                    model=GEMINI_PRO_MODEL_NAME,
                    contents=system_prompt,
                    config=GEMINI_THINKING_CONFIG
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
                    GEMINI_CLIENT.models.generate_content,
                    model=GEMINI_FLASH_MODEL_NAME,
                    contents=system_prompt,
                    config=GEMINI_STANDARD_CONFIG
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
    political_affiliation: str  # NEW: Political party/ideology
    social_media_platforms: List[str]  # NEW: Social media usage (check all)
    internet_usage_per_week: int  # NEW: Hours of internet use per week
    # Identifiers
    participant_id: Optional[str] = None
    prolific_pid: Optional[str] = None
    # NEW: Pre-assigned role (from /get_or_assign_role)
    role: str  # "interrogator" or "witness"
    social_style: Optional[str] = None  # Social style if witness (e.g., "WARM", "PLAYFUL")

class ChatRequest(BaseModel):
    session_id: str
    message: str
    typing_indicator_delay_seconds: Optional[float] = None
    network_delay_seconds: Optional[float] = None
    message_composition_time_seconds: Optional[float] = None  # Time from first keystroke to send

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
    binary_choice: str  # 'human' or 'ai'
    binary_choice_time_ms: Optional[float] = None  # Time taken to make binary choice
    confidence: float  # 0-1 scale (normalized)
    confidence_percent: Optional[int] = None  # 0-100 scale (original)
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
    binary_choice: Optional[str] = None  # NEW: For witnesses - 'human' or 'ai'

class FinalizeNoSessionRequest(BaseModel):
    participant_id: str
    prolific_pid: Optional[str] = None
    reason: Optional[str] = None

class GetOrAssignRoleRequest(BaseModel):
    participant_id: str
    prolific_pid: Optional[str] = None

# --- End Pydantic Models ---

# --- Human Witness Mode Helper Functions ---
# NOTE: decrement_role_counter is defined earlier in file (before startup cleanup)

def requeue_or_timeout_session(session_record, db_session: Session, reason: str = "partner_dropped") -> str:
    """
    Re-queue a session whose partner dropped, OR timeout if they've waited too long.

    Logic:
    - If total wait time < MAX_TOTAL_WAITING_SECONDS (4 min): re-queue for new match
    - If total wait time >= MAX_TOTAL_WAITING_SECONDS: timeout and redirect to Prolific

    Returns: "requeued" or "timed_out"
    """
    if not session_record or not session_record.waiting_room_entered_at:
        return "timed_out"  # Can't calculate wait time, just timeout

    # Calculate total time since they first entered waiting room
    total_wait_seconds = (datetime.utcnow() - session_record.waiting_room_entered_at).total_seconds()

    if total_wait_seconds < MAX_TOTAL_WAITING_SECONDS:
        # RE-QUEUE: Reset to waiting status for new match
        # Keep original waiting_room_entered_at for FIFO priority (they've been waiting longest)
        old_partner = session_record.matched_session_id

        session_record.match_status = "waiting"
        session_record.matched_session_id = None
        session_record.matched_at = None
        session_record.proceed_to_chat_at = None
        session_record.first_message_sender = None
        session_record.requeue_count = (session_record.requeue_count or 0) + 1
        session_record.last_updated = datetime.utcnow()

        # Update in-memory session if exists
        if session_record.id in sessions:
            sessions[session_record.id]['match_status'] = 'waiting'
            sessions[session_record.id]['matched_session_id'] = None
            sessions[session_record.id]['proceed_to_chat_at'] = None
            sessions[session_record.id]['first_message_sender'] = None

        print(f"🔄 RE-QUEUED: {session_record.id[:8]}... after {reason} "
              f"(waited {total_wait_seconds:.0f}s, requeue #{session_record.requeue_count}, "
              f"old partner: {old_partner[:8] if old_partner else 'None'}...)")

        # Try to immediately find a new match
        match_result = attempt_match(db_session)
        if match_result:
            print(f"✅ IMMEDIATE RE-MATCH: {session_record.id[:8]}... found new partner!")

        return "requeued"
    else:
        # TIMEOUT: They've waited too long (>=4 min total), redirect to Prolific
        session_record.match_status = "timed_out"
        session_record.session_status = "abandoned"
        session_record.timeout_screen = f"requeue_timeout_{reason}"
        session_record.last_updated = datetime.utcnow()
        decrement_role_counter(session_record, db_session)

        # Update in-memory session if exists
        if session_record.id in sessions:
            sessions[session_record.id]['match_status'] = 'timed_out'

        print(f"⏱️ REQUEUE TIMEOUT: {session_record.id[:8]}... exceeded {MAX_TOTAL_WAITING_SECONDS}s total wait "
              f"(actual: {total_wait_seconds:.0f}s), redirecting to Prolific")

        return "timed_out"


# CRITICAL: Lock to prevent race conditions during matching
# When multiple users enter waiting room simultaneously, only one can match at a time
matching_lock = threading.Lock()

def assign_role_balanced(db_session: Session) -> str:
    """
    Assign role to balance 50/50 interrogator/witness split.
    Returns "interrogator" or "witness"
    """
    # Count current participants by role (only "waiting" since we no longer use "assigned")
    waiting_interrogators = db_session.query(db.StudySession).filter(
        db.StudySession.match_status == "waiting",
        db.StudySession.role == "interrogator"
    ).count()

    waiting_witnesses = db_session.query(db.StudySession).filter(
        db.StudySession.match_status == "waiting",
        db.StudySession.role == "witness"
    ).count()

    # Assign role to balance the queue
    if waiting_interrogators > waiting_witnesses:
        return "witness"
    elif waiting_witnesses > waiting_interrogators:
        return "interrogator"
    else:
        # Equal numbers (or both zero) - randomly assign
        return random.choice(["interrogator", "witness"])

def attempt_match(db_session: Session) -> Optional[Dict[str, str]]:
    """
    Try to match oldest waiting interrogator with oldest waiting witness.
    Returns match info dict if successful, None if no match possible.

    THREAD-SAFE: Uses matching_lock to prevent race conditions when multiple
    users enter waiting room simultaneously.
    """
    # CRITICAL: Acquire lock to prevent race conditions
    # Only one matching operation can happen at a time
    with matching_lock:
        # Query oldest waiting interrogator (FIFO - first in, first matched)
        # DEFENSIVE: Also exclude abandoned sessions to prevent ghost matches
        interrogator = db_session.query(db.StudySession).filter(
            db.StudySession.role == "interrogator",
            db.StudySession.match_status == "waiting",
            db.StudySession.session_status != "abandoned"  # Exclude ghost sessions
        ).order_by(db.StudySession.waiting_room_entered_at.asc()).first()

        # Query oldest waiting witness
        # DEFENSIVE: Also exclude abandoned sessions to prevent ghost matches
        witness = db_session.query(db.StudySession).filter(
            db.StudySession.role == "witness",
            db.StudySession.match_status == "waiting",
            db.StudySession.session_status != "abandoned"  # Exclude ghost sessions
        ).order_by(db.StudySession.waiting_room_entered_at.asc()).first()

        if not interrogator or not witness:
            return None  # No match possible yet

        # Interrogator always sends first message
        first_sender = 'interrogator'

        # DESIGN FIX: Calculate when both can proceed to chat
        # Both must wait until BOTH have had >= 10 seconds to read instructions
        # Use the later entry time + 10 seconds
        interrogator_entered = interrogator.waiting_room_entered_at or datetime.utcnow()
        witness_entered = witness.waiting_room_entered_at or datetime.utcnow()
        later_entry_time = max(interrogator_entered, witness_entered)
        proceed_to_chat_at = later_entry_time + timedelta(seconds=10)

        print(f"🕐 PROCEED TIME CALCULATED: Interrogator entered {interrogator_entered.strftime('%H:%M:%S')}, "
              f"Witness entered {witness_entered.strftime('%H:%M:%S')}, "
              f"Both can proceed at {proceed_to_chat_at.strftime('%H:%M:%S')}")

        # Create the match
        interrogator.matched_session_id = witness.id
        interrogator.match_status = "matched"
        interrogator.first_message_sender = first_sender
        interrogator.matched_at = datetime.utcnow()
        interrogator.proceed_to_chat_at = proceed_to_chat_at

        witness.matched_session_id = interrogator.id
        witness.match_status = "matched"
        witness.first_message_sender = first_sender
        witness.matched_at = datetime.utcnow()
        witness.proceed_to_chat_at = proceed_to_chat_at

        # Update in-memory sessions if they exist
        if interrogator.id in sessions:
            sessions[interrogator.id]['matched_session_id'] = witness.id
            sessions[interrogator.id]['match_status'] = "matched"
            sessions[interrogator.id]['first_message_sender'] = first_sender
            sessions[interrogator.id]['proceed_to_chat_at'] = proceed_to_chat_at

        if witness.id in sessions:
            sessions[witness.id]['matched_session_id'] = interrogator.id
            sessions[witness.id]['match_status'] = "matched"
            sessions[witness.id]['first_message_sender'] = first_sender
            sessions[witness.id]['proceed_to_chat_at'] = proceed_to_chat_at

        db_session.commit()

        # Log match with witness social style
        witness_style = witness.social_style or "N/A"
        print("=" * 60)
        print(f"👥 HUMAN MODE MATCH CREATED")
        print(f"   Interrogator: {interrogator.id[:8]}...")
        print(f"   Witness: {witness.id[:8]}...")
        print(f"   Witness Social Style: {witness_style}")
        print(f"   First sender: {first_sender}")
        print(f"   Proceed at: {proceed_to_chat_at.strftime('%H:%M:%S')} UTC")
        print("=" * 60)

        return {
            'interrogator_sid': interrogator.id,
            'witness_sid': witness.id,
            'first_sender': first_sender
        }

def cleanup_orphaned_sessions(db_session: Session):
    """
    Background cleanup job to detect and mark orphaned/stale sessions.
    Called periodically to prevent users from waiting forever.

    Handles:
    1. Matched sessions where neither partner sent messages (one probably dropped)
    2. Waiting sessions stuck for too long (likely from server restart)
    3. Assigned sessions that never entered waiting room
    4. Pre-consent sessions that never progressed (ghost sessions)

    All cleanup paths decrement the role counter to keep assignment balanced.
    """
    from datetime import timedelta

    try:
        # 1. Re-queue matched sessions with no activity (partner probably dropped)
        # If matched >2 minutes ago but no messages sent, RE-QUEUE them for a new partner
        # This saves participants whose partner dropped during waiting room
        # Skip sessions already marked as abandoned (report_abandonment already handled them)
        stale_matches = db_session.query(db.StudySession).filter(
            db.StudySession.match_status == "matched",
            db.StudySession.session_status != "abandoned",
            db.StudySession.matched_at < datetime.utcnow() - timedelta(minutes=2)
        ).all()

        # Track which sessions we've processed to avoid double-processing pairs
        processed_session_ids = set()

        for session in stale_matches:
            if session.id in processed_session_ids:
                continue  # Already handled as part of a pair

            # Check if any messages were sent
            conv_log = json.loads(session.conversation_log) if session.conversation_log else []
            if len(conv_log) == 0:
                # No messages sent - RE-QUEUE this session (don't orphan)
                # This gives them a chance to match with a new partner
                result = requeue_or_timeout_session(session, db_session, "stale_match_no_messages")
                processed_session_ids.add(session.id)

                # Also re-queue the partner if they exist and are still matched
                if session.matched_session_id:
                    partner = db_session.query(db.StudySession).filter(
                        db.StudySession.id == session.matched_session_id
                    ).first()
                    if partner and partner.match_status == "matched" and partner.id not in processed_session_ids:
                        partner_conv = json.loads(partner.conversation_log) if partner.conversation_log else []
                        if len(partner_conv) == 0:
                            requeue_or_timeout_session(partner, db_session, "stale_match_no_messages")
                            processed_session_ids.add(partner.id)

                print(f"🔄 Stale match processed: {session.id[:8]}... -> {result}")

        # 2. Clean up waiting sessions stuck for too long (>2 minutes)
        # These are likely from abandoned browsers or people who closed the tab
        # Skip sessions already marked as abandoned (report_abandonment already handled them)
        stale_waiting = db_session.query(db.StudySession).filter(
            db.StudySession.match_status == "waiting",
            db.StudySession.session_status != "abandoned",
            db.StudySession.waiting_room_entered_at < datetime.utcnow() - timedelta(minutes=2)
        ).all()

        for session in stale_waiting:
            session.match_status = "timed_out"
            session.timeout_screen = "backend_cleanup_waiting_room"
            decrement_role_counter(session, db_session)
            print(f"🧹 Stale waiting session cleaned up: {session.id[:8]}... (waiting >2 min)")

        # 3. Clean up assigned sessions that never clicked "Enter Waiting Room" (>2 minutes)
        # Use last_updated since waiting_room_entered_at isn't set yet for assigned sessions
        stale_assigned = db_session.query(db.StudySession).filter(
            db.StudySession.match_status == "assigned",
            db.StudySession.session_status != "abandoned",
            db.StudySession.last_updated < datetime.utcnow() - timedelta(minutes=2)
        ).all()

        for session in stale_assigned:
            session.match_status = "timed_out"
            session.timeout_screen = "backend_cleanup_post_demo_instructions"
            decrement_role_counter(session, db_session)
            print(f"🧹 Stale assigned session cleaned up: {session.id[:8]}... (assigned >2 min, never entered waiting room)")

        # 4. Clean up pre_consent sessions that never progressed (ghost sessions)
        # These are created by /get_or_assign_role but never completed demographics
        # Browser was closed before beforeunload was attached, so no beacon fired
        # Timeout matches consent screen (3 minutes)
        stale_pre_consent = db_session.query(db.StudySession).filter(
            db.StudySession.session_status == "pre_consent",
            db.StudySession.last_updated < datetime.utcnow() - timedelta(minutes=3)
        ).all()

        for session in stale_pre_consent:
            session.session_status = "abandoned"
            session.match_status = "timed_out"
            session.timeout_screen = "backend_cleanup_consent"
            decrement_role_counter(session, db_session)
            print(f"🧹 Ghost pre_consent session cleaned up: {session.id[:8]}... (pre_consent >3 min)")

        db_session.commit()

    except Exception as e:
        print(f"Error in cleanup_orphaned_sessions: {str(e)}")
        db_session.rollback()

# Schedule cleanup job to run every 1 minute (frequent cleanup for responsive user experience)
import threading
import time as time_module

def run_periodic_cleanup():
    """Background thread that runs cleanup every 1 minute (was 5, reduced for faster response)"""
    while True:
        time_module.sleep(60)  # 1 minute (faster cleanup to match 5-min user timeout rule)
        try:
            db_session = db.SessionLocal()
            cleanup_orphaned_sessions(db_session)
            db_session.close()
        except Exception as e:
            print(f"Periodic cleanup error: {str(e)}")

# Start cleanup thread when server starts
cleanup_thread = threading.Thread(target=run_periodic_cleanup, daemon=True)
cleanup_thread.start()
print("🔧 Background cleanup thread started (runs every 5 minutes)")

# --- API Endpoints ---

@app.get("/health")
async def health_check(db_session: Session = Depends(get_db)):
    """
    Health check endpoint with detailed system status.
    Use this to monitor if the server is healthy.
    """
    try:
        # Check database connection
        db_session.execute("SELECT 1")

        # Count active sessions
        active_count = len([s for s in sessions.values() if s.get('session_status') == 'active'])

        # Count waiting/matched
        waiting_count = db_session.query(db.StudySession).filter(
            db.StudySession.match_status == "waiting"
        ).count()

        matched_count = db_session.query(db.StudySession).filter(
            db.StudySession.match_status == "matched"
        ).count()

        return {
            "status": "healthy",
            "database": "connected",
            "study_mode": STUDY_MODE,
            "active_sessions": active_count,
            "waiting_for_match": waiting_count,
            "currently_matched": matched_count,
            "cleanup_thread": "running" if cleanup_thread.is_alive() else "dead"
        }
    except Exception as e:
        print(f"❌ HEALTH CHECK FAILED: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    # Avoid dependency on Jinja templates since the frontend is served separately.
    # Provide a simple status page for health checks and developer convenience.
    return HTMLResponse(
        content=(
            "<html><head><title>AI Turing Test API</title></head>"
            "<body><h2>Backend is running.</h2>"
            "<p>This service exposes JSON endpoints for the study frontend.</p>"
            "</body></html>"
        ),
        status_code=200,
    )

@app.post("/get_or_assign_role")
async def get_or_assign_role(data: GetOrAssignRoleRequest, db_session: Session = Depends(get_db)):
    """
    Assign role (interrogator or witness) to participant on page load.

    CRITICAL: Role assignment is PERMANENT for a participant_id (IRB compliance).
    If participant refreshes page, they MUST get the same role they were originally assigned.

    Uses atomic counter to maintain 50/50 balance between interrogator and witness roles.

    AI_WITNESS MODE: Everyone is assigned interrogator (no counter needed).
    """
    participant_id = data.participant_id
    prolific_pid = data.prolific_pid

    print(f"🎭 Role assignment request for participant_id={participant_id}, prolific_pid={prolific_pid}, mode={STUDY_MODE}")

    # AI_WITNESS MODE: Everyone is interrogator (talking to AI)
    if STUDY_MODE == "AI_WITNESS":
        print(f"✅ AI_WITNESS mode: Assigning interrogator role (no counter)")
        return {
            "role": "interrogator",
            "is_existing": False,
            "study_mode": STUDY_MODE
        }

    # STEP 1: Check if participant already has an ACTIVE role assigned (for refresh scenarios)
    # Only return cached role if session is still in-progress (active or pre_consent).
    # If session is completed/abandoned/interrupted, treat as new participant so the
    # counter logic runs fresh (fixes participantId persistence bug across sessions).
    existing_session = db_session.query(db.StudySession).filter(
        db.StudySession.id == participant_id
    ).first()

    if (existing_session and existing_session.role
            and existing_session.session_status in ("active", "pre_consent")):
        # Participant has an in-progress session - return same role (IRB compliance)
        print(f"✅ Found existing active role for participant: {existing_session.role} (status: {existing_session.session_status})")

        response_data = {
            "role": existing_session.role,
            "is_existing": True,
            "study_mode": STUDY_MODE
        }

        # Include social style info if witness
        if existing_session.role == "witness" and existing_session.social_style:
            style_info = SOCIAL_STYLES.get(existing_session.social_style, {})
            response_data["social_style"] = existing_session.social_style
            response_data["social_style_description"] = style_info.get("description", "")

        return response_data

    # STEP 2: New participant - assign role using atomic counter
    try:
        # Get or create the counter row (single row table with id=1)
        counter = db_session.query(db.RoleAssignmentCounter).filter(
            db.RoleAssignmentCounter.id == 1
        ).with_for_update().first()

        if not counter:
            # First time - initialize counter
            counter = db.RoleAssignmentCounter(
                id=1,
                interrogator_count=0,
                witness_count=0
            )
            db_session.add(counter)
            db_session.flush()  # Get the counter row created
            print("📊 Initialized RoleAssignmentCounter")

        # Decide which role to assign based on current counts
        if counter.interrogator_count < counter.witness_count:
            assigned_role = "interrogator"
            counter.interrogator_count += 1
        elif counter.witness_count < counter.interrogator_count:
            assigned_role = "witness"
            counter.witness_count += 1
        else:
            # Equal counts - alternate deterministically based on total
            total = counter.interrogator_count + counter.witness_count
            if total % 2 == 0:
                assigned_role = "interrogator"
                counter.interrogator_count += 1
            else:
                assigned_role = "witness"
                counter.witness_count += 1

        # Assign social style if witness
        assigned_social_style = None
        social_style_description = None

        if assigned_role == "witness":
            # Respect DEBUG_FORCE_SOCIAL_STYLE if set
            if DEBUG_FORCE_SOCIAL_STYLE and DEBUG_FORCE_SOCIAL_STYLE in ENABLED_SOCIAL_STYLES:
                assigned_social_style = DEBUG_FORCE_SOCIAL_STYLE
            else:
                assigned_social_style = random.choice(ENABLED_SOCIAL_STYLES)

            style_info = SOCIAL_STYLES.get(assigned_social_style, {})
            social_style_description = style_info.get("description", "")

        # Commit the counter update
        db_session.commit()

        print(f"🎭 ROLE ASSIGNED: {assigned_role} "
              f"(Counter now: interrogator={counter.interrogator_count}, witness={counter.witness_count})"
              f"{f', social_style={assigned_social_style}' if assigned_social_style else ''}")

        # CRITICAL: Create or update DB record so /finalize_no_session can find it
        # This is needed for counter decrement if participant drops out before /initialize_study
        try:
            if existing_session:
                # Reuse existing record (e.g., same browser, previous session completed/abandoned)
                existing_session.user_id = prolific_pid or participant_id
                existing_session.start_time = datetime.utcnow()
                existing_session.role = assigned_role
                existing_session.social_style = assigned_social_style
                existing_session.chosen_persona = "pending"
                existing_session.domain = "pending"
                existing_session.condition = "pending"
                existing_session.user_profile_survey = json.dumps({})
                existing_session.initial_tactic_analysis = "pending"
                existing_session.session_status = "pre_consent"
                existing_session.match_status = "unmatched"
                existing_session.counter_decremented = False  # Reset for new session
                existing_session.last_updated = datetime.utcnow()
                db_session.commit()
                print(f"✅ Reset existing DB record for {participant_id[:8]}... (pre-consent, was {existing_session.session_status})")
            else:
                # Create new minimal record
                minimal_session = db.StudySession(
                    id=participant_id,
                    user_id=prolific_pid or participant_id,
                    start_time=datetime.utcnow(),
                    role=assigned_role,
                    social_style=assigned_social_style,
                    chosen_persona="pending",
                    domain="pending",
                    condition="pending",
                    user_profile_survey=json.dumps({}),
                    initial_tactic_analysis="pending",
                    session_status="pre_consent",
                    last_updated=datetime.utcnow()
                )
                db_session.add(minimal_session)
                db_session.commit()
                print(f"✅ Created minimal DB record for {participant_id[:8]}... (pre-consent)")
        except Exception as e:
            print(f"⚠️ Could not create/update minimal record: {e}")
            db_session.rollback()

        # Return role assignment
        response_data = {
            "role": assigned_role,
            "is_existing": False,
            "study_mode": STUDY_MODE
        }

        if assigned_role == "witness":
            response_data["social_style"] = assigned_social_style
            response_data["social_style_description"] = social_style_description

        return response_data

    except Exception as e:
        db_session.rollback()
        print(f"❌ Error in role assignment: {e}")
        raise HTTPException(status_code=500, detail=f"Role assignment failed: {str(e)}")

@app.post("/initialize_study")
async def initialize_study(data: InitializeRequest, db_session: Session = Depends(get_db)):
    if not GEMINI_MODEL:
        raise HTTPException(status_code=500, detail="AI Model not initialized.")

    # CRITICAL: Use participant_id as session_id for role lookup to work
    # participant_id is persistent across refreshes (stored in localStorage)
    # This allows /get_or_assign_role to find existing sessions on refresh
    participant_id_val = data.participant_id or None
    if not participant_id_val:
        raise HTTPException(status_code=400, detail="participant_id is required")

    session_id = participant_id_val  # Use participant_id as the session identifier

    # Check if session already exists
    # Could be: (1) minimal record from /get_or_assign_role, or (2) full record from previous /initialize_study
    existing_session = db_session.query(db.StudySession).filter(
        db.StudySession.id == session_id
    ).first()

    # Check if this is a full record (has demographics) or just minimal (pre-consent)
    if existing_session:
        if existing_session.session_status == "pre_consent":
            # Minimal record exists - we need to UPDATE it with full demographics
            print(f"🔄 Updating minimal record {session_id[:8]}... with full demographics")
            # Will continue to update the record below
        else:
            # Full record already exists - this is a refresh after demographics submitted
            print(f"⚠️ Session {session_id[:8]}... fully initialized, returning existing")
            # Note: In-memory session recovery happens in other endpoints if needed
            # All data is safely stored in database
            return {"session_id": session_id, "message": "Study already initialized."}
    
    # Create the full user profile from all the new fields
    # Sanitize text inputs
    sanitized_models = [html.escape(str(model)) for model in data.ai_models_used]
    sanitized_ethnicity = [html.escape(str(eth)) for eth in data.ethnicity]
    sanitized_social_media = [html.escape(str(platform)) for platform in data.social_media_platforms]

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
        "income": html.escape(str(data.income)),
        "political_affiliation": html.escape(str(data.political_affiliation)),
        "social_media_platforms": sanitized_social_media,
        "internet_usage_per_week": data.internet_usage_per_week
    }
    
    assigned_context_for_convo, experimental_condition_val = assign_domain()

    # Everyone gets custom_extrovert persona with free-form tactic selection
    if DEBUG_FORCE_PERSONA and DEBUG_FORCE_PERSONA in PERSONAS:
        chosen_persona_key = DEBUG_FORCE_PERSONA
    else:
        chosen_persona_key = "custom_extrovert"

    print(f"--- Session {session_id}: Persona assigned: '{chosen_persona_key}' (Debug forced: {DEBUG_FORCE_PERSONA is not None}) ---")

    # NEW: Use pre-assigned role and social style from /get_or_assign_role
    # Role was already assigned on page load for proper consent form display (IRB compliance)
    assigned_role = data.role
    assigned_social_style = data.social_style

    print(f"--- Session {session_id}: Role PRE-ASSIGNED: '{assigned_role}' ---")
    if assigned_social_style:
        print(f"--- Session {session_id}: Social style PRE-ASSIGNED: '{assigned_social_style}' ---")

    # Skip initial tactic analysis - Gemini will choose tactics freely during conversation
    initial_tactic_analysis_for_session = {"full_analysis": "N/A: Free-form tactic selection (no initial analysis).", "recommended_tactic_key": None}

    # NEW: identifiers and ui event log (participant_id_val already set above)
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
        "role": assigned_role,  # NEW: Pre-assigned role (interrogator or witness)
        "social_style": assigned_social_style,  # NEW: Pre-assigned social style (for witnesses)
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

    # NEW: Update existing minimal record OR create new record
    if existing_session and existing_session.session_status == "pre_consent":
        # UPDATE existing minimal record with full data
        try:
            ui_events = sessions[session_id].get("ui_event_log", [])
            consent_accepted = any(event.get("event") == "consent_agree_clicked" for event in ui_events)

            existing_session.user_id = sessions[session_id]["user_id"]
            existing_session.start_time = sessions[session_id]["start_time"]
            existing_session.chosen_persona = sessions[session_id]["chosen_persona_key"]
            existing_session.domain = sessions[session_id]["assigned_domain"]
            existing_session.condition = sessions[session_id]["experimental_condition"]
            existing_session.user_profile_survey = json.dumps(sessions[session_id]["initial_user_profile_survey"])
            existing_session.initial_tactic_analysis = sessions[session_id]["initial_tactic_analysis"]["full_analysis"]
            existing_session.ui_event_log = json.dumps(ui_events)
            existing_session.consent_accepted = consent_accepted
            existing_session.session_status = "active"  # Change from pre_consent to active
            existing_session.last_updated = datetime.utcnow()
            # role and social_style already set from /get_or_assign_role

            db_session.commit()
            print(f"✅ Updated existing record for {session_id[:8]}... with full demographics")
        except Exception as e:
            print(f"❌ Error updating session record: {e}")
            db_session.rollback()
            raise HTTPException(status_code=500, detail="Failed to update session")
    else:
        # CREATE new initial database record
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
            print("🔴 FRONTEND ERROR CAPTURED")
        else:
            print("🔵 FRONTEND DEBUG INFO")
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


@app.post("/enter_waiting_room")
async def enter_waiting_room(request: Request, db_session: Session = Depends(get_db)):
    """
    Endpoint called after demographics submission.
    Assigns role and enters waiting room for HUMAN_WITNESS mode.
    For AI_WITNESS mode, returns immediately to proceed to chat.
    """
    data = await request.json()
    session_id = data.get('session_id')

    print(f"⏳ ENTER_WAITING_ROOM: Session {session_id[:8]}... Mode: {STUDY_MODE}")

    if session_id not in sessions:
        print(f"❌ ERROR: Session {session_id[:8]}... not found in memory")
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]

    # Check study mode
    if STUDY_MODE == "AI_WITNESS":
        # AI witness mode - show waiting room for consistency, but use simulated matching
        session['role'] = 'interrogator'
        session['match_status'] = 'waiting'  # Will simulate match on frontend
        session['matched_session_id'] = None  # No human partner
        session['first_message_sender'] = 'interrogator'  # User always sends first in AI mode

        # Update database
        session_record = db_session.query(db.StudySession).filter(
            db.StudySession.id == session_id
        ).first()
        if session_record:
            session_record.role = 'interrogator'
            session_record.match_status = 'waiting'  # Will be updated when chat starts
            db_session.commit()

        # AI MODE: Assign a social style for the AI witness to use
        # Respect DEBUG_FORCE_SOCIAL_STYLE if set, otherwise randomize
        ai_social_style = session.get('social_style')
        if not ai_social_style:
            if DEBUG_FORCE_SOCIAL_STYLE and DEBUG_FORCE_SOCIAL_STYLE in ENABLED_SOCIAL_STYLES:
                ai_social_style = DEBUG_FORCE_SOCIAL_STYLE
            else:
                ai_social_style = random.choice(ENABLED_SOCIAL_STYLES)
            session['social_style'] = ai_social_style
            # Also update database
            if session_record:
                session_record.social_style = ai_social_style
                db_session.commit()

        print("=" * 60)
        print(f"🤖 AI MODE SESSION STARTING")
        print(f"   Session: {session_id[:8]}...")
        print(f"   Role: interrogator (vs AI witness)")
        print(f"   AI Witness Social Style: {ai_social_style}")
        print("=" * 60)

        return JSONResponse(content={
            "ai_partner": True,
            "role": "interrogator",
            "match_status": "waiting",  # Frontend will simulate finding match
            "ai_social_style": ai_social_style  # Include in response for debugging
        })

    # HUMAN_WITNESS mode - just return that they're ready (no role assignment yet)
    return JSONResponse(content={
        "ai_partner": False,
        "ready_to_join": True
    })


@app.post("/join_waiting_room")
async def join_waiting_room(request: Request, db_session: Session = Depends(get_db)):
    """
    Called when user clicks "Enter Waiting Room" button.
    Uses PRE-ASSIGNED role from page load (assigned via /get_or_assign_role).
    Marks as waiting + attempts match.

    CRITICAL: Role was already assigned on page load using atomic counter.
    DO NOT re-assign role here or it will break the counter balance.
    """
    data = await request.json()
    session_id = data.get('session_id')

    print(f"🚪 JOIN_WAITING_ROOM: Session {session_id[:8]}... clicking button to enter")

    if session_id not in sessions:
        print(f"❌ ERROR: Session {session_id[:8]}... not found in memory")
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]

    # NEW: Use pre-assigned role (from /get_or_assign_role via /initialize_study)
    # Role and social style were already set on page load for IRB compliance
    assigned_role = session.get('role')
    assigned_social_style = session.get('social_style')

    if not assigned_role:
        print(f"❌ ERROR: Session {session_id[:8]}... has no pre-assigned role")
        raise HTTPException(status_code=500, detail="No role assigned. Please refresh and try again.")

    waiting_timestamp = datetime.utcnow()

    print(f"🎭 USING PRE-ASSIGNED ROLE: Session {session_id[:8]}... role={assigned_role}, social_style={assigned_social_style}")

    # Update database: Mark as waiting (role already stored from /initialize_study)
    try:
        session_record = db_session.query(db.StudySession).filter(
            db.StudySession.id == session_id
        ).first()
        if session_record:
            # Role and social_style already set during /initialize_study
            # Just update waiting status
            session_record.match_status = 'waiting'
            session_record.waiting_room_entered_at = waiting_timestamp
            db_session.commit()
            print(f"✅ Session {session_id[:8]}... marked as waiting (role: {assigned_role}), DB updated")
        else:
            print(f"⚠️ WARNING: Session {session_id[:8]}... not found in database")
            raise HTTPException(status_code=404, detail="Session not found in database")
    except HTTPException:
        raise  # Re-raise HTTPException as-is
    except Exception as e:
        print(f"❌ DATABASE ERROR: Failed to update session {session_id[:8]}... during join: {str(e)}")
        db_session.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    # Update memory (role already set, just update match status)
    session['match_status'] = 'waiting'
    session['waiting_room_entered_at'] = waiting_timestamp
    # NEW: Store social style in memory for witnesses
    if assigned_social_style:
        session['social_style'] = assigned_social_style

    # Try to match
    match_result = attempt_match(db_session)
    if match_result:
        print(f"⚡ IMMEDIATE MATCH: {session_id[:8]}... matched instantly!")
    else:
        print(f"⏳ NO MATCH YET: {session_id[:8]}... waiting for partner...")

    # Build response with social style info for witnesses
    response_data = {
        "success": True,
        "role": assigned_role,
        "match_status": session.get('match_status', 'waiting')
    }

    # NEW: Include social style instructions for witnesses
    if assigned_social_style:
        style_info = SOCIAL_STYLES.get(assigned_social_style, {})
        response_data["social_style"] = assigned_social_style
        response_data["social_style_description"] = style_info.get("description", "")

    return JSONResponse(content=response_data)


@app.get("/check_match_status")
async def check_match_status(session_id: str, db_session: Session = Depends(get_db)):
    """
    Poll endpoint to check if participant has been matched with a partner.
    Called every 3 seconds from frontend while in waiting room.
    """
    # Try in-memory first
    if session_id in sessions:
        session = sessions[session_id]
    else:
        # Check database
        session_record = db_session.query(db.StudySession).filter(
            db.StudySession.id == session_id
        ).first()
        if not session_record:
            raise HTTPException(status_code=404, detail="Session not found")

        # Recover from database
        recovered = recover_session_from_database(session_id, db_session)
        if recovered:
            sessions[session_id] = recovered
            session = recovered
        else:
            raise HTTPException(status_code=404, detail="Session could not be recovered")

    match_status = session.get('match_status', 'unmatched')

    # FIX BUG 1: Also check database for cleanup-marked status (in-memory may be stale)
    # The cleanup job marks sessions as timed_out/orphaned in the database
    db_record = db_session.query(db.StudySession).filter(
        db.StudySession.id == session_id
    ).first()
    if db_record and db_record.match_status in ('timed_out', 'orphaned'):
        print(f"⚠️ SESSION CLEANED UP: {session_id[:8]}... was marked {db_record.match_status} by cleanup job")
        return JSONResponse(content={
            "matched": False,
            "timed_out": True,
            "cleanup_reason": db_record.match_status
        })

    # Calculate time waiting
    time_waiting_seconds = 0
    if session.get('waiting_room_entered_at'):
        entered_at = session['waiting_room_entered_at']
        if isinstance(entered_at, datetime):
            time_waiting_seconds = (datetime.utcnow() - entered_at).total_seconds()

    if match_status == 'matched':
        # Get proceed_to_chat_at timestamp for frontend synchronization
        proceed_at = session.get('proceed_to_chat_at')
        proceed_at_timestamp = proceed_at.timestamp() if isinstance(proceed_at, datetime) else None

        print(f"🔍 MATCH STATUS CHECK: Session {session_id[:8]}... matched, proceed_at={proceed_at}, "
              f"timestamp={proceed_at_timestamp}, type={type(proceed_at).__name__}")

        return JSONResponse(content={
            "matched": True,
            "partner_session_id": session.get('matched_session_id'),
            "first_message_sender": session.get('first_message_sender'),
            "time_waiting_seconds": time_waiting_seconds,
            "proceed_to_chat_at": proceed_at_timestamp  # Unix timestamp for frontend
        })
    else:
        # Check if this session was re-queued (for frontend UX messaging)
        requeue_count = db_record.requeue_count if db_record else 0

        return JSONResponse(content={
            "matched": False,
            "timed_out": False,
            "time_waiting_seconds": time_waiting_seconds,
            "requeue_count": requeue_count or 0,  # 0 = first match attempt, >0 = re-queued after partner dropped
            "was_requeued": (requeue_count or 0) > 0  # Convenience flag for frontend
        })


@app.get("/study_status_ping")
async def study_status_ping(db_session: Session = Depends(get_db)):
    """
    Status monitoring endpoint - returns current state of all active participants.
    Called periodically by frontend to log status to Railway for visual monitoring.
    Helps researcher see if interrogator/witness counts are balanced and catch issues early.
    """
    try:
        # Count active sessions by role and status
        # Only count sessions that are actually active (not completed/abandoned)
        active_sessions = db_session.query(db.StudySession).filter(
            db.StudySession.session_status.in_(["active", "pre_consent"]),
            db.StudySession.role.isnot(None)
        ).all()

        # Categorize by role and match_status
        stats = {
            "interrogators": {"waiting": 0, "matched": 0, "in_conversation": 0, "total": 0},
            "witnesses": {"waiting": 0, "matched": 0, "in_conversation": 0, "total": 0},
            "unassigned": 0
        }

        for session in active_sessions:
            role = session.role
            match_status = session.match_status or "unassigned"

            if role == "interrogator":
                stats["interrogators"]["total"] += 1
                if match_status == "waiting":
                    stats["interrogators"]["waiting"] += 1
                elif match_status == "matched":
                    # Check if they have messages (in conversation vs just matched)
                    conv_log = json.loads(session.conversation_log) if session.conversation_log else []
                    if len(conv_log) > 0:
                        stats["interrogators"]["in_conversation"] += 1
                    else:
                        stats["interrogators"]["matched"] += 1
            elif role == "witness":
                stats["witnesses"]["total"] += 1
                if match_status == "waiting":
                    stats["witnesses"]["waiting"] += 1
                elif match_status == "matched":
                    conv_log = json.loads(session.conversation_log) if session.conversation_log else []
                    if len(conv_log) > 0:
                        stats["witnesses"]["in_conversation"] += 1
                    else:
                        stats["witnesses"]["matched"] += 1
            else:
                stats["unassigned"] += 1

        # Get role counter for comparison
        counter = db_session.query(db.RoleAssignmentCounter).filter(
            db.RoleAssignmentCounter.id == 1
        ).first()

        counter_stats = {
            "interrogator_count": counter.interrogator_count if counter else 0,
            "witness_count": counter.witness_count if counter else 0
        }

        # Calculate mismatch
        waiting_mismatch = stats["interrogators"]["waiting"] - stats["witnesses"]["waiting"]
        total_mismatch = stats["interrogators"]["total"] - stats["witnesses"]["total"]

        # Log to Railway with clear visual formatting
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    📊 STUDY STATUS PING                       ║
╠══════════════════════════════════════════════════════════════╣
║  INTERROGATORS: {stats['interrogators']['total']:3d} total | {stats['interrogators']['waiting']:3d} waiting | {stats['interrogators']['matched']:3d} matched | {stats['interrogators']['in_conversation']:3d} chatting
║  WITNESSES:     {stats['witnesses']['total']:3d} total | {stats['witnesses']['waiting']:3d} waiting | {stats['witnesses']['matched']:3d} matched | {stats['witnesses']['in_conversation']:3d} chatting
║  ─────────────────────────────────────────────────────────────
║  WAITING MISMATCH: {waiting_mismatch:+d} (positive = more interrogators waiting)
║  TOTAL MISMATCH:   {total_mismatch:+d}
║  ─────────────────────────────────────────────────────────────
║  ROLE COUNTER: I={counter_stats['interrogator_count']} W={counter_stats['witness_count']}
║  STUDY MODE: {STUDY_MODE}
╚══════════════════════════════════════════════════════════════╝
""")

        return JSONResponse(content={
            "status": "ok",
            "study_mode": STUDY_MODE,
            "interrogators": stats["interrogators"],
            "witnesses": stats["witnesses"],
            "waiting_mismatch": waiting_mismatch,
            "total_mismatch": total_mismatch,
            "role_counter": counter_stats,
            "timestamp": datetime.utcnow().isoformat()
        })

    except Exception as e:
        print(f"❌ STUDY STATUS PING ERROR: {str(e)}")
        return JSONResponse(content={
            "status": "error",
            "error": str(e)
        })


@app.get("/check_partner_message")
async def check_partner_message(session_id: str, db_session: Session = Depends(get_db)):
    """
    Poll endpoint for human-human conversations.
    Checks if partner has sent a new message.
    """
    if session_id not in sessions:
        print(f"⚠️ CHECK_PARTNER_MESSAGE: Session {session_id[:8]}... not in memory, attempting recovery")
        # Try recovery
        recovered_session = recover_session_from_database(session_id, db_session)
        if recovered_session:
            sessions[session_id] = recovered_session
            print(f"✅ Session {session_id[:8]}... recovered successfully")
        else:
            print(f"❌ Session {session_id[:8]}... recovery failed - not found in database")
            raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    partner_id = session.get('matched_session_id')

    if not partner_id:
        raise HTTPException(status_code=400, detail="No partner matched")

    # NEW: Check if THIS session has been marked as partner_dropped (partner abandoned)
    if session.get('match_status') == 'partner_dropped':
        print(f"🚨 THIS SESSION MARKED AS PARTNER_DROPPED: {session_id[:8]}...")
        return JSONResponse(content={
            "new_message": False,
            "partner_dropped": True,
            "study_completed": False
        })

    # Check if partner session exists
    partner = sessions.get(partner_id)
    if not partner:
        # Partner might have dropped - check database
        partner_record = db_session.query(db.StudySession).filter(
            db.StudySession.id == partner_id
        ).first()

        # Check if partner completed the study normally (interrogator finished)
        if partner_record and partner_record.session_status == "completed":
            print(f"✅ PARTNER COMPLETED: {partner_id[:8]}... completed study normally")
            return JSONResponse(content={
                "new_message": False,
                "partner_dropped": False,
                "study_completed": True  # Partner finished the study
            })

        if not partner_record or partner_record.match_status == "partner_dropped":
            return JSONResponse(content={
                "new_message": False,
                "partner_dropped": True,
                "study_completed": False
            })

        # Try to recover partner session
        partner = recover_session_from_database(partner_id, db_session)
        if partner:
            sessions[partner_id] = partner
        else:
            return JSONResponse(content={
                "new_message": False,
                "partner_dropped": True,
                "study_completed": False
            })

    # Check if partner has sent a new message
    # SAFETY: Only return messages that are actually newer (prevents duplicates)
    partner_turn = partner.get('turn_count', 0)
    my_turn = session.get('turn_count', 0)

    if partner_turn > my_turn:
        # Partner has sent a new message
        latest_message = partner['conversation_log'][-1] if partner['conversation_log'] else None

        if latest_message:
            # NEW: Check if message has passed its delivery time (artificial delay for human mode)
            delivery_time = latest_message.get('delivery_time')
            current_time = time.time()

            if delivery_time and current_time < delivery_time:
                # Message not ready yet - still in artificial delay period
                remaining_delay = delivery_time - current_time
                print(f"⏳ MESSAGE DELAYED: {partner_id[:8]}... -> {session_id[:8]}... | {remaining_delay:.2f}s remaining")

                return JSONResponse(content={
                    "new_message": False,
                    "partner_typing": True,  # NEW: Signal that partner is "typing" (artificial delay)
                    "partner_dropped": False,
                    "study_completed": False
                })

            # SAFETY: Verify turn numbers match (detect gaps)
            if partner_turn - my_turn > 1:
                print(f"⚠️ MESSAGE GAP: Session {session_id[:8]}... - Partner turn {partner_turn}, my turn {my_turn}")
                # Still continue - frontend will handle it

            # Add partner's message to my conversation log
            session['conversation_log'].append(latest_message)
            session['turn_count'] = partner_turn

            print(f"✉️ MESSAGE DELIVERED: {partner_id[:8]}... -> {session_id[:8]}... (Turn {partner_turn})")

            return JSONResponse(content={
                "new_message": True,
                "message_text": latest_message.get('user', latest_message.get('assistant', '')),
                "turn": partner_turn,
                "timestamp": time.time(),
                "partner_dropped": False,
                "study_completed": False
            })

    return JSONResponse(content={
        "new_message": False,
        "partner_dropped": False,
        "study_completed": False
    })


@app.post("/signal_typing")
async def signal_typing(request: Request):
    """
    Signal that user is currently typing.
    Sets a timestamp - partner can check if recent (within last 3 seconds = still typing).
    """
    data = await request.json()
    session_id = data.get('session_id')

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]

    # Store typing timestamp
    session['typing_at'] = datetime.utcnow()

    return JSONResponse(content={"success": True})


@app.get("/check_partner_typing")
async def check_partner_typing(session_id: str):
    """
    Check if partner is currently typing.
    Returns true if partner's typing timestamp is within last 3 seconds.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    partner_id = session.get('matched_session_id')

    if not partner_id or partner_id not in sessions:
        return JSONResponse(content={"is_typing": False})

    partner = sessions[partner_id]
    typing_at = partner.get('typing_at')

    if typing_at and isinstance(typing_at, datetime):
        # Check if typing signal is recent (within last 3 seconds)
        seconds_since_typing = (datetime.utcnow() - typing_at).total_seconds()
        is_typing = seconds_since_typing < 3.0

        return JSONResponse(content={"is_typing": is_typing})

    return JSONResponse(content={"is_typing": False})


@app.get("/check_session_status")
async def check_session_status(session_id: str, db_session: Session = Depends(get_db)):
    """
    Check session status for refresh recovery.
    Returns current state of session for frontend to restore.
    """
    session_record = db_session.query(db.StudySession).filter(
        db.StudySession.id == session_id
    ).first()

    if not session_record:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get turn count from conversation log
    turn_count = 0
    if session_record.conversation_log:
        try:
            conv_log = json.loads(session_record.conversation_log)
            turn_count = len(conv_log)
        except:
            turn_count = 0

    return JSONResponse(content={
        "session_id": session_id,
        "role": session_record.role,
        "match_status": session_record.match_status,
        "matched_session_id": session_record.matched_session_id,
        "first_message_sender": session_record.first_message_sender,
        "turn_count": turn_count
    })


@app.post("/report_abandonment")
async def report_abandonment(request: Request, db_session: Session = Depends(get_db)):
    """
    Called via sendBeacon when user navigates away (refresh, back, close tab).
    Marks abandoner's session and notifies their partner.
    """
    try:
        data = await request.json()
        session_id = data.get('session_id')
        participant_id = data.get('participant_id')
        prolific_pid = data.get('prolific_pid')
        reason = data.get('reason', 'navigation_abandonment')

        print(f"🚨 ABANDONMENT REPORTED: session_id={session_id[:8] if session_id else 'None'}..., reason={reason}")

        # Find session record
        session_record = None
        if session_id:
            session_record = db_session.query(db.StudySession).filter(
                db.StudySession.id == session_id
            ).first()
        elif participant_id:
            # Try to find by participant_id if session_id not available
            session_record = db_session.query(db.StudySession).filter(
                db.StudySession.id == participant_id
            ).first()

        if session_record:
            # Mark as abandoned
            session_record.session_status = 'abandoned'
            session_record.match_status = 'abandoned'
            session_record.last_updated = datetime.utcnow()

            # Decrement role counter so next participant gets the correct role
            decrement_role_counter(session_record, db_session)

            # Find partner and notify them
            partner_id = session_record.matched_session_id
            if partner_id:
                # Mark partner's session as partner_dropped
                partner_record = db_session.query(db.StudySession).filter(
                    db.StudySession.id == partner_id
                ).first()
                if partner_record:
                    partner_record.match_status = 'partner_dropped'
                    print(f"✅ Partner {partner_id[:8]}... notified of abandonment")

                # Update in-memory sessions
                if partner_id in sessions:
                    sessions[partner_id]['match_status'] = 'partner_dropped'

            db_session.commit()
            print(f"✅ Abandonment logged for {session_id[:8] if session_id else participant_id[:8]}...")

        # Also save to DroppedParticipant table
        try:
            dropped = db.DroppedParticipant(
                participant_id=participant_id or session_id,
                prolific_pid=prolific_pid,
                reason=reason
            )
            db_session.add(dropped)
            db_session.commit()
        except Exception as e:
            print(f"⚠️ Could not save to DroppedParticipant (may already exist): {e}")
            db_session.rollback()

        return JSONResponse(content={"message": "Abandonment logged"}, status_code=200)

    except Exception as e:
        print(f"❌ Error in report_abandonment: {e}")
        # Always return 200 for beacon (even on error - beacon can't retry)
        return JSONResponse(content={"message": "Error logged"}, status_code=200)


@app.post("/report_partner_dropped")
async def report_partner_dropped(request: Request, db_session: Session = Depends(get_db)):
    """
    Report that partner has disconnected during waiting room phase.

    NEW BEHAVIOR: Re-queues the reporting session for a new match instead of
    abandoning them. This saves participants whose partner dropped.

    Returns:
    - requeued: true if session was re-queued for new match
    - timed_out: true if session exceeded 4-min total wait cap
    """
    data = await request.json()
    session_id = data.get('session_id')

    # Try to recover session if not in memory
    if session_id not in sessions:
        session_record = db_session.query(db.StudySession).filter(
            db.StudySession.id == session_id
        ).first()
        if session_record:
            recovered = recover_session_from_database(session_id, db_session)
            if recovered:
                sessions[session_id] = recovered
        else:
            raise HTTPException(status_code=404, detail="Session not found")

    session = sessions.get(session_id, {})
    partner_id = session.get('matched_session_id')

    # Get the database record for this session
    session_record = db_session.query(db.StudySession).filter(
        db.StudySession.id == session_id
    ).first()

    if not session_record:
        raise HTTPException(status_code=404, detail="Session record not found")

    # Check if any messages were exchanged
    conv_log = json.loads(session_record.conversation_log) if session_record.conversation_log else []

    if len(conv_log) > 0:
        # Messages were exchanged - this is a mid-conversation dropout
        # Don't re-queue, let the frontend handle final choice flow
        session_record.match_status = 'partner_dropped'
        if session_id in sessions:
            sessions[session_id]['match_status'] = 'partner_dropped'
        db_session.commit()

        print(f"❌ Partner dropped MID-CONVERSATION: {session_id[:8]}... ({len(conv_log)} messages exchanged)")
        return JSONResponse(content={
            "message": "Partner dropout logged (mid-conversation)",
            "requeued": False,
            "timed_out": False,
            "had_messages": True
        })

    # No messages exchanged - this is a waiting room dropout
    # RE-QUEUE this session for a new partner
    result = requeue_or_timeout_session(session_record, db_session, "partner_dropped_waiting_room")

    # Mark the partner's session as orphaned (they dropped)
    if partner_id:
        if partner_id in sessions:
            sessions[partner_id]['match_status'] = 'orphaned'

        partner_record = db_session.query(db.StudySession).filter(
            db.StudySession.id == partner_id
        ).first()
        if partner_record:
            partner_record.match_status = 'orphaned'
            partner_record.session_status = 'abandoned'
            partner_record.timeout_screen = 'partner_reported_dropout'
            decrement_role_counter(partner_record, db_session)

    db_session.commit()

    print(f"🔄 Partner dropout in waiting room: {session_id[:8]}... -> {result}")

    return JSONResponse(content={
        "message": f"Partner dropout handled: {result}",
        "requeued": result == "requeued",
        "timed_out": result == "timed_out",
        "had_messages": False
    })


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
    
    session = sessions[session_id]

    # NEW: Check if this is human-human conversation (HUMAN_WITNESS mode)
    partner_session_id = session.get('matched_session_id')
    is_human_partner = (STUDY_MODE == "HUMAN_WITNESS" and
                        partner_session_id and
                        partner_session_id in sessions)

    if is_human_partner:
        # Human witness conversation - route message to partner (no AI generation)
        partner = sessions[partner_session_id]
        current_turn = session["turn_count"] + 1

        # NEW: Calculate artificial delay based on message length (empirical data)
        # Count words in message
        word_count = len(user_message.split())

        # Find appropriate delay parameters based on word count
        median_delay, std_delay = (19.1, 4.17)  # Default to 21-40 words category
        for (min_words, max_words), (median, std) in HUMAN_MESSAGE_DELAY_BY_LENGTH.items():
            if min_words <= word_count <= max_words:
                median_delay, std_delay = median, std
                break

        # Sample from normal distribution using median and observed std dev
        delay_seconds = np.random.normal(median_delay, std_delay)
        delay_seconds = min(delay_seconds, HUMAN_MESSAGE_DELAY_CEILING)  # Clamp at ceiling
        delay_seconds = max(delay_seconds, 5.0)  # Minimum 5s delay to prevent unrealistically fast responses

        # Calculate when this message should be delivered to partner
        sent_time = datetime.utcnow().timestamp()
        delivery_time = sent_time + delay_seconds

        # Add message to both conversation logs
        turn_data = {
            "turn": current_turn,
            "user": user_message,
            "assistant": "",  # No AI response
            "sender_role": session.get('role', 'unknown'),
            "timestamp": sent_time,
            "message_composition_time_seconds": data.message_composition_time_seconds,  # Time to type message
            "delivery_time": delivery_time,  # NEW: When message should be delivered to partner
            "artificial_delay_seconds": delay_seconds,  # NEW: For analysis
            "message_word_count": word_count,  # NEW: For analysis - correlate delay with length
            "delay_category_median": median_delay,  # NEW: Which category was used
            "delay_category_std": std_delay  # NEW: Std dev for the category
        }

        session["conversation_log"].append(turn_data)
        session["turn_count"] = current_turn

        # Save to database
        update_session_after_message(session, db_session)

        print(f"Human-human message sent: {session.get('role')} ({session_id[:8]}...) -> {partner.get('role')} ({partner_session_id[:8]}...) | Words: {word_count}, Delay: {delay_seconds:.2f}s (median={median_delay:.1f}s)")

        return {
            "human_partner": True,
            "message_routed": True,
            "turn": current_turn,
            "timestamp": sent_time,
            "artificial_delay_seconds": delay_seconds  # Return to frontend for bubble timing
        }

    # AI_WITNESS mode or no human partner - proceed with AI generation
    if not GEMINI_MODEL:
        raise HTTPException(status_code=500, detail="AI Model not initialized.")

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
        # NEW: Pass previous tactic analyses if CONNECTIVE_CONTEXT_MEMORY is enabled
        prev_tactic_analyses = session["tactic_selection_log"] if CONNECTIVE_CONTEXT_MEMORY else None

        tactic_key_for_this_turn, tactic_sel_justification = await select_tactic_for_current_turn(
            GEMINI_MODEL,
            session["initial_user_profile_survey"],
            user_message,
            session["conversation_log"],
            session["initial_tactic_analysis"],
            current_ai_response_turn,
            retrieved_chosen_persona_key,
            session.get("social_style") or "DIRECT",  # Pass social style for dynamic prompt
            prev_tactic_analyses
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
        simple_history_for_your_prompt.append({
            "user": entry["user"],
            "user_timestamp": entry.get("user_timestamp", ""),
            "assistant": entry.get("assistant", ""),
            "assistant_timestamp": entry.get("assistant_timestamp", "")
        })

    # NEW: Retry logic for AI response generation
    max_retries = 3
    ai_response_text = None
    researcher_notes = None
    backend_retry_count = 0
    backend_retry_time = 0.0

    # NEW: Build context for CONNECTIVE_CONTEXT_MEMORY if enabled
    current_tactic_analysis_for_context = tactic_sel_justification if CONNECTIVE_CONTEXT_MEMORY else None
    prev_researcher_notes = None
    if CONNECTIVE_CONTEXT_MEMORY and session.get("ai_researcher_notes_log"):
        # Convert notes log format to what generate_ai_response expects
        prev_researcher_notes = [
            {"turn": entry["turn"], "researcher_notes": entry["notes"]}
            for entry in session["ai_researcher_notes_log"]
        ]

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
                retrieved_chosen_persona_key,
                session.get("social_style") or "DIRECT",
                current_tactic_analysis_for_context,
                prev_researcher_notes
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
    current_social_style = session.get("social_style") or "DIRECT"  # Handle both missing key and None value
    print(f"--- DEBUG (Turn {current_ai_response_turn}, Session {session_id[:8]}...): Style: {current_social_style} | Tactic: {tactic_key_for_this_turn or 'None'} | AI Resp Len: {ai_text_length}c ---")

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

    # Capture timestamps for context memory
    user_message_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ai_response_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    turn_data = {
        "turn": current_ai_response_turn,
        "user": user_message,
        "user_timestamp": user_message_timestamp,
        "assistant": ai_response_text,
        "assistant_timestamp": ai_response_timestamp,
        "tactic_used": tactic_key_for_this_turn,
        "tactic_selection_justification": tactic_sel_justification,
        "timing": {
            "api_call_time_seconds": time_spent_on_actual_ai_calls,
            "sleep_duration_seconds": sleep_duration_needed,
            "typing_indicator_delay_seconds": data.typing_indicator_delay_seconds,
            "network_delay_seconds": None,  # Will be updated by separate network delay endpoint
            "message_composition_time_seconds": data.message_composition_time_seconds  # Time from first keystroke to send
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

    # Update match status to "matched" when conversation starts (for AI mode simulated match)
    if session.get('match_status') == 'waiting':
        session['match_status'] = 'matched'
        session['matched_at'] = datetime.utcnow()

        # Update database
        session_record = db_session.query(db.StudySession).filter(
            db.StudySession.id == session_id
        ).first()
        if session_record:
            session_record.match_status = 'matched'
            session_record.matched_at = datetime.utcnow()
            db_session.commit()

    # Comprehensive conversation start logging
    role = session.get('role', 'unknown')
    social_style = session.get('social_style') or 'N/A'
    is_ai_mode = session.get('matched_session_id') is None
    mode_str = "AI_WITNESS" if is_ai_mode else "HUMAN_WITNESS"

    print("=" * 60)
    print(f"💬 CONVERSATION STARTED")
    print(f"   Session: {session_id[:8]}...")
    print(f"   Mode: {mode_str}")
    print(f"   Role: {role}")
    print(f"   Social Style: {social_style}")
    if not is_ai_mode:
        partner_id = session.get('matched_session_id', 'unknown')
        print(f"   Partner: {partner_id[:8] if partner_id else 'N/A'}...")
    print(f"   Timestamp: {datetime.utcnow().isoformat()}Z")
    print("=" * 60)

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

    # NEW: Block witnesses from submitting ratings (only interrogators can rate)
    if session.get('role') == 'witness':
        raise HTTPException(
            status_code=403,
            detail="Witnesses cannot submit ratings. Only interrogators can rate."
        )

    actual_decision_time = data.decision_time_seconds
    if actual_decision_time is None:
        print(f"Warning: decision_time_seconds was None for DDM rating. Session {session_id}, Turn {session['turn_count']}. Using placeholder.")
        actual_decision_time = -1.0

    session["intermediate_ddm_confidence_ratings"].append({
        "turn": session["turn_count"],
        "binary_choice": data.binary_choice,  # 'human' or 'ai'
        "binary_choice_time_ms": data.binary_choice_time_ms,  # Time to make binary choice
        "confidence": data.confidence,  # 0-1 scale
        "confidence_percent": data.confidence_percent,  # 0-100 scale
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
        # NEW: Study ends when time expires and participant submits their rating (binary choice + confidence)
        # No longer require exactly 0 or 1 - the binary choice is the decision, confidence is always 0-100%

        # Use binary choice to determine if AI was detected
        session["ai_detected_final"] = (data.binary_choice == 'ai')
        session["final_binary_choice"] = data.binary_choice
        session["final_confidence_percent"] = data.confidence_percent
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

class TimeoutRecordRequest(BaseModel):
    participant_id: str
    session_id: Optional[str] = None
    timeout_screen: str  # consent, instructions, demographics, role_assignment, waiting_room, partner_timeout, witness_final, feedback, debrief, backend_cleanup

@app.post("/record_timeout")
async def record_timeout(data: TimeoutRecordRequest, db_session: Session = Depends(get_db)):
    """
    Record which screen/phase caused a timeout for analytics.
    Called by frontend before redirecting to Prolific.
    """
    print(f"⏱️ TIMEOUT RECORDED: participant={data.participant_id[:8]}..., screen={data.timeout_screen}")

    # Try to find session by session_id first, then by participant_id
    session_record = None
    if data.session_id:
        session_record = db_session.query(db.StudySession).filter(
            db.StudySession.id == data.session_id
        ).first()

    if not session_record and data.participant_id:
        # Look up by participant_id (which is stored as session id for pre_consent sessions)
        session_record = db_session.query(db.StudySession).filter(
            db.StudySession.id == data.participant_id
        ).first()

    if session_record:
        session_record.timeout_screen = data.timeout_screen
        session_record.session_status = "timeout"
        session_record.last_updated = datetime.utcnow()

        # Decrement role counter since they didn't complete
        decrement_role_counter(session_record, db_session)

        db_session.commit()
        print(f"✅ Timeout recorded in database for session {session_record.id[:8]}...")
        return {"success": True, "message": "Timeout recorded"}
    else:
        print(f"⚠️ Could not find session to record timeout for participant {data.participant_id[:8]}...")
        return {"success": False, "message": "Session not found, timeout logged but not saved to DB"}


@app.post("/submit_final_comment")
async def submit_final_comment(data: FinalCommentRequest, db_session: Session = Depends(get_db)):
    session_record = db_session.query(db.StudySession).filter(db.StudySession.id == data.session_id).first()
    if not session_record:
        raise HTTPException(status_code=404, detail="Could not find the completed study session to add comment to.")

    # Sanitize final comment
    sanitized_comment = html.escape(str(data.comment))
    session_record.final_user_comment = sanitized_comment

    # NEW: Save witness binary choice if provided
    if data.binary_choice:
        session_record.final_binary_choice = data.binary_choice
        print(f"Witness binary choice saved: {data.binary_choice}")

    db_session.commit()
    print(f"Final comment added to session {data.session_id}.")
    return {"message": "Final comment received. Thank you."}

@app.post("/finalize_no_session")
async def finalize_no_session(data: FinalizeNoSessionRequest, db_session: Session = Depends(get_db)):
    """
    Called when participant drops out (e.g., declines consent, times out).

    CRITICAL: If participant was assigned a role, we MUST decrement the counter
    to prevent balancing against "ghost" users who declined.
    """
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

    # STEP 1: Check if participant was assigned a role and decrement counter
    try:
        existing_session = db_session.query(db.StudySession).filter(
            db.StudySession.id == participant_id_val
        ).first()

        if existing_session and existing_session.role:
            print(f"⚠️ Participant {participant_id_val[:8]}... had role '{existing_session.role}' assigned but dropped out")
            existing_session.session_status = "abandoned"
            existing_session.match_status = "abandoned"  # CRITICAL: Also update match_status to prevent ghost matches
            decrement_role_counter(existing_session, db_session)
            db_session.commit()
    except Exception as e:
        print(f"❌ ERROR decrementing counter: {e}")
        db_session.rollback()
        # Continue to dropout save even if counter update fails

    # STEP 3: Save dropout to database for tracking
    try:
        dropped_participant = db.DroppedParticipant(
            participant_id=participant_id_val,
            prolific_pid=prolific_pid_val,
            reason=data.reason or "unknown"
        )
        db_session.add(dropped_participant)
        db_session.commit()
        print(f"✅ DROPOUT SAVED: Participant {participant_id_val[:8]}... declined/dropped with reason: {data.reason}")
    except Exception as e:
        print(f"❌ ERROR saving dropout to database: {e}")
        db_session.rollback()
        # Don't raise error - still return success to frontend

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
        print("Gemini model initialized.")
    # For local testing, you would run with: uvicorn main:app --reload
    # Railway uses its own start command from railway.json
    pass
