import os
from sqlalchemy import create_engine, Column, String, Text, Float, Boolean, DateTime, Integer, Index, inspect, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json

DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Configure connection pooling for better concurrent performance
if DATABASE_URL and DATABASE_URL.startswith("postgresql://"):
    # Production: PostgreSQL with optimized pooling
    engine = create_engine(
        DATABASE_URL,
        pool_size=10,              # Number of connections to keep open
        max_overflow=20,           # Additional connections when pool is exhausted
        pool_timeout=30,           # Seconds to wait before timing out getting a connection
        pool_recycle=3600,         # Recycle connections after 1 hour
        pool_pre_ping=True,        # Test connections before using them
        connect_args={
            "connect_timeout": 10,  # Connection timeout in seconds
            "options": "-c statement_timeout=30000"  # 30 second query timeout
        }
    )
else:
    # Development: SQLite without pooling
    engine = create_engine(DATABASE_URL or "sqlite:///./test.db")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class StudySession(Base):
    __tablename__ = "study_sessions"

    id = Column(String, primary_key=True)
    user_id = Column(String, index=True)  # Add index for faster lookups
    start_time = Column(DateTime, default=datetime.utcnow)
    chosen_persona = Column(String)
    social_style = Column(String, nullable=True)  # Social style assigned (WARM, PLAYFUL, DIRECT, GUARDED, CONTRARIAN, ADAPTIVE, HYBRID, NEUTRAL)
    domain = Column(String)
    condition = Column(String)
    user_profile_survey = Column(Text)  # JSON
    ai_detected_final = Column(Boolean)
    ddm_confidence_ratings = Column(Text)  # JSON
    conversation_log = Column(Text)  # JSON
    initial_tactic_analysis = Column(Text)
    tactic_selection_log = Column(Text)  # JSON
    ai_researcher_notes = Column(Text)  # JSON
    feels_off_comments = Column(Text)  # JSON
    final_decision_time = Column(Float)
    final_user_comment = Column(Text, nullable=True)
    ui_event_log = Column(Text, nullable=True)  # JSON string of UI events
    consent_accepted = Column(Boolean, default=False)  # Explicit consent flag
    total_study_time_minutes = Column(Float, nullable=True)  # Total time spent in study
    forced_completion = Column(Boolean, default=False)  # Whether study ended due to time limit

    # Confirmatory study phase tracking
    conversation_phase_reached = Column(Boolean, default=False)
    conversation_started_at = Column(DateTime, nullable=True)

    # Legacy DDM-era fields kept for backward compatibility. In the confirmatory
    # study, prefer the explicitly named judgment fields below.
    pure_ddm_decision = Column(Float, nullable=True)  # First 0.0 or 1.0 selection
    pure_ddm_timestamp = Column(DateTime, nullable=True)  # When they made it
    pure_ddm_turn_number = Column(Integer, nullable=True)  # Which message turn
    pure_ddm_decision_time_seconds = Column(Float, nullable=True)  # Time to make pure decision
    first_confidence_slider_endpoint_value_percent = Column(Integer, nullable=True)
    first_confidence_slider_endpoint_choice = Column(String, nullable=True)
    first_confidence_slider_endpoint_timestamp = Column(DateTime, nullable=True)
    first_confidence_slider_endpoint_turn_number = Column(Integer, nullable=True)
    first_confidence_slider_endpoint_decision_time_seconds = Column(Float, nullable=True)

    # Confirmatory per-turn judgment log. This mirrors ddm_confidence_ratings
    # with clearer naming for new exports while keeping old data intact.
    interrogator_turn_judgment_log = Column(Text, nullable=True)  # JSON
    reading_time_seconds = Column(Float, nullable=True)  # Time from AI response to first slider touch
    active_decision_time_seconds = Column(Float, nullable=True)  # Time from first slider touch to submit
    slider_interaction_log = Column(Text, nullable=True)  # JSON of all slider interactions per turn
    # NEW: Session status tracking for incremental saves
    session_status = Column(String, default="active", index=True)  # active, completed, interrupted - indexed for faster queries
    last_updated = Column(DateTime, default=datetime.utcnow, index=True)  # Track when session was last updated - indexed for queries
    recovered_from_restart = Column(Boolean, default=False)  # Flag if session continued after Railway restart
    # NEW: Binary choice tracking
    final_binary_choice = Column(String, nullable=True)  # 'human' or 'ai' - final decision at time expiry
    final_confidence_percent = Column(Integer, nullable=True)  # 0-100 confidence in final choice

    # Confirmatory final responses: interrogator and witness are intentionally
    # separate because they answer different final questions.
    interrogator_final_binary_choice = Column(String, nullable=True)  # 'human' or 'ai'
    interrogator_final_confidence_percent = Column(Integer, nullable=True)
    interrogator_final_decision_time_seconds = Column(Float, nullable=True)
    interrogator_final_response_collected = Column(Boolean, default=False)
    interrogator_final_response_reason = Column(String, nullable=True)
    interrogator_final_response_not_collected_reason = Column(String, nullable=True)

    witness_final_partner_belief = Column(String, nullable=True)  # 'human' or 'ai'
    witness_final_partner_belief_choice_time_ms = Column(Float, nullable=True)
    witness_final_response_collected = Column(Boolean, default=False)
    witness_final_response_reason = Column(String, nullable=True)
    witness_final_response_not_collected_reason = Column(String, nullable=True)

    # Suspicious-behavior summaries. Raw event details remain in ui_event_log.
    suspicious_behavior_event_count = Column(Integer, default=0)
    tab_hidden_count = Column(Integer, default=0)
    total_tab_hidden_ms = Column(Float, default=0)
    window_blur_count = Column(Integer, default=0)
    paste_event_count = Column(Integer, default=0)
    copy_event_count = Column(Integer, default=0)
    context_menu_event_count = Column(Integer, default=0)
    text_selection_event_count = Column(Integer, default=0)
    page_exit_event_count = Column(Integer, default=0)
    pasted_text_log = Column(Text, nullable=True)  # JSON list of pasted text events
    beforeinput_event_count = Column(Integer, default=0)
    beforeinput_paste_event_count = Column(Integer, default=0)
    beforeinput_drop_event_count = Column(Integer, default=0)
    beforeinput_replacement_event_count = Column(Integer, default=0)
    text_growth_anomaly_count = Column(Integer, default=0)
    large_message_after_inactivity_count = Column(Integer, default=0)
    drop_event_count = Column(Integer, default=0)
    textarea_focus_count = Column(Integer, default=0)
    textarea_blur_count = Column(Integer, default=0)
    untrusted_input_event_count = Column(Integer, default=0)
    automation_webdriver_detected = Column(Boolean, default=False)
    automation_fingerprint_event_count = Column(Integer, default=0)
    page_lifecycle_freeze_count = Column(Integer, default=0)
    page_lifecycle_resume_count = Column(Integer, default=0)
    page_lifecycle_pageshow_count = Column(Integer, default=0)
    page_lifecycle_beforeunload_count = Column(Integer, default=0)
    input_provenance_typed_only_message_count = Column(Integer, default=0)
    input_provenance_pasted_message_count = Column(Integer, default=0)
    input_provenance_dropped_message_count = Column(Integer, default=0)
    input_provenance_large_jump_message_count = Column(Integer, default=0)
    input_provenance_mixed_message_count = Column(Integer, default=0)
    input_provenance_unknown_message_count = Column(Integer, default=0)
    max_message_chars_per_second = Column(Float, nullable=True)
    max_message_length_chars = Column(Integer, default=0)
    total_message_keydown_count = Column(Integer, default=0)
    total_message_backspace_delete_count = Column(Integer, default=0)
    total_message_long_pause_count = Column(Integer, default=0)

    has_excessive_delays = Column(Boolean, default=False)  # Flag if session had network delays >40s
    counter_decremented = Column(Boolean, default=False)  # Prevent double-decrement of role counter on dropout
    requeue_count = Column(Integer, default=0)  # How many times re-queued after partner dropped (for analytics)

    # NEW: Timeout tracking - which screen/phase caused timeout (null if completed normally)
    # Values: consent, instructions, demographics, role_assignment, waiting_room,
    #         partner_timeout, witness_final, feedback, debrief, backend_cleanup
    timeout_screen = Column(String, nullable=True)

    # NEW: Human witness mode - Role and matching fields
    role = Column(String, nullable=True)  # "interrogator" or "witness"
    matched_session_id = Column(String, nullable=True, index=True)  # Partner's session_id
    match_status = Column(String, default="unmatched", index=True)  # "unmatched", "waiting", "matched", "partner_dropped", "completed"
    waiting_room_entered_at = Column(DateTime, nullable=True, index=True)  # When entered waiting room (for FIFO)
    matched_at = Column(DateTime, nullable=True)  # When successfully matched with partner
    first_message_sender = Column(String, nullable=True)  # "interrogator" or "witness" (randomly assigned)
    proceed_to_chat_at = Column(DateTime, nullable=True)  # When both can proceed (10s after later entry)
    witness_instructions_version = Column(String, nullable=True)  # Track which instructions shown to witness
    prolific_completion_code = Column(String, nullable=True)  # Which Prolific code was sent: CR0KFVQO, C120SCQ9, C1B54A7Q, C19WFTZR, CZSGWT2I
    study_mode = Column(String, nullable=True, index=True)  # "AI_WITNESS" or "HUMAN_WITNESS" — which condition this session ran under


class DroppedParticipant(Base):
    """
    Track participants who viewed the study but did not consent/complete.
    Only stores minimal data since they didn't consent to full study.
    """
    __tablename__ = "dropped_participants"

    id = Column(Integer, primary_key=True, autoincrement=True)
    participant_id = Column(String, nullable=False, index=True)  # Internal UUID
    prolific_pid = Column(String, nullable=True, index=True)  # Prolific ID (for payment tracking)
    declined_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    reason = Column(String, nullable=False)  # e.g., "consent_disagreed", "timeout", etc.


class RoleAssignmentCounter(Base):
    """
    Atomic counter for balancing interrogator/witness role assignments.
    Single row table - incremented when roles assigned, decremented when consent declined.
    """
    __tablename__ = "role_assignment_counter"

    id = Column(Integer, primary_key=True, default=1)
    interrogator_count = Column(Integer, default=0, nullable=False)
    witness_count = Column(Integer, default=0, nullable=False)


def ensure_study_session_columns():
    """Add nullable columns introduced after table creation.

    SQLAlchemy's create_all() does not migrate existing tables. This lightweight
    additive migration keeps Railway/Postgres and local SQLite usable without
    introducing destructive schema changes.
    """
    try:
        inspector = inspect(engine)
        if "study_sessions" not in inspector.get_table_names():
            return

        existing_columns = {
            column_info["name"]
            for column_info in inspector.get_columns("study_sessions")
        }

        missing_columns = [
            column
            for column in StudySession.__table__.columns
            if column.name not in existing_columns
        ]

        if not missing_columns:
            return

        with engine.begin() as connection:
            for column in missing_columns:
                column_type = column.type.compile(dialect=engine.dialect)
                connection.execute(
                    text(
                        f"ALTER TABLE study_sessions "
                        f"ADD COLUMN {column.name} {column_type}"
                    )
                )
                print(f"Added missing study_sessions column: {column.name}")
    except Exception as e:
        print(f"WARNING: Could not add missing study_sessions columns: {e}")


# Create tables - wrapped in try/except to prevent import failures
try:
    Base.metadata.create_all(bind=engine)
    ensure_study_session_columns()
    print("Database tables created/verified successfully")
except Exception as e:
    print(f"WARNING: Could not create database tables during import: {e}")
    print("Tables will be created on first database access")
