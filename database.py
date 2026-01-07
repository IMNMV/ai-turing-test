import os
from sqlalchemy import create_engine, Column, String, Text, Float, Boolean, DateTime, Integer, Index
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
    social_style = Column(String, nullable=True)  # Social style assigned (WARM, PLAYFUL, DIRECT, GUARDED, CONTRARIAN)
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
    pure_ddm_decision = Column(Float, nullable=True)  # First 0.0 or 1.0 selection
    pure_ddm_timestamp = Column(DateTime, nullable=True)  # When they made it
    pure_ddm_turn_number = Column(Integer, nullable=True)  # Which message turn
    pure_ddm_decision_time_seconds = Column(Float, nullable=True)  # Time to make pure decision
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
    has_excessive_delays = Column(Boolean, default=False)  # Flag if session had network delays >40s


Base.metadata.create_all(bind=engine)
