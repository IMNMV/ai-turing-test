import os
from sqlalchemy import create_engine, Column, String, Text, Float, Boolean, DateTime, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json

DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL or "sqlite:///./test.db")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class StudySession(Base):
    __tablename__ = "study_sessions"
    
    id = Column(String, primary_key=True)
    user_id = Column(String)
    start_time = Column(DateTime, default=datetime.utcnow)
    chosen_persona = Column(String)
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


Base.metadata.create_all(bind=engine)
