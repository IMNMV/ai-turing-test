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


Base.metadata.create_all(bind=engine)
