import datetime as dt
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, JSON, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
from .config import DATABASE_URL

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
Session = sessionmaker(bind=engine, expire_on_commit=False)
Base = declarative_base()
now = lambda: dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String(60), unique=True)
    password_hash = Column(String(200))
    role = Column(String(20))                      # employee | hr | finance | admin


class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True)
    name = Column(String(260), index=True)
    ext = Column(String(10))
    allowed_roles = Column(JSON)                   # e.g. ["all"] or ["hr"]
    pages = Column(Integer)
    n_chunks = Column(Integer)
    uploaded_by = Column(String(60))
    created_at = Column(DateTime, default=now)

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


class Chunk(Base):
    __tablename__ = "chunks"
    id = Column(Integer, primary_key=True)
    doc_id = Column(Integer, index=True)
    page = Column(Integer)
    idx = Column(Integer)
    text = Column(Text)


class Message(Base):
    __tablename__ = "messages"                     # conversational memory
    id = Column(Integer, primary_key=True)
    conversation_id = Column(String(40), index=True)
    username = Column(String(60))
    role = Column(String(12))                      # user | assistant
    content = Column(Text)
    created_at = Column(DateTime, default=now)


class QueryLog(Base):
    __tablename__ = "query_logs"                   # observability
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=now, index=True)
    username = Column(String(60))
    user_role = Column(String(20))
    question = Column(Text)
    rewritten = Column(Text)
    latency_ms = Column(Float)
    tokens = Column(Integer)
    attempts = Column(Integer)
    faithfulness = Column(Float)
    n_flagged = Column(Integer)
    abstained = Column(Boolean)
    mode = Column(String(12))
    cited_docs = Column(JSON)
