import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class User(Base):
    """User account details populated via OAuth provider."""

    __tablename__ = "users"

    id = Column(String, primary_key=True)  # Format: "provider|id" (e.g. "hf|username")
    name = Column(String, nullable=False)
    email = Column(String, nullable=True)  # User's email from OAuth provider
    first_name = Column(String, nullable=True)
    last_name = Column(String, nullable=True)
    institution = Column(String, nullable=True)  # Free text institution/organisation
    avatar_url = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    annotations = relationship(
        "Annotation", back_populates="user", cascade="all, delete-orphan"
    )
    saved_papers = relationship(
        "SavedPaper", back_populates="user", cascade="all, delete-orphan"
    )


class PaperRecord(Base):
    """Research papers loaded for classification or user-defined custom entries."""

    __tablename__ = "paper_records"

    # Normalized DOI (lowercase, stripped of 'https://doi.org/') as primary key
    doi = Column(String, primary_key=True, index=True)
    title = Column(String, nullable=False)
    abstract = Column(String, nullable=False)
    initial_intent = Column(String, nullable=True)  # Label predicted by local model
    source = Column(String, default="arxiv")  # 'arxiv', 'openaire', 'custom'
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    annotations = relationship(
        "Annotation", back_populates="paper", cascade="all, delete-orphan"
    )


class Annotation(Base):
    """
    Label correction / flag submitted by a human user OR an LLM judge.

    Uniqueness rules
    ────────────────
    • Human:  one annotation per (paper_doi, user_id)  — user_id IS NOT NULL, llm_model IS NULL
    • LLM:    one annotation per (paper_doi, llm_model) — llm_model IS NOT NULL, user_id IS NULL

    The two partial-unique constraints below enforce this at the DB level.
    SQLAlchemy/Alembic will render them as regular UniqueConstraints; the
    CHECK constraint ensures exactly one of the two identifiers is non-null.
    """

    __tablename__ = "annotations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    paper_doi = Column(
        String, ForeignKey("paper_records.doi", ondelete="CASCADE"), nullable=False
    )

    # Exactly one of these must be non-null (enforced by CHECK constraint below)
    user_id = Column(
        String,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,  # nullable for LLM
    )
    llm_model = Column(String, nullable=True)  # e.g. "Gemma-4-26B-A4B-it-GGUF"

    # Discriminator: 'human' | 'llm'
    annotator_type = Column(String, nullable=False, default="human")

    proposed_label = Column(String, nullable=True)  # "Methodology", "Dataset", etc.
    is_flagged = Column(Boolean, default=False)  # noisy text, formatting issue, etc.
    flag_reason = Column(String, nullable=True)  # "Garbled text", "Language mismatch" …
    comment = Column(String, nullable=True)  # LLM rationale or human critique

    updated_at = Column(
        DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow
    )

    user = relationship("User", back_populates="annotations")
    paper = relationship("PaperRecord", back_populates="annotations")

    __table_args__ = (
        # One human annotation per paper
        UniqueConstraint("paper_doi", "user_id", name="uq_annotation_human"),
        # One LLM annotation per model per paper
        UniqueConstraint("paper_doi", "llm_model", name="uq_annotation_llm"),
        # Exactly one of user_id / llm_model must be set
        CheckConstraint(
            "(user_id IS NOT NULL AND llm_model IS NULL) OR "
            "(user_id IS NULL AND llm_model IS NOT NULL)",
            name="ck_annotation_single_annotator",
        ),
    )


class SavedPaper(Base):
    """Papers saved to history by authenticated users."""

    __tablename__ = "saved_papers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title = Column(String, nullable=False)
    abstract = Column(String, nullable=True)
    label = Column(String, nullable=True)
    probabilities = Column(String, nullable=True)  # JSON-serialised softmax dict
    doi = Column(String, nullable=True)
    link = Column(String, nullable=True)
    lang = Column(String, nullable=True)
    openaire = Column(Boolean, default=False)
    source = Column(String, default="unknown")
    timestamp = Column(Integer, nullable=True)  # epoch milliseconds
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    user = relationship("User", back_populates="saved_papers")

    __table_args__ = (
        UniqueConstraint("user_id", "title", name="uq_saved_paper_user_title"),
    )
