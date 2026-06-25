"""add_saved_papers

Revision ID: 3a7f8c9d1e2b
Revises: 2d6a8f810146
Create Date: 2026-06-22 11:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "3a7f8c9d1e2b"
down_revision: Union[str, Sequence[str], None] = "2d6a8f810146"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "saved_papers",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("abstract", sa.String(), nullable=True),
        sa.Column("label", sa.String(), nullable=True),
        sa.Column("probabilities", sa.String(), nullable=True),
        sa.Column("doi", sa.String(), nullable=True),
        sa.Column("link", sa.String(), nullable=True),
        sa.Column("lang", sa.String(), nullable=True),
        sa.Column("openaire", sa.Boolean(), nullable=True),
        sa.Column("source", sa.String(), nullable=True),
        sa.Column("timestamp", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "title", name="uq_saved_paper_user_title"),
    )


def downgrade() -> None:
    op.drop_table("saved_papers")
