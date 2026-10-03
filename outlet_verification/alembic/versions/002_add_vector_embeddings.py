"""add vector embeddings to outlets table

Revision ID: 002_add_vector_embeddings
Revises: 001_create_outlets
Create Date: 2026-10-03 09:10:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '002_add_vector_embeddings'
down_revision: Union[str, None] = '001_create_outlets'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()

    # 1. Ensure pgvector extension is enabled
    conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector;"))

    # 2. Add name_embedding (vector(384)) and image_embedding (vector(512))
    # Preserves existing rows with NULL values
    conn.execute(sa.text("""
        ALTER TABLE outlets 
        ADD COLUMN IF NOT EXISTS name_embedding vector(384) NULL;
    """))

    conn.execute(sa.text("""
        ALTER TABLE outlets 
        ADD COLUMN IF NOT EXISTS image_embedding vector(512) NULL;
    """))


def downgrade() -> None:
    conn = op.get_bind()

    # Safely drop embedding columns
    conn.execute(sa.text("ALTER TABLE outlets DROP COLUMN IF EXISTS image_embedding;"))
    conn.execute(sa.text("ALTER TABLE outlets DROP COLUMN IF EXISTS name_embedding;"))
