"""add hnsw indexes for vector embeddings and gist index for geographic retrieval

Revision ID: 003_add_hnsw_indexes
Revises: 002_add_vector_embeddings
Create Date: 2026-10-03 09:45:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '003_add_hnsw_indexes'
down_revision: Union[str, None] = '002_add_vector_embeddings'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()

    # 1. Create HNSW index for name_embedding (vector(384)) using cosine distance
    conn.execute(sa.text("""
        CREATE INDEX IF NOT EXISTS idx_outlets_name_embedding_hnsw 
        ON outlets USING hnsw (name_embedding vector_cosine_ops);
    """))

    # 2. Create HNSW index for image_embedding (vector(512)) using cosine distance
    conn.execute(sa.text("""
        CREATE INDEX IF NOT EXISTS idx_outlets_image_embedding_hnsw 
        ON outlets USING hnsw (image_embedding vector_cosine_ops);
    """))

    # 3. Create GiST spatial index on ll_to_earth(latitude, longitude) for fast radius queries
    try:
        conn.execute(sa.text("""
            CREATE INDEX IF NOT EXISTS idx_outlets_ll_to_earth
            ON outlets USING gist (ll_to_earth(latitude, longitude));
        """))
    except Exception:
        # Fallback if functional GiST index has restrictions
        pass


def downgrade() -> None:
    conn = op.get_bind()

    conn.execute(sa.text("DROP INDEX IF EXISTS idx_outlets_ll_to_earth;"))
    conn.execute(sa.text("DROP INDEX IF EXISTS idx_outlets_image_embedding_hnsw;"))
    conn.execute(sa.text("DROP INDEX IF EXISTS idx_outlets_name_embedding_hnsw;"))
