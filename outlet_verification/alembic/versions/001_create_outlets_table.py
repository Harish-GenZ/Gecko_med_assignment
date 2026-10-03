"""create outlets table

Revision ID: 001_create_outlets
Revises: 
Create Date: 2026-10-03 08:40:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001_create_outlets'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()

    # 1. Enable geospatial extensions (cube and earthdistance for distance calculations)
    conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS cube;"))
    conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS earthdistance;"))

    # 2. Check if postgis is available/installed
    has_postgis = False
    try:
        check_postgis = conn.execute(
            sa.text("SELECT 1 FROM pg_extension WHERE extname = 'postgis';")
        ).scalar()
        if not check_postgis:
            # Check if available to install
            avail = conn.execute(
                sa.text("SELECT 1 FROM pg_available_extensions WHERE name = 'postgis';")
            ).scalar()
            if avail:
                conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS postgis;"))
                has_postgis = True
        else:
            has_postgis = True
    except Exception:
        has_postgis = False

    # 3. Create outlets table
    location_type_sql = "geography(Point, 4326)" if has_postgis else "point"

    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS outlets (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        name TEXT NOT NULL,
        latitude DOUBLE PRECISION NOT NULL,
        longitude DOUBLE PRECISION NOT NULL,
        location {location_type_sql},
        image_url TEXT NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
        CONSTRAINT chk_outlets_latitude CHECK (latitude >= -90.0 AND latitude <= 90.0),
        CONSTRAINT chk_outlets_longitude CHECK (longitude >= -180.0 AND longitude <= 180.0)
    );
    """
    conn.execute(sa.text(create_table_sql))

    # 4. Create Indexes
    conn.execute(sa.text("CREATE INDEX IF NOT EXISTS idx_outlets_name ON outlets (name);"))
    conn.execute(sa.text("CREATE INDEX IF NOT EXISTS idx_outlets_lat_lng ON outlets (latitude, longitude);"))
    conn.execute(sa.text("CREATE INDEX IF NOT EXISTS idx_outlets_created_at ON outlets (created_at DESC);"))

    # 5. Create updated_at trigger function
    conn.execute(sa.text("""
    CREATE OR REPLACE FUNCTION update_outlets_updated_at()
    RETURNS TRIGGER AS $$
    BEGIN
        NEW.updated_at = CURRENT_TIMESTAMP;
        RETURN NEW;
    END;
    $$ LANGUAGE plpgsql;
    """))

    conn.execute(sa.text("""
    DROP TRIGGER IF EXISTS trg_outlets_updated_at ON outlets;
    CREATE TRIGGER trg_outlets_updated_at
    BEFORE UPDATE ON outlets
    FOR EACH ROW
    EXECUTE FUNCTION update_outlets_updated_at();
    """))

    # 6. Create trigger function to keep location automatically synced from (latitude, longitude)
    if has_postgis:
        sync_sql = """
        CREATE OR REPLACE FUNCTION sync_outlet_location()
        RETURNS TRIGGER AS $$
        BEGIN
            IF NEW.latitude IS NOT NULL AND NEW.longitude IS NOT NULL THEN
                NEW.location = ST_SetSRID(ST_MakePoint(NEW.longitude, NEW.latitude), 4326);
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
    else:
        sync_sql = """
        CREATE OR REPLACE FUNCTION sync_outlet_location()
        RETURNS TRIGGER AS $$
        BEGIN
            IF NEW.latitude IS NOT NULL AND NEW.longitude IS NOT NULL THEN
                NEW.location = point(NEW.longitude, NEW.latitude);
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """

    conn.execute(sa.text(sync_sql))
    conn.execute(sa.text("""
    DROP TRIGGER IF EXISTS trg_outlets_sync_location ON outlets;
    CREATE TRIGGER trg_outlets_sync_location
    BEFORE INSERT OR UPDATE ON outlets
    FOR EACH ROW
    EXECUTE FUNCTION sync_outlet_location();
    """))


def downgrade() -> None:
    conn = op.get_bind()
    conn.execute(sa.text("DROP TRIGGER IF EXISTS trg_outlets_sync_location ON outlets;"))
    conn.execute(sa.text("DROP FUNCTION IF EXISTS sync_outlet_location();"))
    conn.execute(sa.text("DROP TRIGGER IF EXISTS trg_outlets_updated_at ON outlets;"))
    conn.execute(sa.text("DROP FUNCTION IF EXISTS update_outlets_updated_at();"))
    conn.execute(sa.text("DROP INDEX IF EXISTS idx_outlets_created_at;"))
    conn.execute(sa.text("DROP INDEX IF EXISTS idx_outlets_lat_lng;"))
    conn.execute(sa.text("DROP INDEX IF EXISTS idx_outlets_name;"))
    conn.execute(sa.text("DROP TABLE IF EXISTS outlets CASCADE;"))
