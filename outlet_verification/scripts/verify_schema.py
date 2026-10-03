import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from sqlalchemy import text
from app.db.session import engine


def verify_schema():
    print("\n========================================================")
    print("      OUTLET DATA SCHEMA VERIFICATION (PHASE 2)")
    print("========================================================\n")

    with engine.connect() as conn:
        # 1. Query table existence
        table_exists = conn.execute(
            text("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'outlets');")
        ).scalar()

        print(f"Table 'outlets' exists: {table_exists}")
        if not table_exists:
            print("ERROR: outlets table does not exist!")
            sys.exit(1)

        # 2. Query columns and datatypes
        print("\n--- Columns in 'outlets' table ---")
        cols = conn.execute(
            text("""
                SELECT 
                    column_name, 
                    data_type, 
                    udt_name, 
                    is_nullable, 
                    column_default
                FROM information_schema.columns 
                WHERE table_name = 'outlets' 
                ORDER BY ordinal_position;
            """)
        ).fetchall()

        for c in cols:
            print(f"  • {c[0]:<15} | type: {c[1]:<25} (udt: {c[2]}) | nullable: {c[3]:<5} | default: {c[4]}")

        # 3. Query constraints
        print("\n--- Constraints on 'outlets' table ---")
        constraints = conn.execute(
            text("""
                SELECT conname, pg_get_constraintdef(oid) 
                FROM pg_constraint 
                WHERE conrelid = 'outlets'::regclass;
            """)
        ).fetchall()
        for con in constraints:
            print(f"  • {con[0]}: {con[1]}")

        # 4. Query indexes
        print("\n--- Indexes on 'outlets' table ---")
        indexes = conn.execute(
            text("""
                SELECT indexname, indexdef 
                FROM pg_indexes 
                WHERE tablename = 'outlets';
            """)
        ).fetchall()
        for idx in indexes:
            print(f"  • {idx[0]}: {idx[1]}")

        # 5. Query triggers
        print("\n--- Triggers on 'outlets' table ---")
        triggers = conn.execute(
            text("""
                SELECT trigger_name, event_manipulation, action_timing 
                FROM information_schema.triggers 
                WHERE event_object_table = 'outlets';
            """)
        ).fetchall()
        for trg in triggers:
            print(f"  • {trg[0]} ({trg[2]} {trg[1]})")

        # 6. Functional test: Insert a sample row to verify location sync & triggers
        print("\n--- Testing Insert & Location Sync Trigger ---")
        insert_res = conn.execute(
            text("""
                INSERT INTO outlets (name, latitude, longitude, image_url)
                VALUES ('Apollo Pharmacy - Indiranagar', 12.9715987, 77.5945627, 'https://example.com/apollo.jpg')
                RETURNING id, name, latitude, longitude, location, created_at, updated_at;
            """)
        ).fetchone()
        conn.commit()

        sample_id = insert_res[0]
        print(f"Inserted sample outlet ID: {sample_id}")
        print(f"Name:                      {insert_res[1]}")
        print(f"Coordinates:               ({insert_res[2]}, {insert_res[3]})")
        print(f"Auto-synced location:      {insert_res[4]}")
        print(f"Created At:                {insert_res[5]}")
        print(f"Updated At:                {insert_res[6]}")

        # Clean up test row
        conn.execute(text("DELETE FROM outlets WHERE id = :id;"), {"id": sample_id})
        conn.commit()
        print("\nTest row deleted cleanly.")

    print("\n--------------------------------------------------------")
    print("      SCHEMA VERIFICATION COMPLETED SUCCESSFULLY!")
    print("--------------------------------------------------------\n")


if __name__ == "__main__":
    verify_schema()
