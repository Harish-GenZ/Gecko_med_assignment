import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.core.logging import setup_logging
from app.db.init_db import inspect_and_initialize_db

logger = setup_logging(debug=True)


def main():
    print("\n========================================================")
    print(" OUTLET VERIFICATION - RAILWAY POSTGRESQL & PGVECTOR CHECK")
    print("========================================================\n")

    result = inspect_and_initialize_db()

    print("\n------------------- VERIFICATION REPORT -------------------")
    print(f"Target Database URL: {result['target_url']}")
    print(f"Connected:           {'SUCCESS' if result['connected'] else 'FAILED'}")

    if result["connected"]:
        print(f"PostgreSQL Version:  {result['postgres_version']}")
        print(f"pgvector Available:  {'YES' if result['pgvector_available'] else 'NO'} "
              f"(default: {result['pgvector_available_version']})")
        print(f"pgvector Enabled:    {'YES' if result['pgvector_enabled'] else 'NO'} "
              f"(installed: {result['pgvector_installed_version']})")
    else:
        print(f"Error Details:       {result['error']}")

    print("-----------------------------------------------------------\n")

    if not result["connected"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
