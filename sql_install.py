import sqlite3
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("app_activity.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

DB_NAME = 'cars.db'

def setup_database():
    logger.info(f"Attempting to connect to the database and set up the 'cars' table in: {DB_NAME}")
    try:
        with sqlite3.connect(DB_NAME) as conn:
            c = conn.cursor()
            c.execute('''
                CREATE TABLE IF NOT EXISTS cars (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ad_id TEXT UNIQUE,
                    make TEXT,
                    model TEXT,
                    price INTEGER,
                    year INTEGER,
                    body_type TEXT,
                    fuel TEXT,
                    gearbox TEXT,
                    engine_volume REAL,
                    engine_power INTEGER,
                    mileage INTEGER
                )
            ''')
            conn.commit()
            logger.info(f"Table 'cars' successfully created or already exists in '{DB_NAME}'.")
    except sqlite3.Error as e:
        logger.critical(f"A critical SQLite error occurred while setting up the database table: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.critical(f"An unexpected error occurred during database setup: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    logger.info("Executing sql_install.py as a standalone script.")
    setup_database()
    logger.info("Database setup script finished.")