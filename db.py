import sqlite3

DB_NAME = 'cars.db'

def insert_car(car):
    with sqlite3.connect(DB_NAME) as conn:
        cur = conn.cursor()
        cur.execute('''
            INSERT OR IGNORE INTO cars 
            (ad_id, make, model, price, year, body_type, fuel, gearbox, engine_volume, engine_power, mileage)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', car)
        conn.commit()

def get_cars_by_make(make):
    with sqlite3.connect(DB_NAME) as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM cars WHERE make = ?", (make,))
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        return rows, columns

def get_all_car_makes():
    with sqlite3.connect(DB_NAME) as conn:
            cur = conn.cursor()
            cur.execute("SELECT DISTINCT make FROM cars WHERE make IS NOT NULL")
            makes = [row[0] for row in cur.fetchall()]
            return makes