import os
import sqlite3

from src.common import config


conn = sqlite3.connect(config.DB_PATH)
cursor = conn.cursor()

fonts_dir = os.path.join("data", "fonts")
csv_path = os.path.join(fonts_dir, "dafonts-free-v1", "info.csv")
with open(csv_path, "r") as f:
    lines = f.read().splitlines()

n = len(lines[0].split(","))
cursor.executemany(f"""
        INSERT OR IGNORE INTO fonts ({lines[0]})
        VALUES ({','.join(["?"]*n)})
    """, (vals for vals in (line.split(",") for line in lines[1:]) if len(vals)==6)
)

conn.commit()
conn.close()
