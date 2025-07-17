#!/usr/bin/env python3
# Created by Sean L. on Jul. 16.
# Last Updated by Sean L. on Jul. 16.
# 
# TeaML
# embedding/load_training.py
# 
# Makabaka1880, 2025. All rights reserved.

import os
import sqlite3
import csv
import sys

def load_csv_to_db(csv_path: str, db_path: str = "main.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    with open(csv_path, newline='') as csvfile:
        reader = list(csv.reader(csvfile))
        tea_names = reader[0][1:]     # skip step_idx
        label_values = reader[1][1:]  # skip label

        procedure_ids = []
        for tea, label in zip(tea_names, label_values):
            phenol, caffeine, price = map(float, label.split())
            cursor.execute(
                "INSERT INTO procedures (tea_name, label_phenol, label_caffeine, label_price) VALUES (?, ?, ?, ?)",
                (tea, phenol, caffeine, price)
            )
            procedure_ids.append(cursor.lastrowid)

        for row in reader[2:]:
            step_idx = int(row[0])
            for i, step in enumerate(row[1:]):
                if not step.strip():
                    continue
                parts = step.strip().split()
                op = parts[0]
                p1 = float(parts[1]) if len(parts) > 1 else 0.0
                p2 = float(parts[2]) if len(parts) > 2 else 0.0
                p3 = float(parts[3]) if len(parts) > 3 else 0.0
                cursor.execute(
                    "INSERT INTO steps (procedure_id, step_idx, operation, param1, param2, param3) VALUES (?, ?, ?, ?, ?, ?)",
                    (procedure_ids[i], step_idx, op, p1, p2, p3)
                )

    conn.commit()
    conn.close()
    print(f"✅ Loaded {csv_path} into {db_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: load_procedures.py <working_dir> <csv_file>")
        sys.exit(1)

    working_dir = sys.argv[1]
    csv_file = sys.argv[2]

    os.chdir(working_dir)
    load_csv_to_db(csv_file)