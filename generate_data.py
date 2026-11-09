# -*- coding: utf-8 -*-
"""
Генерация данных варианта 1 для ЛР №3 по методам оптимизации.
Задача: размещение складов (facility location) с фиксированными затратами
открытия и транспортными затратами.
Минимальный масштаб варианта: 6 кандидатов, 8 клиентов.

Seed: 10000 + 100*Q + N = 10131.
"""

from pathlib import Path
import csv
import json
import numpy as np

SEED = 10131
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"

CANDIDATES = ["Кандидат C1", "Кандидат C2", "Кандидат C3",
              "Кандидат C4", "Кандидат C5", "Кандидат C6"]
CLIENTS = [f"Клиент K{j}" for j in range(1, 9)]
FIXED = np.array([800.0, 900.0, 700.0, 950.0, 850.0, 750.0])
CAPACITY = np.array([120.0, 100.0, 90.0, 110.0, 80.0, 100.0])
DEMAND = np.array([50.0, 45.0, 60.0, 55.0, 40.0, 65.0, 50.0, 45.0])
BASE_COST = np.array([
    [3, 4, 5, 7, 8, 9, 10, 9],
    [8, 7, 6, 3, 4, 6, 8, 9],
    [9, 9, 8, 7, 6, 3, 4, 5],
    [6, 5, 6, 8, 9, 7, 5, 4],
    [7, 6, 7, 9, 8, 5, 4, 3],
    [5, 4, 5, 6, 7, 8, 9, 8],
], dtype=float)


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    transport = np.round(BASE_COST * rng.normal(1.0, 0.06, BASE_COST.shape), 2)
    cand_x = np.round(rng.uniform(0, 3, len(CANDIDATES)), 2)
    cand_y = np.round(np.linspace(1, 7, len(CANDIDATES)) + rng.normal(0, 0.4, len(CANDIDATES)), 2)
    cli_x = np.round(rng.uniform(6, 9, len(CLIENTS)), 2)
    cli_y = np.round(np.linspace(0.5, 7.5, len(CLIENTS)) + rng.normal(0, 0.5, len(CLIENTS)), 2)

    with open(DATA / "candidates.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "candidate", "fixed_cost", "capacity", "x", "y"])
        for i, name in enumerate(CANDIDATES):
            w.writerow([i + 1, name, float(FIXED[i]), float(CAPACITY[i]),
                        float(cand_x[i]), float(cand_y[i])])

    with open(DATA / "clients.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["client_id", "client", "demand", "x", "y"])
        for j, name in enumerate(CLIENTS):
            w.writerow([j + 1, name, float(DEMAND[j]), float(cli_x[j]), float(cli_y[j])])

    with open(DATA / "transport.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["candidate", "client", "cost"])
        for i, cn in enumerate(CANDIDATES):
            for j, kn in enumerate(CLIENTS):
                w.writerow([cn, kn, float(transport[i, j])])

    with open(DATA / "params.json", "w", encoding="utf-8") as f:
        json.dump({
            "seed": SEED,
            "n_candidates": len(CANDIDATES),
            "n_clients": len(CLIENTS),
            "total_capacity": float(CAPACITY.sum()),
            "total_demand": float(DEMAND.sum()),
            "units": {"flow": "ед. груза", "fixed": "тыс. руб.",
                      "transport": "тыс. руб. на ед.", "capacity": "ед.",
                      "demand": "ед."},
            "note": "Синтетический кейс facility location; seed фиксирован.",
        }, f, ensure_ascii=False, indent=2)

    print("Данные сохранены в", DATA)
    print("Транспортные затраты:\n", transport)
    print("Суммарная мощность:", CAPACITY.sum(), "суммарный спрос:", DEMAND.sum())


if __name__ == "__main__":
    main()
