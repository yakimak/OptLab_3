# -*- coding: utf-8 -*-
"""
Лабораторная работа №3 по дисциплине «Методы оптимизации. Линейное
программирование». Вариант 1.

Задача: размещение складов (facility location): фиксированные затраты
открытия + транспортные затраты (6 кандидатов, 8 клиентов).

Реализовано:
    * MILP-модель с бинарными y_i и непрерывными x_ij;
    * LP-релаксация и интегральный зазор;
    * слабая и сильная big-M формулировки, сравнение силы LP-релаксаций;
    * независимая проверка допустимости и целевой функции;
    * сценарии: рост спроса, отказ кандидата, диагностика несовместимости;
    * жадная эвристика;
    * кривая «время решения / зазор» при масштабировании;
    * визуализация выбранных складов и потоков.

Запуск:
    python generate_data.py
    python main.py
"""

import csv
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog, milp, LinearConstraint, Bounds

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
OUT = BASE / "results"


# ---------------------------------------------------------------------------
# Данные
# ---------------------------------------------------------------------------
def load_data():
    candidates, clients = [], []
    with open(DATA / "candidates.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            candidates.append({
                "id": int(row["candidate_id"]), "name": row["candidate"],
                "fixed": float(row["fixed_cost"]), "cap": float(row["capacity"]),
                "x": float(row["x"]), "y": float(row["y"]),
            })
    with open(DATA / "clients.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            clients.append({
                "id": int(row["client_id"]), "name": row["client"],
                "demand": float(row["demand"]),
                "x": float(row["x"]), "y": float(row["y"]),
            })
    costs = np.zeros((len(candidates), len(clients)))
    with open(DATA / "transport.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            i = [c["name"] for c in candidates].index(row["candidate"])
            j = [c["name"] for c in clients].index(row["client"])
            costs[i, j] = float(row["cost"])
    with open(DATA / "params.json", encoding="utf-8") as f:
        params = json.load(f)
    return candidates, clients, costs, params


def arrays(candidates, clients, costs):
    fixed = np.array([c["fixed"] for c in candidates])
    cap = np.array([c["cap"] for c in candidates])
    demand = np.array([c["demand"] for c in clients])
    return fixed, cap, demand, costs


# ---------------------------------------------------------------------------
# MILP-модель
# ---------------------------------------------------------------------------
def build_model(costs, fixed, cap, demand, strong=True):
    n, m = costs.shape
    N = n * m
    A_ub, b_ub = [], []
    # связка мощности: sum_j x_ij <= cap_i * y_i
    for i in range(n):
        row = np.zeros(N + n)
        row[i * m:(i + 1) * m] = 1.0
        row[N + i] = -cap[i]
        A_ub.append(row)
        b_ub.append(0.0)
    # сильная формулировка: x_ij <= demand_j * y_i
    if strong:
        for i in range(n):
            for j in range(m):
                row = np.zeros(N + n)
                row[i * m + j] = 1.0
                row[N + i] = -demand[j]
                A_ub.append(row)
                b_ub.append(0.0)
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    A_eq, b_eq = [], []
    for j in range(m):
        row = np.zeros(N + n)
        row[j:N:m] = 1.0
        A_eq.append(row)
        b_eq.append(demand[j])
    A_eq = np.array(A_eq)
    b_eq = np.array(b_eq)

    c_obj = np.concatenate([costs.flatten(), fixed])
    integrality = np.concatenate([np.zeros(N), np.ones(n)])
    lb = np.zeros(N + n)
    ub = np.concatenate([np.full(N, np.inf), np.ones(n)])
    return A_ub, b_ub, A_eq, b_eq, c_obj, integrality, lb, ub


def solve_milp_model(costs, fixed, cap, demand, strong=True, availability=None,
                     disp=False):
    A_ub, b_ub, A_eq, b_eq, c_obj, integrality, lb, ub = build_model(
        costs, fixed, cap, demand, strong)
    n = len(fixed)
    N = len(c_obj) - n
    if availability is not None:
        for i in availability:
            ub[N + i] = 0.0
    constraints = [LinearConstraint(A_ub, -np.inf, b_ub),
                   LinearConstraint(A_eq, b_eq, b_eq)]
    args = dict(constraints=constraints, integrality=integrality,
                bounds=Bounds(lb, ub))
    if not disp:
        return milp(c_obj, **args), ""
    old_fd = os.dup(1)
    tmp = tempfile.TemporaryFile(mode="w+b")
    os.dup2(tmp.fileno(), 1)
    try:
        res = milp(c_obj, options={"disp": True}, **args)
    finally:
        sys.stdout.flush()
        os.dup2(old_fd, 1)
        os.close(old_fd)
        tmp.seek(0)
        log = tmp.read().decode("utf-8", errors="replace")
        tmp.close()
    return res, log


def solve_lp_relax(costs, fixed, cap, demand, strong=True, availability=None):
    A_ub, b_ub, A_eq, b_eq, c_obj, _, lb, ub = build_model(
        costs, fixed, cap, demand, strong)
    n = len(fixed)
    N = len(c_obj) - n
    if availability is not None:
        for i in availability:
            ub[N + i] = 0.0
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=list(zip(lb, ub)), method="highs")
    return res


def extract(res, n, m):
    N = n * m
    x = res.x[:N].reshape(n, m)
    y = res.x[N:N + n]
    return x, y


def verify_milp(res, costs, fixed, cap, demand, tol=1e-6):
    n, m = costs.shape
    x, y = extract(res, n, m)
    demand_met = x.sum(axis=0)
    capacity_used = x.sum(axis=1)
    return {
        "x": x, "y": y,
        "max_demand_violation": float(np.max(np.abs(demand_met - demand))),
        "max_capacity_violation": float(np.max(np.maximum(0, capacity_used - cap * y))),
        "max_binary_violation": float(np.max(np.minimum(np.abs(y), np.abs(1 - y)))),
        "objective": float(np.sum(x * costs) + fixed @ y),
        "opened": [i for i in range(n) if y[i] > 0.5],
    }


# ---------------------------------------------------------------------------
# Диагностика несовместимости
# ---------------------------------------------------------------------------
def min_unmet(costs, fixed, cap, demand, availability=None):
    n, m = costs.shape
    N = n * m
    # переменные: x (N), y (n), u (m)
    A_ub, b_ub = [], []
    for i in range(n):
        row = np.zeros(N + n + m)
        row[i * m:(i + 1) * m] = 1.0
        row[N + i] = -cap[i]
        A_ub.append(row)
        b_ub.append(0.0)
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    A_eq, b_eq = [], []
    for j in range(m):
        row = np.zeros(N + n + m)
        row[j:N:m] = 1.0
        row[N + n + j] = 1.0
        A_eq.append(row)
        b_eq.append(demand[j])
    A_eq = np.array(A_eq)
    b_eq = np.array(b_eq)
    c_obj = np.concatenate([np.zeros(N + n), np.ones(m)])
    integrality = np.concatenate([np.zeros(N), np.ones(n), np.zeros(m)])
    lb = np.zeros(N + n + m)
    ub = np.concatenate([np.full(N, np.inf), np.ones(n), np.full(m, np.inf)])
    if availability is not None:
        for i in availability:
            ub[N + i] = 0.0
    res = milp(c_obj, constraints=[LinearConstraint(A_ub, -np.inf, b_ub),
                                   LinearConstraint(A_eq, b_eq, b_eq)],
               integrality=integrality, bounds=Bounds(lb, ub))
    if not res.success:
        return {"status": int(res.status), "message": res.message}
    u = res.x[N + n:]
    return {"status": int(res.status), "message": res.message,
            "unmet": u, "total_unmet": float(u.sum()),
            "x": res.x[:N].reshape(n, m), "y": res.x[N:N + n]}


# ---------------------------------------------------------------------------
# Жадная эвристика
# ---------------------------------------------------------------------------
def greedy(costs, fixed, cap, demand):
    n, m = costs.shape
    score = fixed / cap + costs.mean(axis=1)
    order = np.argsort(score)
    opened, total = [], 0.0
    for i in order:
        opened.append(int(i))
        total += cap[i]
        if total >= demand.sum():
            break
    remaining = {i: cap[i] for i in opened}
    x = np.zeros((n, m))
    unmet = 0.0
    for j in np.argsort(-demand):
        need = demand[j]
        for i in sorted(opened, key=lambda k: costs[k, j]):
            if need <= 1e-9:
                break
            if remaining[i] <= 0:
                continue
            amount = min(need, remaining[i])
            x[i, j] += amount
            remaining[i] -= amount
            need -= amount
        unmet += max(0.0, need)
    obj = float(np.sum(x * costs) + fixed[opened].sum())
    return {"x": x, "opened": opened, "cost": obj, "unmet": float(unmet)}


# ---------------------------------------------------------------------------
# Масштабирование
# ---------------------------------------------------------------------------
def random_instance(n_candidates, n_clients, seed):
    rng = np.random.default_rng(seed)
    fixed = rng.uniform(500, 1000, n_candidates)
    cap = rng.uniform(80, 150, n_candidates)
    demand = rng.uniform(30, 70, n_clients)
    while cap.sum() < 1.25 * demand.sum():
        cap *= 1.15
    costs = rng.uniform(2, 12, (n_candidates, n_clients))
    return costs, fixed, cap, demand


def scaling_experiment():
    cases = [(4, 10), (6, 20), (8, 40)]
    rows = []
    for k, (n, m) in enumerate(cases):
        costs, fixed, cap, demand = random_instance(n, m, 20000 + k)
        t0 = time.perf_counter()
        lp = solve_lp_relax(costs, fixed, cap, demand, strong=True)
        t_lp = time.perf_counter() - t0
        t0 = time.perf_counter()
        mi, _ = solve_milp_model(costs, fixed, cap, demand, strong=True)
        t_milp = time.perf_counter() - t0
        if mi.success and lp.success:
            gap = (mi.fun - lp.fun) / abs(mi.fun) * 100
        else:
            gap = np.nan
        rows.append({"candidates": n, "clients": m,
                     "lp_obj": float(lp.fun) if lp.success else None,
                     "milp_obj": float(mi.fun) if mi.success else None,
                     "lp_time": t_lp, "milp_time": t_milp,
                     "gap": float(gap) if np.isfinite(gap) else None})
    return rows


# ---------------------------------------------------------------------------
# Визуализация
# ---------------------------------------------------------------------------
def plot_map(candidates, clients, x, y, path):
    fig, ax = plt.subplots(figsize=(11, 7))
    for j, cl in enumerate(clients):
        ax.scatter(cl["x"], cl["y"], s=140, marker="s", color="#2E86C1",
                   edgecolor="black", zorder=3)
        ax.text(cl["x"] + 0.15, cl["y"], cl["name"], fontsize=8.5)
    for i, c in enumerate(candidates):
        opened = y[i] > 0.5
        ax.scatter(c["x"], c["y"], s=260, marker="o",
                   color="#27AE60" if opened else "#D5D8DC",
                   edgecolor="black", zorder=4)
        ax.text(c["x"] - 0.1, c["y"] + 0.25, c["name"], fontsize=9,
                fontweight="bold" if opened else "normal")
        if opened:
            for j, cl in enumerate(clients):
                if x[i, j] > 1e-6:
                    ax.plot([c["x"], cl["x"]], [c["y"], cl["y"]],
                            color="#7F8C8D", lw=0.8 + 3.0 * x[i, j] / max(1, cl["demand"]),
                            alpha=0.65, zorder=2)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Размещение складов: зелёные — открыты, серые — закрыты")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_open(candidates, y, cap, path):
    n = len(candidates)
    colors = ["#27AE60" if y[i] > 0.5 else "#D5D8DC" for i in range(n)]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar([c["name"] for c in candidates], [c["fixed"] for c in candidates],
                  color=colors, edgecolor="white")
    for bar, c, yi in zip(bars, candidates, y):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
                f"{'открыт' if yi > 0.5 else 'закрыт'}\ncap={c['cap']:.0f}",
                ha="center", fontsize=8.5)
    ax.set_ylabel("Фиксированные затраты, тыс. руб.")
    ax.set_title("Выбор складов")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_formulations(form_rows, path):
    labels = ["weak LP", "strong LP", "MILP"]
    vals = [form_rows["weak_lp"], form_rows["strong_lp"], form_rows["milp"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, vals, color=["#95A5A6", "#2F5D8A", "#27AE60"], alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v * 0.99, f"{v:.1f}",
                ha="center", va="top", color="white", fontsize=9)
    ax.set_ylabel("Целевая функция")
    ax.set_title("LP-релаксации и MILP: сила формулировок")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_scenarios(records, path):
    keys = ["base", "deficit", "stress", "recovery"]
    titles = ["базовый", "спрос +20%", "стресс: C1 нет", "восстановление"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    costs = [records[k]["cost"] if records[k]["cost"] is not None else 0 for k in keys]
    bars = axes[0].bar(titles, costs, color=["#2F5D8A", "#F39C12", "#C0392B", "#27AE60"],
                       alpha=0.85)
    top = max(costs + [1])
    for bar, k in zip(bars, keys):
        rec = records[k]
        if rec["cost"] is None:
            bar.set_height(0.04 * top)
            axes[0].text(bar.get_x() + bar.get_width() / 2, 0.05 * top, "нет",
                         rotation=90, ha="center", va="bottom", color="#C0392B",
                         fontsize=9)
        else:
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 0.5,
                         f"{rec['cost']:.0f}", ha="center", va="center",
                         color="white", fontsize=9)
    axes[0].set_ylabel("Затраты, тыс. руб.")
    axes[0].set_title("Сценарии")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].grid(alpha=0.3, axis="y")

    counts = [len(records[k]["opened"]) if records[k]["opened"] else 0 for k in keys]
    axes[1].bar(titles, counts, color=["#2F5D8A", "#F39C12", "#C0392B", "#27AE60"],
                alpha=0.85)
    for i, v in enumerate(counts):
        axes[1].text(i, v + 0.1, str(v), ha="center", fontsize=9)
    axes[1].set_ylabel("Число открытых складов")
    axes[1].set_title("Состав выбранных складов")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_scaling(rows, path):
    clients = [r["clients"] for r in rows]
    times = [r["milp_time"] for r in rows]
    gaps = [r["gap"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(clients, times, "o-", color="#2F5D8A", lw=2, label="время MILP")
    ax1.set_xlabel("Число клиентов")
    ax1.set_ylabel("Время MILP, с", color="#2F5D8A")
    ax1.tick_params(axis="y", labelcolor="#2F5D8A")
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(clients, gaps, "s--", color="#C0392B", lw=2, label="gap, %")
    ax2.set_ylabel("Интегральный зазор, %", color="#C0392B")
    ax2.tick_params(axis="y", labelcolor="#C0392B")
    plt.title("Масштабирование: время решения и зазор")
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_dashboard(candidates, clients, x, y, records, form_rows, scaling,
                   output, path):
    from matplotlib.gridspec import GridSpec
    import matplotlib.image as mpimg

    tmp = OUT / "_map_tmp.png"
    plot_map(candidates, clients, x, y, tmp)
    img = mpimg.imread(tmp)

    fig = plt.figure(figsize=(17, 11))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.28)
    fig.suptitle("ЛР №3 · Размещение складов: fixed-charge MILP (вариант 1)",
                 fontsize=16, fontweight="bold", color="#12263A", y=0.98)

    ax1 = fig.add_subplot(gs[0, :2])
    ax1.imshow(img)
    ax1.axis("off")
    ax1.set_title("Выбранные склады и потоки")

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis("off")
    lines = ["Сводка", "",
             f"MILP: {output['cost']:.1f} тыс. руб.",
             f"LP weak: {form_rows['weak_lp']:.1f}",
             f"LP strong: {form_rows['strong_lp']:.1f}",
             f"Gap weak: {form_rows['weak_gap']:.2f}%",
             f"Gap strong: {form_rows['strong_gap']:.2f}%", "",
             "Открыты: " + ", ".join(candidates[i]["name"] for i in output["opened"])]
    ax2.text(0.02, 0.98, "\n".join(lines), ha="left", va="top", fontsize=9.5,
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#EAF2FB", edgecolor="#2F5D8A"))

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.bar([c["name"] for c in candidates], [c["fixed"] for c in candidates],
            color=["#27AE60" if y[i] > 0.5 else "#D5D8DC" for i in range(len(y))])
    ax3.set_title("Фиксированные затраты")
    ax3.tick_params(axis="x", rotation=15)
    ax3.grid(alpha=0.3, axis="y")

    ax4 = fig.add_subplot(gs[1, 1])
    keys = ["base", "deficit", "stress", "recovery"]
    vals = [records[k]["cost"] if records[k]["cost"] is not None else 0 for k in keys]
    ax4.bar(["base", "+20%", "stress", "recover"], vals,
            color=["#2F5D8A", "#F39C12", "#C0392B", "#27AE60"], alpha=0.85)
    for i, v in enumerate(vals):
        ax4.text(i, v + 0.01 * max(vals + [1]), f"{v:.0f}", ha="center", fontsize=8.5)
    ax4.set_title("Сценарии")
    ax4.grid(alpha=0.3, axis="y")

    ax5 = fig.add_subplot(gs[1, 2])
    clients_n = [r["clients"] for r in scaling]
    times = [r["milp_time"] for r in scaling]
    gaps = [r["gap"] for r in scaling]
    ax5.plot(clients_n, times, "o-", color="#2F5D8A", label="time")
    ax5.set_xlabel("Клиентов")
    ax5.set_ylabel("Время, с", color="#2F5D8A")
    ax5b = ax5.twinx()
    ax5b.plot(clients_n, gaps, "s--", color="#C0392B", label="gap")
    ax5b.set_ylabel("Gap, %", color="#C0392B")
    ax5.set_title("Масштабирование")

    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Основной сценарий
# ---------------------------------------------------------------------------
def save_verification(path: str, candidates, clients, ver, solver_obj, cap, demand):
    rows = []
    x, y = ver["x"], ver["y"]
    for j, cl in enumerate(clients):
        delivered = float(x[:, j].sum())
        rows.append(["demand", cl["name"], round(delivered, 6),
                     round(float(demand[j]), 6),
                     round(abs(delivered - demand[j]), 9),
                     "OK" if abs(delivered - demand[j]) <= 1e-6 else "FAIL"])
    for i, cand in enumerate(candidates):
        used = float(x[i].sum())
        limit = float(cap[i] * y[i])
        rows.append(["capacity", cand["name"], round(used, 6), round(limit, 6),
                     round(float(max(0.0, used - limit)), 9),
                     "OK" if used <= limit + 1e-6 else "FAIL"])
        rows.append(["binary", cand["name"], round(float(y[i]), 6), 1.0,
                     round(float(min(abs(y[i]), abs(1.0 - y[i]))), 9),
                     "OK" if min(abs(y[i]), abs(1.0 - y[i])) <= 1e-6 else "FAIL"])
    delta = abs(float(ver["objective"]) - float(solver_obj))
    rows.append(["objective", "recomputed_vs_solver",
                 round(float(ver["objective"]), 6), round(float(solver_obj), 6),
                 round(delta, 9), "OK" if delta <= 1e-6 else "FAIL"])
    overall = max(ver["max_demand_violation"], ver["max_capacity_violation"],
                  ver["max_binary_violation"])
    rows.append(["overall", "max_violation", round(float(overall), 9), 1e-6,
                 round(float(overall), 9), "OK" if overall <= 1e-6 else "FAIL"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["check", "entity", "value", "limit", "violation", "status"])
        w.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    candidates, clients, costs, params = load_data()
    fixed, cap, demand, costs = arrays(candidates, clients, costs)
    n, m = costs.shape

    res, log = solve_milp_model(costs, fixed, cap, demand, strong=True, disp=True)
    if not res.success:
        raise SystemExit(f"Базовая задача не решена: {res.message}")
    ver = verify_milp(res, costs, fixed, cap, demand)
    save_verification(OUT / "verification.csv", candidates, clients, ver,
                      float(res.fun), cap, demand)
    x, y = ver["x"], ver["y"]

    # LP-релаксации
    lp_weak = solve_lp_relax(costs, fixed, cap, demand, strong=False)
    lp_strong = solve_lp_relax(costs, fixed, cap, demand, strong=True)
    form_rows = {
        "weak_lp": float(lp_weak.fun), "strong_lp": float(lp_strong.fun),
        "milp": float(res.fun),
        "weak_gap": (float(res.fun) - float(lp_weak.fun)) / float(res.fun) * 100,
        "strong_gap": (float(res.fun) - float(lp_strong.fun)) / float(res.fun) * 100,
    }

    # Сценарии
    records = {}
    records["base"] = {"status": int(res.status), "cost": float(res.fun),
                       "opened": ver["opened"], "x": x, "y": y}

    res_def, _ = solve_milp_model(costs, fixed, cap, demand * 1.20, strong=True)
    if res_def.success:
        vd = verify_milp(res_def, costs, fixed, cap, demand * 1.20)
        records["deficit"] = {"status": int(res_def.status), "cost": float(res_def.fun),
                              "opened": vd["opened"], "x": vd["x"], "y": vd["y"]}
    else:
        records["deficit"] = {"status": int(res_def.status), "cost": None,
                              "opened": [], "x": None, "y": None}

    # Стресс: спрос +30%, кандидат C1 недоступен.
    stress_demand = demand * 1.30
    res_str, _ = solve_milp_model(costs, fixed, cap, stress_demand, strong=True,
                                  availability=[0])
    if res_str.success:
        vs = verify_milp(res_str, costs, fixed, cap, stress_demand)
        records["stress"] = {"status": int(res_str.status), "cost": float(res_str.fun),
                             "opened": vs["opened"], "x": vs["x"], "y": vs["y"]}
    else:
        records["stress"] = {"status": int(res_str.status), "cost": None,
                             "opened": [], "x": None, "y": None}
    stress_unmet = min_unmet(costs, fixed, cap, stress_demand, availability=[0])

    # Восстановление: C1 недоступен, но спрос базовый.
    res_rec, _ = solve_milp_model(costs, fixed, cap, demand, strong=True,
                                  availability=[0])
    if res_rec.success:
        vr = verify_milp(res_rec, costs, fixed, cap, demand)
        records["recovery"] = {"status": int(res_rec.status), "cost": float(res_rec.fun),
                               "opened": vr["opened"], "x": vr["x"], "y": vr["y"]}
    else:
        records["recovery"] = {"status": int(res_rec.status), "cost": None,
                               "opened": [], "x": None, "y": None}

    greedy_res = greedy(costs, fixed, cap, demand)
    scaling = scaling_experiment()

    # ---------------- CSV ----------------
    with open(OUT / "lab3_solution.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["candidate", "open", "fixed_cost", "capacity", "used"] +
                   [c["name"] for c in clients])
        for i, c in enumerate(candidates):
            w.writerow([c["name"], int(y[i] > 0.5), c["fixed"], c["cap"],
                        round(float(x[i].sum()), 3)] +
                       [round(float(v), 2) for v in x[i]])
        w.writerow(["ИТОГО", "", round(float(fixed @ y), 2), "",
                    round(float(x.sum()), 2), "", "", "", "", "", "", "", ""])

    with open(OUT / "lab3_formulations.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["formulation", "lp_objective", "milp_objective", "gap_percent"])
        w.writerow(["weak", round(form_rows["weak_lp"], 3), round(form_rows["milp"], 3),
                    round(form_rows["weak_gap"], 3)])
        w.writerow(["strong", round(form_rows["strong_lp"], 3), round(form_rows["milp"], 3),
                    round(form_rows["strong_gap"], 3)])

    with open(OUT / "lab3_scenarios.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "status", "cost", "opened", "unmet_demand"])
        for key, rec in records.items():
            unmet = stress_unmet.get("total_unmet", 0) if key == "stress" else 0
            w.writerow([key, rec["status"], rec["cost"],
                        "; ".join(candidates[i]["name"] for i in rec["opened"]),
                        round(float(unmet), 3)])

    with open(OUT / "lab3_greedy.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "cost", "opened", "unmet", "gap_percent"])
        w.writerow(["MILP", round(float(res.fun), 3),
                    "; ".join(candidates[i]["name"] for i in ver["opened"]), 0.0, 0.0])
        w.writerow(["greedy", round(greedy_res["cost"], 3),
                    "; ".join(candidates[i]["name"] for i in greedy_res["opened"]),
                    round(greedy_res["unmet"], 3),
                    round((greedy_res["cost"] - float(res.fun)) / float(res.fun) * 100, 3)])

    with open(OUT / "lab3_scaling.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["candidates", "clients", "lp_objective", "milp_objective",
                    "lp_time_sec", "milp_time_sec", "gap_percent"])
        for r in scaling:
            w.writerow([r["candidates"], r["clients"],
                        round(r["lp_obj"], 3) if r["lp_obj"] is not None else "",
                        round(r["milp_obj"], 3) if r["milp_obj"] is not None else "",
                        round(r["lp_time"], 6), round(r["milp_time"], 6),
                        round(r["gap"], 3) if r["gap"] is not None else ""])

    summary = {
        "variant": 1,
        "student": "Якимов Максим Андреевич, Б24-517, № 31",
        "n_candidates": n, "n_clients": m,
        "total_capacity": float(cap.sum()), "total_demand": float(demand.sum()),
        "milp_cost": float(res.fun),
        "status": res.message,
        "opened": [candidates[i]["name"] for i in ver["opened"]],
        "max_demand_violation": ver["max_demand_violation"],
        "max_capacity_violation": ver["max_capacity_violation"],
        "max_binary_violation": ver["max_binary_violation"],
        "formulations": form_rows,
        "stress_unmet": stress_unmet.get("total_unmet"),
        "greedy_cost": greedy_res["cost"],
        "greedy_gap_percent": (greedy_res["cost"] - float(res.fun)) / float(res.fun) * 100,
        "scaling": scaling,
        "seed": params["seed"],
    }
    with open(OUT / "lab3_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)

    with open(OUT / "solver.log", "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\nБАЗОВАЯ MILP-ЗАДАЧА\n" + "=" * 70 + "\n")
        f.write(log)
        f.write("\nLP-релаксация (weak): " + str(lp_weak.fun) + "\n")
        f.write("LP-релаксация (strong): " + str(lp_strong.fun) + "\n")
        f.write("MILP: " + str(res.fun) + "\n")

    # ---------------- Графики ----------------
    plot_map(candidates, clients, x, y, OUT / "lab3_map.png")
    plot_open(candidates, y, cap, OUT / "lab3_open.png")
    plot_formulations(form_rows, OUT / "lab3_formulations.png")
    plot_scenarios(records, OUT / "lab3_scenarios.png")
    plot_scaling(scaling, OUT / "lab3_scaling.png")
    output = {"cost": float(res.fun), "opened": ver["opened"]}
    plot_dashboard(candidates, clients, x, y, records, form_rows, scaling,
                   output, OUT / "lab3_dashboard.png")

    # ---------------- Консоль ----------------
    print("=" * 65)
    print("ЛР №3. Вариант 1: размещение складов (facility location)")
    print("=" * 65)
    print(f"Статус: {res.message}")
    print(f"MILP-стоимость: {res.fun:.2f} тыс. руб.")
    print(f"Открыты: {', '.join(candidates[i]['name'] for i in ver['opened'])}")
    print(f"Проверка: спрос {ver['max_demand_violation']:.2e}, "
          f"мощность {ver['max_capacity_violation']:.2e}, "
          f"бинарность {ver['max_binary_violation']:.2e}")
    print("\nФормулировки:")
    print(f"  weak LP:   {form_rows['weak_lp']:.2f}, gap {form_rows['weak_gap']:.2f}%")
    print(f"  strong LP: {form_rows['strong_lp']:.2f}, gap {form_rows['strong_gap']:.2f}%")
    print("\nСценарии:")
    for key, rec in records.items():
        cost = f"{rec['cost']:.0f}" if rec["cost"] is not None else "—"
        print(f"  {key:<10} cost={cost}, открыто={len(rec['opened'])}")
    if stress_unmet.get("status") == 0:
        print(f"  стресс: минимальный дефицит спроса = {stress_unmet['total_unmet']:.1f}")
    print(f"\nЖадная эвристика: {greedy_res['cost']:.2f} "
          f"(зазор {(greedy_res['cost'] - res.fun) / res.fun * 100:.2f}%)")
    print("\nМасштабирование:")
    for r in scaling:
        print(f"  {r['candidates']}x{r['clients']}: MILP {r['milp_time']:.3f} с, "
              f"gap {r['gap']:.2f}%")
    print(f"\nФайлы сохранены в {OUT}")


if __name__ == "__main__":
    main()
