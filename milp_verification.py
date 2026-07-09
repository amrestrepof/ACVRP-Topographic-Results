"""
MILP verification for the ACVRP-TC formulation (Eqs. 4-11 of the manuscript).
=============================================================================
Reviewer 2, Comment 5 (Adjustment 17): solve the complete MILP exactly on
small subsampled Seoul instances, verify that all constraints are
simultaneously satisfiable and non-contradictory, check strong connectivity
of the pruned graph, and compare the exact optima against H-MA-LNS.

Data: real SLAS100 instance (Lee & Chae, 2021) + SRTM 1 Arc-Second DEM
(n37_e126 / n37_e127, same mosaics as the public repository).

The GCF cost matrix is built EXACTLY as in Eqs. (1)-(3) of the manuscript:
    theta_ij = (z_j - z_i) / d_ij            (Eq. 1, d_ij = real road distance)
    Omega    = 1 + alpha*theta   (ascent)    (Eq. 3, alpha = 10)
             = max(kappa, 1 + beta*theta)    (descent, beta = 5, kappa = 0.6)
    c_ij     = d_ij * Omega(theta_ij)        (Eq. 2)
    arcs with |theta_ij| > theta_max = 25% are infeasible (Eq. 10).
The same matrix (exported to CSV) is fed to both the exact solver and the
H-MA-LNS, so the comparison is on identical data.

Requirements:  pip install pulp rasterio numpy pandas networkx
Solver: CBC (bundled with PuLP).  For Gurobi (academic licence) replace the
solver line marked  # <-- GUROBI  below.

Output: milp_verification_results.csv  ->  Table for Section 3.2.1,
plus subinstance GCF matrices (gcf_subinstance_nXX.csv) and a per-instance
constraint-check report printed to console.
"""

import os
import time
import random
import numpy as np
import pandas as pd
import networkx as nx
import pulp

# ---------------------------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------------------------
DATA_DIR   = "/"          # instance CSVs
DEM_DIR    = "/" # SRTM tiles
OUT_DIR    = "/"
INSTANCE   = "SLAS100"                          # mother instance
DEMAND_CSVS = [f"{INSTANCE}_Volume_V1M5.csv",   # DV1M5 (capacity slack)
               f"{INSTANCE}_Volume_V1M20.csv"]  # DV1M20 (capacity binding)
DEMAND_CSV = DEMAND_CSVS[0]                     # set per run in __main__
SIZES      = [10, 12, 15, 18, 20]               # n customers per subinstance
TIME_LIMIT = 600                                # seconds per MILP (CBC)
ALPHA, BETA, KAPPA, THETA_MAX = 10.0, 5.0, 0.6, 0.25
HEUR_SEEDS = 20                                 # H-MA-LNS runs per instance
HEUR_LNS_ITERS = 8000

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Real-instance loader (replaces load_instance_synthetic)
# ---------------------------------------------------------------------------

def build_full_gcf_matrix():
    """Elevations from SRTM + GCF (Eqs. 1-3) over the full 100-node instance."""
    import rasterio
    from rasterio.merge import merge

    coords = pd.read_csv(os.path.join(DATA_DIR, f"{INSTANCE}_Coordinates.csv"),
                         header=None).values            # id, lon, lat
    D = pd.read_csv(os.path.join(DATA_DIR, f"{INSTANCE}_Cost_Distance.csv"),
                    header=None).values.astype(float)   # metres, asymmetric

    srcs = [rasterio.open(os.path.join(DEM_DIR, f))
            for f in ("n37_e126_1arc_v3.tif", "n37_e127_1arc_v3.tif")]
    mosaic, trans = merge(srcs)
    band = mosaic[0]
    elev = np.empty(len(coords))
    for k, (_, lon, lat) in enumerate(coords):
        r, c = rasterio.transform.rowcol(trans, lon, lat)
        v = band[r, c]
        elev[k] = float(v) if not np.isnan(v) else 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        theta = (elev[None, :] - elev[:, None]) / D      # Eq. (1)
    np.fill_diagonal(theta, 0.0)
    infeas = np.abs(theta) > THETA_MAX                    # Eq. (10) pruning
    Omega = np.where(theta >= 0.0,
                     1.0 + ALPHA * theta,
                     np.maximum(KAPPA, 1.0 + BETA * theta))   # Eq. (3)
    Omega = np.clip(Omega, KAPPA, 5.0)   # same numerical caps as the solver code
    C = D * Omega                                         # Eq. (2)
    C[infeas] = np.inf
    np.fill_diagonal(C, np.inf)          # no self-loops
    return C, D, elev


def load_subinstance(C_full, n_customers):
    """First n customers (nodes 1..n) of the mother instance + depot 0.

    Returns c (with np.inf on pruned arcs), q, Q, K -- the SAME data seen by
    the heuristic (the subinstance matrix is also exported to CSV).
    """
    idx = list(range(0, n_customers + 1))
    c = C_full[np.ix_(idx, idx)].copy()

    q = pd.read_csv(os.path.join(DATA_DIR, DEMAND_CSV),
                    header=None)[1].values[: n_customers + 1].astype(float)
    veh = open(os.path.join(DATA_DIR, f"{INSTANCE}_Vehicle_V1.csv")).read()
    Q = float(veh.strip().split(",")[1])          # V1: Q = 5 m^3
    K = n_customers                                # uncapped fleet, as in H-MA-LNS

    # export so the heuristic can consume the identical matrix
    out = os.path.join(OUT_DIR, f"gcf_subinstance_{INSTANCE}_n{n_customers}.csv")
    pd.DataFrame(c).to_csv(out, header=False, index=False)
    return c, q, Q, K


# ---------------------------------------------------------------------------
# 2. Strong-connectivity check of the pruned graph (Section 3.2.1, analytical)
# ---------------------------------------------------------------------------

def check_strong_connectivity(c):
    n = c.shape[0]
    Gp = nx.DiGraph()
    Gp.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if i != j and np.isfinite(c[i, j]):
                Gp.add_edge(i, j)
    return nx.is_strongly_connected(Gp)


# ---------------------------------------------------------------------------
# 3. Exact MILP (Eqs. 4-11), PuLP + CBC
# ---------------------------------------------------------------------------

def solve_milp(c, q, Q, K, time_limit=TIME_LIMIT, msg=False):
    n = c.shape[0]              # nodes incl. depot
    Vc = range(1, n)
    arcs = [(i, j) for i in range(n) for j in range(n)
            if i != j and np.isfinite(c[i, j])]

    prob = pulp.LpProblem("ACVRP_TC", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", arcs, cat="Binary")           # Eq. (11)
    u = pulp.LpVariable.dicts("u", Vc, lowBound=0, cat="Continuous")

    prob += pulp.lpSum(c[i, j] * x[(i, j)] for (i, j) in arcs)   # Eq. (4)

    for i in Vc:                                                 # Eq. (5)
        prob += pulp.lpSum(x[(i, j)] for j in range(n)
                           if (i, j) in x) == 1
    for j in Vc:                                                 # Eq. (6)
        prob += pulp.lpSum(x[(i, j)] for i in range(n)
                           if (i, j) in x) == 1
    prob += pulp.lpSum(x[(0, j)] for j in Vc if (0, j) in x) <= K   # Eq. (7)
    prob += pulp.lpSum(x[(i, 0)] for i in Vc if (i, 0) in x) <= K
    # Standard valid inequality (bin-packing bound): at least ceil(sum q / Q)
    # routes are required.  It does not alter the feasible set of the ACVRP-TC
    # (any feasible solution satisfies it) but tightens the LP relaxation,
    # closing n = 20 with binding capacity in seconds instead of hours.
    kmin = int(np.ceil(sum(q[1:]) / Q))
    prob += pulp.lpSum(x[(0, j)] for j in Vc if (0, j) in x) >= kmin
    prob += pulp.lpSum(x[(i, 0)] for i in Vc if (i, 0) in x) >= kmin

    for i in Vc:                                                 # Eq. (8) MTZ
        for j in Vc:
            if i != j and (i, j) in x:
                prob += u[j] - u[i] >= q[j] - Q * (1 - x[(i, j)])
    for i in Vc:                                                 # Eq. (9)
        prob += u[i] >= q[i]
        prob += u[i] <= Q
    # Eq. (10) is enforced by construction: pruned arcs have no variable.

    solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit)
    # solver = pulp.GUROBI_CMD(msg=msg, timeLimit=time_limit)    # <-- GUROBI
    t0 = time.perf_counter()
    prob.solve(solver)
    t = time.perf_counter() - t0

    status = pulp.LpStatus[prob.status]
    xsol = {(i, j): int(round(pulp.value(x[(i, j)]) or 0)) for (i, j) in arcs}
    obj = pulp.value(prob.objective)
    return status, obj, t, xsol


# ---------------------------------------------------------------------------
# 4. Independent post-hoc constraint verification (Section 3.2.1)
# ---------------------------------------------------------------------------

def posthoc_check(xsol, c, q, Q, K):
    """Re-verifies Eqs. (5)-(10) on the solution, independently of the solver."""
    n = c.shape[0]
    report = {}
    out_deg = {i: sum(v for (a, b), v in xsol.items() if a == i) for i in range(n)}
    in_deg  = {j: sum(v for (a, b), v in xsol.items() if b == j) for j in range(n)}
    report["assignment (5)-(6)"] = all(out_deg[i] == 1 and in_deg[i] == 1
                                       for i in range(1, n))
    report["fleet (7)"] = out_deg[0] <= K and in_deg[0] <= K

    # rebuild routes from the arcs and check capacity + depot-connected tours
    succ = {a: b for (a, b), v in xsol.items() if v == 1}
    routes, visited = [], set()
    starts = [b for (a, b), v in xsol.items() if v == 1 and a == 0]
    for s in starts:
        route, node = [0, s], s
        while node != 0:
            node = succ[node]
            route.append(node)
        routes.append(route)
        visited.update(route[1:-1])
    all_customers = set(range(1, n))
    report["no subtours (8)"] = visited == all_customers
    report["capacity (8)-(9)"] = all(
        sum(q[v] for v in r[1:-1]) <= Q + 1e-9 for r in routes)
    report["pruned arcs excluded (10)"] = all(
        np.isfinite(c[a, b]) for (a, b), v in xsol.items() if v == 1)
    cost = sum(c[a, b] for (a, b), v in xsol.items() if v == 1)
    return report, routes, cost


# ---------------------------------------------------------------------------
# 5. H-MA-LNS on the SAME subinstances (same operators as the repository code:
#    biased Clarke-Wright init, shaw/worst removal, regret-2 insertion,
#    intra-route improvement, greedy split), reduced budget for n <= 20.
#    Replace HMALNS_COST with the output of the full V5-fast engine if desired.
# ---------------------------------------------------------------------------

def split_routes(perm, d, Q):
    routes, cur, load = [], [0], 0.0
    for cl in perm:
        cl = int(cl)
        if load + d[cl] > Q:
            cur.append(0); routes.append(cur); cur = [0, cl]; load = d[cl]
        else:
            cur.append(cl); load += d[cl]
    cur.append(0); routes.append(cur)
    return routes


def costo_total(routes, C):
    return sum(C[r[i], r[i + 1]] for r in routes for i in range(len(r) - 1))


def cw_savings_init(C, d, Q):
    n = len(d)
    savings = []
    for i in range(1, n):
        for j in range(i + 1, n):
            s_ij = C[i, 0] + C[0, j] - C[i, j]
            s_ji = C[j, 0] + C[0, i] - C[j, i]
            if not (np.isfinite(s_ij) and np.isfinite(s_ji)):
                continue
            val = 0.6 * max(s_ij, s_ji) + 0.4 * min(s_ij, s_ji)
            if val > 0:
                savings.append((val, i, j))
    savings.sort(reverse=True, key=lambda x: x[0])
    routes = {i: [0, i, 0] for i in range(1, n)}
    loads = {i: d[i] for i in range(1, n)}
    for _, i, j in savings:
        ri = rj = None; i_end = j_start = False
        for k, r in routes.items():
            if r and i == r[-2]: ri, i_end = k, True
            if r and j == r[1]: rj, j_start = k, True
        if i_end and j_start and ri != rj and loads[ri] + loads[rj] <= Q:
            routes[ri] = routes[ri][:-1] + routes[rj][1:]
            loads[ri] += loads[rj]; routes[rj] = None
    return [x for r in routes.values() if r for x in r if x != 0]


def swap_intra(route, C):
    """Directional swap (no segment reversal) - asymmetric-safe."""
    best = list(route); improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 1):
            for j in range(i + 1, len(best) - 1):
                cand = list(best); cand[i], cand[j] = cand[j], cand[i]
                d_old = sum(C[best[k], best[k+1]] for k in range(len(best)-1))
                d_new = sum(C[cand[k], cand[k+1]] for k in range(len(cand)-1))
                if d_new < d_old - 1e-9:
                    best = cand; improved = True; break
            if improved: break
    return best


def shaw_removal(routes, C, d, qn, p=6):
    clients = [c for r in routes for c in r if c != 0]
    seed = random.choice(clients)
    removed = [seed]; pool = set(clients); pool.discard(seed)
    finite = C[np.isfinite(C)]
    max_c = finite.max() if finite.size else 1.0
    max_d = max(d.max(), 1e-9)
    while len(removed) < qn and pool:
        ref = random.choice(removed)
        ranks = sorted(((c, (min(C[ref, c], max_c) / max_c)
                         + abs(d[ref] - d[c]) / max_d) for c in pool),
                       key=lambda t: t[1])
        idx = int(len(ranks) * (random.random() ** p))
        chosen = ranks[idx][0]
        removed.append(chosen); pool.remove(chosen)
    return removed


def worst_removal(routes, C, qn, p=3):
    scores = []
    for r in routes:
        for i in range(1, len(r) - 1):
            c0, prev, nxt = r[i], r[i - 1], r[i + 1]
            scores.append((c0, (C[prev, c0] + C[c0, nxt]) - C[prev, nxt]))
    scores.sort(key=lambda t: t[1], reverse=True)
    removed = []
    while len(removed) < qn and scores:
        idx = int(len(scores) * (random.random() ** p))
        removed.append(scores.pop(idx)[0])
    return removed


def regret_insertion(routes, removed, C, d, Q, k=2):
    rutas = [list(r) for r in routes]; unassigned = set(removed)
    while unassigned:
        candidates = []
        for c in unassigned:
            ins = []
            for r_idx, r in enumerate(rutas):
                load = sum(d[x] for x in r if x != 0)
                if load + d[c] > Q: continue
                for pos in range(1, len(r)):
                    a, b = C[r[pos-1], c], C[c, r[pos]]
                    if not (np.isfinite(a) and np.isfinite(b)):
                        continue          # never insert through a pruned arc
                    base = C[r[pos-1], r[pos]]
                    # base may be inf when removals joined two nodes whose
                    # direct arc is pruned; repairing it costs a + b in full
                    cost = a + b - (base if np.isfinite(base) else 0.0)
                    ins.append({"cost": cost, "r": r_idx, "p": pos})
            if np.isfinite(C[0, c] + C[c, 0]):
                ins.append({"cost": C[0, c] + C[c, 0], "r": -1, "p": -1})
            ins.sort(key=lambda t: t["cost"])
            regret = (ins[k-1]["cost"] - ins[0]["cost"]) if len(ins) >= k else 0.0
            candidates.append((regret, c, ins[0]))
        candidates.sort(key=lambda t: t[0], reverse=True)
        _, best_c, best_ins = candidates[0]
        if best_ins["r"] == -1:
            rutas.append([0, best_c, 0])
        else:
            rutas[best_ins["r"]].insert(best_ins["p"], best_c)
        unassigned.remove(best_c)
    return rutas


def relocate_inter(routes, C, d, Q):
    """Inter-route relocate (part of LS_light in the manuscript, Eq. 24)."""
    routes = [list(r) for r in routes]
    improved = True
    while improved:
        improved = False
        for ri in range(len(routes)):
            for i in range(1, len(routes[ri]) - 1):
                cnode = routes[ri][i]
                rem = (C[routes[ri][i-1], routes[ri][i+1]]
                       - C[routes[ri][i-1], cnode] - C[cnode, routes[ri][i+1]])
                if not np.isfinite(rem):
                    continue
                for rj in range(len(routes)):
                    if rj == ri:
                        continue
                    load = sum(d[x] for x in routes[rj] if x != 0)
                    if load + d[cnode] > Q:
                        continue
                    for pos in range(1, len(routes[rj])):
                        a = C[routes[rj][pos-1], cnode]
                        b = C[cnode, routes[rj][pos]]
                        base = C[routes[rj][pos-1], routes[rj][pos]]
                        if not (np.isfinite(a) and np.isfinite(b)
                                and np.isfinite(base)):
                            continue
                        if rem + a + b - base < -1e-9:
                            routes[rj].insert(pos, cnode)
                            del routes[ri][i]
                            improved = True
                            break
                    if improved: break
                if improved: break
            if improved: break
        routes = [r for r in routes if len(r) > 2]
    return routes


def hmalns(C, d, Q, iterations=HEUR_LNS_ITERS, seed=0):
    random.seed(seed); np.random.seed(seed)
    n_clients = len(d) - 1
    perm = cw_savings_init(C, d, Q)
    best_routes = [swap_intra(r, C) for r in split_routes(perm, d, Q)]
    best_cost = costo_total(best_routes, C)
    curr = best_routes
    for _ in range(iterations):
        qn = random.randint(2, max(3, int(n_clients * 0.4)))
        removed = (shaw_removal if random.random() < 0.5 else
                   lambda r, C_, d_, q_: worst_removal(r, C_, q_)
                   )(curr, C, d, qn)
        partial = [[x for x in r if x not in set(removed)] for r in curr]
        partial = [r for r in partial if len(r) > 2]
        new_r = regret_insertion(partial, removed, C, d, Q)
        new_r = [swap_intra(r, C) for r in new_r]
        new_c = costo_total(new_r, C)
        if new_c < best_cost - 1e-9:
            best_cost, best_routes, curr = new_c, new_r, new_r
        elif random.random() < 0.05:
            curr = new_r
    # LS_light polish: inter-route relocate + directional swap-intra
    best_routes = relocate_inter(best_routes, C, d, Q)
    best_routes = [swap_intra(r, C) for r in best_routes]
    best_cost = costo_total(best_routes, C)
    return best_cost, best_routes


# ---------------------------------------------------------------------------
# 6. Main experiment
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Building full GCF matrix (Eqs. 1-3, alpha=10, beta=5, kappa=0.6, "
          "theta_max=25%) ...")
    C_full, D_full, elev = build_full_gcf_matrix()
    pruned_full = int(np.isinf(C_full).sum() - C_full.shape[0])
    print(f"  full instance: {pruned_full} pruned arcs of "
          f"{C_full.shape[0]*(C_full.shape[0]-1)}")

    rows = []
    for demand_csv in DEMAND_CSVS:
      globals()["DEMAND_CSV"] = demand_csv
      cfg = "DV1M5" if "M5" in demand_csv else "DV1M20"
      for n in SIZES:
        name = f"{INSTANCE}-n{n}"
        c, q, Q, K = load_subinstance(C_full, n)
        pruned = int(np.isinf(c).sum() - c.shape[0])
        connected = check_strong_connectivity(c)
        print(f"\n=== {name}: Q={Q}, K={K}, sum(q)={q.sum():.3f}, "
              f"pruned arcs={pruned}, strongly connected={connected} ===")

        status, obj, t, xsol = solve_milp(c, q, Q, K)
        report, routes, cost_check = posthoc_check(xsol, c, q, Q, K)
        ok = all(report.values())
        print(f"  MILP: status={status}, obj={obj:.2f}, time={t:.2f}s, "
              f"post-hoc={'ALL SATISFIED' if ok else report}")
        print(f"  routes: {routes}")

        # H-MA-LNS on the same matrix, HEUR_SEEDS independent seeds
        t0 = time.perf_counter()
        heur_costs = [hmalns(c, q, Q, seed=s)[0] for s in range(HEUR_SEEDS)]
        t_heur = time.perf_counter() - t0
        heur_best = min(heur_costs)
        gap = 100.0 * (heur_best - obj) / obj if obj else float("nan")
        print(f"  H-MA-LNS best of {HEUR_SEEDS} seeds: {heur_best:.2f} "
              f"(gap {gap:.2f}%, {t_heur:.1f}s total)")

        rows.append({
            "Instance": name, "Config": cfg, "n": n,
            "Connected": "Yes" if connected else "No",
            "Pruned arcs": pruned, "MILP status": status,
            "MILP opt": round(obj, 2), "Time (s)": round(t, 2),
            "Constraints": "All satisfied" if ok else "VIOLATION",
            "H-MA-LNS": round(heur_best, 2), "Gap (%)": round(gap, 2),
        })
        pd.DataFrame(rows).to_csv(
            os.path.join(OUT_DIR, "milp_verification_results.csv"), index=False)

    print("\nDone. Results in", os.path.join(OUT_DIR,
          "milp_verification_results.csv"))
