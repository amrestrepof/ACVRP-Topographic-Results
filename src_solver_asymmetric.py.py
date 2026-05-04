# -*- coding: utf-8 -*-
"""
VRP asimétrico con GA + LNS (versión unificada y mejorada).

Este script combina la potencia algorítmica del GA+LNS con una estructura
de código robusta, modular y con mejor manejo de errores.

Incluye:
- Estructura principal con función main() y manejo de errores.
- Población inicial híbrida (Clarke & Wright + Cluster-First/Sweep).
- Algoritmo Genético Memético con búsqueda local intra-ruta (2-opt) e 
  inter-ruta (swap).
- Fase final de refinamiento intensivo con Large Neighborhood Search (LNS).
- Cálculo opcional de costos sobre la malla vial de OSMnx.
- Visualización completa y reporte a Excel.
- **NUEVO: Medición detallada de tiempos por sección.**

"""

import os
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.core.callback import Callback
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.optimize import minimize
from pymoo.termination import get_termination

import folium
from folium import plugins
import networkx as nx
import osmnx as ox
from joblib import Parallel, delayed


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os

# Configuración de OSMnx
ox.settings.use_cache = True
ox.settings.timeout = 300
ox.settings.memory = 0.8

# ────────────────────────────────────────────────────────────────────────────
# LECTURA DE DATOS
# ────────────────────────────────────────────────────────────────────────────

def cargar_datos_instancia(R_VEHICLE, R_DEMAND, R_COORDS, R_COST_MATRIX):
    """Carga los datos de la instancia desde archivos CSV."""
    try:
        # 1. Leer Capacidad (Q)
        df_vehicle = pd.read_csv(R_VEHICLE, header=None)
        Q = df_vehicle.iloc[0, 1]
        
        # 2. Leer Demandas (d)
        df_demand = pd.read_csv(R_DEMAND, header=None)
        d = df_demand.iloc[:, 1].values
        
        # 3. Leer Coordenadas (coords_latlon) -> [Lat, Lon]
        df_coords = pd.read_csv(R_COORDS, header=None)
        coords_latlon = df_coords.iloc[:, [2, 1]].values
        
        # 4. Leer Matriz de Costos (C) -> Convertir de Metros a KM
        C_m = pd.read_csv(R_COST_MATRIX, header=None).values
        C = C_m / 1000.0
        
        print(f"✅ Datos cargados: {len(d)} nodos, Q={Q}")
        return C, d, Q, coords_latlon
        
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        raise

# ────────────────────────────────────────────────────────────────────────────
# UTILIDADES VRP
# ────────────────────────────────────────────────────────────────────────────

def reparar_permutacion(perm, n_clientes):
    seen, out = set(), []
    for x in perm:
        xi = int(x)
        if xi != 0 and 1 <= xi <= n_clientes and xi not in seen:
            seen.add(xi)
            out.append(xi)
    missing = [i for i in range(1, n_clientes + 1) if i not in seen]
    return out + missing

def split_routes(permutation, d, Q):
    routes, current_route, current_load = [], [0], 0.0
    for client in permutation:
        c = int(client)
        dem = d[c]
        if current_load + dem > Q:
            current_route.append(0)
            routes.append(current_route)
            current_route = [0, c]
            current_load = dem
        else:
            current_route.append(c)
            current_load += dem
    if len(current_route) > 1:
        current_route.append(0)
        routes.append(current_route)
    return routes

def costo_total(routes, C):
    total = 0.0
    for r in routes:
        for i in range(len(r)-1):
            total += C[int(r[i]), int(r[i+1])]
    return float(total)

# ────────────────────────────────────────────────────────────────────────────
# OSMnx: Red vial y Matriz de costos
# ────────────────────────────────────────────────────────────────────────────

SENTINEL = 9.9e11

def build_drive_graph(coords_latlon, pad_km=5.0, graphml_path=None):
    """Construye o carga el grafo de calles desde OSM."""
    try:
        center = (float(coords_latlon[0, 0]), float(coords_latlon[0, 1]))
        from math import radians, cos, sin, asin, sqrt
        def haversine_m(p, q):
            lat1, lon1 = map(radians, p); lat2, lon2 = map(radians, q)
            dlat, dlon = lat2 - lat1, lon2 - lon1
            a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
            return 2 * 6371000 * asin(sqrt(a))
        
        R = max(haversine_m(center, (float(lat), float(lon))) for lat, lon in coords_latlon[1:]) + pad_km * 1000
        
        G = None
        if graphml_path and os.path.exists(graphml_path):
            try:
                G = ox.load_graphml(graphml_path)
                print(f"✅ Grafo cargado desde cache: {graphml_path}")
            except Exception as e:
                print(f"⚠️ No se pudo cargar cache del grafo: {e}")
        
        if G is None:
            print(f"🌐 Descargando grafo de OpenStreetMap (radio {R/1000:.2f} km)...")
            G = ox.graph_from_point(center, dist=R, network_type="drive", retain_all=False)
            if graphml_path:
                Path(graphml_path).parent.mkdir(parents=True, exist_ok=True)
                ox.save_graphml(G, graphml_path)
                print(f"💾 Grafo guardado en cache: {graphml_path}")
        
        print("⚡ Calculando velocidades y tiempos de viaje...")
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)
        print(f"✅ Grafo listo: {len(G.nodes)} nodos, {len(G.edges)} aristas")
        return G
        
    except Exception as e:
        print(f"❌ Error construyendo grafo: {e}")
        return None

def snap_nodes(G, coords_latlon):
    """Encuentra los nodos más cercanos en el grafo para cada coordenada."""
    return [ox.distance.nearest_nodes(G, lon, lat) for (lat, lon) in coords_latlon]

def validar_matriz_costos(C):
    """Valida que la matriz de costos sea correcta."""
    if not np.all(np.isfinite(C)): raise RuntimeError("C tiene NaN/Inf.")
    if np.any(C >= SENTINEL):
        imposibles = int(np.sum(C >= SENTINEL))
        raise RuntimeError(f"Quedan {imposibles} pares sin camino en C.")
    if np.any(np.diag(C) != 0): np.fill_diagonal(C, 0.0)
    print("✅ Matriz de costos validada")

def _dijkstra_row(i, u, nodes, G, weight):
    """Calcula una fila de la matriz de costos usando Dijkstra."""
    lengths = nx.single_source_dijkstra_path_length(G, u, weight=weight)
    row = np.full(len(nodes), SENTINEL, dtype=float)
    for j, v in enumerate(nodes):
        if i == j: row[j] = 0.0
        else:      row[j] = lengths.get(v, SENTINEL)
    return row

def street_cost_matrix_fast(G, snapped_nodes, weight='length', cache_path=None, n_jobs=1):
    """Calcula la matriz de costos usando caminos reales en el grafo."""
    n = len(snapped_nodes)
    if cache_path and os.path.exists(cache_path):
        try:
            C = np.load(cache_path)
            if C.shape == (n, n) and np.all(np.isfinite(C)) and np.all(C < SENTINEL):
                print(f"✅ Matriz cargada desde cache: {cache_path}")
                return C
        except Exception as e:
            print(f"⚠️ Cache de matriz inválido: {e}")
    
    print(f"🔄 Calculando matriz de costos ({n}×{n}, weight='{weight}', jobs={n_jobs})...")
    start_time = time.time()
    
    rows = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_dijkstra_row)(i, snapped_nodes[i], snapped_nodes, G, weight) for i in range(n)
    )
    
    C = np.vstack(rows)
    validar_matriz_costos(C)
    
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, C)
        print(f"💾 Matriz guardada en cache: {cache_path}")
    
    print(f"✅ Matriz calculada en {time.time() - start_time:.2f} segundos")
    return C

# ────────────────────────────────────────────────────────────────────────────
# HEURÍSTICAS Y OPERADORES
# ────────────────────────────────────────────────────────────────────────────

def cluster_first_sweep(coords_latlon, d, Q):
    n_clientes = len(d) - 1
    depot_lat, depot_lon = coords_latlon[0, 0], coords_latlon[0, 1]
    angles = {i: np.arctan2(coords_latlon[i, 0] - depot_lat, coords_latlon[i, 1] - depot_lon) for i in range(1, n_clientes + 1)}
    sorted_clients = sorted(angles, key=angles.get)
    
    clusters, current_cluster, current_load = [], [], 0.0
    for client in sorted_clients:
        if current_load + d[client] <= Q:
            current_cluster.append(client)
            current_load += d[client]
        else:
            clusters.append(current_cluster)
            current_cluster = [client]
            current_load = d[client]
    if current_cluster: clusters.append(current_cluster)
    return clusters

def clarke_wright_savings_asymmetric_biased(C, d, Q, beta=0.6):
    n = len(d)
    savings = []
    for i in range(1, n):
        for j in range(i+1, n):
            s_ij = C[i, 0] + C[0, j] - C[i, j]
            s_ji = C[j, 0] + C[0, i] - C[j, i]
            w = beta * max(s_ij, s_ji) + (1 - beta) * min(s_ij, s_ji)
            if w > 0: savings.append((w, i, j))
    
    savings.sort(reverse=True, key=lambda x: x[0])
    routes = {i: [0, i, 0] for i in range(1, n)}
    loads = {i: d[i] for i in range(1, n)}
    
    for _, i, j in savings:
        ri = rj = None; is_i_end = is_j_start = False
        for key, r in routes.items():
            if r and i == r[-2]: ri, is_i_end = key, True
            if r and j == r[1]:  rj, is_j_start = key, True
        
        if is_i_end and is_j_start and ri != rj:
            if loads.get(ri, 0) + loads.get(rj, 0) <= Q:
                routes[ri] = routes[ri][:-1] + routes[rj][1:]
                loads[ri] += loads.get(rj, 0)
                routes[rj], loads[rj] = None, 0
    
    final_routes = [r for r in routes.values() if r]
    perm = [c for r in final_routes for c in r if c != 0]
    return reparar_permutacion(perm, n-1)

def two_opt_intra_route(route, C):
    best = list(route); improved = True
    while improved:
        improved = False
        for i in range(1, len(best)-2):
            for j in range(i+1, len(best)-1):
                p1, p2, p3, p4 = best[i-1], best[i], best[j], best[j+1]
                seg = best[i:j+1]
                if len(seg) < 2: continue
                rev = seg[::-1]
                orig_cost = C[p1, p2] + C[p3, p4] + sum(C[seg[k], seg[k+1]] for k in range(len(seg)-1))
                new_cost = C[p1, p3] + C[p2, p4] + sum(C[rev[k], rev[k+1]] for k in range(len(rev)-1))
                if new_cost < orig_cost - 1e-9:
                    best = best[:i] + rev + best[j+1:]
                    improved = True; break
            if improved: break
    return best

def swap_inter_route(routes, C, d, Q):
    # --- ¡NUEVO! Traído del primer script ---
    R = len(routes)
    for r1 in range(R):
        for r2 in range(r1+1, R):
            for i in range(1, len(routes[r1])-1):
                for j in range(1, len(routes[r2])-1):
                    a, b = routes[r1][i], routes[r2][j]
                    load1 = sum(d[c] for c in routes[r1] if c!=0)
                    load2 = sum(d[c] for c in routes[r2] if c!=0)
                    if load1 - d[a] + d[b] > Q or load2 - d[b] + d[a] > Q: continue
                    
                    p1, n1 = routes[r1][i-1], routes[r1][i+1]
                    p2, n2 = routes[r2][j-1], routes[r2][j+1]
                    removed = C[p1,a] + C[a,n1] + C[p2,b] + C[b,n2]
                    added   = C[p1,b] + C[b,n1] + C[p2,a] + C[a,n2]
                    
                    if added < removed - 1e-9:
                        routes[r1][i], routes[r2][j] = b, a
                        return routes # Retorna tras la primera mejora
    return routes

def shaw_removal(clientes_en_rutas, C, d, q, p=6):
    # --- ¡NUEVO! Componente para LNS ---
    if not clientes_en_rutas: return [], []
    destruidos, pool = [], set(clientes_en_rutas)
    seed = random.choice(clientes_en_rutas)
    destruidos.append(seed); pool.discard(seed)
    
    max_cost = np.max(C[np.nonzero(C)]) if np.any(C > 0) else 1.0
    max_demand = np.max(d) if np.any(d > 0) else 1.0

    while len(destruidos) < q and pool:
        ref = random.choice(destruidos)
        rel = []
        for c in pool:
            r = 0.7 * (C[ref, c] / max_cost) + 0.3 * (abs(d[ref] - d[c]) / max_demand)
            rel.append((c, r))
        rel.sort(key=lambda x: x[1])
        idx = int(len(rel) * (random.random() ** p))
        pick = rel[idx][0]
        destruidos.append(pick); pool.remove(pick)
    return destruidos, list(pool)

def worst_removal(routes, C, q, p=3):
    # --- ¡NUEVO! Componente para LNS ---
    if not routes: return [], []
    scores, allc = [], []
    for r in routes:
        for i in range(1, len(r)-1):
            c, prev, nxt = r[i], r[i-1], r[i+1]
            saving = (C[prev,c] + C[c,nxt]) - C[prev,nxt]
            scores.append((c, saving)); allc.append(c)
    
    scores.sort(key=lambda x: x[1], reverse=True)
    destruidos = []
    while len(destruidos) < q and scores:
        idx = int(len(scores) * (random.random() ** p))
        c = scores.pop(idx)[0]
        if c not in destruidos: destruidos.append(c)
    
    restantes = [c for c in allc if c not in destruidos]
    return destruidos, restantes

def large_neighborhood_search(perm, C, d, Q, n_clientes, params):
    # --- ¡NUEVO! Fase completa de LNS traída del primer script ---
    current_routes = split_routes(perm, d, Q)
    best_cost = costo_total(current_routes, C)
    
    for it in range(params['lns_iterations']):
        temp_routes = [list(r) for r in current_routes]
        clientes = [c for r in temp_routes for c in r if c != 0]
        if not clientes: continue
        
        q = max(1, min(int(len(clientes) * params['lns_destruction_frac']), len(clientes)-1))
        
        # Fase de Destrucción
        if random.random() < 0.5:
            removed, _ = shaw_removal(clientes, C, d, q, p=params['shaw_p'])
        else:
            removed, _ = worst_removal(temp_routes, C, q, p=params['worst_p'])
            
        unassigned = set(removed)
        rutas = [ [x for x in r if x not in unassigned] for r in temp_routes ]
        rutas = [ r for r in rutas if len(r) > 2 ]

        # Fase de Reparación (Regret-k Insertion)
        k = params['regret_k']
        while unassigned:
            best_client, best_ins, best_regret = None, None, -float('inf')
            
            for c in list(unassigned):
                insertions = []
                # Evaluar inserción en rutas existentes
                for ridx, r in enumerate(rutas):
                    load = sum(d[x] for x in r if x != 0)
                    if load + d[c] <= Q:
                        for pos in range(1, len(r)):
                            delta = (C[r[pos-1], c] + C[c, r[pos]]) - C[r[pos-1], r[pos]]
                            insertions.append({'cost': delta, 'ridx': ridx, 'pos': pos})
                # Evaluar creación de nueva ruta
                insertions.append({'cost': C[0, c] + C[c, 0], 'ridx': -1, 'pos': -1})
                
                insertions.sort(key=lambda x: x['cost'])
                if not insertions: continue
                
                regret = sum(insertions[i]['cost'] - insertions[0]['cost'] for i in range(1, min(k, len(insertions))))
                
                if regret > best_regret:
                    best_regret, best_client, best_ins = regret, c, insertions[0]

            if best_client is None: break 
            
            ridx, pos = best_ins['ridx'], best_ins['pos']
            if ridx == -1:
                rutas.append([0, best_client, 0])
            else:
                rutas[ridx].insert(pos, best_client)
            
            unassigned.remove(best_client)

        new_cost = costo_total(rutas, C)
        if new_cost < best_cost - 1e-9:
            best_cost, current_routes = new_cost, rutas
            if it % 20 == 0: print(f"    LNS iter {it}: Nuevo mejor costo {best_cost:.2f}")

    final_perm = [c for r in current_routes for c in r if c != 0]
    return reparar_permutacion(final_perm, n_clientes), best_cost


# ────────────────────────────────────────────────────────────────────────────
# PYMOO Problem Definition
# ────────────────────────────────────────────────────────────────────────────

class AVRPProblem(ElementwiseProblem):
    def __init__(self, C, d, Q):
        super().__init__(n_var=len(d)-1, n_obj=1, n_constr=0)
        self.C, self.d, self.Q = C, d, Q

    def _evaluate(self, x, out, *args, **kwargs):
        perm = reparar_permutacion(list(map(int, x)), len(self.d)-1)
        routes = split_routes(perm, self.d, self.Q)
        out["F"] = costo_total(routes, self.C)

class InitialPopulationAVRP(Sampling):
    def __init__(self, C, d, Q, coords_latlon, n_cw_individuals, n_cf_rs_individuals):
        super().__init__()
        self.C, self.d, self.Q = C, d, Q
        self.coords_latlon = coords_latlon
        self.n_cw_individuals = n_cw_individuals
        self.n_cf_rs_individuals = n_cf_rs_individuals

    def _do(self, problem, n_samples, **kwargs):
        nvar = problem.n_var
        X = np.zeros((n_samples, nvar), dtype=np.int32)
        
        # Clarke & Wright
        for i in range(self.n_cw_individuals):
            perm = clarke_wright_savings_asymmetric_biased(self.C, self.d, self.Q)
            X[i, :] = np.array(reparar_permutacion(perm, nvar), dtype=np.int32)
        
        # Cluster-First, Route-Second
        offset = self.n_cw_individuals
        for i in range(self.n_cf_rs_individuals):
            clusters = cluster_first_sweep(self.coords_latlon, self.d, self.Q)
            final_perm = []
            for cluster in clusters:
                if len(cluster) > 0:
                    route_cluster = [0] + cluster + [0]
                    optimized_route = two_opt_intra_route(route_cluster, self.C)
                    final_perm.extend(optimized_route[1:-1])
            X[offset + i, :] = np.array(reparar_permutacion(final_perm, nvar), dtype=np.int32)

        # Permutaciones aleatorias
        for i in range(offset + self.n_cf_rs_individuals, n_samples):
            X[i, :] = np.random.permutation(np.arange(1, nvar + 1))
        
        return X

class MemeticCallback(Callback):
    def __init__(self, C, d, Q, frequency, elite_frac):
        super().__init__()
        self.C, self.d, self.Q, self.frequency, self.elite_frac = C, d, Q, frequency, elite_frac

    def notify(self, algorithm):
        if algorithm.n_gen > 1 and algorithm.n_gen % self.frequency == 0:
            pop = algorithm.pop
            elite_size = max(1, int(self.elite_frac * len(pop)))
            idxs = np.argsort([ind.F[0] for ind in pop])[:elite_size]
            
            for i in idxs:
                perm = reparar_permutacion(pop[i].X.astype(int), algorithm.problem.n_var)
                rutas = split_routes(perm, self.d, self.Q)
                # --- ¡MEJORADO! Se aplican ambas búsquedas locales ---
                rutas = [two_opt_intra_route(r, self.C) for r in rutas]
                rutas = swap_inter_route(rutas, self.C, self.d, self.Q) # Mejora inter-ruta
                
                perm2 = [c for r in rutas for c in r if c != 0]
                pop[i].X = np.array(reparar_permutacion(perm2, algorithm.problem.n_var), dtype=np.int32)
            
            algorithm.evaluator.eval(algorithm.problem, pop)

# ────────────────────────────────────────────────────────────────────────────
# VISUALIZACIÓN
# ────────────────────────────────────────────────────────────────────────────

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os

def plot_routes_png(routes, nodes, instance_name, output_path=None,
                    plot_final_street_map=True, dpi=150):
    """
    Visualiza rutas de VRP o ACVRP en un gráfico limpio y profesional.
    Acepta tanto DataFrame como numpy.ndarray para coordenadas de nodos.
    """

    # ─────────────────────────────────────────────
    # Normalizar entrada de nodos
    # ─────────────────────────────────────────────
    if isinstance(nodes, np.ndarray):
        nodes_df = None
        X, Y = nodes[:, 0], nodes[:, 1]
    else:
        nodes_df = nodes.copy()
        if 'x' in nodes_df.columns and 'y' in nodes_df.columns:
            X, Y = nodes_df['x'].values, nodes_df['y'].values
        else:
            raise ValueError("El DataFrame debe tener columnas 'x' y 'y'.")

    # ─────────────────────────────────────────────
    # Preparar color map moderno
    # ─────────────────────────────────────────────
    cmap = plt.colormaps.get_cmap('tab20').resampled(len(routes))
    norm = mcolors.Normalize(vmin=0, vmax=len(routes) - 1)

    # ─────────────────────────────────────────────
    # Configurar figura
    # ─────────────────────────────────────────────
    plt.figure(figsize=(10, 10))
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.gca().set_facecolor('white')

    # ─────────────────────────────────────────────
    # Dibujar cada ruta
    # ─────────────────────────────────────────────
    for i, route in enumerate(routes):
        color = cmap(norm(i))
        route = np.array(route, dtype=int)

        if nodes_df is None:
            # Caso numpy
            x, y = X[route], Y[route]
        else:
            # Caso DataFrame
            x, y = nodes_df.loc[route, 'x'], nodes_df.loc[route, 'y']

        plt.plot(x, y, color=color, linewidth=2.5, alpha=0.9, label=f'Route {i+1}')
        plt.scatter(x, y, color=color, s=25, alpha=0.9)

        # 🔢 Etiquetas de cada nodo
        for j, (xi, yi) in enumerate(zip(x, y)):
            plt.text(xi, yi, str(route[j]), fontsize=8, color='black',
                     ha='center', va='center', weight='bold',
                     bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, boxstyle='round,pad=0.2'))

    # ─────────────────────────────────────────────
    # Destacar depósito si está marcado
    # ─────────────────────────────────────────────
    if nodes_df is not None and 'type' in nodes_df.columns:
        depots = nodes_df[nodes_df['type'].str.lower().str.contains('dep')]
        if not depots.empty:
            plt.scatter(depots['x'], depots['y'], c='black', s=120, marker='*', label='Deposit')

    # ─────────────────────────────────────────────
    # Estilo del gráfico
    # ─────────────────────────────────────────────
    plt.title(f"Routes - {instance_name}", fontsize=14, fontweight='bold')
    plt.xlabel(f"Longitude")
    plt.ylabel(f"Latitude")
    plt.legend(loc='best', fontsize=9, frameon=True)
    plt.tight_layout()

    # ─────────────────────────────────────────────
    # Guardar resultado
    # ─────────────────────────────────────────────
    if output_path is None:
        os.makedirs("./resultados_vrp", exist_ok=True)
        output_path = f"./resultados_vrp/{instance_name}_rutas.png"

    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"✅ Mapa de rutas guardado en: {output_path}")


def plot_routes_map_streets(rutas, coords_latlon, G, snapped_nodes, weight, titulo, fname):
    """
    Genera un mapa interactivo con rutas que siguen la geometría real de las calles
    y muestran la dirección del recorrido.
    """
    m = folium.Map(location=[coords_latlon[0, 0], coords_latlon[0, 1]], zoom_start=13, tiles="OpenStreetMap")
    folium.Marker([coords_latlon[0, 0], coords_latlon[0, 1]], icon=folium.Icon(color="red", icon="home"), tooltip="Depósito (*)").add_to(m)
    
    # --- CORRECCIÓN 1: Obtener geometrías detalladas de las calles ---
    # Convertimos las aristas del grafo a un GeoDataFrame para acceder a su forma real.
    gdf_edges = ox.graph_to_gdfs(G, nodes=False)
    
    colores = list(mcolors.TABLEAU_COLORS.values())
    if len(colores) < len(rutas):
        random.seed(42)
        colores += [f"#{random.randint(0, 0xFFFFFF):06x}" for _ in range(len(rutas)-len(colores))]
    
    for ridx, ruta in enumerate(rutas):
        if len(ruta) < 2: continue
        
        capa = folium.FeatureGroup(name=f"Route {ridx+1}")
        color = colores[ridx % len(colores)]
        
        # Lista para almacenar todos los puntos de la ruta detallada
        puntos_de_ruta_completos = []
        
        for k in range(len(ruta)-1):
            u, v = snapped_nodes[ruta[k]], snapped_nodes[ruta[k+1]]
            
            try:
                # Obtenemos la ruta como una lista de nodos
                path_nodes = nx.shortest_path(G, u, v, weight=weight)
                
                # --- CORRECCIÓN 1 (continuación): Iterar sobre las aristas de la ruta ---
                for i in range(len(path_nodes) - 1):
                    nodo_origen, nodo_destino = path_nodes[i], path_nodes[i+1]
                    
                    # Buscamos la geometría de la arista
                    # Usamos .get(0) en caso de calles paralelas
                    edge_geom = gdf_edges.loc[(nodo_origen, nodo_destino, 0)].geometry
                    
                    # Extraemos las coordenadas de la geometría y las añadimos a nuestra lista
                    puntos = list(edge_geom.coords)
                    puntos_de_ruta_completos.extend(puntos)

            except nx.NetworkXNoPath:
                print(f"⚠️ No hay camino entre {ruta[k]} y {ruta[k+1]}")
                continue
        
        if puntos_de_ruta_completos:
            # Folium espera (lat, lon), pero las geometrías de OSMnx están en (lon, lat)
            puntos_folium = [(y, x) for x, y in puntos_de_ruta_completos]
            
            # Dibujamos la línea completa que sigue las calles
            line = folium.PolyLine(puntos_folium, color=color, weight=5, opacity=0.9, tooltip=f"Route {ridx+1}")
            capa.add_child(line)

            # --- CORRECCIÓN 2: Añadir flechas de dirección ---
            try:
                plugins.PolyLineTextPath(
                    line,
                    "▶",  # Símbolo de flecha
                    repeat=True,
                    offset=8,
                    attributes={'fill': color, 'font-weight': 'bold', 'font-size': '18'}
                ).add_to(capa)
            except Exception as e:
                print(f"No se pudieron añadir las flechas de dirección: {e}")

        # Dibujamos los marcadores de los clientes
        for n in ruta:
            if n != 0:
                folium.CircleMarker([coords_latlon[n, 0], coords_latlon[n, 1]], radius=6, color='white', fill_color=color, fill_opacity=1.0, tooltip=f"Cliente {n}").add_to(capa)
        
        capa.add_to(m)
    
    folium.LayerControl(collapsed=False).add_to(m)
    Path(fname).parent.mkdir(parents=True, exist_ok=True)
    m.save(fname)
    print(f"🗺️  Mapa mejorado guardado: {fname}")



# ────────────────────────────────────────────────────────────────────────────
# ALGORITMO PRINCIPAL
# ────────────────────────────────────────────────────────────────────────────

def run_experiment(C, d, Q, coords_latlon, params, G=None, snapped=None):
    """Ejecuta el ciclo completo de optimización (GA + LNS)."""
    print("\n" + "="*60 + "\n🚀 INICIANDO OPTIMIZACIÓN HÍBRIDA (GA + LNS)\n" + "="*60)
    
    costos_finales, mejores_perms = [], []
    detalle_corridas = []
    outdir = Path(params.get('plot_dir', './plots_vrp'))
    outdir.mkdir(parents=True, exist_ok=True)
    u = params.get('unit_label', 'u')
    
    for run in range(params['n_runs']):
        print(f"\n{'━'*25} Corrida {run+1}/{params['n_runs']} {'━'*25}")
        start_run = time.time()
        
        # ──────────────────────────────────────────────────────────
        # FASE 1: Algoritmo Genético (Cronometrado)
        # ──────────────────────────────────────────────────────────
        print("🧬 Fase 1: Ejecutando Algoritmo Genético Memético...")
        t_ga_start = time.time()
        
        algorithm = GA(
            pop_size=params['pop_size'],
            sampling=InitialPopulationAVRP(C, d, Q, coords_latlon, params['n_cw_individuals'], params['n_cf_rs_individuals']),
            crossover=OrderCrossover(prob=params['pc']),
            mutation=InversionMutation(prob=params['pm']),
            eliminate_duplicates=True,
            callback=MemeticCallback(C, d, Q, params['ls_freq'], params['elite_frac'])
        )
        
        res = minimize(
            problem=AVRPProblem(C, d, Q),
            algorithm=algorithm,
            termination=get_termination("n_gen", params['n_gen']),
            seed=random.randint(1, 10_000),
            verbose=True
        )
        t_ga_end = time.time()
        duracion_ga = t_ga_end - t_ga_start
        
        print(f"  > Fin GA. Costo provisional: {res.F[0]:.2f} {u} (Tiempo: {duracion_ga:.2f}s)")

        # ──────────────────────────────────────────────────────────
        # FASE 2: Large Neighborhood Search (Cronometrado)
        # ──────────────────────────────────────────────────────────
        print("\n🔍 Fase 2: Refinando con Búsqueda de Vecindad Amplia (LNS)...")
        t_lns_start = time.time()
        
        perm_from_ga = list(map(int, res.X))
        final_perm, costo_final = large_neighborhood_search(
            perm_from_ga, C, d, Q, len(d)-1, params
        )
        
        t_lns_end = time.time()
        duracion_lns = t_lns_end - t_lns_start

        elapsed = time.time() - start_run
        print(f"\n Corrida {run+1} completada:")
        print(f"  • Costo Final: {costo_final:.2f} {u}")
        print(f"  • Tiempo GA:   {duracion_ga:.2f}s")
        print(f"  • Tiempo LNS:  {duracion_lns:.2f}s")
        print(f"  • Tiempo Total:{elapsed:.2f}s")
        
        costos_finales.append(costo_final)
        mejores_perms.append(final_perm)

        detalle_corridas.append({
            'Corrida #': run + 1,
            f'Costo Final ({u})': costo_final,
            'Tiempo GA (s)': round(duracion_ga, 2),
            'Tiempo LNS (s)': round(duracion_lns, 2),
            'Tiempo Total (s)': round(elapsed, 2)
        })

    # Encontrar y visualizar mejor solución global
    best_idx = np.argmin(costos_finales)
    best_perm, best_costo = mejores_perms[best_idx], costos_finales[best_idx]
    
    print(f"\n🎯 MEJOR SOLUCIÓN GLOBAL (de {params['n_runs']} corridas):")
    print(f"  • Costo: {best_costo:.2f} {u}")
    
    rutas_best = split_routes(best_perm, d, Q)
    if params.get('plot_final_png', True):
        plot_routes_png(rutas_best, coords_latlon, "SLAM250_DV1M20", str(outdir / "mejor_solucion_SLAM250_DV1M20.png"))
    
    if params.get('plot_final_street_map', True) and G is not None and snapped is not None:
        plot_routes_map_streets(rutas_best, coords_latlon, G, snapped, params['street_weight'], "SLAM250_DV1M20", str(outdir / "mejor_solucion_calles_SLAM250_DV1M20.html"))
    
    return costos_finales, best_perm, detalle_corridas

# ────────────────────────────────────────────────────────────────────────────
# EXPORTACIÓN DE RESULTADOS
# ────────────────────────────────────────────────────────────────────────────

def guardar_resultados_excel(costos, mejor_perm, detalle_corridas, C, d, Q, tiempo_total, params, tiempos_preproc):
    u = params.get('unit_label', 'u')
    fname = f"resultados_vrp_{params.get('instance_name', 'solucion')}.xlsx"
    rutas = split_routes(mejor_perm, d, Q)
    filas = []
    
    for i, r in enumerate(rutas, start=1):
        costo_r = sum(C[r[j], r[j+1]] for j in range(len(r)-1))
        carga_r = sum(d[c] for c in r if c != 0)
        filas.append({
            'Ruta #': i, 'Secuencia': ' -> '.join(map(str, r)),
            'Paradas': max(0, len([c for c in r if c != 0])),
            'Carga Total': carga_r, 'Capacidad': Q,
            'Ocupación (%)': round(100.0 * carga_r / Q, 2),
            f'Distancia ({u})': round(costo_r, 2)
        })
    
    print(f"\n💾 Guardando resultados en '{fname}'...")
    with pd.ExcelWriter(fname) as writer:
        # Hoja de Resumen
        df_resumen = pd.DataFrame({
            'Métrica': ['Mejor Costo', 'Costo Promedio', 'Peor Costo', 'Desv. Estándar', 'Tiempo Total Ejecución (s)', 'Unidad',
                        '--- TIEMPOS PREPROCESAMIENTO ---',
                        'Carga Datos (s)', 'Construcción Grafo (s)', 'Snapping Nodos (s)', 'Cálculo Matriz (s)'],
            'Valor': [np.min(costos), np.mean(costos), np.max(costos), np.std(costos), tiempo_total, u,
                      '',
                      tiempos_preproc.get('carga', 0), tiempos_preproc.get('grafo', 0), tiempos_preproc.get('snapping', 0), tiempos_preproc.get('matriz', 0)]
        })
        df_resumen.to_excel(writer, sheet_name='Resumen', index=False)
        
        # Hoja de Mejor Solución (sin cambios)
        pd.DataFrame(filas).to_excel(writer, sheet_name='Mejor Solucion', index=False)
        
        # <-- NUEVO: Escribir la nueva hoja en el Excel con los tiempos desglosados
        pd.DataFrame(detalle_corridas).to_excel(writer, sheet_name='Detalle Corridas', index=False)

        # Hoja de Parámetros (sin cambios)
        pd.DataFrame(list(params.items()), columns=['Parámetro', 'Valor']).to_excel(writer, sheet_name='Parametros', index=False)

    print("Resultados guardados exitosamente.")


# ────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN Y EJECUCIÓN PRINCIPAL
# ────────────────────────────────────────────────────────────────────────────

def main():
    """Función principal que orquesta la ejecución."""
    # --- CONFIGURACIÓN DE LA INSTANCIA ---
    # NOTA: Para portabilidad, es mejor usar rutas relativas o argumentos de línea de comandos.
    BASE_PATH = BASE_PATH = r'C:\Users\arestrepo\Desktop\doc_amrf\Algoritmos\Prueba instancias asimetricas SLA\ACVRP Benchmark Instances\ACVRP Benchmark Instances\02. SLAM250' # Ejemplo con ruta relativa
    
    R_VEHICLE = os.path.join(BASE_PATH, 'SLAM250_Vehicle_V1.csv')
    R_DEMAND = os.path.join(BASE_PATH, 'SLAM250_Volume_V1M20.csv')
    R_COORDS = os.path.join(BASE_PATH, 'SLAM250_Coordinates.csv')
    R_COST_MATRIX = os.path.join(BASE_PATH, 'SLAM250_Cost_Distance.csv')
    INSTANCE_NAME = "SLAM250_DV1M20"
    
    # Diccionario para guardar tiempos de preprocesamiento
    tiempos_preproc = {}

    # 1. Cargar datos
    print("📂 Cargando datos de la instancia...")
    t0 = time.time()
    C_csv, d, Q, coords_latlon = cargar_datos_instancia(R_VEHICLE, R_DEMAND, R_COORDS, R_COST_MATRIX)
    tiempos_preproc['carga'] = round(time.time() - t0, 4)
    print(f"   ⏱️ Tiempo carga: {tiempos_preproc['carga']}s")

    # 2. Configurar parámetros del algoritmo
    n_clientes = len(d) - 1
    parametros = {
        'instance_name': INSTANCE_NAME, 'n_runs':10 ,
        'pop_size': 150, 'n_gen': 300,
        'pc': 0.8, 'pm': 0.25, 'ls_freq': 8, 'elite_frac': 0.2,
        
        # Parámetros de Población Inicial
        'n_cw_individuals': 15, 'n_cf_rs_individuals': 15,
        
        # --- ¡NUEVO! Parámetros para LNS ---
        'lns_iterations': 200, 'lns_destruction_frac': 0.30,
        'regret_k': 5, 'shaw_p': 4, 'worst_p': 3,
        
        # Configuración de Visualización y OSMnx
        'plot_dir': './resultados_vrp', 'save_dpi': 150,
        'plot_final_png': True, 'plot_final_street_map': True,
        'use_street_network': True, 'street_weight': 'length', 'unit_label': 'km',
        'pad_km': 8.0, 
        'graphml_path': f'./cache/grafo_osm_{INSTANCE_NAME}.graphml', # <-- CACHÉ DINÁMICA
        'cost_cache_path': f'./cache/matriz_costos_{INSTANCE_NAME}.npy', # <-- CACHÉ DINÁMICA
        'n_jobs_dijkstra': -1,
    }
    
    # 3. Preparar Matriz de Costos (OSMnx o CSV)
    G, snapped, C = None, None, C_csv
    
    # Inicializar tiempos en 0 por si no se usa OSMnx
    tiempos_preproc['grafo'] = 0.0
    tiempos_preproc['snapping'] = 0.0
    tiempos_preproc['matriz'] = 0.0

    if parametros['use_street_network']:
        print("\n🌐 Preparando red vial de OpenStreetMap...")
        try:
            # Cronometrar Construcción Grafo
            t0 = time.time()
            G = build_drive_graph(coords_latlon, parametros['pad_km'], parametros['graphml_path'])
            tiempos_preproc['grafo'] = round(time.time() - t0, 4)
            print(f"   ⏱️ Tiempo grafo: {tiempos_preproc['grafo']}s")
            
            if G: 
                # Cronometrar Snapping
                t0 = time.time()
                snapped = snap_nodes(G, coords_latlon)
                tiempos_preproc['snapping'] = round(time.time() - t0, 4)
                print(f"   ⏱️ Tiempo snapping: {tiempos_preproc['snapping']}s")

            if G and snapped:
                # Cronometrar Matriz
                t0 = time.time()
                C_m = street_cost_matrix_fast(G, snapped, parametros['street_weight'], parametros['cost_cache_path'], parametros['n_jobs_dijkstra'])
                tiempos_preproc['matriz'] = round(time.time() - t0, 4)
                print(f"   ⏱️ Tiempo matriz: {tiempos_preproc['matriz']}s")
                
                if parametros['street_weight'] == 'length': C = C_m / 1000.0; parametros['unit_label'] = 'km'
                elif parametros['street_weight'] == 'travel_time': C = C_m / 60.0; parametros['unit_label'] = 'min'
                else: C = C_m
        except Exception as e:
            print(f"❌ Error con OSMnx: {e}. Se usará la matriz CSV por defecto.")
            G, snapped, C = None, None, C_csv
            parametros['use_street_network'] = False

    validar_matriz_costos(C)
    
    # 4. Ejecutar optimización y medir tiempo
    start_time = time.time()
    costos, mejor_perm, detalle_corridas = run_experiment(C, d, Q, coords_latlon, parametros, G, snapped)
    tiempo_total = time.time() - start_time
    
    # 5. Reporte Final
    print("\n" + "="*60 + "\n📊 RESUMEN FINAL DE LA EJECUCIÓN\n" + "="*60)
    print(f"Mejor costo encontrado: {np.min(costos):.2f} {parametros['unit_label']}")
    print(f"Costo promedio: {np.mean(costos):.2f} {parametros['unit_label']}")
    print(f"Tiempo total de ejecución: {tiempo_total:.2f} segundos")
    print("="*60)
    
    # Se pasa el diccionario de tiempos de preprocesamiento a la función de guardado
    guardar_resultados_excel(costos, mejor_perm, detalle_corridas, C, d, Q, tiempo_total, parametros, tiempos_preproc)


if __name__ == "__main__":
    GLOBAL_SEED = 42
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    main()