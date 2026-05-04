# -*- coding: utf-8 -*-
"""
VRP FUSIONADO "ULTIMATE GRAPHICS": 
- Estructura: 10 Corridas + Estadísticas (Código B).
- Visualización: Geometría real de calles + Flechas (Código A).
- Motor: GA Memético + LNS Completo.
"""

import os
import time
import random
import math
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import folium
from folium import plugins # Importante para las flechas

# ────────────────────────────────────────────────────────────────────────────
# IMPORTACIONES GEOESPACIALES
# ────────────────────────────────────────────────────────────────────────────
try:
    import rasterio
    from rasterio.merge import merge as raster_merge
    from rasterio.transform import rowcol
except ImportError:
    print("❌ ERROR CRÍTICO: Instala rasterio (pip install rasterio)")

import osmnx as ox
from joblib import Parallel, delayed

# ────────────────────────────────────────────────────────────────────────────
# IMPORTACIONES OPTIMIZACIÓN (PYMOO)
# ────────────────────────────────────────────────────────────────────────────
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.core.callback import Callback
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.optimize import minimize
from pymoo.termination import get_termination

# Configuración Global
ox.settings.use_cache = True
ox.settings.timeout = 300
SENTINEL = 9.9e11

# ────────────────────────────────────────────────────────────────────────────
# 1. LECTURA DE DATOS Y GRAFOS
# ────────────────────────────────────────────────────────────────────────────

def cargar_datos_csv(base_path, f_veh, f_dem, f_coord, f_cost):
    try:
        path_v = os.path.join(base_path, f_veh)
        Q = float(pd.read_csv(path_v, header=None).iloc[0, 1])
        path_d = os.path.join(base_path, f_dem)
        d = pd.read_csv(path_d, header=None).iloc[:, 1].values.astype(float)
        path_c = os.path.join(base_path, f_coord)
        # Coordenadas: [Lat, Lon]
        coords = pd.read_csv(path_c, header=None).iloc[:, [2, 1]].values.astype(float)
        path_m = os.path.join(base_path, f_cost)
        if os.path.exists(path_m):
            C = pd.read_csv(path_m, header=None).values / 1000.0
        else:
            C = None
        print(f"✅ Datos Cargados: {len(d)} nodos, Q={Q}")
        return C, d, Q, coords
    except Exception as e:
        print(f"❌ Error leyendo archivos: {e}")
        raise

def procesar_dems(tiff_paths):
    validos = [f for f in tiff_paths if os.path.exists(f)]
    if not validos: return None, None, None
    srcs = [rasterio.open(f) for f in validos]
    mosaic, out_trans = raster_merge(srcs)
    out_meta = srcs[0].meta.copy()
    out_meta.update({"height": mosaic.shape[1], "width": mosaic.shape[2], "transform": out_trans})
    for s in srcs: s.close()
    return mosaic, out_trans, out_meta

def asignar_elevaciones(G, mosaic, trans):
    if mosaic is None: return G
    band = mosaic[0]
    for n, data in G.nodes(data=True):
        try:
            r, c = rowcol(trans, data['x'], data['y'])
            if 0 <= r < band.shape[0] and 0 <= c < band.shape[1]:
                val = band[r, c]
                G.nodes[n]['elevation'] = float(val) if not np.isnan(val) else 0.0
            else: G.nodes[n]['elevation'] = 0.0
        except: G.nodes[n]['elevation'] = 0.0
    return G

def costo_fisico_hibrido(G, alpha=12.0, beta=6.0, max_grade=0.20, min_len=50.0):
    remove_edges = []
    count = 0
    for u, v, k, data in G.edges(keys=True, data=True):
        length_m = data.get('length', 1.0)
        dist_km = length_m / 1000.0
        elev_u = G.nodes[u].get('elevation', 0)
        elev_v = G.nodes[v].get('elevation', 0)
        rise = elev_v - elev_u
        slope = rise / (length_m + 1e-6)
        
        if length_m < min_len and abs(slope) > max_grade: slope = 0.0
        elif abs(slope) > max_grade:
            remove_edges.append((u, v, k))
            continue
            
        if slope >= 0: factor = 1.0 + (slope * alpha)
        else: factor = max(0.6, 1.0 + (slope * beta))
        
        factor = min(5.0, max(0.6, factor))
        data['costo_ajustado'] = dist_km * factor
        data['slope'] = slope
        data['factor'] = factor
        count += 1
        
    G.remove_edges_from(remove_edges)
    print(f"🏔️ Terreno Procesado: {len(remove_edges)} aristas eliminadas, {count} ajustadas.")
    return G

def haversine_dist(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def obtener_matriz_grafos(coords, tiff_list, params):
    center = (float(coords[0,0]), float(coords[0,1]))
    if os.path.exists(params['graph_cache']):
        G = ox.load_graphml(params['graph_cache'])
    else:
        G = ox.graph_from_point(center, dist=params['pad_km']*1000, network_type='drive')
        # Corrección de versión OSMnx
        try:
            G = ox.truncate.largest_component(G, strongly=True)
        except AttributeError:
            try:
                G = ox.utils_graph.get_largest_component(G, strongly=True)
            except AttributeError:
                largest_cc = max(nx.strongly_connected_components(G), key=len)
                G = G.subgraph(largest_cc).copy()
        
        ox.save_graphml(G, params['graph_cache'])
        
    if params['use_dem']:
        mosaic, trans, _ = procesar_dems(tiff_list)
        if mosaic is not None:
            G = asignar_elevaciones(G, mosaic, trans)
            G = costo_fisico_hibrido(G, params['alpha'], params['beta'], params['max_grade'])
            
    snapped = ox.distance.nearest_nodes(G, coords[:,1], coords[:,0])
    weight = 'costo_ajustado' if params['use_dem'] else 'length'
    node_coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in snapped]
    
    def _dijkstra_safe(i):
        source = snapped[i]
        try: dists = nx.single_source_dijkstra_path_length(G, source, weight=weight)
        except: dists = {}
        row = []
        src_lat, src_lon = node_coords[i]
        for j in range(len(snapped)):
            target = snapped[j]
            if i == j: row.append(0.0); continue
            val = dists.get(target, SENTINEL)
            if val >= SENTINEL:
                tgt_lat, tgt_lon = node_coords[j]
                row.append(haversine_dist(src_lat, src_lon, tgt_lat, tgt_lon) * 3.0)
            else: row.append(val)
        return row
        
    print("🔄 Calculando matriz (con reparación automática de islas)...")
    rows = Parallel(n_jobs=-1)(delayed(_dijkstra_safe)(i) for i in range(len(snapped)))
    C = np.array(rows)
    if not params['use_dem']: C = C / 1000.0 
    print(f"✅ Matriz calculada. Max costo: {np.max(C):.2f}")
    return G, snapped, C

# ────────────────────────────────────────────────────────────────────────────
# 3. UTILIDADES VRP Y HEURÍSTICAS
# ────────────────────────────────────────────────────────────────────────────

def reparar_permutacion(perm, n_clientes):
    seen, out = set(), []
    for x in perm:
        xi = int(x)
        if xi != 0 and 1 <= xi <= n_clientes and xi not in seen:
            seen.add(xi); out.append(xi)
    missing = [i for i in range(1, n_clientes + 1) if i not in seen]
    return out + missing

def split_routes(permutation, d, Q):
    routes, current_route, current_load = [], [0], 0.0
    for client in permutation:
        c = int(client); dem = d[c]
        if current_load + dem > Q:
            current_route.append(0); routes.append(current_route)
            current_route = [0, c]; current_load = dem
        else: current_route.append(c); current_load += dem
    current_route.append(0); routes.append(current_route)
    return routes

def costo_total(routes, C):
    return sum(C[r[i], r[i+1]] for r in routes for i in range(len(r)-1))

def cw_savings_init(C, d, Q):
    n = len(d); savings = []
    for i in range(1, n):
        for j in range(i+1, n):
            s_ij = C[i,0] + C[0,j] - C[i,j]
            s_ji = C[j,0] + C[0,i] - C[j,i]
            val = 0.6 * max(s_ij, s_ji) + 0.4 * min(s_ij, s_ji)
            if val > 0: savings.append((val, i, j))
    savings.sort(reverse=True, key=lambda x: x[0])
    routes = {i: [0, i, 0] for i in range(1, n)}
    loads = {i: d[i] for i in range(1, n)}
    for _, i, j in savings:
        ri = rj = None; i_end = j_start = False
        for k, r in routes.items():
            if r and i == r[-2]: ri, i_end = k, True
            if r and j == r[1]: rj, j_start = k, True
        if i_end and j_start and ri != rj:
            if loads[ri] + loads[rj] <= Q:
                routes[ri] = routes[ri][:-1] + routes[rj][1:]
                loads[ri] += loads[rj]; routes[rj] = None
    final = [x for r in routes.values() if r for x in r if x != 0]
    return reparar_permutacion(final, n-1)

def sweep_init(coords, n_clientes):
    depot = coords[0]
    angles = []
    for i in range(1, n_clientes+1):
        ang = math.atan2(coords[i,0]-depot[0], coords[i,1]-depot[1])
        angles.append((i, ang))
    angles.sort(key=lambda x: x[1])
    return [x[0] for x in angles]

# ────────────────────────────────────────────────────────────────────────────
# 4. LNS (OPERADORES DE OPTIMIZACIÓN)
# ────────────────────────────────────────────────────────────────────────────

def two_opt_intra(route, C):
    best = list(route); improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                if j - i < 1: continue
                new_r = best[:i] + best[i:j+1][::-1] + best[j+1:]
                old_c = sum(C[best[k], best[k+1]] for k in range(i-1, j+1))
                new_c = sum(C[new_r[k], new_r[k+1]] for k in range(i-1, j+1))
                if new_c < old_c - 1e-9:
                    best = new_r; improved = True; break
            if improved: break
    return best

def shaw_removal(routes, C, d, q, p=6):
    clients = [c for r in routes for c in r if c!=0]
    if not clients: return []
    seed = random.choice(clients)
    removed = [seed]; pool = set(clients); pool.discard(seed)
    max_c = np.max(C) if np.max(C) > 0 else 1
    max_d = np.max(d) if np.max(d) > 0 else 1
    while len(removed) < q and pool:
        ref = random.choice(removed)
        ranks = []
        for c in pool:
            rel = (C[ref,c]/max_c) + (abs(d[ref]-d[c])/max_d)
            ranks.append((c, rel))
        ranks.sort(key=lambda x: x[1])
        idx = int(len(ranks) * (random.random()**p))
        chosen = ranks[idx][0]
        removed.append(chosen); pool.remove(chosen)
    return removed

def worst_removal(routes, C, q, p=3):
    scores = []
    for r in routes:
        for i in range(1, len(r)-1):
            c = r[i]; prev = r[i-1]; nxt = r[i+1]
            saving = (C[prev,c] + C[c,nxt]) - C[prev,nxt]
            scores.append((c, saving))
    scores.sort(key=lambda x: x[1], reverse=True)
    removed = []
    while len(removed) < q and scores:
        idx = int(len(scores) * (random.random()**p))
        chosen = scores.pop(idx)[0]; removed.append(chosen)
    return removed

def regret_insertion(routes, removed, C, d, Q, k=2):
    rutas = [list(r) for r in routes]; unassigned = set(removed)
    while unassigned:
        candidates = []
        for c in unassigned:
            insertions = []
            for r_idx, r in enumerate(rutas):
                load = sum(d[x] for x in r if x!=0)
                if load + d[c] > Q: continue
                for pos in range(1, len(r)):
                    cost = C[r[pos-1], c] + C[c, r[pos]] - C[r[pos-1], r[pos]]
                    insertions.append({'cost': cost, 'r': r_idx, 'p': pos})
            insertions.append({'cost': C[0,c]+C[c,0], 'r': -1, 'p': -1})
            insertions.sort(key=lambda x: x['cost'])
            if not insertions: continue
            regret = (insertions[k-1]['cost'] - insertions[0]['cost']) if len(insertions) >= k else (insertions[-1]['cost'] - insertions[0]['cost'])
            candidates.append((regret, c, insertions[0]))
        if not candidates: break
        candidates.sort(key=lambda x: x[0], reverse=True)
        _, best_c, best_ins = candidates[0]
        if best_ins['r'] == -1: rutas.append([0, best_c, 0])
        else: rutas[best_ins['r']].insert(best_ins['p'], best_c)
        unassigned.remove(best_c)
    return rutas

def large_neighborhood_search_full(perm, C, d, Q, iterations):
    best_routes = split_routes(perm, d, Q)
    best_cost = costo_total(best_routes, C)
    curr_routes = best_routes
    n_clients = len(d)-1
    for i in range(iterations):
        q = random.randint(4, min(30, int(n_clients*0.4)))
        if random.random() < 0.5: removed = shaw_removal(curr_routes, C, d, q)
        else: removed = worst_removal(curr_routes, C, q)
        partial = [ [x for x in r if x not in set(removed)] for r in curr_routes]
        partial = [r for r in partial if len(r)>2]
        new_r = regret_insertion(partial, removed, C, d, Q)
        new_r = [two_opt_intra(r, C) for r in new_r]
        new_c = costo_total(new_r, C)
        if new_c < best_cost: best_cost = new_c; best_routes = new_r; curr_routes = new_r
        else:
            if random.random() < 0.05: curr_routes = new_r
    final_perm = [c for r in best_routes for c in r if c!=0]
    return reparar_permutacion(final_perm, n_clients), best_cost

# ────────────────────────────────────────────────────────────────────────────
# 5. GA CLASES
# ────────────────────────────────────────────────────────────────────────────

class AVRPProblem(ElementwiseProblem):
    def __init__(self, C, d, Q):
        super().__init__(n_var=len(d)-1, n_obj=1, n_constr=0)
        self.C, self.d, self.Q = C, d, Q
    def _evaluate(self, x, out, *args, **kwargs):
        perm = reparar_permutacion(list(map(int, x)), self.n_var)
        routes = split_routes(perm, self.d, self.Q)
        out["F"] = costo_total(routes, self.C)

class HybridPopulation(Sampling):
    def __init__(self, C, d, Q, coords, n_cw, n_sweep):
        super().__init__(); self.C, self.d, self.Q, self.coords = C, d, Q, coords; self.n_cw, self.n_sweep = n_cw, n_sweep
    def _do(self, problem, n_samples, **kwargs):
        X = np.zeros((n_samples, problem.n_var), dtype=int)
        cw_sol = np.array(cw_savings_init(self.C, self.d, self.Q))
        sw_sol = np.array(reparar_permutacion(sweep_init(self.coords, problem.n_var), problem.n_var))
        for i in range(n_samples):
            if i < self.n_cw: X[i] = cw_sol
            elif i < self.n_cw + self.n_sweep: X[i] = sw_sol
            else: X[i] = np.random.permutation(np.arange(1, problem.n_var+1))
        return X

class MemeticCallback(Callback):
    def __init__(self, C, d, Q, freq, elite_frac):
        super().__init__(); self.C, self.d, self.Q = C, d, Q; self.freq, self.elite = freq, elite_frac
    def notify(self, algo):
        if algo.n_gen > 1 and algo.n_gen % self.freq == 0:
            pop = algo.pop; n_elite = int(len(pop) * self.elite)
            I = np.argsort([ind.F[0] for ind in pop])[:n_elite]
            for i in I:
                perm = pop[i].X.astype(int)
                new_perm, _ = large_neighborhood_search_full(perm, self.C, self.d, self.Q, 15)
                pop[i].X = np.array(new_perm)
            algo.evaluator.eval(algo.problem, pop)

# ────────────────────────────────────────────────────────────────────────────
# 6. REPORTES Y VISUALIZACIÓN MEJORADA (ESTILO CÓDIGO A)
# ────────────────────────────────────────────────────────────────────────────

def generar_excel_detallado(rutas, C, d, Q, G, snapped, params, filename, historial_corridas=None):
    """Genera Excel con hojas de Resumen, Estadísticas y Detalle de Terreno"""
    print(f"📝 Generando reporte detallado: {filename}")
    data_rutas = []
    for idx, r in enumerate(rutas, 1):
        cost = sum(C[r[i], r[i+1]] for i in range(len(r)-1))
        load = sum(d[c] for c in r if c!=0)
        data_rutas.append({'Ruta': idx, 'Secuencia': str(r), 'Paradas': len(r)-2, 'Carga': load, 'Ocupacion %': round(load/Q*100, 1), 'Costo': round(cost, 2)})
    
    data_edges = []
    if G is not None and snapped is not None:
        for r_idx, r in enumerate(rutas, 1):
            for i in range(len(r)-1):
                try:
                    u_idx, v_idx = int(r[i]), int(r[i+1])
                    if u_idx >= len(snapped) or v_idx >= len(snapped): continue
                    u, v = snapped[u_idx], snapped[v_idx]
                    path = nx.shortest_path(G, u, v, weight='costo_ajustado')
                    for k in range(len(path)-1):
                        n1, n2 = path[k], path[k+1]
                        edge_data = G.get_edge_data(n1, n2)
                        if edge_data:
                            vals = min(edge_data.values(), key=lambda x: x.get('costo_ajustado', 999))
                            data_edges.append({
                                'Ruta': r_idx, 'Cliente_Origen': r[i], 'Cliente_Destino': r[i+1],
                                'Distancia_m': round(vals.get('length', 0), 2),
                                'Pendiente %': round(vals.get('slope', 0)*100, 2),
                                'Factor': round(vals.get('factor', 1.0), 3),
                                'Costo_Ajustado': round(vals.get('costo_ajustado', 0), 4)
                            })
                except: pass

    with pd.ExcelWriter(filename) as writer:
        pd.DataFrame(data_rutas).to_excel(writer, sheet_name='Mejor_Solucion', index=False)
        if historial_corridas:
            df_stats = pd.DataFrame(historial_corridas)
            avg_row = {'Corrida': 'PROMEDIO', 'Costo Final': df_stats['Costo Final'].mean(), 'Tiempo Total (s)': df_stats['Tiempo Total (s)'].mean()}
            df_stats = pd.concat([df_stats, pd.DataFrame([avg_row])], ignore_index=True)
            df_stats.to_excel(writer, sheet_name='Estadisticas', index=False)
        if data_edges: pd.DataFrame(data_edges).to_excel(writer, sheet_name='Detalle_Terreno', index=False)
        pd.DataFrame([params]).to_excel(writer, sheet_name='Parametros', index=False)

def plot_routes_png(routes, nodes, instance_name, output_path=None, dpi=150):
    """
    Visualiza rutas de VRP en un gráfico limpio y profesional (Estilo Código A).
    """
    # 1. Preparar Coordenadas (Numpy [Lat, Lon] -> Plot [Lon, Lat])
    # nodes tiene [Lat, Lon]. Matplotlib espera [X, Y] = [Lon, Lat]
    X, Y = nodes[:, 1], nodes[:, 0]

    # 2. Configurar colores
    cmap = plt.colormaps.get_cmap('tab20').resampled(len(routes))
    norm = mcolors.Normalize(vmin=0, vmax=len(routes) - 1)

    # 3. Configurar figura
    plt.figure(figsize=(10, 10))
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.gca().set_facecolor('white')

    # 4. Dibujar rutas
    for i, route in enumerate(routes):
        color = cmap(norm(i))
        route = np.array(route, dtype=int)
        
        # Extraer puntos
        rx, ry = X[route], Y[route]

        plt.plot(rx, ry, color=color, linewidth=2.5, alpha=0.9, label=f'Route {i+1}')
        plt.scatter(rx, ry, color=color, s=25, alpha=0.9)

        # Etiquetas
        for j, (xi, yi) in enumerate(zip(rx, ry)):
            if route[j] != 0: # No etiquetar depósito multiples veces
                plt.text(xi, yi, str(route[j]), fontsize=8, color='black',
                         ha='center', va='center', weight='bold',
                         bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, boxstyle='round,pad=0.2'))

    # 5. Destacar Depósito
    plt.scatter(X[0], Y[0], c='black', s=150, marker='*', label='Deposit', zorder=10)

    # 6. Estilos finales
    plt.title(f"Routes- {instance_name}", fontsize=14, fontweight='bold')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend(loc='best', fontsize=9, frameon=True)
    plt.tight_layout()

    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"🗺️ Mapa PNG guardado: {output_path}")

def plot_routes_map_streets(rutas, coords_latlon, G, snapped_nodes, weight, titulo, fname):
    """
    Genera mapa interactivo HTML con geometría real de calles y flechas (Estilo Código A).
    """
    # Centrar mapa
    m = folium.Map(location=[coords_latlon[0, 0], coords_latlon[0, 1]], zoom_start=13, tiles="OpenStreetMap")
    folium.Marker([coords_latlon[0, 0], coords_latlon[0, 1]], icon=folium.Icon(color="red", icon="home"), tooltip="Deposit").add_to(m)
    
    # Obtener geometría de aristas para trazado real
    gdf_edges = ox.graph_to_gdfs(G, nodes=False)
    
    colores = list(mcolors.TABLEAU_COLORS.values())
    if len(colores) < len(rutas):
        random.seed(42)
        colores += [f"#{random.randint(0, 0xFFFFFF):06x}" for _ in range(len(rutas)-len(colores))]
    
    for ridx, ruta in enumerate(rutas):
        if len(ruta) < 2: continue
        capa = folium.FeatureGroup(name=f"Ruta {ridx+1}")
        color = colores[ridx % len(colores)]
        
        puntos_ruta_completa = []
        
        for k in range(len(ruta)-1):
            u, v = snapped_nodes[ruta[k]], snapped_nodes[ruta[k+1]]
            try:
                # Camino de nodos
                path_nodes = nx.shortest_path(G, u, v, weight=weight)
                
                # Extraer geometría real de cada tramo
                for i in range(len(path_nodes) - 1):
                    orig, dest = path_nodes[i], path_nodes[i+1]
                    # Acceder al GeoDataFrame (maneja multigrafo con key=0)
                    try:
                        edge_geom = gdf_edges.loc[(orig, dest, 0)].geometry
                        # Convertir geometría a lista de puntos
                        puntos = list(edge_geom.coords)
                        puntos_ruta_completa.extend(puntos)
                    except KeyError:
                        # Fallback si no encuentra la key 0 (raro pero posible)
                        pass
                        
            except nx.NetworkXNoPath:
                continue
        
        if puntos_ruta_completa:
            # Folium necesita (Lat, Lon), OSMnx da (Lon, Lat). Invertimos.
            puntos_folium = [(y, x) for x, y in puntos_ruta_completa]
            
            # Línea Principal
            line = folium.PolyLine(puntos_folium, color=color, weight=5, opacity=0.8, tooltip=f"Ruta {ridx+1}")
            capa.add_child(line)
            
            # Flechas de Dirección
            try:
                plugins.PolyLineTextPath(
                    line, "▶", repeat=True, offset=8, attributes={'fill': color, 'font-weight': 'bold', 'font-size': '18'}
                ).add_to(capa)
            except: pass

        # Marcadores de clientes
        for n in ruta:
            if n != 0:
                folium.CircleMarker([coords_latlon[n, 0], coords_latlon[n, 1]], radius=6, color='white', fill_color=color, fill_opacity=1.0, tooltip=f"Cliente {n}").add_to(capa)
        
        capa.add_to(m)
    
    folium.LayerControl(collapsed=False).add_to(m)
    Path(fname).parent.mkdir(parents=True, exist_ok=True)
    m.save(fname)
    print(f"🗺️ Mapa HTML guardado: {fname}")

# ────────────────────────────────────────────────────────────────────────────
# 7. MAIN (CON BUCLE DE 10 CORRIDAS)
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    
    # --- CONFIGURACIÓN USUARIO ---
    BASE_DIR = r'C:\Users\arestrepo\Desktop\doc_amrf\Algoritmos\Prueba instancias asimetricas SLA\ACVRP Benchmark Instances\ACVRP Benchmark Instances\02. SLAM250'
    FILES = {
        'v': 'SLAM250_Vehicle_V1.csv',
        'd': 'SLAM250_Volume_V1M20.csv',
        'c': 'SLAM250_Coordinates.csv',
        'm': 'SLAM250_Cost_Distance.csv'
    }
    TIFFS = [
        os.path.join(BASE_DIR, 'n37_e126_1arc_v3.tif'),
        os.path.join(BASE_DIR, 'n37_e127_1arc_v3.tif')
    ]
    
    PARAMS = {
        'n_runs': 10,
        'use_dem': True,
        'pad_km': 6.0,
        'alpha': 10.0, 'beta': 5.0,
        'max_grade': 0.30,
        'pop': 100, 'gen': 200, 'elite': 0.15,
        'lns_final_iter': 100,
        'instance_name': "SLAM250_DV1M20_v2" # Cambia esto en cada corrida
    }
    # La caché ahora es dinámica:
    PARAMS['graph_cache'] = f"./cache/graph_{PARAMS['instance_name']}.graphml"
    
    print(f"🚀 INICIANDO VRP ULTIMATE ({PARAMS['n_runs']} Corridas)...")
    
    # 1. Cargar Datos y Procesar Terreno
    C_csv, d, Q, coords = cargar_datos_csv(BASE_DIR, FILES['v'], FILES['d'], FILES['c'], FILES['m'])
    G, snapped, C_calc = obtener_matriz_grafos(coords, TIFFS, PARAMS)
    C_final = C_calc if C_calc is not None else C_csv
    
    historial = []
    mejor_costo_global = float('inf')
    mejor_perm_global = None
    
    # 2. Bucle de Optimización
    for run in range(1, PARAMS['n_runs'] + 1):
        print(f"\n--- Corrida {run}/{PARAMS['n_runs']} ---")
        seed_run = 42 + run 
        t_inicio = time.time()
        
        prob = AVRPProblem(C_final, d, Q)
        algo = GA(
            pop_size=PARAMS['pop'],
            sampling=HybridPopulation(C_final, d, Q, coords, n_cw=15, n_sweep=15),
            crossover=OrderCrossover(prob=0.85),
            mutation=InversionMutation(prob=0.15),
            eliminate_duplicates=True,
            callback=MemeticCallback(C_final, d, Q, freq=20, elite_frac=PARAMS['elite'])
        )
        
        res = minimize(prob, algo, get_termination("n_gen", PARAMS['gen']), seed=seed_run, verbose=False)
        t_ga = time.time() - t_inicio
        costo_ga = res.F[0]
        
        t_lns_start = time.time()
        best_perm_ga = list(map(int, res.X))
        best_perm_run, costo_run = large_neighborhood_search_full(best_perm_ga, C_final, d, Q, PARAMS['lns_final_iter'])
        t_lns = time.time() - t_lns_start
        
        tiempo_total = t_ga + t_lns
        print(f"   Resultado Run {run}: Costo GA={costo_ga:.2f} -> Final={costo_run:.2f} (Tiempo: {tiempo_total:.2f}s)")
        
        historial.append({'Corrida': run, 'Costo GA': costo_ga, 'Costo Final': costo_run, 'Mejora LNS': costo_ga - costo_run, 'Tiempo GA (s)': round(t_ga, 2), 'Tiempo LNS (s)': round(t_lns, 2), 'Tiempo Total (s)': round(tiempo_total, 2)})
        
        if costo_run < mejor_costo_global:
            mejor_costo_global = costo_run
            mejor_perm_global = best_perm_run
            print("   🌟 ¡Nuevo mejor global encontrado!")

    print(f"\n🏆 MEJOR COSTO GLOBAL: {mejor_costo_global:.4f}")
    
    # 3. Salidas
    out_folder = Path("Resultados_Ultimate_Final")
    out_folder.mkdir(exist_ok=True)
    rutas_finales = split_routes(mejor_perm_global, d, Q)
    
    # Excel
    generar_excel_detallado(rutas_finales, C_final, d, Q, G, snapped, PARAMS, out_folder / "Reporte_Multicorrida_SLAM250_DV1M20_v2.xlsx", historial_corridas=historial)
    
    # Mapa PNG (Estilo A mejorado)
    plot_routes_png(rutas_finales, coords, PARAMS['instance_name'], str(out_folder / "Mejor_Mapa_SLAM250_DV1M20_2.png"))
    
    # Mapa HTML (Estilo A con geometría real y flechas)
    if G is not None and snapped is not None:
        plot_routes_map_streets(rutas_finales, coords, G, snapped, 'costo_ajustado', PARAMS['instance_name'], str(out_folder / "Mejor_Mapa_Interactivo_SLAM250_DV1M20_v2.html"))

    print("✅ PROCESO COMPLETADO.")