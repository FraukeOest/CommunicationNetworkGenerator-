import networkx as nx
from PhysGraph import PhysGraph
import numpy as np
import pickle
import time
from scipy.spatial import cKDTree
from re import search, sub
import logging
import pandapower.topology as top
import simbench as sb
import json
import pandas as pd
import networkx as nx
import pandapower.plotting as ppplot
from datetime import datetime as dt
logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                                  '%Y-%m-%d:%H:%M:%S')
logger.setLevel(logging.DEBUG)
now_dt = dt.now()
formatted_dt = now_dt.strftime("%Y-%m-%d_%H_%M")
now = time.perf_counter()
file_handler = logging.FileHandler(f'log/log_{formatted_dt}.log', 'w')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def _fix_graph_for_all_cycles(MG: nx.Graph):
    """some edges might be wrongly generaged and need to readded to the graph
        Params:
            mg: Modelled Graph from pandapower-based networks
        Returns:
            G: PhysGraph()
    """
    G = PhysGraph()
    edgesMG = list(MG.edges(data=False))

    ctr = 0
    for edge in edgesMG:
        G.add_edge(str(edge[0]), str(edge[1]))
        try:
            cycle_edges = list(nx.minimum_cycle_basis(G))
            ctr += 1
        except Exception as e:
            print(f"failed at {edge} number {ctr}")
            print(f"Typen: {type(edge[0])} -- {type(edge[1])}")
            G.remove_edge(edge[0], edge[1])
    isolates = list(nx.isolates(MG))
    if len(isolates) >= 1:
        raise ValueError("There should not be any isolated nodes left ")
    print(f"isolates: {isolates}")
    G.add_nodes_from(isolates)
    pos = nx.get_node_attributes(MG, 'pos')
    p_mw = nx.get_node_attributes(MG, 'p_mw')
    nx.set_node_attributes(G, values=pos, name='pos')
    nx.set_node_attributes(G, values=p_mw, name='p_mw')

    if len(list(G.nodes)) == 0:
        raise ValueError("fixing failed, no nodes in graph")
    return G



def _remove_middle_compoents(graph: PhysGraph):
    """removes parts that are connected in the power grid line that are connected differently in the communication
    network (star-like)"""


    if not list(graph.nodes):
        raise ValueError("no nodes in Graph to be remodelled")
    switches = [n for n in graph.nodes if 'switch' in n]
    for n in switches:
        neighbor = list(graph.neighbors(n))
        #print(neighbor)
        graph.add_edge(neighbor[0], neighbor[1])
        if "Trafo" in neighbor[1]:
            print("found trafo")
            graph.remove_edge(n, neighbor[1])
        else:
            graph.remove_edge(n, neighbor[0])

    # busses = [n for n in graph.nodes if "Bus" in n]
    # for n in busses:
    #     neighbor = list(graph.neighbors(n))
    #     if len(neighbor) > 1:
    #         if "Bus" in neighbor[0] and "Bus" in neighbor[1]:
    #             nx.contracted_edge(graph, (neighbor[0], n), self_loops=False, copy=False)
    MV_Load = [n for n in graph.nodes if "MV" in n and "Load" in n]
    for n in MV_Load:
        neighbor = list(graph.neighbors(n))
        if len(neighbor) > 1:
            if "Bus" in neighbor[0] and "Bus" in neighbor[1]:
                print(neighbor)
    degrees = [(n, d) for n, d in nx.degree(graph) if not "Bus" in n and d > 1]
    if degrees:
        for n, d in degrees:
            bus_node = None
            H = graph.copy()
            neighbors = nx.neighbors(H, n)
            if "Load" in n:
                for nei in neighbors:
                    if "Bus" in nei:
                        bus_node = nei
                    else:
                        graph.remove_edge(n, nei)
                if bus_node:
                    for nei in neighbors:
                        graph.add_edge(nei, bus_node)

    Trafo = [n for n in graph.nodes if 'Trafo' in n]
    for n in Trafo:
        neighbor = [n for n in graph.neighbors(n) if "Trafo" not in n]
        graph.add_edge(neighbor[0], neighbor[1])
        graph.remove_edge(n, neighbor[0])

    lv_busses = [n for n in graph.nodes if "bus" in n or ("Bus" in n and "LV" in n)]
    for n in lv_busses:
        graph.remove_node(n)

    degrees4 = [(n, d) for n, d in nx.degree(graph) if "Bus" in n and d <= 1]
    while degrees4:
        graph.remove_nodes_from(degrees4)
        degrees4 = [n for n, d in graph.degree() if "Bus" in str(n) and d <= 1]

    router = [n for n in graph.nodes() if "busbar" in n]
    pos = nx.get_node_attributes(graph, 'pos')
    isolates = list(nx.isolates(graph))
    coords_router = np.array([pos[n] for n in router])
    kdtree = cKDTree(coords_router)
    coords_iso = np.array([pos[n] for n in isolates])
    if len(coords_iso) > 1:
        dists, idxs = kdtree.query(coords_iso, k=1)
        for iso_node, connected_index in zip(isolates, idxs):
            target = router[connected_index]
            graph.add_edge(iso_node, target, weight=10, lat=20)

    degrees = [(n, d) for n, d in nx.degree(graph) if not "Bus" in n and d > 1]
    logger.info("finished fixing")
    if degrees:
        logger.error(f"didnt delete all irrevant nodes that could act as router (but aren't): {degrees}")
        raise ValueError("didnt delete all irrevant nodes that could act as router (but aren't)")
    return graph


def _rename_components(G):
    """renames former pandapower components to make easier distinguising between MV and LV, and to unify naming pattern"""

    nodes = G.nodes

    if ('Residential' in nodes) and ('CHP diesel' in nodes) and ('Fuel' in nodes):
        mapping = {n: n.replace('Residential ', 'LV_CHP_') for n in G.nodes if 'Residential' in n}
        G = nx.relabel_nodes(G, mapping)
        mapping = {n: n.replace('Load R', 'MV_Load_') for n in G.nodes if 'Load R' in n}
        G = nx.relabel_nodes(G, mapping)
        mapping = {n: n.replace('Load', 'MV_Load') for n in G.nodes if n.startswith('Load')}
        G = nx.relabel_nodes(G, mapping)
        mapping = {n: n.replace('PV', 'MV_PV') for n in G.nodes if 'PV' in n}
        G = nx.relabel_nodes(G, mapping)
        mapping = {n: n.replace('Battery ', 'MV_Bat_') for n in G.nodes if 'Battery' in n}
        G = nx.relabel_nodes(G, mapping)
        mapping = {n: n.replace('load', 'LV_Load_') for n in G.nodes if 'load' in n}
        G = nx.relabel_nodes(G, mapping)
        mapping = {n: n.replace('fuel cell ', 'LV_CHP_') for n in G.nodes if 'fuel' in n}
        G = nx.relabel_nodes(G, mapping)
        mapping = {n: n.replace('Fuel cell ', 'MV_CHP_') for n in G.nodes if 'Fuel' in n}
        G = nx.relabel_nodes(G, mapping)
        mapping = {n: n.replace('CHP diesel ', 'LV_CHP_') for n in G.nodes if 'CHP diesel' in n}
        G = nx.relabel_nodes(G, mapping)
        mapping = {n: n.replace('gen', 'LV_PV_') for n in G.nodes if 'gen' in n}
        G = nx.relabel_nodes(G, mapping)
        mapping = {n: n.replace('Bus ', 'R') for n in G.nodes if 'Bus' in n}
        G = nx.relabel_nodes(G, mapping)
        mapping = {n: n.replace('Trafo', 'HVMV_Trafo') for n in G.nodes if 'Trafo' in n}
        G = nx.relabel_nodes(G, mapping)
        mapping = {n: n.replace('trafo', 'MVLV_trafo') for n in G.nodes if 'trafo' in n}
        G = nx.relabel_nodes(G, mapping)
    else:
        mapping = {
            n: sub(r'(LV\d+)\.(\d+)\s+Bus\s*(\d+)', r'R\2\3_0', n)
            for n in G.nodes()
            if 'Bus' in n
        }
        G = nx.relabel_nodes(G, mapping)

        mapping = {
            n: sub(r'(MV\d+)\.(\d+)\s+Bus\s*(\d+)', r'R\2\3_1', n)
            for n in G.nodes()
            if 'Bus' in n
        }
        G = nx.relabel_nodes(G, mapping)

        mapping = {
            n: sub(r'HV(\d+)\s+Bus\s*(\d+)', r'R\2_2', n)
            for n in G.nodes()
            if 'Bus' in n
        }
        G = nx.relabel_nodes(G, mapping)

        mapping = {
            n: sub(r'LV(\d+)\.(\d+)\s+SGen\s*(\d+)', r'LV_PV_\1\2\3', n)
            for n in G.nodes
            if "SGen" in n
        }
        G = nx.relabel_nodes(G, mapping)
        mapping = {n: sub(r'MV(\d+)\.(\d+)-(LV)(\d+)\.(\d+)-Trafo\s*(\d*)', r'MVLV_trafo_\2\4\5', n)
                   for n in G.nodes
                   if "Trafo" in n and "LV" in n
                   }
        G = nx.relabel_nodes(G, mapping)

        degrees4 = [(n, d) for n, d in nx.degree(G) if not "R" in n and d > 1]
        if degrees4:
            raise ValueError(f"router is end device L 307 {degrees4}")


        mapping = {n: sub(r'(HV)(\d+)-(MV)(\d+)\.(\d+)-Trafo\s*(\d*)', r'\1\3_Trafo_\2\5', n)
                   for n in G.nodes
                   if "Trafo" in n and "HV" in n
                   }
        G = nx.relabel_nodes(G, mapping)

        mapping = {n: sub(r'MV(\d+)\.(\d+) MV\s*Load\s*(\d+)', r'MV_Load_\1\2\3_1', n)
                   for n in G.nodes
                   if "Load" in n and "MV" in n
                   }
        G = nx.relabel_nodes(G, mapping)

        mapping = {n: sub(r'MV(\d+)\.(\d+) \s*Load\s*(\d+)', r'MV_Load_\1\2\3', n)
                   for n in G.nodes
                   if "Load" in n and "MV" in n
                   }
        G = nx.relabel_nodes(G, mapping)

        mapping = {n: sub(r'LV(\d+)\.(\d+)\s*Load\s*(\d+)', r'LV_Load_\1\2\3', n)
                   for n in G.nodes
                   if "Load" in n and "LV" in n
                   }
        G = nx.relabel_nodes(G, mapping)

        mapping = {n: sub(r'MV(\d+)\.(\d+) (MV)*\s*SGen\s*(\d+)', r'MV_PV_\1\2\4_\3', n)
                   for n in G.nodes
                   if "SGen" in n and "MV" in n
                   }
        G = nx.relabel_nodes(G, mapping)

        mapping = {n: sub(r'MV\d+\.\d+ loop_line_switch (\d)\.(\d)', r'switch_\1\2', n)
                   for n in G.nodes
                   if 'switch' in n
                   }
        G = nx.relabel_nodes(G, mapping)

        mapping = {n: sub(r'HV(\d+)_MV(\d*).(\d+)_load', r'MV_Load_\1\2\3_2', n)
                   for n in G.nodes
                   if 'load' in n and "HV" in n and "MV" in n
                   }

        G = nx.relabel_nodes(G, mapping)

    accepted_keys = ['HVMV_Trafo', 'MVLV_trafo', 'switch', 'MV_Bat', 'MV_Load', 'LV_CHP_', 'MV_CHP', 'R', 'MV_PV', 'LV_Load',
                     'WKA','LV_PV', 'S']
    accepted = []
    for k in G.nodes:
        for ak in accepted_keys:
            if k.startswith(ak):
                accepted.append(k)
    if len(accepted) < len(list(G.nodes)):
        dif = set(list(G.nodes)) - set(accepted)
        raise KeyError(f"missed components to consider renaming by {dif} ")


    return G


def repairing_graph(MG):
    G = PhysGraph(MG)
    components = list(nx.connected_components(G))  # Liste von Sets mit Knoten
    # try:
    #     logger.info("test cycle edges")
    #     cycle_edges = list(nx.minimum_cycle_basis(MG))
    #     G = PhysGraph(MG)
    # except:
    #     logger.info("fix graph")
    #     G = _fix_graph_for_all_cycles(MG)
    if not nx.is_connected(G):
        raise nx.NetworkXError("Graph is not connected after fixing")
    G.remove_nodes_from(['S1', 'S2', 'S3'])  # these are not servers, but names for switches
    logger.info("remove middle components")
    G = _remove_middle_compoents(G)
    if not nx.is_connected(G):
        raise nx.NetworkXError("Graph is not connected after removing middle components")
    #degrees2 = [(n, d) for n, d in nx.degree(G) if d > 1]
    G = _rename_components(G)
    if not nx.is_connected(G):
        raise nx.NetworkXError("Graph is not connected after renaming components")
    degrees3 = [(n, d) for n, d in nx.degree(G) if not "R" in n and d > 1]
    if degrees3:
        raise ValueError(f"OT device pretending to be a router {degrees3}")
    degrees4 = [(n, d) for n, d in nx.degree(G) if "R" in n and d <= 1]
    if degrees4:
        raise ValueError(f"router is end device {degrees4}")
    return G


def determine_smallest_grid():
    # Alle kombinierten MV+LV-Netze (MVLV) holen
    codes = sb.collect_all_simbench_codes(
        hv_level="MV",
        lv_level="LV",
        scenario=0,          # Szenario fixieren (Bus-Anzahl bleibt i.d.R. gleich)
        breaker_rep=None,    # beide Varianten zulassen (sw / no_sw)
        all_data=True
    )

    best = None  # (n_buses, code)

    for code in codes:
        net = sb.get_simbench_net(code)
        n_buses = len(net.bus)
        if best is None or n_buses < best[0]:
            best = (n_buses, code)

    print("Kleinstes MV+LV-Netz:", best[1])
    print("Anzahl Busse:", best[0])


    #Kleinstes MV+LV-Netz: 1-MVLV-rural-1.108-0-no_sw
    #Anzahl Busse: 109

def voltage_level(vn_kv: float) -> str:
    if vn_kv < 1.0:
        return "LV"          # Niederspannung
    elif vn_kv < 60.0:
        return "MV"          # Mittelspannung
    else:
        return "HV"          # Hochspannung


def power(G, net):
    def sum_pq_by_bus(df: pd.DataFrame, bus_col="bus", p_col="p_mw", q_col="q_mvar"):
        if df is None or len(df) == 0 or bus_col not in df.columns:
            return pd.Series(dtype=float), pd.Series(dtype=float)

        d = df.copy()
        if "in_service" in d.columns:
            d = d[d["in_service"].astype(bool)]

        p = d.groupby(bus_col)[p_col].sum() if p_col in d.columns else pd.Series(dtype=float)
        q = d.groupby(bus_col)[q_col].sum() if q_col in d.columns else pd.Series(dtype=float)
        return p, q

    p_load, q_load = sum_pq_by_bus(net.load)
    p_sgen, q_sgen = sum_pq_by_bus(net.sgen)
    p_gen, q_gen = sum_pq_by_bus(net.gen)
    p_sto, q_sto = sum_pq_by_bus(getattr(net, "storage", pd.DataFrame()))

    attrs = {}
    for b in G.nodes:
        pl = float(p_load.get(b, 0.0));
        ql = float(q_load.get(b, 0.0))
        pg = float(p_gen.get(b, 0.0));
        qg = float(q_gen.get(b, 0.0))
        ps = float(p_sgen.get(b, 0.0));
        qs = float(q_sgen.get(b, 0.0))
        pst = float(p_sto.get(b, 0.0));
        qst = float(q_sto.get(b, 0.0))

        # Netto-Injektion: Erzeugung minus Last (Storage hier als "Erzeuger" mit Vorzeichen aus Tabelle)
        p_net = (pg + ps + pst) - pl
        q_net = (qg + qs + qst) - ql

        attrs[b] = {
            "p_load_mw": pl, "q_load_mvar": ql,
            "p_gen_mw": pg, "q_gen_mvar": qg,
            "p_sgen_mw": ps, "q_sgen_mvar": qs,
            "p_net_mw": p_net, "q_net_mvar": q_net,
        }

    nx.set_node_attributes(G, attrs)
    return G


def determine_type(G, net):
    bus_ids = list(G.nodes)
    bus_df = net.bus.loc[bus_ids, ["name", "vn_kv", "zone"]]
    # print(bus_df.head())
    mapping = {i: f"{net.bus.at[i, 'name']} ({i})" for i in net.bus.index}
    types = {bus: "bus" for bus in G.nodes}

    for bus in net.ext_grid["bus"].tolist():
        types[bus] = "slack"

    for bus in net.gen["bus"].tolist():
        types[bus] = "gen"

    for bus in net.sgen["bus"].tolist():
        types[bus] = "sgen"
    for bus in net.storage["bus"].tolist():
        types[bus] = "storage"
    for bus in net.load["bus"].tolist():
        types[bus] = "load" if types.get(bus) == "bus" else types[bus] + "+load"

    level = {bus: voltage_level(net.bus.at[bus, "vn_kv"]) for bus in G.nodes}
    nx.set_node_attributes(G, level, name="level")

    nx.set_node_attributes(G, {b: (b in set(net.trafo["hv_bus"])) for b in G.nodes}, name="hv_trafo")
    nx.set_node_attributes(G, {b: (b in set(net.trafo["lv_bus"])) for b in G.nodes}, name="lv_trafo")
    nx.set_node_attributes(G, types, name="type")
    # nx.set_node_attributes(G, net.bus["type"].to_dict(), name="type")
    MG = nx.relabel_nodes(G, mapping, copy=True)

    return MG


def determine_pos(G, net):
    # 1) Falls keine Geoda, ten vorhanden sind: generische Koordinaten erzeugen
    if "geo" not in net.bus.columns or net.bus["geo"].isna().all():
        ppplot.create_generic_coordinates(net, geodata_table="bus", overwrite=True)  # schreibt nach net.bus.geo
        #                                  ^^^^^^^^^^^^^^^
        # siehe Doku (geodata_table) :contentReference[oaicite:1]{index=1}

    def point_from_geo(val):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        obj = json.loads(val) if isinstance(val, str) else val
        if obj.get("type") != "Point":
            return None
        x, y = obj["coordinates"][:2]
        return (x, y)

    # 2) pos-Attribut setzen (Bus-Index = Node-ID)
    pos = {}
    for bus in G.nodes:
        if bus in net.bus.index:
            p = point_from_geo(net.bus.at[bus, "geo"])
            if p is not None:
                pos[bus] = p

    nx.set_node_attributes(G, pos, name="pos")

    # # optional: x/y separat
    # nx.set_node_attributes(G, {b: p[0] for b, p in pos.items()}, name="x")
    # nx.set_node_attributes(G, {b: p[1] for b, p in pos.items()}, name="y")
    return G

sb_code = "1-MVLV-rural-1.108-0-no_sw"   # Beispiel 1-MVLV-urban-all-0-sw
net = sb.get_simbench_net(sb_code)

G = top.create_nxgraph(
    net,
    respect_switches=True,   # offene Schalter trennen Kanten
    include_lines=True,
    include_trafos=True
)
if not nx.is_connected(G):
    raise nx.NetworkXError("Graph is not connected after fixing")
G = determine_pos(G, net)
MG = determine_type(G, net)
#nx.set_node_attributes(G, net.bus["type"].to_dict(), name="type")
#file = "1-MVLV-rural-1.108-0-no_sw_graph.pkl"
# with open(file, 'rb') as outfile:
#     MG = pickle.load(outfile)

G = repairing_graph(MG)

file2 = f"{sb_code}_fixed.pkl"
pickle.dump(G, open(file2, "wb"))
