import networkx as nx
from PhysGraph import PhysGraph
import numpy as np
import pickle
import time
from scipy.spatial import cKDTree
from re import search, sub
import logging
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

    router = [n for n in graph.nodes() if "Bus" in n]
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
    try:
        logger.info("test cycle edges")
        cycle_edges = list(nx.minimum_cycle_basis(MG))
        G = PhysGraph(MG)
    except:
        logger.info("fix graph")
        G = _fix_graph_for_all_cycles(MG)
    G.remove_nodes_from(['S1', 'S2', 'S3'])  # these are not servers, but names for switches
    logger.info("remove middle components")
    G = _remove_middle_compoents(G)
    #degrees2 = [(n, d) for n, d in nx.degree(G) if d > 1]
    G = _rename_components(G)
    degrees3 = [(n, d) for n, d in nx.degree(G) if not "R" in n and d > 1]
    if degrees3:
        raise ValueError(f"OT device pretending to be a router {degrees3}")
    degrees4 = [(n, d) for n, d in nx.degree(G) if "R" in n and d <= 1]
    if degrees4:
        raise ValueError(f"router is end device {degrees4}")
    return G

file = "1-MVLV-rural-all-0-no_sw_graph.pkl"
with open(file, 'rb') as outfile:
    MG = pickle.load(outfile)

G = repairing_graph(MG)

file2 = "1-MVLV-rural-all-0-no_sw_graph_fixed.pkl"
pickle.dump(G, open(file2, "wb"))
