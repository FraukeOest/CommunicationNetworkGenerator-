import networkx as nx
from PhysGraph import PhysGraph
import numpy as np
import pickle
from scipy.spatial import cKDTree
from re import search, sub
import os
import IctConfig
from pathlib import Path
import simbench as sb
import matplotlib.pyplot as plt
import json
import time
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
import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*DataFrame concatenation with empty or all-NA entries.*"
)

def calc_transmission_lat_s(datarate):
    """calculates transmission delay in kByte"""
    p_size = IctConfig.max_p_size * 8
    transmission_lat = p_size / (datarate * 1000)
    return transmission_lat * 1000


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

def determine_crossconnection(G, nodes_list, r_n, a, b, c):
    """
        Params: G: physical graph
                nodeslist: list of nodes in a cycle
                n: max number of cross connections
                a: weight of geolocation
                b: weight of hops
                c: weight of degree
        Returns:
            list of n crossconnection edges
    """
    nodes = nodes_list #= list(G.nodes)
    pos = nx.get_node_attributes(G, 'pos')
    coords = np.array([pos[u] for u in nodes])
    idx_of = {u: i for i, u in enumerate(nodes)}

        # Distanzmatrix (symmetrisch)
        # Für große Graphen optional durch KD-Tree ersetzen.
    geo_dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    dic_geo_dist = {
        (nodes[i], nodes[j]): float(geo_dist[i][j])
        for i in range(len(nodes))
        for j in range(len(nodes))
        if i != j
    }
    sorted_d_geo_dist = dict(sorted(dic_geo_dist.items(), key=lambda item: item[1]))
    # avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
    # for (u, v), val in sorted_d.items():
    #     if not G.has_edge(u, v):
    #         G.add_edge(u, v)
    #     avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
    #     if avg_degree > 4:
    #         break

    G_sub = G.subgraph(nodes_list).copy()
    hops = dict(nx.all_pairs_shortest_path_length(G_sub))
    dict_hops = {}
    for k, v in hops.items():
        for i, j in v.items():
            if j > 1 and (i, k) not in dict_hops:
                dict_hops[(k, i)] = j
    sorted_d_hops = dict(sorted(dict_hops.items(), key=lambda item: item[1]))
    print(sorted_d_hops)
    max_geo_local = max(sorted_d_geo_dist.values())
    norm_geo_local = {k: v / max_geo_local for k, v in sorted_d_geo_dist.items() if k in sorted_d_hops.keys()}

    max_hops = max(sorted_d_hops.values())
    norm_hops = {k: v / max_hops for k, v in sorted_d_hops.items()}

    edge_degree = {k: G.degree[k[0]] + G.degree[k[1]] for k in sorted_d_hops.keys()}
    max_edge_degree = max(edge_degree.values())
    norm_edge_degree = {k: v / max_edge_degree for k, v in edge_degree.items()}

    # uniform_sorting = {k:(round(b * norm_hops[k] + c * norm_edge_degree[k], 2), 1 - norm_geo_local[k]) for k in
    #                    norm_hops.keys()}
    # solutions_sorted = sorted(
    #     uniform_sorting.items(),
    #     key=lambda x: (x[1][0], x[1][1]),
    #     reverse=True
    # )
    # n = int(round(len(nodes_list) * r_n, 0))
    # first_n = dict(solutions_sorted[:n])
    uniform_sorting = {k: a * (1 - norm_geo_local[k]) + b * norm_hops[k] + c * norm_edge_degree[k] for k in
                       norm_hops.keys()}
    solutions_sorted = sorted(
        uniform_sorting.items(),
        key=lambda x: (x[1]),
        reverse=True
    )
    n = int(round(len(nodes_list) * r_n, 0)) #n = int(round(len(solutions_sorted) * r_n, 0))
    first_n = dict(solutions_sorted[:n])
    return first_n

def add_crossconnetions(G, list_of_overlay_links, br_core):
    for (u, v) in list_of_overlay_links:
        lat = calc_transmission_lat_s(br_core)
        G.add_edges_from([(u, v, {"weight": br_core, "lat": lat})])
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

def __create_public_topology(G):
    """ create one backbone node and let all acces router connect to it in a star topology
    :param G: PhysicalGrpah
    :return:
    """
    router = G.routers
    pos = nx.get_node_attributes(G, 'pos')
    core_edges = [edge for edge in G.edges if "R" in edge[0] and "R" in edge[1]]
    coords = np.array(list(pos.values()))  # [[x1, y1], [x2, y2], ...]
    centroid = coords.mean(axis=0)
    G.add_node("Backbone", pos=tuple(centroid))
    for (u, v) in core_edges:
        G.remove_edge(u, v)
    for r in router:
        G.add_edge(r, "Backbone", weight=10e10, lat=0)
    return G


def _contract_router(G, reduced_to_factor):
    """contracts edges in order to reduce the amount of router by approx a third"""
    edges = [edge for edge in G.edges() if "R" in edge[0] and "R" in edge[1]]
    H = G.copy()
    H.routers = [node for node in G.routers if node in H.nodes]
    if reduced_to_factor > 1:
        raise ValueError(f"reduced_to_factor must be below 1 but is currently {reduced_to_factor}")
    n_reduced_router = max(int(round(len(G.routers) * reduced_to_factor)), 4)
    #sorted_router = sorted(G.routers, key=lambda n: G.degree(n))
    while len(H.routers) > n_reduced_router:
        sorted_router = sorted(H.routers, key=lambda n: H.degree(n))
        r = sorted_router[0]
        #for r in sorted_router:
        neighbors = [n for n in H.neighbors(r) if n in H.routers]
        sorted_neighbors = sorted(neighbors, key=lambda n: H.degree(n))
        v = sorted_neighbors[0]
        if H.has_edge(r, v):
            H = nx.contracted_edge(H, (r, v), copy=False, self_loops=False)
        H.routers = [node for node in G.routers if node in H.nodes]
    # if 0.6 < reduced_to_factor < 0.7:
    #     contract_every = 3
    # if reduced_to_factor == 0.5:
    #     contract_every = 2
    # for i, (u, v) in enumerate(edges):
    #     if i % contract_every == 0 and H.has_edge(u, v):
    #         H = nx.contracted_edge(H, (u, v), copy=False, self_loops=False)
    #         print(f"contracted {u, v}")
    H.servers = G.servers
    H.routers = [node for node in G.routers if node in H.nodes]
    H.ot_devices = G.ot_devices
    #print({r: H.degree(r) for r in H.routers})
    return H


def _create_small_worlds_for_areas(G, cycle_edges, k, p, br_core):
    """adds random cross-connections via newman-watts-strogatz to areas"""
    conn = []
    for cycle in cycle_edges:
        # k = max(int(len(cycle) / 2), 2)
        n = len(cycle)
        small_world = nx.newman_watts_strogatz_graph(n, k=k, p=p)
        a = nx.algebraic_connectivity(small_world)
        conn.append(a)
        avg_degree = np.mean([small_world.degree[n] for n in small_world.nodes])
        # if not (1 < a > 0.5 * avg_degree):
        #     raise ValueError(f"Small world is not robust enough and violates {a} > {avg_degree * 0.5}")
        print(f"Small world is robust with 1 < {a} < {avg_degree * 0.5}")
        mapping = {i: node for i, node in enumerate(cycle)}
        small_world = nx.relabel_nodes(small_world, mapping)
        for edge in small_world.edges():
            if edge not in G.edges():
                lat = calc_transmission_lat_s(br_core)
                G.add_edges_from([(edge[0], edge[1], {"weight": br_core, "lat": lat})])
    G.avg_alg_conn_cycle = np.mean(conn)
    return G


def find_server_place(G):
    """Ranks routers for potential server placements
    considers node degree of the router, close switch or transformer, and PV in the connected LV-grid"""
    H = nx.Graph()
    # create graph with where routers are involved
    for (u, v) in G.edges():
        if "R" in u and "R" in v:
            H.add_edge(u, v)
    # sort nodes in descending order according to node degree
    sorted_nodes = sorted(H.degree(), key=lambda x: x[1], reverse=True) # ToDo das Sort scheint unnötig, wenn später nochmal gesortet wird

    # just regard routers
    filtered = [n for n in sorted_nodes if 'R' in n[0]]
    nodes = [n[0] for n in filtered]
    degree = [n[1] for n in filtered]
    swtiches = []
    ee_occurances = []

    # check for nodes with PV or switches attached
    for n in nodes:
        neigbhors = list(G.neighbors(n))
        num_ee = sum("LV_PV" in s for s in neigbhors)
        ee_occurances.append(num_ee)
        #print(neigbhors)
        for neighbor in neigbhors:
            sw_exists = 0
            if "switch" in neighbor or "Trafo" in neighbor:
                sw_exists = 1
                break
        swtiches.append(sw_exists)

    normalized_ee = [1 if e > 0 else 0 for e in ee_occurances]
    output = [(n, d + s + e) for n, d, s, e in zip(nodes, degree, swtiches, normalized_ee)]
    return output


def predetermine_cigre_sampled(router_reduced=False, r_n=1, w_geo=1, w_hops=1, w_degree=1, comp_factor=1, br_edge=10, br_core=100, regenerate=False):
    """only generates a graph once for one parameterization combination.
    If it already exists, it loads the pickled graph"""
    graph_name = (f"CigreMVLV_router_reducted={router_reduced}_r_n={r_n}_w_geo={w_geo}_w_hops={ w_hops}_w_degrees={w_degree}"
                  f"_comp_factor={comp_factor}_br_edge={br_edge}_core={br_core}.pkl")
    cwd = Path.cwd()
    parent = cwd.parent
    graph_dir = parent / "graphs"
    path = graph_dir / graph_name
    if os.path.exists(path) and not regenerate:
        graph = pickle.load(open(path, "rb"))
        print(f"{graph_name} already.")
    else:
        BASE_DIR = Path(__file__).resolve().parent
        pkl_path = BASE_DIR / "cigre_MV_LV_Graph.pkl"
        with open(pkl_path,'rb') as outfile:
                MG = pickle.load(outfile)
        graph = Cigre_Sampled(router_reduced=router_reduced, rel_n_crosslinks=r_n, w_geo=w_geo, w_hops=w_hops, w_degree=w_degree,
                                   comp_factor=comp_factor, br_edge=br_edge, br_core=br_core, MG=MG)
        if not regenerate:
            pickle.dump(graph, open(path, "wb"))
            print(f"{graph_name} was not found.")
    return graph

def Cigre_Sampled(router_reduced=1, rel_n_crosslinks=1,w_geo=1, w_hops= 1, w_degree=1,
                  comp_factor=1, br_edge=10, br_core=100, MG:nx.Graph = None):
    # try:
    #     logger.info("test cycle edges")
    #     cycle_edges = list(nx.minimum_cycle_basis(MG))
    #     G = PhysGraph(MG)
    # except:
    #     logger.info("fix graph")
    #     G = _fix_graph_for_all_cycles(MG)
    G = PhysGraph(MG)
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
    router_for_server = find_server_place(G)
    ordered_places = sorted(router_for_server, key=lambda x: x[1], reverse=True) # todo sort in function


    all_nodes = G.nodes()
    #servers = [f'S{i}' for i in range(len(router_for_server))]
    res = dict()
    max_p_size = 255  # [Byte]
    pi_ressources = {key: value * comp_factor for key, value in IctConfig.S_PI.items()}
    server_with_rank = dict()
    servers=[]
    for router, rank in ordered_places:
        match_router = search(r"[-+]?\d*\.?\d+", router)
        router_number = match_router.group()

        lat = calc_transmission_lat_s(br_core)
        r_pos = nx.get_node_attributes(G, 'pos')[router]
        s = f"S{router_number}"

        G.add_node(s, pos=(r_pos[0] + 1, r_pos[1] + 1))
        G.add_edge(s, router, weight=br_core, lat=lat)
        server_with_rank[s] = rank
        res[s] = pi_ressources
        servers.append(s)

    #res = {"S1": Config.IctConfig.S1_RES, "S2": Config.IctConfig.S2_RES, "S3": Config.IctConfig.S3_RES}
    G.set_servers(servers, res)
    G.server_with_rank = server_with_rank

    degrees = [(n, d) for n, d in nx.degree(G) if d > 1]

    routers = [n for n in G.nodes() if "R" in n]
    G.set_routers(routers)
    ot_devies = [v for v in all_nodes if v not in routers and v not in servers]
    G.set_ot_devices(ot_devies)
    logger.info("adjusting edge weight")
    for u, v, d in G.edges(data=True):
        if 'R' in u and 'R' in v: # Core Network
            G[u][v]["weight"] = br_core
            G[u][v]["lat"] = calc_transmission_lat_s(br_core)
        elif u in servers or v in servers:
            G[u][v]["weight"] = br_core
            G[u][v]["lat"] = calc_transmission_lat_s(br_core)
        else: # Access Network
            G[u][v]["weight"] = br_edge
            G[u][v]["lat"] = calc_transmission_lat_s(br_edge)

    logger.info("creating finegrained topology")
    regard_rings = True
    if router_reduced < 1:
        G = _contract_router(G, router_reduced)
        # create small worlds
    if regard_rings:
        cycle_edges = list(nx.minimum_cycle_basis(G, weight=None))
        print(len(cycle_edges))
        print(cycle_edges)
    else:
        router = [n for n in G.nodes() if "R" in n]
        cycle_edges = [router]
    G.n_areas = len(cycle_edges)
    #G = _create_small_worlds_for_areas(G, cycle_edges, sw_k, sw_p, br_core)
    for cycle in cycle_edges:
        crossy_list = determine_crossconnection(G, nodes_list=cycle, r_n=rel_n_crosslinks, a=w_geo, b=w_hops, c=w_degree)
        G.cross_connections.extend(list(crossy_list))
        G = add_crossconnetions(G, crossy_list, br_core)

    for u ,v, d in G.edges(data=True):
        if "weight" not in G[u][v]:
            raise ValueError(f"edge {u, v} doesnt have an edge weight")
    print(f"graph contains {len(list(G.nodes))} nodes and {len(list(G.edges))} edges")
    #G.plot()
    return G


def positions(mv_net):
    pos = {}
    # 1) Falls bus_geodata existiert (bei dir offenbar nicht)
    if hasattr(mv_net, "bus_geodata"):
        if len(mv_net.bus_geodata.index) > 0 and {"x", "y"}.issubset(mv_net.bus_geodata.columns):
            pos = {b: (mv_net.bus_geodata.at[b, "x"], mv_net.bus_geodata.at[b, "y"])
                   for b in mv_net.bus_geodata.index if b in MG}

    # 2) Manche Netze haben x/y direkt in net.bus
    if not pos and {"x", "y"}.issubset(getattr(mv_net, "bus", []).columns):
        pos = {b: (mv_net.bus.at[b, "x"], mv_net.bus.at[b, "y"])
               for b in mv_net.bus.index if b in MG}

    # 3) Oder GeoJSON-artig in net.bus["geo"]
    if not pos and "geo" in mv_net.bus.columns:
        for b in mv_net.bus.index:
            if b not in MG:
                continue
            g = mv_net.bus.at[b, "geo"]
            if g is None:
                continue
            # oft als String im GeoJSON-Format
            if isinstance(g, str):
                try:
                    d = json.loads(g)
                    x, y = d["coordinates"]
                    pos[b] = (x, y)
                except Exception:
                    pass

    # 4) Fallback: schnelles Layout (NICHT spring_layout)
    if not pos:
        print("Keine Geodaten gefunden -> random_layout als Fallback.")
        t0 = time.perf_counter()
        pos = nx.random_layout(MG, seed=1)
        print("Layout seconds:", time.perf_counter() - t0)
    return pos


if __name__ == '__main__':
    #
    #codes = sb.collect_all_simbench_codes()
    #print(codes)
    ##SGS 1-MVLV-rural-all-2 ~ 100
#SGS 1-MVLV-urban-all-2-sw ~ 170

    # grid_mv1= "1-MVLV-rural-all-0-no_sw"
    # # #grid0 = '1-LV-rural1--1-no_sw'
    # mv_net = sb.get_simbench_net(grid_mv1)
    # # print("get Simbench")
    # MG = pandapower.topology.create_nxgraph(mv_net, multi=True, include_switches=True, respect_switches=False)
    # print("simbench to networkx")
    # print("Nodes:", MG.number_of_nodes())
    # print("Edges:", MG.number_of_edges())
    # print("Is empty:", MG.number_of_nodes() == 0)
    # pos = positions(mv_net)
    # plt.figure(figsize=(12, 9), dpi=150)
    # nx.draw_networkx_edges(MG, pos, width=0.2, alpha=0.3)
    # nx.draw_networkx_nodes(MG, pos, node_size=2)
    # plt.axis("off")
    # plt.tight_layout()
    # plt.show()
    print("load graph")
    file = "1-MVLV-rural-all-0-no_sw_graph.pkl"
    #file = "cigre_MV_LV_Graph.pkl"
    #file = "1-MVLV-rural-1.108-0-no_sw_graph.pkl"
    with open(file, 'rb') as outfile:
        MG = pickle.load(outfile)
    print("create phys graph")
    G = Cigre_Sampled(MG=MG, router_reduced=1)
    G.plot(legend=True)
    # pos2 = nx.get_node_attributes(MG, "pos")
    # nx.draw(MG, pos2, with_labels=True)
    # nx.draw_networkx_edges(MG, pos2, width=2)
    #
    # plt.show()

