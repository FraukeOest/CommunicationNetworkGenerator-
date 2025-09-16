import networkx as nx
from PhysGraph import PhysGraph
import numpy as np
import pickle
from scipy.spatial import cKDTree
from re import search
import os
import IctConfig
import simbench as sb
import pandapower


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
    return G


def _remove_middle_compoents(graph: PhysGraph):
    """removes parts that are connected in the power grid line that are connected differently in the communication
    network (star-like)"""
    switches = [n for n in graph.nodes if 'switch' in n]
    for n in switches:
        neighbor = list(graph.neighbors(n))
        #print(neighbor)
        graph.add_edge(neighbor[0], neighbor[1])
        if "Trafo" in neighbor[1]:
            graph.remove_edge(n, neighbor[1])
        else:
            graph.remove_edge(n, neighbor[0])
    Trafo = [n for n in graph.nodes if 'Trafo' in n]

    for n in Trafo:
        neighbor = [n for n in graph.neighbors(n) if "Trafo" not in n]
        graph.add_edge(neighbor[0], neighbor[1])
        graph.remove_edge(n, neighbor[0])

    lv_busses = [n for n in graph.nodes if "bus" in n]
    for n in lv_busses:
        graph.remove_node(n)

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
    if degrees:
        raise ValueError("didnt delete all irrevant nodes that could act as router (but aren't)")
    return graph


def _rename_components(G):
    """renames former pandapower components to make easier distinguising between MV and LV, and to unify naming pattern"""
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


def _create_public_topology(G):
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


def _contract_router(G):
    """contracts edges in order to reduce the amount of router by approx a third"""
    edges = [edge for edge in G.edges() if "R" in edge[0] and "R" in edge[1]]
    H = G.copy()
    for i, (u, v) in enumerate(edges):
        if i % 3 == 0 and H.has_edge(u, v):
            H = nx.contracted_edge(H, (u, v), copy=False, self_loops=False)
            print(f"contracted {u, v}")
    H.servers = G.servers
    H.routers = [node for node in G.routers if node in H.nodes]
    H.ot_devices = G.ot_devices

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


def predetermine_cigre_sampled(router_reduced=False, sw_p=0.5, sw_k=2, regard_rings=False, public_topo=False,
                  comp_factor=1, br_edge=10, br_core=100):
    """only generates a graph once for one parameterization combination.
    If it already exists, it loads the pickled graph"""
    graph_name = (f"CigreMVLV_router_reducted={router_reduced}_swp={sw_p}_swk={sw_k}_regard_rings={regard_rings}"
                  f"_public_topo={public_topo}_comp_factor={1}_br_edge={br_edge}_core={br_core}.pkl")
    directory = "graphs/"

    path = directory+graph_name
    if os.path.exists(path):
        graph = pickle.load(open(path, "rb"))
        print(f"{graph_name} already.")
    else:
        graph = Cigre_Sampled(router_reduced=router_reduced, sw_p=sw_p, sw_k=sw_k, regard_rings=regard_rings,
                              public_topo=public_topo, comp_factor=comp_factor, br_edge=br_edge, br_core=br_core)
        pickle.dump(graph, open(path, "wb"))
        print(f"{graph_name} was not found.")
    return graph

def Cigre_Sampled(router_reduced=False, sw_p=0.5, sw_k=2, regard_rings=False, public_topo=False,
                  comp_factor=1, br_edge=10, br_core=100, MG:nx.Graph = None):
    try:
        cycle_edges = list(nx.minimum_cycle_basis(MG))
        G = PhysGraph(MG)
    except:
        print("fix graph")
        G = _fix_graph_for_all_cycles(MG)

    G.remove_nodes_from(['S1', 'S2', 'S3'])  # these are not servers, but names for switches
    print("remove middle components")
    G = _remove_middle_compoents(G)
    #degrees2 = [(n, d) for n, d in nx.degree(G) if d > 1]
    G = _rename_components(G)
    degrees3 = [(n, d) for n, d in nx.degree(G) if not "R" in n and d > 1]
    if degrees3:
        raise ValueError(f"OT device pretending to be a router {degrees3}")
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
    print("adjusting edge weight")
    for u, v, d in G.edges(data=True):
        if 'R' in u and 'R' in v: # Core Network
            G[u][v]["weight"] = br_core
            G[u][v]["lat"] = calc_transmission_lat_s(br_core)
        else: # Access Network
            G[u][v]["weight"] = br_edge
            G[u][v]["lat"] = calc_transmission_lat_s(br_edge)

    print("creating finegrained topology")
    if public_topo:
        G = _create_public_topology(G)
    else:
        if router_reduced:
            G = _contract_router(G)
        # create small worlds
        if regard_rings:
            cycle_edges = list(nx.minimum_cycle_basis(G, weight=None))
            print(len(cycle_edges))
            print(cycle_edges)
        else:
            router = [n for n in G.nodes() if "R" in n]
            cycle_edges = [router]
        G.n_areas = len(cycle_edges)
        G = _create_small_worlds_for_areas(G, cycle_edges, sw_k, sw_p, br_core)

    for u ,v, d in G.edges(data=True):
        if "weight" not in G[u][v]:
            raise ValueError(f"edge {u, v} doesnt have an edge weight")
    print(f"graph contains {len(list(G.nodes))} nodes and {len(list(G.edges))} edges")
    #G.plot()
    return G


if __name__ == '__main__':
    grid = '1-LV-rural1--1-no_sw'
    mv_net = sb.get_simbench_net(grid)
    MG = pandapower.topology.create_nxgraph(mv_net, multi=True, include_switches=True, respect_switches=False)
    # with open("cigre_MV_LV_Graph.pkl", 'rb') as outfile:
    #     MG = pickle.load(outfile)

    G = Cigre_Sampled(sw_k=0, sw_p=0, regard_rings=True, MG=MG)
    unused_server = G.servers[11:]
    G.servers = G.servers[0:11]
    for s in unused_server:
        G.remove_node(s.name)
    G.plot(legend=True)