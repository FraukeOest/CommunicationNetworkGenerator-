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
import matplotlib.pyplot as plt
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

def calculate_cycles(graph, MH):
    print("calculate cycles")
    H = graph.copy()
    J = graph.copy()
    for node, attr in J.nodes(data=True):
        if attr['type'] == "bus":
            if attr['b_id'] not in MH.nodes():
                H.remove_node(node)
    cycles = list(nx.minimum_cycle_basis(H, weight=None))
    print(f"cycles: {len(cycles)}")

def _remove_middle_compoents(graph: PhysGraph, MH):
    """removes parts that are connected in the power grid line that are connected differently in the communication
    network (star-like)"""


    if not list(graph.nodes):
        raise ValueError("no nodes in Graph to be remodelled")
    calculate_cycles(graph, MH)
    ctr = 0
    H = graph.copy()
    router = [n[0] for n in graph.nodes(data=True) if n[1]["type"] == "bus" and "R" in n[0]]
    ot = [n[0] for n in graph.nodes(data=True) if n[1]["type"] != "bus"]
    graph.ot_devices = ot
    graph.routers = router
    for node, attr in H.nodes(data=True):
        if attr["type"] != "bus":
            nbrs = list(nx.neighbors(graph, node))
            if any("R" in n for n in nbrs):
                for nbr in nbrs:
                    if "R" not in nbr:
                        graph.remove_edge(node, nbr)
            else:
                closest = {r: nx.shortest_path_length(graph, source=node, target=r) for r in router}
                best_r = min(closest, key=closest.get)
                for nbr in nbrs:
                    graph.remove_edge(node, nbr)
                graph.add_edge(best_r, node)
    calculate_cycles(graph, MH)

    while True:
        finished = True
        ctr += 1
        H = graph.copy()
        print(graph.number_of_nodes())
        for node, attr in H.nodes(data=True):
            if attr["type"] == "bus":
                if "R" not in node:
                    graph, deleted = delete_node(graph, node, 2)
                    finished = not deleted
                    # print("try to delete switch node", node)
                    #calculate_cycles(graph, MH)
                    if deleted:
                        continue
                    #print("delete base on degree 2")
                if H.degree[node] <= 1:
                    graph.remove_node(node)
                    finished = False
                    continue
        ctr += 1
        print("counter: ", ctr)
        if finished:
            break
    l_connected_buses = dict()
    # for node, attr in H.nodes(data=True):
    #     if attr["type"] == "bus" and not "R" in node:
    #         nbrs = list(nx.neighbors(graph, node))
    #         if len(nbrs) == 3:
    #             sub = "R"
    #             l_connected_buses[node] = {'degree': len(nbrs),
    #                                        'neighbors': nbrs}
    #             idx = next((i for i, s in enumerate(nbrs) if sub in s), -1)
    #             r1 = nbrs.pop(idx)
    #             idx = next((i for i, s in enumerate(nbrs) if sub in s), -1)
    #             r2 = nbrs[idx]
    #             if r1 != r2:
    #                 graph.add_edge(r1, r2)
    #                 graph.remove_edge(node, r1)
    #                 graph.remove_edge(node, r2)
    #                 break
    #         print(f"node: {node} nbrs: {nbrs}")
    H = graph.copy()
    # for node, attr in H.nodes(data=True):
    #     if attr["type"] == "bus" and not "R" in node:
    #         if H.degree[node] <= 1:
    #             graph.remove_node(node)
    for r in router:
        neigbors = list(nx.neighbors(graph, r))
        for n in neigbors:
            if graph.nodes[n]['type'] == "bus" and "R" not in n:
                graph = nx.contracted_nodes(graph, r, n, self_loops=False, copy=True)


    isolates = list(nx.isolates(graph))
    graph.remove_nodes_from(isolates)
    calculate_cycles(graph, MH)
    keep = [n for n, a in graph.nodes(data=True) if a.get("type") == "bus"]
    H = graph.copy()
    J = H.copy()
    for node in J.nodes():
        if node not in keep:
            H.remove_node(node)
    cycles = list(nx.minimum_cycle_basis(H, weight=None))
    print("calculated cycles: ", len(cycles))
    # for i, c in enumerate(cycles):
    #     print(f"------ cycle ----- {i}\n")
    #     print(c)
    # keep = cycles[4]
    # H = nx.subgraph(H, keep)
    # print(f"cycles graph: {len(cycles)}")
    # nx.draw(graph)
    # plt.show()

    # busses = [n for n in graph.nodes if "Bus" in n]
    # for n in busses:
    #     neighbor = list(graph.neighbors(n))
    #     if len(neighbor) > 1:
    #         if "Bus" in neighbor[0] and "Bus" in neighbor[1]:
    #             nx.contracted_edge(graph, (neighbor[0], n), self_loops=False, copy=False)

    degrees4 = [(n, d) for n, d in nx.degree(graph) if "Bus" in n and d <= 1]
    while degrees4:
        graph.remove_nodes_from(degrees4)
        degrees4 = [n for n, d in graph.degree() if "Bus" in str(n) and d <= 1]


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

    degrees = [(n, d) for n, d in nx.degree(graph) if not "R" in n and d > 1]
    logger.info("finished fixing")
    if degrees:
        logger.error(f"didnt delete all irrevant nodes that could act as router (but aren't): {degrees}")
        raise ValueError("didnt delete all irrevant nodes that could act as router (but aren't)")
    graph.ot_devices = ot
    graph.routers = router
    return graph


def _rename_components(G):
    """renames former pandapower components to make easier distinguising between MV and LV, and to unify naming pattern"""
    H = G.copy()
    for node in H.nodes(data=True):
            if node[1]["type"] == "trafo":
                nbrs = list(G.neighbors(node[0]))
                if any('R' in n for n in nbrs):
                    print(nbrs)
                new_name = f"R_{G.nodes[nbrs[0]]['b_id']}"
                nx.relabel_nodes(G, {nbrs[0]: new_name}, copy=False)
                if new_name == "R_2":
                    print(node[0])


    accepted_keys = ['HVMV_Trafo', 'MVLV_Trafo', 'switch', 'MV_Bat', 'LV_Bat', 'MV_Load', 'LV_CHP_', 'MV_CHP', 'R', 'MV_PV', 'LV_Load',
                     'WKA','LV_PV', 'S', 'Bus', 'MV_BM', 'MV_Hydro', 'MV_WP']
    accepted = []
    for k in G.nodes:
        for ak in accepted_keys:
            if k.startswith(ak):
                accepted.append(k)
    if len(accepted) < len(list(G.nodes)):
        dif = set(list(G.nodes)) - set(accepted)
        raise KeyError(f"missed components to consider renaming by {dif} ")

    return G



def repairing_graph(MG, MH):
    G = PhysGraph(MG)
    components = list(nx.connected_components(G))  # Liste von Sets mit Knoten
    # try:
    #     logger.info("test cycle edges")
    #     cycle_edges = list(nx.minimum_cycle_basis(MG))
    #     G = PhysGraph(MG)
    # except:
    #     logger.info("fix graph")
    #     G = _fix_graph_for_all_cycles(MG)
    # if not nx.is_connected(G):
    #     raise nx.NetworkXError("Graph is not connected after fixing")
    # G.remove_nodes_from(['S1', 'S2', 'S3'])  # these are not servers, but names for switches
    # logger.info("remove middle components")
    G = _rename_components(G)
    if not nx.is_connected(G):
        raise nx.NetworkXError("Graph is not connected after renaming components")
    G = _remove_middle_compoents(G, MH)
    if not nx.is_connected(G):
        raise nx.NetworkXError("Graph is not connected after removing middle components")
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
        scenario=2,          # Szenario fixieren (Bus-Anzahl bleibt i.d.R. gleich)
        breaker_rep=None,    # beide Varianten zulassen (sw / no_sw)
        all_data=True
    )
    sizes = []
    for code in codes:
        net = sb.get_simbench_net(code)
        sizes.append((len(net.bus), code))

    sizes.sort()  # sortiert standardmäßig nach dem 1. Element im Tupel, dann nach dem 2.

    for n_buses, code in sizes:
        print(f"{n_buses:6d}  {code}")

    print("\nKleinstes MV+LV-Netz:", sizes[0][1])
    print("Anzahl Busse:", sizes[0][0])


    #Kleinstes MV+LV-Netz: 1-MVLV-rural-1.108-0-no_sw
    #Anzahl Busse: 109



def create_node(G, asset, type):
    bus = asset["bus"]
    name = str(asset.get("name", ""))
    vl = "LV" if "LV" in name else ("MV" if "MV" in name else "HV")
    subnet = asset["subnet"][2:]
    if "profile" in asset:
        if "lv_rural" in asset["profile"] or "lv_semiurb" in asset["profile"] or "lv_urban" in asset["profile"]:
            return G
        if type == "storage":
            asset_name = f"{vl}_Bat_{asset["profile"]}_{subnet}"
        elif type != "load":
            asset_name = f"{vl}_{asset["profile"]}_{subnet}"
        else:
            asset_name = f"{vl}_Load_{asset["profile"]}_{subnet}"
    else:
        if "switch" in name or "Switch" in name:
            asset_name = sub(r'MV\d+\.\d+ loop_line_switch (\d+)\.(\d+)', r'switch_\1\2', name)
        else:
            raise ValueError("no rule for this asset")
    pos = G.nodes[bus]['pos']
    pos = (pos[0], pos[0] + 1)
    if "p_mw" in asset:
        p_mw = asset["p_mw"]
        G.add_node(asset_name, pos=pos, p_mw=p_mw, type=type)
    else:
        G.add_node(asset_name, pos=pos, type=type)
    G.add_edge(bus, asset_name)
    return G

def delete_node(G, node, degree):
    try:
        nbrs = list(G.neighbors(node))
    except AttributeError:
        print("Attribute Error ")
    deleted = False
    if degree >= len(nbrs) > 1:
        #print("delete node ", node, "with degree ", len(nbrs))
        deleted = True
        for ix in range(1, len(nbrs)):
            G.add_edge(nbrs[0], nbrs[ix])
        G.remove_node(node)
    return G, deleted


def determine_type(G, net):
    bus_ids = list(G.nodes)
    #bus_df = net.bus.loc[bus_ids, ["name", "vn_kv", "zone"]]
    # print(bus_df.head())
    mapping = {i: f"{net.bus.at[i, 'name']} ({i})" for i in net.bus.index}
    for ix, bus in net.bus.iterrows():
        name = bus['name']
        G.nodes[ix]['b_id'] = ix
        G.nodes[ix]['type'] = "bus"
        vl = ""
        if "LV" in name:
            vl = "LV"
        elif "MV" in name:
            vl = "MV"
        elif "HV" in name:
            vl = "HV"
        else:
            raise ValueError("no voltage level rule for this asset")
        G.nodes[ix]["voltage_level"] = vl
        new_name = f"Bus_{ix}"
        mapping[ix] = new_name


    # for asset in net.ext_grid:
    #     bus = asset["bus"]
    #     vl = "LV" if "LV" in asset['name'] else "HV"
    #     sgen_name = f"{vl}_{asset["profile"]}"
    #     pos = G.nodes[bus]['pos']
    #     p_mv = asset["p_mw"]
    #     G.add_node(sgen_name, pos=pos, p_mv=p_mv)
    #     G.add_edge(bus, asset)

    accepted_keys = ['HVMV_Trafo', 'MVLV_Trafo', 'switch', 'MV_Bat', 'MV_Load', 'LV_CHP_', 'MV_CHP', 'R', 'MV_PV', 'LV_Load',
                     'WKA','LV_PV', 'S']

    for asset in net.gen.iterrows():
        G = create_node(G, asset, "gen")
    for ix, sgen in net.sgen.iterrows():
        G = create_node(G, sgen, "sgen")
    for ix, asset in net.storage.iterrows():
        G = create_node(G, asset, "storage")
    for ix, asset in net.switch.iterrows():
        G = create_node(G, asset, "switch")
    for ix, asset in net.load.iterrows():
        G = create_node(G, asset, "load")
    for ix, trafo in net.trafo.iterrows():
        ctr = 1
        lvbus = trafo["lv_bus"]
        hvbus = trafo["hv_bus"]
        name = str(trafo.get("name", ""))
        if "EHV" in name:
            raise ValueError("no concept for EHV")
        if "LV" in name:
            asset_type = "MVLV"
        else:
            asset_type = "HVMV"
        subnet = trafo["subnet"][2:]
        asset_name = asset_type + "_Trafo_" + subnet
        if asset_name in G.nodes():
            asset_name = asset_name + str(ctr)
            ctr += 1
        pos = G.nodes[hvbus]['pos']
        G.add_node(asset_name, pos=pos, type="trafo", model=asset_type)
        G.add_edge(lvbus, asset_name)
        #G.add_edge(hvbus, asset_name)
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

#determine_smallest_grid()
sb_code ="1-MVLV-urban-all-2-no_sw" #"1-MVLV-rural-1.108-2-no_sw" #   # Beispiel 1-MVLV-urban-all-0-sw    1-MVLV-rural-4.101-2-no_sw

net = sb.get_simbench_net(sb_code)

G = top.create_nxgraph(
    net,
    multi=True, include_switches=True, respect_switches=False
)
netH = sb.get_simbench_net('1-MV-urban--2-no_sw')
MH = top.create_nxgraph(
    netH,
    multi=True, include_switches=True, respect_switches=False
)
# H = nx.Graph(G)
# cycles = list(nx.minimum_cycle_basis(H, weight=None))
# print(f"first cycles: {len(cycles)}")

if not nx.is_connected(G):
    raise nx.NetworkXError("Graph is not connected after fixing")
G = determine_pos(G, net)

MG = determine_type(G, net)
#nx.set_node_attributes(G, net.bus["type"].to_dict(), name="type")
#file = "1-MVLV-rural-1.108-0-no_sw_graph.pkl"
# with open(file, 'rb') as outfile:
#     MG = pickle.load(outfile)
G = PhysGraph(MG)
G = repairing_graph(G, MH)

file2 = f"{sb_code}_fixed.pkl"
pickle.dump(G, open(file2, "wb"))
