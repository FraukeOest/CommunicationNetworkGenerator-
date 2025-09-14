# - * - coding: utf - 8
# Created By: Frauke Oest

import networkx as nx
import itertools
from sklearn.metrics import r2_score, root_mean_squared_error
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from re import search
import numpy as np
from matplotlib.colors import Normalize
class Server():

    def __init__(self, name:str, res:dict()):
        self.name = name
        self.cpu = res['cpu']
        self.mem = res['mem']
        self.sto = res['sto']


class PhysGraph(nx.Graph):

    def __init__(self, data=None,**attr):
        super(PhysGraph, self).__init__()
        self.__max_bitrate = None
        self.servers = None
        self.ot_devices = None
        self.routers = None
        self.__max_delay_in_system = None
        self.__max_number_paths = None
        self.avg_alg_conn_cycle = None
        self.n_areas = None
        self.server_with_rank = None

    def set_servers(self, server_names: list, ressorces:dict()):
        self.servers = [Server(s, ressorces[s]) for s in server_names]

    def set_routers(self, routers:list):
        self.routers = routers

    def set_ot_devices(self, ot_devices: list):
        self.ot_devices = ot_devices
    def get_adj_br(self):
        G_array = nx.to_numpy_array(self, weight='weight')
        return G_array.astype(int)

    def get_adj_lat(self):
        G_array = nx.to_numpy_array(self, weight='lat')
        return G_array.astype(int)


    def get_server_index(self, server_name):
        for ix, el in enumerate(self.servers):
            if server_name == el.name:
                return ix

    def get_max_bitrate(self):
        if not self.__max_bitrate:
            self.__max_bitrate = max(dict(self.edges).items(), key=lambda x: x[1]['weight'])[1]['weight']
        return self.__max_bitrate

    def get_number_of_edges(self):
        return len(self.edges)

    def get_max_delay_in_system(self):
        """Calculates the maximum transmission delay by calculating all end-to-end paths and adding the individual
        link latencies"""
        if self.__max_delay_in_system:
            return self.__max_delay_in_system
        else:
            self.__max_delay_in_system = 0
            longest_path = []
            nodelist = self.ot_devices + [s.name for s in self.servers]
            for i, j in itertools.product(nodelist, nodelist):
                if i != j:
                    paths = list(nx.all_simple_paths(self, i, j)) #ToDo get the latency attributes of the path
                    for path in paths:
                        delay = 0
                        sub_graph = self.subgraph(path).copy()
                        for u, v, d in sub_graph.edges(data=True):
                            delay += d['lat']
                        if delay > self.__max_delay_in_system:
                            self.__max_delay_in_system = delay
                            longest_path = path
            return self.__max_delay_in_system

    def get_max_number_of_paths(self):
        if self.__max_number_paths:
            return self.__max_number_paths
        else:
            self.__max_number_paths = 0
            nodelist = self.ot_devices + [s.name for s in self.servers]
            for i, j in itertools.product(nodelist, nodelist):
                if i != j:
                    paths = list(nx.all_simple_paths(self, i, j))
                    if len(paths) > self.__max_number_paths:
                        self.__max_number_paths = len(paths)
            return self.__max_number_paths


    def create_sub_graphs_of_paths(self, source, destination, k):
        shortest_paths = []
        G = nx.Graph()
        for i, path in enumerate(nx.shortest_simple_paths(self, source, destination, weight="lat")):
            shortest_paths.append(path)
            if i == k:
                break
        for path in shortest_paths:
            for i in range(len(path) - 1):
                #if not G.has_edge(path[i], path[i+1]):
                G.add_edge(path[i], path[i+1])
        G.remove_nodes_from(nx.isolates(G))
        return G


    def get_neighbor_enddevice(self, ref_node, with_servers = False):
        """
            Defines the end_devices of the ni
            Params: ref_node (str): node in graph to whom neighbors should be searched for
                    k (int): number of neighbors to be returned
        """
        if with_servers:
            end_devices = self.ot_devices + [server.name for server in self.servers]
        else:
            end_devices = self.ot_devices
        betweeness_centrality = {k: v for k, v in nx.betweenness_centrality(self).items() if k in end_devices}
        eigenvector_centrality = {k: v for k, v in nx.eigenvector_centrality(self).items() if k in end_devices}
        #path_graphs = {self.create_sub_graphs_of_paths(ref_node, n, k=5) for n in end_devices if n is not ref_node}

        # The diameter is the maximum eccentricity.
        diameter = nx.diameter(self)
        # The eccentricity of a node v is the maximum distance from v to all other nodes in G.
        eccentricity = nx.eccentricity(self)
        print("Diameter:", diameter)
        absolute_hops = {n: len (nx.shortest_path(self, source=ref_node, target=n, weight=None))-2 for n in end_devices if n is not ref_node}
        relative_hops = {n: absolute_hops[n]/diameter for n in end_devices if n is not ref_node}
        #total_neighborhood = {n: (betweeness_centrality[n] + eigenvector_centrality[n] + relative_hops[n])/3 for n in end_devices if n is not ref_node}
        total_neighborhood = {n: eigenvector_centrality[n] for n in end_devices}
        sorted_neighborhood = {k: v for k, v in sorted(total_neighborhood.items(), key=lambda item: item[1])}
        relative_hops.update({ref_node: None})
        return sorted_neighborhood, relative_hops

    def get_sorted_eigenvector_centrality_for_ot_devices(self):
        """calculates the eigenvector centrality for all ot devices (without server) and sorts them according to the eigenvalue
            Returns: sorted_neighborhood (dict): sorted dictionary according to values with nodes as keys"""
        eigenvector_centrality = {k: v for k, v in nx.eigenvector_centrality(self).items() if k in self.ot_devices}
        #total_neighborhood = {n: eigenvector_centrality[n] for n in self.ot_devices}
        sorted_neighborhood = {k: v for k, v in sorted(eigenvector_centrality.items(), key=lambda item: item[1])}
        sorted_neighborhood_keys = [k for k,v in sorted(eigenvector_centrality.items(), key=lambda item: item[1])]
        return sorted_neighborhood, sorted_neighborhood_keys


    def node_neighborhood(self):
        """calculate various neighborhood / centrality metrics of nodes in the graph """
        betweeness_centrality = nx.betweenness_centrality(self)
        degree_centrality = nx.degree_centrality(self)
        closeness_centrality = nx.closeness_centrality(self)
        eigenvector_centrality = nx.eigenvector_centrality(self)
        sorted_betweenness = {k: v for k, v in sorted(betweeness_centrality.items(), key=lambda item: item[1])}
        sorted_eigenvector = {k: v for k, v in sorted(eigenvector_centrality.items(), key=lambda item: item[1])}
        print(sorted_betweenness)
        print(sorted_eigenvector)

        nodes_ = list(self.nodes())
        nodes = self.ot_devices + [server.name for server in self.servers]
        routers = [node for node in nodes_ if node not in nodes]

        nodes = nodes + routers

        x = [betweeness_centrality[k] for k in nodes]
        y = [degree_centrality[key] for key in nodes]
        z = [eigenvector_centrality[key] for key in nodes]
        a = [closeness_centrality[key] for key in nodes]
        print(f"avg of eigenvector: {np.average(z)} with std: {np.std(z)}")

        error_metric = root_mean_squared_error
        print(f"error of betweenness and degree: {error_metric(x, y)}")
        print(f"error of betweenness and eigenvector: {error_metric(x, z)}")
        print(f"error of betweenness and closeness: {error_metric(x, a)}")
        print(f"error of degree and eigenvector: {error_metric(x, y)}")
        print(f"error of degree and closeness: {error_metric(x, y)}")
        print(f"error of eigenvector and closeness: {error_metric(x, y)}")

        # n = nodes
        # fig, ax = plt.subplots()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(x, z, a)
        # for i, txt in enumerate(n):
        #     #ax.annotate(txt, (x[i], y[i]))
        #     ax.text(x[i], z[i], a[i], txt)
        #
        # ax.set_xlabel('betweenness')
        # ax.set_ylabel('eigenvector')
        # ax.set_zlabel('closeness')

        # fig, ax = plt.subplots()
        # all = {'betweenness': x,
        #        'degree': y,
        #        'eigenvector': z,
        #        'closeness': a}
        # for k, v in all.items():
        #     #ax.scatter(nodes, v, label=k)
        #     ax.plot(nodes, v, label=k)
        # ax.legend()
        # ax.set_title("all node attributes")
        # fig.tight_layout()
        # plt.show()
        self.plot_centrality_values_in_nodes(closeness_centrality, "closeness centrality")

    def plot_centrality_values_in_nodes(self, centrality: dict(), label):
        """"
        Params:
            centrality: dict of precalculated centrality values to a node (key)
            label: title of the plot
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        nodes_ = list(self.nodes())
        node_values = [centrality[k] for k in nodes_]
        pos = nx.get_node_attributes(self, 'pos')
        color = plt.cm.RdYlGn.reversed()
        #sc = plt.scatter([], [], c=[], cmap=color, vmin=min(node_values), vmax=max(node_values))
        nx.draw(self, pos, node_color=node_values, cmap=color, with_labels=True, node_size=500)
        # Erstelle eine Farbskala (colorbar)
        sm = plt.cm.ScalarMappable(cmap=color, norm=plt.Normalize(vmin=min(node_values), vmax=max(node_values)))
        plt.colorbar(sm, label=label, ax=ax)
        plt.title(label)
        plt.tight_layout()
        plt.show()



    def plot(self, utilization:list=None, legend=True):
        """Plot the Communication Graph
            Params:
                utilization: list of edge utilization / traffic (communication load on line / bitrate capacity of the line)
                legend: determined if legend for graph should be shown
        """
        pos = nx.get_node_attributes(self, 'pos')
        color_map = []
        size_map = []
        big_node = 100
        small_node = 50
        figure_size_p = (16, 10)
        fig, ax = plt.subplots(num="Physical Graph", figsize=figure_size_p)
        cmap=plt.cm.RdYlGn.reversed()
        for node in self:
            if search(r'R\d+', node) or search(r'RC\d+', node):
                color_map.append('lightgray')
                size_map.append(big_node)
            elif search(r'Backbone', node):
                color_map.append('lightgray')
                size_map.append(big_node * 4)
            elif search(r'^S\d+', node):
                color_map.append('lightblue')
                size_map.append(big_node)
            elif search(r'LV_PV.+', node):
                color_map.append('lightgreen')
                size_map.append(small_node)
            elif search(r'MV_PV.+', node) or search(r'WKA.+', node):
                color_map.append('lightgreen')
                size_map.append(big_node)
            elif search('LV_Load.+', node):
                color_map.append('lightcoral')
                size_map.append(small_node)
            elif search(r'MV_Load.+', node):
                color_map.append('lightcoral')
                size_map.append(big_node)
            elif search(r'HVMV_Trafo.+', node):
                color_map.append('palevioletred')
                size_map.append(big_node)
            elif search(r'MVLV_trafo.+', node):
                color_map.append('palevioletred')
                size_map.append(small_node)
            elif search(r'switch.+', node):
                color_map.append('darkorchid')
                size_map.append(big_node)
            elif search(r'MV_Bat.+', node):
                size_map.append(big_node)
                color_map.append('gold')
            elif search(r'MV_CHP.*', node):
                size_map.append(big_node)
                color_map.append('forestgreen')
            else:
                color_map.append('red')
                size_map.append(big_node)

        weight_labels = nx.get_edge_attributes(self, 'weight')
        latency_labels = nx.get_edge_attributes(self, 'lat')
        mixed_label = weight_labels.copy()
        for key, value in mixed_label.items():
            mixed_label[key] = (weight_labels[key] / 1000, latency_labels[key])
        if utilization:
            utilization_np = np.array(utilization)
            U = nx.from_numpy_array(utilization_np, nodelist=self.nodes())
            for u, v in self.edges():
                if U.has_edge(u, v):
                    self[u][v]['utilization'] = round(U[u][v]['weight'] / self[u][v]['weight'], 2)
                else:
                    self[u][v]['utilization'] = 0
            edge_colors = [self[u][v]['utilization'] for u, v in self.edges()]
            label = nx.get_edge_attributes(self, 'utilization')
            adj = nx.to_numpy_array(self, weight='utilization')
            #print(f"utilization: {adj}")
            #print(f"max value utilization: {np.max(adj)}")

            nx.draw(self, pos, with_labels=True, font_weight='bold', node_size=800, font_size=12,
                    node_color=color_map, edge_color=edge_colors, edge_cmap=cmap, width=5, edge_vmin=0, edge_vmax=1)
            nx.draw_networkx_edge_labels(self, pos, edge_labels=label, font_size=8)
            norm = Normalize(vmin=0, vmax=1)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm._A = []
            plt.colorbar(sm, ax=ax)
        else:
            nx.draw(self, pos, with_labels=False, font_weight='bold', node_size=size_map, font_size=8,
                    node_color=color_map)

            legend_colors = dict(
                {'lightgray': 'router', 'lightblue': 'server', 'lightgreen': 'DER',
                 'lightcoral': 'load', 'palevioletred': 'trafo',
                 'darkorchid': 'switch', 'gold': 'battery', 'forestgreen': 'CHP', 'red': 'misc'})
            legend_nodes = [plt.Line2D([], [], marker='o', color=color, label=label) for color, label in
                            legend_colors.items()]
            legend_nodes.append(
                plt.scatter([], [], s=big_node, marker='o', color='black', label='MV component'))
            legend_nodes.append(
                plt.scatter([], [], s=small_node, marker='o', color='black', label='LV component'))
            if legend:
                plt.legend(handles=legend_nodes)
            #nx.draw_networkx_edge_labels(self, pos, edge_labels=mixed_label, font_size=8)
        # plt.savefig("Fau_graph_big.png"
        plt.tight_layout()
        plt.show()




