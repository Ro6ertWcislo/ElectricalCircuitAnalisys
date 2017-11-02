import csv
import networkx as nx


def read_as_graph(csv_path):
    with open(csv_path) as csvfile:
        G = nx.Graph()
        data_reader = csv.reader(csvfile, delimiter=';')
        data_reader = [(int(row[0]), int(row[1]), float(row[2])) for row in data_reader]
        start_node, end_node, U = data_reader[0]
        G.add_weighted_edges_from(
            [row for row in data_reader]
        )
        return G, start_node, end_node, U
