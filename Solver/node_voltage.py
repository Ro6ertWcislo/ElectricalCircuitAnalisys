from Solver.reader import read_as_graph
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx


class Analyzer(object):
    def __init__(self, graph_path):
        self.G, self.zero_node, self.U_node, self.U = read_as_graph(graph_path)
        self.nodes = {}
        for node in self.G.nodes():
            self.nodes[node] = 0.0

    def compute(self):
        partial_matrix = np.zeros((self.G.number_of_nodes(), self.G.number_of_nodes()))
        result = np.zeros(self.G.number_of_nodes())
        self.num = {node: node_num for node_num, node in enumerate(self.G.nodes())}
        for node in self.G.nodes():
            if node == self.zero_node:
                partial_matrix[self.num[node], self.num[node]] = 1.0
            elif node == self.U_node:
                partial_matrix[self.num[node], self.num[node]] = 1.0
                result[self.num[node]] = self.U
            else:
                edge_dict = {}
                for edge in self.G.edges([node]):
                    node, n2 = edge
                    edge_dict[n2] = self.G[node][n2]['weight']
                partial_matrix[self.num[node], self.num[node]] = sum([1 / R for R in edge_dict.values()])
                for other_node, R in edge_dict.items():
                    partial_matrix[self.num[node], self.num[other_node]] = -1 * (1 / R)

        return np.linalg.solve(partial_matrix, result)

    def assign(self):
        res = self.compute()
        self.I = []
        for edge in self.G.edges():
            n1, n2 = edge
            R = self.G[n1][n2]['weight']
            V1 = res[self.num[n1]]
            V2 = res[self.num[n2]]
            I = abs(V1 - V2) / R
            self.G[n1][n2]['weight'] = I
            self.I.append(I)


x = Analyzer("abc.csv")
x.assign()
y = x.I
print(y)

jet = cm = plt.get_cmap('jet')
cNorm = colors.Normalize(vmin=np.min(y), vmax=np.max(y))
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
colorList = []

for i in range(len(y)):
    colorVal = scalarMap.to_rgba(y[i])
    colorList.append(colorVal)

nx.draw(x.G)
plt.subplot(122)

nx.draw_shell(x.G, edge_color=colorList, width='3')
plt.show()
