import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from Solver.reader import read_as_graph




class Analyzer(object):
    def __init__(self, graph_path):
        self.G, self.n_positive, self.n_negative, self.U = read_as_graph(graph_path)
        self.edge_dict = self.assign_edge_dict()
        # self.valid I kirchoff nodes

    def get_I_kirchoff_equations(self):
        partial_matrix = np.zeros((self.G.number_of_nodes(), self.G.number_of_edges()))
        for node_num, node in enumerate(self.G.nodes()):
            if not self.special_node(node):
                for edge in self.G.edges([node]):
                    n1, n2 = edge
                    edge_num, value = self.edge_dict[(n1, n2)]
                    partial_matrix[node_num, edge_num] = value
        return partial_matrix

    def special_node(self, node):
        return node == self.n_negative or node == self.n_positive

    def get_II_kirchoff_equations(self):
        cycles = nx.cycle_basis(self.G)
        partial_matrix = np.zeros((len(cycles), self.G.number_of_edges()))
        result = np.zeros(len(cycles))
        for cycle_num, cycle in enumerate(cycles):
            for i in range(len(cycle)):
                n1, n2 = cycle[i], cycle[(i + 1) % len(cycle)]
                edge_num, sign = self.edge_dict[(n1, n2)]
                if edge_num == "special edge":
                    result[cycle_num] = self.U
                else:
                    partial_matrix[cycle_num, edge_num] = sign * self.G.get_edge_data(n1, n2)["weight"]

        return partial_matrix,result

    def assign_edge_dict(self):
        y = {}
        for num, edge in enumerate(self.G.edges()):
            (a, b) = edge
            y[(a, b)] = (num, 1)
            y[(b, a)] = (num, -1) ##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        y[(self.n_positive, self.n_negative)] = ("special edge", 0)
        y[(self.n_negative, self.n_positive)] = ("special edge", 0)
        return y


# G = nx.DiGraph([(0, 0), (0, 1), (0, 2), (1, 2), (2, 0), (2, 1), (2, 2)])
# for cycle in nx.simple_cycles(G):
#     print(cycle)

x = Analyzer("abc.csv")
a1 = x.get_I_kirchoff_equations()
f,_ = a1.shape
b1 = np.zeros(f)
#print(nx.cycle_basis(x.G))

a2,b2 = x.get_II_kirchoff_equations()

b = np.concatenate((b1,b2))

a = np.vstack((a1,a2))


y = np.absolute(np.linalg.lstsq(a,b)[0])
print(y)
for (a,b),(num,_) in x.edge_dict.items():
    if num != 'special edge':
        x.G[a][b]['weight'] = y[num]





# G, s, e, d = read_as_graph("abc.csv")
# print(G.nodes()[0],
#       G.number_of_nodes())
#
# for node in G.nodes():
#     print(G.edges([node]))
#
# G = nx.petersen_graph()
# plt.subplot(121)
#

jet = cm = plt.get_cmap('jet')
cNorm  = colors.Normalize(vmin=np.min(y), vmax=np.max(y))
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
colorList = []

for i in range(len(y)):
  colorVal = scalarMap.to_rgba(y[i])
  colorList.append(colorVal)

print(colorList)
print(np.argmin(y))


nx.draw(x.G)
plt.subplot(122)

nx.draw_shell(x.G,edge_color = colorList, width='3')
plt.show()

for node in x.G.nodes():
    print(sum([x.G[edge[0]][edge[1]]['weight'] for edge in x.G.edges([node])]))

# nx.draw(x.G,edge_color=colorList)
# plt.savefig("simple_path.png") # save as png
# plt.show() # display