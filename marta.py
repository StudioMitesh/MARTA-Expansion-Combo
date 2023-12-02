# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %%
martabase = pd.read_csv("martabase_final.csv")

# %%
martabase.head()

# %%
martabase.shape

# %%
import folium

mapmarta=folium.Map(location=[33.934150, -84.246706])

for ind in range(len(martabase["Station"])):
    name = martabase.iloc[ind,1]
    lat = martabase.iloc[ind,2]
    longi = martabase.iloc[ind,3]
    folium.Marker([lat,longi],tooltip=name, icon=folium.Icon(color='black',icon_color='red')).add_to(mapmarta)
mapmarta

# %%
from collections import defaultdict

class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = defaultdict(list)

    def addEdge(self, v1, v2):
        self.graph[v1].append(v2)
        self.graph[v2].append(v1)

    def DepthFirst(self, v, visited_list):
        visited_list[v] = True
        for i in self.graph[v]:
            if visited_list[i] == False:
                self.DepthFirst(i, visited_list)
    
    def isConnected(self):
        visited_list = False*(self.V)
        for i in range(self.V):
            if(len(self.graph[i]) != 0):
                break
        if i == self.V-1:
            return True
       
        self.DepthFirst(i, visited_list)
 
        # Check if all non-zero degree vertices are visited
        for i in range(self.V):
            if visited_list[i] == False and len(self.graph[i]) > 0:
                return False
 
        return True
    
    def isEulerian(self):
        # Check if all non-zero degree vertices are connected
        if self.isConnected() == False:
            return 0
        else:
            # Count vertices with odd degree
            odd = 0
            for i in range(self.V):
                if len(self.graph[i]) % 2 != 0:
                    odd += 1
            #number of odds:
                # 0 -> euler cycle
                # 2 -> euler path 
                # >2 -> not eulerian 
            if odd == 0:
                return 2 #euler cycle
            elif odd == 2:
                return 1 #euler path
            elif odd > 2:
                return 0 #no euler
    
    def test(self):
        res = self.isEulerian()
        if res == 0:
            print("Graph is not Eulerian")
        elif res == 1:
            print("Graph has a Euler path")
        else:
            print("Graph has a Euler cycle")

# %%
g = Graph(160)
g.addEdge(1,2)

# %%
names = martabase.iloc[:,1]
latitudes = martabase.iloc[:,2]
longitudes = martabase.iloc[:,3]
longitudes.head()

# %%
npnames = names.to_numpy()
nplats = latitudes.to_numpy()
nplongs = longitudes.to_numpy()

# %%
from sklearn.preprocessing import MinMaxScaler

nplats = nplats.reshape(-1, 1)
nplongs = nplongs.reshape(-1, 1)

lats_fin = MinMaxScaler().fit_transform(nplats)
longs_fin = MinMaxScaler().fit_transform(nplongs)

# %%
lats_fin[0]

# %%
longs_fin[0]

# %%
longs = longs_fin.flatten()
lats = lats_fin.flatten()

# %%
vertices_finnn = pd.DataFrame({'Station': npnames, 'Latitude': lats, 'Longitude': longs})

martagraph = vertices_finnn.set_index('Station')[['Longitude', 'Latitude']].apply(tuple, axis=1).to_dict()


# %%
vertices_f = martabase.iloc[:,2:4]
vertices_f.head()

# %%
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math

# %%
vertices_final = vertices_f.to_numpy().tolist()
vertices_final[0]

# %%
def distance_formula(x1, y1, x2, y2):
   dist = math.sqrt( (math.pow( (x1 - x2) ,2)) + (math.pow( (y1 - y2) ,2)) )
   return dist

def knn_edges(n, k, vertices):
    edges = []
    for key, v in vertices.items():
        # Values for x1,x2
        name = key
        y1 = v[0]
        x1 = v[1]

        # Calculate distances to other vertices
        distances = []
        for key2, u in vertices.items():
            if u[0]!=y1 and u[1]!=x1: 
                try:
                    distances.append((u, distance_formula(x1, y1, u[1], u[0])))
                except Exception as e:
                    print(f"Error calculating distance: {e}")
        
        # Sort distances
        distances.sort(key=lambda x: x[1])

        # Select k nearest neighbors and add edges
        for j in range(k):
            u, _ = distances[j]
            val = {i for i in martagraph if martagraph[i]==u}
            va = val.pop()
            edges.append((name, va, distances[j][1]))

    return edges

G = nx.Graph()

# Step 3: Add nodes and edges
for name, coordinates in martagraph.items():
    G.add_node(name, pos=coordinates)  # Add node with 'pos' attribute

edges = knn_edges(160, 8, martagraph)
#print(edges)
G.add_weighted_edges_from(edges)

# Plot the graph
pos = nx.get_node_attributes(G, 'pos')
nx.draw(G, pos, with_labels=False, node_size=10)
plt.show()

# %%
for k in range(20):
    G2 = nx.Graph()

    # Step 3: Add nodes and edges
    for name, coordinates in martagraph.items():
        G2.add_node(name, pos=coordinates)  # Add node with 'pos' attribute

    edges = knn_edges(160, k, martagraph)
    G2.add_weighted_edges_from(edges)
    try: 
        eulerian_circuit = list(nx.eulerian_circuit(G))
        # Print the Eulerian circuit
        print("Eulerian Circuit:", eulerian_circuit)

        # Visualize the graph and the Eulerian circuit
        pos = nx.spring_layout(G2)
        nx.draw(G2, pos, with_labels=True, font_weight='bold')
        nx.draw_networkx_edges(G2, pos, edgelist=eulerian_circuit, edge_color='r', width=2)
        plt.show()
    except Exception as e:
        print(f"Error not Eulerian: {e}")

# %%
'''def hamiltonian(graph, start, path=[]):
    path = path + [start]
    if len(path) == len(graph.nodes):
        return path
    for node in graph.neighbors(start):
        if node not in path:
            new_path = hamiltonian(graph, node, path)
            if new_path:
                return new_path
    return None

starting_node = list(G.nodes)[0]

hamiltonian_path = hamiltonian(graph=G, start=starting_node)

print("Hamiltonian Path:", hamiltonian_path)
'''
# Visualize the graph and the Hamiltonian path
'''
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, font_weight='bold')
nx.draw_networkx_nodes(G, pos, nodelist=hamiltonian_path, node_color='r')
nx.draw_networkx_edges(G, pos, edgelist=[(hamiltonian_path[i], hamiltonian_path[i + 1]) for i in range(len(hamiltonian_path) - 1)], edge_color='r', width=2)
plt.show()
'''

# %%
Ge = nx.euler.eulerize(G)

eulerian_circuit = list(nx.eulerian_circuit(Ge))

# Print the Eulerian circuit
print("Eulerian Circuit:", eulerian_circuit)

# Visualize the graph and the Eulerian circuit
pos = nx.get_node_attributes(Ge, 'pos')
nx.draw(Ge, pos, with_labels=False, node_size=12)
nx.draw_networkx_edges(Ge, pos, edgelist=eulerian_circuit, edge_color='r', width=1.5)
plt.show()

# %%
# example path provided by app from Avalon to Airport station using weighted Dijkstra's algo
lp = True
while lp:
    try:
        start = input("Enter your starting location: ")
        end = input("Enter your destination: ")
        print(nx.dijkstra_path(G, start, end))
        lp = False
    except Exception as e:
            print(f"Error: {e}")


# %%
for u, v, data in G.edges(data=True):
    if 'weight' not in data:
        G[u][v]['weight'] = 0.05
    #print(f"Edge ({u}, {v}): {data}")


# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric
import tensorflow as tf

node_features = torch.randn(160, 16)
data = torch_geometric.utils.from_networkx(G)
data.x = node_features
print(data)

# Define and train a GNN model
class GNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GNN, self).__init__()
        self.conv1 = torch_geometric.nn.GraphConv(input_size, hidden_size)
        self.conv2 = torch_geometric.nn.GraphConv(hidden_size, output_size)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

model = GNN(input_size=16, hidden_size=64, output_size=2)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Train the model on graph data
for epoch in range(20):
    optimizer.zero_grad()
    output = model(data.x, data.edge_index)
    target = torch.ones_like(output)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch + 1}/{20}, Loss: {loss.item()}')

# uses the learned embeddings to guide Hamiltonian path search
node_embeddings = model.conv1(data.x, data.edge_index)
print(node_embeddings)

# %% [markdown]
# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=b3686541-b591-45d5-90b4-564457237abf' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>


