import time
import pandas as pd
import numpy as np
import igraph as ig
import leidenalg
import networkit as nk
import matplotlib.pyplot as plt
from collections import defaultdict 

def leiden_cpm(graph, node_mapping, reverse_mapping, resolution):
    print(f"Running Leiden with CPM ({resolution}).", flush=True)
    start_time = time.time()
    
    partition = leidenalg.find_partition(
        graph, 
        leidenalg.CPMVertexPartition, 
        resolution_parameter=resolution
    )
    
    comm_to_nodes = defaultdict(list)
    for vertex_idx, comm_idx in enumerate(partition.membership):
        original_node_id = reverse_mapping[vertex_idx]
        comm_to_nodes[comm_idx].append(original_node_id)
    
    print(f"Runtime: {time.time() - start_time:.2f} s\n")
    
    num_nodes2, num_edges2 = 0, 0
    for comm, nodes in comm_to_nodes.items():
        vertex_indices = [node_mapping[node] for node in nodes if node in node_mapping]
        subgraph = graph.subgraph(vertex_indices)
        num_nodes2 += subgraph.vcount()
        num_edges2 += subgraph.ecount()    

    return partition, comm_to_nodes, num_nodes2, num_edges2

def leiden_modularity(graph, node_mapping, reverse_mapping):
    print(f"Running Leiden with Modularity.", flush=True)
    start_time = time.time()
    
    partition = leidenalg.find_partition(
        graph, 
        leidenalg.ModularityVertexPartition
    )
    
    comm_to_nodes = defaultdict(list)
    for vertex_idx, comm_idx in enumerate(partition.membership):
        original_node_id = reverse_mapping[vertex_idx]
        comm_to_nodes[comm_idx].append(original_node_id)
    
    print(f"Runtime: {time.time() - start_time:.2f} seconds\n")
    
    num_nodes2, num_edges2 = 0, 0
    for comm, nodes in comm_to_nodes.items():
        vertex_indices = [node_mapping[node] for node in nodes if node in node_mapping]
        subgraph = graph.subgraph(vertex_indices)
        num_nodes2 += subgraph.vcount()
        num_edges2 += subgraph.ecount()    

    return partition, comm_to_nodes, num_nodes2, num_edges2

def output_information(comm_to_nodes, num_nodes2, num_edges2, alg_name):
    cluster_sizes = [len(nodes) for comm, nodes in comm_to_nodes.items()]
    singleton_clusters = sum(1 for size in cluster_sizes if size == 1)
    non_singleton_clusters = sum(1 for size in cluster_sizes if size > 1)
    total_clusters = len(cluster_sizes)

    # Percentage    
    if total_clusters > 0:
        percentage_singleton = (singleton_clusters / total_clusters) * 100
        percentage_non_singleton = (non_singleton_clusters / total_clusters) * 100

    # Number of nodes in non-singleton clusters
    nodes_in_non_singleton = sum(size for size in cluster_sizes if size > 1)
    node_coverage = (nodes_in_non_singleton / num_nodes2) * 100 if num_nodes2 > 0 else 0
    non_singleton_sizes = [size for size in cluster_sizes if size > 1]
    
    if non_singleton_sizes:
        min_size = min(non_singleton_sizes)
        max_size = max(non_singleton_sizes)
        median_size = np.median(non_singleton_sizes)
        q1 = np.percentile(non_singleton_sizes, 25)
        q3 = np.percentile(non_singleton_sizes, 75)
    else:
        min_size = max_size = median_size = q1 = q3 = 0

    # Print output information
    print(f"-{alg_name}-", flush=True)
    print(f"Clustering:", flush=True)
    print(f"Total nodes: {num_nodes2}", flush=True)
    print(f"Total edges: {num_edges2}", flush=True)
    print(f"Total clusters: {total_clusters}", flush=True)
    print(f"Singleton clusters: {singleton_clusters}", flush=True)
    print(f"Non-singleton clusters: {non_singleton_clusters}", flush=True)
    print(f"Singleton clusters percentage: {percentage_singleton:.2f}%", flush=True)
    print(f"Non-singleton clusters percentage: {percentage_non_singleton:.2f}%", flush=True)
        
    print(f"\nNon-Singleton Cluster Size Distribution:", flush=True)
    if non_singleton_sizes:
        print(f"Minimum size: {min_size}", flush=True)
        print(f"First quartile: {q1:.2f}", flush=True)
        print(f"Median size: {median_size:.2f}", flush=True)
        print(f"Third quartile: {q3:.2f}", flush=True)
        print(f"Maximum size: {max_size}", flush=True)
    else:
        print(f"\nNo non-singleton clusters found.", flush=True)
        
    print(f"\nNode-Coverage:", flush=True)
    print(f"Nodes in non-singleton clusters: {nodes_in_non_singleton}", flush=True)
    print(f"Node-coverage percentage: {node_coverage:.2f}%", flush=True)
    


files = [
    'cen_cleaned.tsv',
    'cit_patents_cleaned.tsv',
    'wiki_topcats_cleaned.tsv',
    'cit_hepph_cleaned.tsv',
    'wiki_talk_cleaned.tsv'
]

resolutions = [0.01, 0.001]

dir_path = "/projects/illinois/eng/shared/shared/CS598GCK-SP25/assig2_networks/"

# Change i in files[i] to change object network, i in [0,4]
edgelist_df = pd.read_csv(dir_path+files[0], sep='\t', header=None, names=['from', 'to'], dtype=str)
edgelist_df["unique_id"] = edgelist_df.apply(lambda row: "-".join(np.sort(row)), axis=1)
edgelist_df = edgelist_df.drop_duplicates(subset=["unique_id"], keep="first")
num_edges1 = len(edgelist_df)

nodelist_array = edgelist_df[["from", "to"]].stack().unique()
nodelist_array = np.array(nodelist_array, dtype=int)
num_nodes1 = len(nodelist_array)

unique_nodes = sorted(nodelist_array)
node_mapping = {node: idx for idx, node in enumerate(unique_nodes)}
reverse_mapping = {idx: node for node, idx in node_mapping.items()}

# Create networkit graph
G = nk.graph.Graph(num_nodes1, weighted=False, directed=False)	# Create an empty undirected graph
for _, row in edgelist_df.iterrows():
        fr0m = int(row['from'])
        to = int(row['to'])
        G.addEdge(node_mapping[fr0m], node_mapping[to])

print(f"Graph created with {num_edges1} edges and {num_nodes1} nodes.", flush=True)

# Convert to igraph
edges = []
for u, v in G.iterEdges():
    edges.append((u, v))

mapped_edges = edges    

# Create igraph
ig_graph = ig.Graph()
ig_graph.add_vertices(len(node_mapping))
ig_graph.add_edges(mapped_edges)

# Run Leiden CPM (0.01)
print(f"\n" + "="*50)
partition_cpm_001, comm_cpm_001, num_nodes2, num_edges2 = leiden_cpm(
        ig_graph, 
        node_mapping,
        reverse_mapping,
        resolution=resolutions[0]
)
output_information(comm_cpm_001, num_nodes2, num_edges2, "Leiden CPM (0.01)")

# Run Leiden CPM (0.001)
print(f"\n" + "="*50)
partition_cpm_0001, comm_cpm_0001, num_nodes2, num_edges2 = leiden_cpm(
        ig_graph, 
        node_mapping,
        reverse_mapping,
        resolution=resolutions[1]
)
output_information(comm_cpm_0001, num_nodes2, num_edges2, "Leiden CPM (0.001)")

# Run Leiden modularity
print(f"\n" + "="*50)
partition_mod, comm_mod, num_nodes2, num_edges2 = leiden_modularity(
        ig_graph, 
        node_mapping,
        reverse_mapping
)
output_information(comm_mod, num_nodes2, num_edges2, "Leiden Modularity")

print(f"\n" + "="*50)
print(f"\n- MISSION COMPLETED -", flush=True)

