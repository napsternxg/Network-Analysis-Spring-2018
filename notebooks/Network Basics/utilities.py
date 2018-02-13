import pandas as pd
import networkx as nx

def read_UCINET_matrix(xlsx_file, sheet_name):
    df = pd.read_excel(xlsx_file, sheet_name=sheet_name, skiprows=1, index_col=1).drop("Unnamed: 0", axis=1)
    G = nx.from_numpy_matrix(df.values)
    G = nx.relabel_nodes(G, {i: v for i, v in enumerate(df.index.values)})
    return G


def get_all_node_metrics(G):
    df = pd.DataFrame(index=G.nodes)
    df["degree"] = pd.Series(nx.degree_centrality(G))
    df["betweenness"] = pd.Series(nx.betweenness_centrality(G))
    df["closeness"] = pd.Series(nx.closeness_centrality(G))
    df["eigenvector"] = pd.Series(nx.eigenvector_centrality(G))
    df["clustering"] = pd.Series(nx.clustering(G))
    return df


def dict_to_values(G, dict_data):
    return [dict_data[n] for n in G.nodes]

def plot_network(G, node_size_dict, factor=10, **kwargs):
    nx.draw(
        G, 
        pos=nx.spring_layout(G),
        with_labels=True,
        node_size=[v*factor for v in dict_to_values(G, node_size_dict)],
        **kwargs
    )