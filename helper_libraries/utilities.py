import tweepy as tw
import json
import pandas as pd
import numpy as np

from collections import defaultdict, Counter
import os
from IPython.display import clear_output

import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

ENTITY_TYPES_TEXT_PROP={
    "user_mentions": "screen_name",
    "hashtags": "text",
}

DATA_DIR=os.path.join(os.path.dirname(__file__), "../data")
TWITTER_CONFIG_FILE=os.path.join(DATA_DIR, "twitter_config.json")

api=None
def setup_twitter():
    global api
    if not os.path.isfile(TWITTER_CONFIG_FILE):
        with open(os.path.join(DATA_DIR, "twitter_config.sample.json")) as fp:
            creds = json.load(fp)
            for k in sorted(creds.keys()):
                v = input("Enter %s:\t" % k)
                creds[k] = v
        print(creds)
        with open(TWITTER_CONFIG_FILE, "w+") as fp:
            json.dump(creds, fp, indent=4, sort_keys=True)
        clear_output()
        print("Printed credentials to file %s" % TWITTER_CONFIG_FILE)
    with open(TWITTER_CONFIG_FILE) as fp:
        creds = json.load(fp)
    print(creds.keys())

    auth = tw.OAuthHandler(creds["consumer_key"], creds["consumer_secret"])
    auth.set_access_token(creds["access_token"], creds["access_token_secret"])
    api = tw.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True,
                 retry_count=5, retry_delay=100, 
                )
    print("Tweepy ready for search")

def get_statuses(query_term, num_statuses=100):
    """Collect statuses using tweepy
    
    Input:
    ------
        query_term: query term to use for searching twitter
        num_statuses: number of statuses to collect. (default=100)
        
    Returns:
    --------
        statuses: list of status objects
    """
    if api is None:
        setup_twitter()
    statuses = [
        status
        for status in tw.Cursor(
            api.search,
            q=query_term, count=num_statuses
        ).items(num_statuses)
    ]
    return statuses


def get_network_from_count_data(node_counts, edge_counts):
    """Create entity networks from data
    
    Inputs:
    -------
        node_counts: a dict with nodes as keys and values as attributes
        edge_counts: a dict with edges as keys (source, target) and values as attributes
        
        
    Returns:
    -------
        df_entities, df_entity_pairs, G
    
    """
    # Convert data to dataframes          
    df_entities = pd.DataFrame(
        list(node_counts.items()),
        columns=["entity", "counts"]
    ).sort_values(
        "counts",
        ascending=False
    ).reset_index(drop=True)
    
    df_entity_pairs = pd.DataFrame([
        (k1, k2, v)
        for (k1,k2), v in edge_counts.items()
    ],
        columns=[
            "source",
            "target",
            "counts"
        ]).sort_values(
        "counts",
        ascending=False
    ).reset_index(drop=True)
    
    ## Generate nx graph instance
    G = nx.Graph()
    G.add_nodes_from(df_entities.entity)
    G = nx.from_pandas_edgelist(
        df_entity_pairs,
        "source",
        "target",
        edge_attr="counts",
        create_using=G
    )
    return df_entities, df_entity_pairs, G




def get_entity_networks_from_statuses(statuses, entity_type):
    """Extract entity networks from tweet statuses
    
    Input:
    ------
        statuses: a list or iterable containing status json from the twitter API
        entity_type: "hashtags", "user_mentions"
        
    Returns:
    --------
        df_entities, df_entity_pairs, G
    
    """
    assert entity_type in ENTITY_TYPES_TEXT_PROP, "entity_type should be one of '{}'".format(
        ",".join(ENTITY_TYPES_TEXT_PROP.keys())
    )
    text_property = ENTITY_TYPES_TEXT_PROP[entity_type]
    entity_counts = defaultdict(int)
    entity_network = defaultdict(int)
    for status in statuses:
        for i, entity in enumerate(status.entities[entity_type]):
            entity_counts[entity[text_property].lower()] += 1
            for j, entity_2 in enumerate(status.entities[entity_type][i+1:], start=i+1):
                entity_network[(
                    entity[text_property].lower(),
                    entity_2[text_property].lower()
                )] += 1
                
    return get_network_from_count_data(entity_counts, entity_network)


def get_connected_components(G):
    """Get connected components of a Graph
    
    Input:
    ------
        G: networkx graph
        
    Return:
    -------
        connected_components: a list with subgraphs of each connected components sorted by size (largest first)
    """
    connected_components = sorted(
        nx.connected_component_subgraphs(G),
        key = len,
        reverse=True
    )
    print("{} connected components found.".format(
        len(connected_components)
    ))
    return connected_components



def get_all_node_metrics(G):
    """Return node metrics for the graph
    
    Input:
    ------
        G: networkx graph
    
    Returns:
    --------
        a dataframe containing metrics for each node. 
        currently following centrality measures along with 
        node level clustering coefficients are supported:
            degree, betweenness, closeness, eigenvector
    """
    df = pd.DataFrame(index=G.nodes)
    metric_dict = {
        "degree": nx.degree_centrality,
        "betweenness": nx.betweenness_centrality,
        "closeness": nx.closeness_centrality,
        "eigenvector": nx.eigenvector_centrality,
        "clustering": nx.clustering
    }
    for k, f in metric_dict.items():
        try:
            value_dict = f(G)
            nx.set_node_attributes(G, value_dict, k)
            df[k] = pd.Series(value_dict)
        except:
            df[k] = np.nan
    return df

def get_all_graph_metrics(G):
    """Return graph level metrics
    
    Input:
    ------
        G: networkx graph
    
    Returns:
    --------
        a dataframe containing metrics for the graph. 
        currently following centrality measures along with 
        node level clustering coefficients are supported:
            degree, betweenness, closeness, eigenvector
    """
    index = [
        "avg. path length",
        "clustering coefficents",
        "density",
        "transitivity",
        "connected components",
        "diameter",
        "radius",
    ]
    values = []
    try:
        values.append(nx.average_shortest_path_length(G))
    except nx.NetworkXError:
        values.append(np.nan)
    values.append(nx.average_clustering(G))
    values.append(nx.density(G))
    values.append(nx.transitivity(G))
    values.append(nx.number_connected_components(G))
    try:
        values.append(nx.diameter(G))
    except nx.NetworkXError:
        values.append(np.nan)
    try:
        values.append(nx.radius(G))
    except nx.NetworkXError:
        values.append(np.nan)
    df_measures = pd.DataFrame(
        np.array([values]).T,
        columns=["values"],
        index=index
    )
    return df_measures

def dict_to_values(G, dict_data):
    """Helper function to convert a dictionary of node values to list
    
    Inputs:
    -------
        G: networkX graph
        dict_data: a dictionary like structure which has keys as nodes and values
    
    Returns:
    --------
        a list containing the values in the same order as graph nodes
    """
    return [dict_data[n] for n in G.nodes]

def plot_network(
    G,
    node_sizes=None,
    factor=10,
    node_color_col=None,
    **kwargs
):
    """Plot graph with node sizes depending on specific values
    
    Inputs:
    -------
        G: networkX graph
        node_size_dict: can be dict, int, or a string identifying node attribute
            if dict it should have nodes as keys and values used for sizing the node
            if string the value should be present in the node attribute
        factor: multiply dict values to get node size, if no node_size_dict is provided this will be used to size nodes
        node_color: node attribute to color by
        **kwargs: all the other arguments as expected by networkx.draw_networkx function. 
             Details at: https://networkx.github.io/documentation/stable/reference/generated/networkx.drawing.nx_pylab.draw_networkx.html#networkx.drawing.nx_pylab.draw_networkx
             
        
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    node_size_values=factor
    if isinstance(node_sizes, str):
        node_sizes=G.nodes.data(node_sizes)
    if node_sizes is not None:
        node_size_values = [
            v*factor
            for v in dict_to_values(
                G,
                node_sizes
            )
        ]
    node_color_values = None
    if node_color_col:
        palette = sns.color_palette("Set2", 10)
        color_col_values = list(dict(G.node.data(node_color_col)).values())
        node_color_values = list(map({
            k: palette[i]
            for i, k in enumerate(np.unique(color_col_values))
        }.get, color_col_values))
    nx.draw(
        G, 
        pos=nx.spring_layout(G),
        with_labels=True,
        node_size=node_size_values,
        node_color=node_color_values,
        ax=ax,
        width=0.1,
        **kwargs
    )
    
    
## UCINET helpers

def read_UCINET_matrix(xlsx_file, sheet_name, attribute_file=None, directed=False):
    """Read xlsx files from UCINET as graph
    
    Inputs:
    -------
        xlsx_file: path to excel file created by exporting network from UCINET matrix editor
        sheet_name: name of the sheet
        attribute_file: excel file with attribute information
        
    Returns:
    --------
        G: networkx graph
    """
    df = pd.read_excel(
        xlsx_file,
        sheet_name=sheet_name,
        skiprows=1,
        index_col=1
    ).drop("Unnamed: 0", axis=1)
    G = nx.Graph()
    if directed:
        G = nx.DiGraph()
    G = nx.from_numpy_matrix(df.values, create_using=G)
    G = nx.relabel_nodes(G, {i: v for i, v in enumerate(df.index.values)})
    if attribute_file:
        df_attr = pd.read_excel(
            attribute_file,
            sheet_name="1",
            skiprows=1,
            index_col=1
        ).drop("Unnamed: 0", axis=1)
        for c in df_attr.columns:
            nx.set_node_attributes(G, df_attr[c].to_dict(), c)
    return G


def get_nodes_as_dataframe(G):
    """Return node data as dataframe
    
    Input:
    ------
        G: networkx graph
       
    Returns:
    --------
        dataframe with node attributes
    """
    df = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient="index")
    return df

## Simulation helpers
def run_all_simulations(G, iters=10):
    n=len(G.nodes)
    fig, ax = plt.subplots(2,2, figsize=(10,10))
    ax = ax.flatten()
    def run_small_world_simulations():
        simulated_apl = []
        simulated_cc = []
        for i in range(iters):
            try:
                g = nx.watts_strogatz_graph(n, k=2, p=1)
                apl = nx.average_shortest_path_length(g)
                simulated_apl.append(apl)
                cc = nx.average_clustering(g)
                simulated_cc.append(cc)
            except:
                pass
        return simulated_apl, simulated_cc
    
    def run_simulations_pref():
        degree_dist = []
        for i in range(iters):
            try:
                g = nx.barabasi_albert_graph(n, m=1)
                degree_dist.append([k[1] for k in g.degree])
            except:
                pass
        return degree_dist
    
    def run_simulations_random():
        degree_dist = []
        for i in range(iters):
            try:
                g = nx.erdos_renyi_graph(n, p=nx.density(G))
                degree_dist.append([k[1] for k in g.degree])
            except:
                pass
        return degree_dist
    
    simulated_apls, simulated_cc = run_small_world_simulations()
    
    ax[0].hist(simulated_apls, bins=20, facecolor="0.7")
    try:
        ax[0].axvline(nx.average_shortest_path_length(G), color="r")
    except nx.NetworkXError:
        print("Graph not connected. Hence can't comput avg. path length.")
    ax[0].set_xlabel("Average path length")
    ax[0].set_ylabel("Frequency")
    ax[0].set_title("Small World")
    
    ax[1].hist(simulated_cc, bins=20, facecolor="0.7")
    ax[1].axvline(nx.average_clustering(G), color="r")
    ax[1].set_xlabel("Clustering coefficient")
    ax[1].set_ylabel("Frequency")
    ax[1].set_title("Small World")
    
    def get_degree_dist(dd, probs=False):
        degree, counts = tuple(zip(*[
            k for k in sorted(Counter(dd).items(), key=lambda x: x[0])
        ]))
        degree = np.array(degree)
        counts = np.array(counts)
        if probs:
            counts = counts/counts.sum()
            #counts = 1-(counts.cumsum() / counts.sum())
        return degree, counts
    
    def plot_degree_dist(f, i):
        degree_dists = f()
        for dd in degree_dists:
            degree, counts = get_degree_dist(dd, probs=True)
            ax[i].plot(degree, counts, color="0.7", linestyle="-", alpha=0.3)
        degree, counts = get_degree_dist([k[1] for k in G.degree], probs=True)
        ax[i].plot(
            degree,
            counts,
            color="red",
            marker="s",
            linestyle="none",
            alpha=0.9,
            ms=5
        )
        ax[i].set_xlabel("degree$(X)$")
        ax[i].set_ylabel("P(X)")
    
    plot_degree_dist(run_simulations_random, 2)
    ax[2].set_title("Random Graph")
    
    plot_degree_dist(run_simulations_pref, 3)
    ax[3].set_title("Pref Attachement")
    
    sns.despine(offset=10)
    fig.tight_layout()
    
    print("Grey points and lines are from the simulation, red points and lines are from actual data.")
    