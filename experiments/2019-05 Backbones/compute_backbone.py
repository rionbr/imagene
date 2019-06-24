# coding=utf-8
# Author: Rion B Correia
# Date: May 21, 2019
#
# Description:
# Loads a network adjacecy matrix and computes its backbone.
#
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 40)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)
import networkx as nx
from distanceclosure.utils import _prox2dist as prox2dist
from distanceclosure.dijkstra import Dijkstra
from distanceclosure.cython._dijkstra import _cy_single_source_shortest_distances


if __name__ == '__main__':

    #
    # Init variables
    #
    # uinput = 'correlation'  # correlation, powerlaw, tom
    uinput = input("Which file to load [correlation,powerlaw,tom]:")

    # Paths
    data_path = '../../data/'
    result_path = 'results/'

    # Which file are we computing the backbone
    if uinput == 'correlation':
        file = '1-updatedCorrMatfile'
    elif uinput == 'powerlaw':
        file = '2-powerlawmat'
    elif uinput == 'tom':
        file = '3-TOMfile'
    else:
        raise Exception('Input matrix must be either "correlation", "powerlaw", or "tom".')
    #
    csv_extension = '.csv'
    gpickle_extension = '.gpickle'

    # Read/Write file names
    print("Loading file: '{:s}'.".format(file + csv_extension))
    rCSVFile = data_path + file + csv_extension
    wGPickleFile = result_path + file + '-backbone' + gpickle_extension
    wCSVFile = result_path + file + '-backbone' + csv_extension

    # Load File
    df = pd.read_csv(rCSVFile, index_col=0, nrows=None)

    # Debug
    # df = df.iloc[:5, :5].copy()
    n_nodes = df.shape[0]

    # Rename Nodes (node names must start from 0)
    index_bkp = df.index
    columns_bkp = df.columns
    df.index = range(0, n_nodes)
    df.columns = range(0, n_nodes)



    df.columns = df.index
    nodelist = df.index.values

    # Build Networkx object
    print('--- Building Network ---')
    # C for Correlation network
    G = nx.from_pandas_adjacency(df, create_using=nx.Graph)

    # P for Proximity network (which in this case is a Correlation)
    P = [w for i, j, w in G.edges.data('weight')]

    # Converts (P)roximity to (D)istance using a map.
    D_dict = dict(zip(G.edges(), map(prox2dist, P)))
    # Set the distance value for each edge
    nx.set_edge_attributes(G, name='distance', values=D_dict)
    # Compute closure (Using the Dijkstra Class directly)
    print('--- Computing Dijkstra APSP ---')
    dij = Dijkstra.from_edgelist(D_dict, directed=False, verbose=10)
    # Serial Computation
    poolresults = list(range(len(dij.N)))
    for node in dij.N:
        print('> Dijkstra node %s of %s' % (node + 1, len(dij.N)))
        # Computes the shortest distance and paths between a source node and every other node
        poolresults[node] = _cy_single_source_shortest_distances(node, dij.N, dij.E, dij.neighbours, ('min', 'sum'), verbose=10)
    shortest_distances, local_paths = map(list, zip(*poolresults))
    dij.shortest_distances = dict(zip(dij.N, shortest_distances))
    # (S)hortest (D)istances
    SD = dij.get_shortest_distances(format='dict', translate=True)
    print('> Done.')

    print('> Populating (G)raph')
    # Mapping results to triplet (i, j): v -  Convert Dict-of-Dicts to Dict
    Cm = {(i, j): v for i, jv in SD.items() for j, v in jv.items()}

    # Cm contains two edges of each. Make sure we are only inserting one
    edges_seen = set()
    for (i, j), cm in Cm.items():
        # Knowledge Network is undirected. Small ids come first.
        if (i, j) in edges_seen or (i == j):
            continue
        else:
            edges_seen.add((i, j))

            # New Edge?
            if not G.has_edge(i, j):
                # Self-loops have proximity 1, non-existent have 0
                proximity = 1 if i == j else 0
                G.add_edge(i, j, distance=np.inf, proximity=proximity, distance_metric_closure=cm, metric_backbone=False)
            else:
                G[i][j]['distance_metric_closure'] = cm
                G[i][j]['metric_backbone'] = True if ((cm == G[i][j]['distance']) and (cm != np.inf)) else False

    print('--- Calculating S Values ---')
    S = {
        (i, j): d['distance'] / d['distance_metric_closure']
        for i, j, d in G.edges(data=True, default=0)
        if ((d.get('distance') < np.inf) and (d.get('distance_metric_closure') is not None))
    }
    nx.set_edge_attributes(G, name='s_value', values=S)

    print('--- Calculating B Values ---')
    mean_distance = {
        k: np.mean(
            [d['distance'] for i, j, d in G.edges(nbunch=k, data=True)]
        )
        for k in G.nodes()
    }
    print('> b_ij')
    B_ij = {
        (i, j): float(mean_distance[i] / d['distance_metric_closure'])
        for i, j, d in G.edges(data=True)
        if (d.get('distance') == np.inf)
    }
    nx.set_edge_attributes(G, name='b_ij_value', values=B_ij)

    print('> b_ji')
    B_ji = {
        (i, j): float(mean_distance[j] / d['distance_metric_closure'])
        for i, j, d in G.edges(data=True)
        if (d.get('distance') == np.inf)
    }
    nx.set_edge_attributes(G, name='b_ji_value', values=B_ji)

    print('--- Saving files ---')

    # Save Graph Object
    nx.write_gpickle(G, wGPickleFile)

    # Only extract the metric backbone
    Bm = G.copy()
    Bm.remove_edges_from([(i, j) for i, j, d in Bm.edges(data=True) if d.get('metric_backbone') is False])
    
    # Save Adjacency Object
    dfB = nx.to_pandas_adjacency(Bm, nodelist=nodelist, weight='weight', nonedge=np.nan)
    dfB.index = index_bkp
    dfB.columns = columns_bkp
    dfB.to_csv(wCSVFile)

    print("done.")
