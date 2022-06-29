from typing import Iterable, Dict, List
import numpy as np
from copy import copy, deepcopy
from data/


def kmers(seq: str, k: int, stride: int = 1) -> Iterable[str]:
    """Generate a list of k-mers from a given string.

    Parameters
    ----------
    seq: str
        A string which will be decomposed into k-mers.
    k: int
    stride: int

    Returns
    -------
    Iterable[str]

    """
    k_mers = []
    for i in range(0, len(seq)-k+1, stride):
        k_mers.append(seq[i:i+k])    
    return k_mers


def de_bruijn_graph(sequences: Iterable[str], k: int) -> Dict[str, List[str]]:
    """Construct a De Bruijn graph from a list of sequences.

    Given a list of strings, decompose each string into its respective k-mers
    (each string independently). When constructing k-mers, always use stride=1
    Then, construct the corresponding De Bruijn graph using all the k-mers
    produced in the first step. The De Bruijn graph should be given in edge-list
    format using a dictionary, where each dictionary entry corresponds to a node
    and its corresponding neighbors.

    For instance, given a string "abcde", you should return a graph
    corresponding to "abc" -> "bcd" -> "cde". If multiple strings are given with
    no overlap, your graph will contain disconnected components, e.g. for
    sequences "aaaa" and "bbbb", the graph should contain two disconnected
    components "aaa" -> "aaa" and "bbb" -> "bbb".

    Please use the tests to verify your implementation.

    Parameters
    ----------
    sequences: Iterable[str]
    k: int
        k for k-mers

    """
    list_kmers = []
    res = {}
    for seq in sequences:
        k_mers = kmers(seq, k, 1)
        graph = {}
        for i in range(len(k_mers)-1):
            key = k_mers[i]
            val = k_mers[i+1]
            if key not in graph:
                graph[key] = [val]
            else:
                graph[key].append(val)
                            
        for key in graph.keys():
            val = graph[key]
            if key not in res and val != []:
                res[key] = val
            elif val != []:
                res[key].extend(val)
                 
    return res


def find_starting_node(graph, seq, k):
    values = np.unique(graph.values())
    for node in graph.keys():
        for edges in values:
            if node in edges:
                break
            elif edges == values[-1] and node not in edges:
                return node  
            
    k_mers = kmers(seq, k)
    return k_mers[0]


def get_path_length(values):
    L = 1
    for val in values:
        L += len(val)
    return L


def assemble_path(eulr_paths):
    paths = []
    for p in eulr_paths:
        path = p[0]
        for i in range(1, len(p)):
            path += p[i][-1]
        paths.append(path)
    return paths


def find_euler_paths(graph, seqs, k):
    global all_euler_paths
    all_euler_paths = []

    start_node = find_starting_node(graph, seqs[0], k)
    
    L = get_path_length(graph.values())

    dfs_rec(graph, start_node, [], L)

    return all_euler_paths

    #dfs(graph, start_node, [])
    #return [path for path in all_euler_paths if len(path) == L]


def dfs_rec(graph, start_node, seq, k):
    if start_node not in graph: # no edges left
        if len(seq) == k:
            seq.append(start_node)
            all_euler_paths.append(seq)
        return

    edges = list(np.unique(graph[start_node]))
    seq.append(start_node)
    while len(edges) > 0:
        g = deepcopy(graph)
        next_edge = edges.pop()
        g[start_node].remove(next_edge)
        dfs_rec(g, next_edge, deepcopy(seq), k)


def go_back(graph, current, seq):
    previous = seq.pop()
    graph[previous].append(current)

    while len(seq) > 0:
        current = previous
        previous = seq.pop()
        graph[previous].insert(0, current)
        
        if len(graph[previous]) <= 1:
            continue
            
        elif len(graph[previous]) > 1:
            edges = [True for p in graph[previous]]
            for p in all_euler_paths:
                for i in range(len(graph[previous])):
                    t = copy(seq)
                    t.append(previous)
                    if len(p) >= len(t) and t == p[:len(t)] and graph[previous][i] == p[len(t)]:
                        edges[i] = False

            if True in edges:
                i = edges.index(True)
                edge = graph[previous][i]
                graph[previous].remove(edge)
                graph[previous].append(edge)
                return graph, previous, seq
            else:
                continue
    return graph, None, seq


def dfs(graph, start_node, seq):
    edges = [start_node]
    while (len(edges) > 0):
        current = edges.pop()
        if current not in graph:
            seq.append(current)
            all_euler_paths.append(seq)
            seq = deepcopy(seq)
            seq.pop()

            graph, current, seq = go_back(graph, current, seq)
            if current is None:
                break

        all_edges = graph[current]
        edges.append(all_edges.pop())
        seq.append(current)


def assemble_genome(seqs: Iterable[str], k: int) -> List[str]:
    """Perform genome assembly using the Eulerian path algorithm.

    The overall algorithm should follow the following general structure:

    1. For an input list of sequences, construct a corresponding De Bruijn graph.
    2. Find all possible Euerlian paths through the graph, i.e. all possible paths
       which visit each edge exactly once. Your paths should all start from a
       source node with in-degree zero. In case no such node exists, you should
       start using the first k-mer of the first sequence in `seqs`. You can
       assume you will never encounter a disconnected graph.
    3. Decode your obtained paths into sequences, and return all unique genome
       assemblies.

    Parameters
    ----------
    seqs: List[str]
    k: int
        The k to be used to decompose the input strings into k-mers. The stride
        should always be set to 1.

    Returns
    -------
    List[str]
        A list of unique assemblies for the given `seqs`.

    """
    graph = de_bruijn_graph(seqs, k)
    paths = find_euler_paths(graph, seqs, k)
    res = assemble_path(paths)        
    return res