import networkx as nx
import random
import numpy as np
import argparse

import pickle
import sys
from typing import List

DEFAULTOUTPUTFILE = r"./data/network"
DEFAULTGRAPHTYPE = "zhang"
DEFAULTSEED = None

def saveGraph(
        graph: nx.Graph,
        outputfile: str = DEFAULTOUTPUTFILE,
        ) -> None:
    """
    Function that saves the supplied graph to the supplied outputfile, both a .gpickle and .txt variant
    """

    #https://stackoverflow.com/a/77377332
    with open(f"{outputfile}.gpickle", 'wb') as f:
        pickle.dump(graph, f, pickle.HIGHEST_PROTOCOL)

    with open(f"{outputfile}.txt", 'w') as f:
        oldSys = sys.stdout
        sys.stdout = f

        print("nodes:")
        print(graph.nodes(data=True))
        print()
        print("edges:")
        print(nx.to_dict_of_dicts(graph))

        sys.stdout = oldSys

def generateToyGraph() -> nx.Graph:
    """
    Function that creates a graph that corresponds to the specifications of the toy implementation.
    """
    returnGraph = nx.Graph()

    CPUs = [50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0] #These default values imitate results from a U[50, 100] distribution

    for (nodeIndex, theCPU) in enumerate(CPUs):
        returnGraph.add_node(nodeIndex, cpu=theCPU)

    return returnGraph

def generateZhangGraph(
        seed: int = DEFAULTSEED
        ) -> nx.Graph:
    """
    Function that emulates the network found in https://ieeexplore.ieee.org/document/9749937 by Zhang
    """
    
    returnGraph = nx.Graph()
    nNodes = 20

    nodeCPUs = generateNUniformValues(nNodes, [10, 50], seed)
    nodeCPUs = np.rint(nodeCPUs).tolist()

    nodeLatencies = generateNUniformValues(nNodes, [10, 40], seed)

    for nodeIndex in range(nNodes):
        returnGraph.add_node(nodeIndex, cpu=nodeCPUs[nodeIndex], latency=nodeLatencies[nodeIndex])

    #Refer to figure 4 from: https://ieeexplore.ieee.org/document/9749937
    links: List[tuple[int, int]] = [
        (0, 1), (0, 2), (0, 5), (0, 8),
        (1, 3), (1, 7), (1, 10),
        (2, 3), (2, 4), (2, 9),
        (3, 6), (3, 11),

        (4, 5), (4, 8), (4, 12),
        (5, 9), (5, 13),
        (6, 7), (6, 10), (6, 14),
        (7, 11), (7, 15),
        (8, 9), (8, 16),
        (9, 17),
        (10, 11), (10, 18),
        (11, 19),
        
        (12, 13), (12, 16),
        (13, 17),
        (14, 15), (14, 18),
        (15, 19),
        (16, 17),
        #17
        (18, 19)
        #19
    ]
    nLinks = len(links)

    linkBWs = generateNUniformValues(nLinks, [15, 30], seed)
    linkLatencies = generateNUniformValues(nLinks, [15, 30], seed)

    for linkIndex, linkTuple in enumerate(links):
        returnGraph.add_edge(linkTuple[0], linkTuple[1], bandwidth=linkBWs[linkIndex], latency=linkLatencies[linkIndex])

    return returnGraph


def generateNUniformValues(
        n: int = 10,
        distributionEndPoints: List[int] = [50, 100],
        seed: int = DEFAULTSEED   
        ) -> List[float]:
    """
    Function that creates a list of length n of randomly distributed values
    """

    if seed != None:
        random.seed(seed)

    values = []

    for _ in range(n):
        values.append(random.uniform(distributionEndPoints[0], distributionEndPoints[1]))

    return values


def makeFullMesh(graph: nx.Graph) -> None: 
    """
    Function that makes any nx.Graph a fully connected mesh topology.
    \n
    Not utilized, included for the sake of reference.
    """
    graphSize = len(graph)

    for (mainIndex, _) in enumerate(graph):
        firstNeighborIndex = mainIndex + 1
        
        if firstNeighborIndex < graphSize:
            for secondIndex in range(firstNeighborIndex, graphSize):
                graph.add_edge(mainIndex, secondIndex, latency=50.0) #NOTE: Latency placeholder for now


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outputfile", type=str, default=DEFAULTOUTPUTFILE
    )
    parser.add_argument(
        "--CPUDist", type=List[int], nargs=2, default=[50, 100]
    )
    parser.add_argument(
        "--nNodes", type=int, default=10
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULTSEED
    )
    parser.add_argument(
        "--graphType", type=str, default=DEFAULTGRAPHTYPE
    )
    args = parser.parse_args()

    if args.graphType == "toy":
        graph = generateToyGraph()
    elif args.graphType == "zhang":
        graph = generateZhangGraph(args.seed)
    elif args.graphType == "proper":
        sys.exit()
    else:
        raise ValueError("Not valid graphtype!")


    saveGraph(graph, args.outputfile)
