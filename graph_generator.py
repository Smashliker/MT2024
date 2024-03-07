import networkx as nx
from geopy.distance import geodesic
import random
import numpy as np
import argparse

import pickle
import sys

DEFAULTOUTPUTFILE = r"./data/network"


def generate_graph(
        outputfile: str = DEFAULTOUTPUTFILE,
        CPUs: list[float] = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95] #These default values imitate results from a U[50, 100] distribution
        ) -> None:
    """
    Function that creates a graph in outputfile with amount of nodes equal to length of the lists.
    """
    
    graph = nx.Graph()

    for (nodeIndex, _) in enumerate(CPUs):
        graph.add_node(nodeIndex, cpu=CPUs[nodeIndex])
    
    #makeFullMesh(graph) 

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


#TODO: Make generic
def generateCPUs(
        nNodes: int = 10,
        CPUDist: list[int] = [50, 100]   
        ) -> list[float]:
    """
    Function that creates a list with uniformly distributed values in the range defined by CPUDist.
    """

    CPUs = []

    for _ in range(nNodes):
        CPUs.append(random.uniform(CPUDist[0], CPUDist[1]))

    return CPUs


def makeFullMesh(graph: nx.Graph) -> None: 
    """
    Function that makes any nx.Graph a fully connected mesh topology.
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
        "--CPUDist", type=list[int], nargs=2, default=[50, 100]
    )
    parser.add_argument(
        "--nNodes", type=int, default=10
    )
    args = parser.parse_args()

    CPUs = generateCPUs(args.nNodes, args.CPUDist)

    generate_graph(args.outputfile, CPUs)
