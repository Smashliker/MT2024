import networkx as nx
import random
import numpy as np
import argparse

import pickle
import sys
from typing import List

DEFAULTOUTPUTFILE = r"./data/network"
DEFAULTGRAPHTYPE = "enhancedFederated"
DEFAULTSEED = 0

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
        print()
        print("overview:")
        print(graph)

        sys.stdout = oldSys

def generateToyGraph() -> List[nx.Graph]:
    """
    Function that creates a graph that corresponds to the specifications of the toy implementation.
    """
    returnGraph = nx.Graph()

    CPUs = [50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0] #These default values imitate results from a U[50, 100] distribution

    for (nodeIndex, theCPU) in enumerate(CPUs):
        returnGraph.add_node(nodeIndex, cpu=theCPU)

    return [returnGraph]

def generateZhangGraph(
        seed: int = DEFAULTSEED
        ) -> List[nx.Graph]:
    """
    Function that emulates the network found in https://ieeexplore.ieee.org/document/9749937 by Zhang
    """
    if seed != None:
        random.seed(seed)
    
    returnGraph = nx.Graph()
    nNodes = 20

    nodeCPUs = generateNUniformValues(nNodes, [10, 50])
    nodeCPUs = np.rint(nodeCPUs).tolist()

    nodeLatencies = generateNUniformValues(nNodes, [10, 40])

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

    linkBWs = generateNUniformValues(nLinks, [15, 30])
    linkLatencies = generateNUniformValues(nLinks, [15, 30])

    for linkIndex, linkTuple in enumerate(links):
        returnGraph.add_edge(linkTuple[0], linkTuple[1], bandwidth=linkBWs[linkIndex], latency=linkLatencies[linkIndex])

    return [returnGraph]

def generateEnhancedGraph(
        seed: int = DEFAULTSEED,
        singleNetwork: bool = False
        ) -> List[nx.Graph]:
    
    if seed != None:
        random.seed(seed)

    returnGraph = nx.Graph()

    spaceIndices = range(0, 10)
    airIndices = range(10, 40)
    groundIndices = range(40, 100)

    totalNodes = len(spaceIndices) + len(airIndices) + len(groundIndices)

    threeDomains = [spaceIndices, airIndices, groundIndices]
    threeCPURanges = [[20, 40], [20, 40], [50, 100]]
    threeStorageRanges = [[50, 80], [50, 80], [50, 80]]

    #Creation of nodes
    for domainIndex, domain in enumerate(threeDomains):
        domainCPURange = threeCPURanges[domainIndex]
        domainStorageRange = threeStorageRanges[domainIndex]

        for nodeIndex in domain:
            addSingleNode(returnGraph, nodeIndex, domainCPURange, [10, 40], domainStorageRange)

    sumOfEdges = 600

    domainEdges = [45, sumOfEdges * len(airIndices) / totalNodes, sumOfEdges * len(groundIndices) / totalNodes]
    domainEdges = np.cumsum(domainEdges).tolist()

    domainBWRanges = [[50, 100], [60, 100], [50, 100]]

    #TODO: Consider BFS like Fredrik did
    #This creates a direct line of links for each node: 0<->1<->2... as a baseline so that all nodes have at least one connection
    for sourceNode in range(totalNodes - 1):
        destinationNode = sourceNode + 1

        domainTuple = (mapNodeToDomain(sourceNode, threeDomains), mapNodeToDomain(destinationNode, threeDomains))

        if domainTuple[0] == domainTuple[1]:
            linkBW = generateNUniformValues(1, domainBWRanges[domainTuple[0]])[0]
            linkLatency = generateNUniformValues(1, [15, 30])[0]
            
            returnGraph.add_edge(sourceNode, destinationNode, bandwidth=linkBW, latency=linkLatency)

    #Creation of links
    while returnGraph.number_of_edges() < sumOfEdges:
        node = random.randint(0, totalNodes - 1)

        #This way of determining the relevant domain makes the links uniformly distributed (not taking the link-saturation of the space domain into account)
        domainIndex = mapNodeToDomain(node, threeDomains)

        nodes = threeDomains[domainIndex]
        
        linkNodes = random.sample(nodes, 2)

        if returnGraph.number_of_edges(linkNodes[0], linkNodes[1]) < 1:
            linkBW = generateNUniformValues(1, domainBWRanges[domainIndex])[0]
            linkLatency = generateNUniformValues(1, [15, 30])[0]

            returnGraph.add_edge(linkNodes[0], linkNodes[1], bandwidth=linkBW, latency=linkLatency)

    interDomainBW = generateNUniformValues(2, [50, 100])
    interDomainLatency = generateNUniformValues(2, [30, 60])

    returnGraphs = []
    if not singleNetwork:
        padding = totalNodes

        for _, domain in enumerate(threeDomains):
            toAppend = returnGraph.subgraph(domain).copy() #NOTE: The importance of creating a mutable copy of the immuatable subgraph

            padGraphToN(toAppend, padding)

            returnGraphs.append(toAppend) 



    #Creation of inter-domain links must happen after division of network into subgraphs
    for interDomainIndex in range(len(threeDomains) - 1):
        secondDomainIndex = interDomainIndex + 1

        domainNodeTuple = (random.sample(threeDomains[interDomainIndex], 1)[0], random.sample(threeDomains[secondDomainIndex], 1)[0])

        returnGraph.add_edge(domainNodeTuple[0], domainNodeTuple[1], bandwidth=interDomainBW[interDomainIndex], latency=interDomainLatency[interDomainIndex])

    

    returnGraphs.insert(0, returnGraph)
    return returnGraphs


def addSingleNode(
        graph: nx.Graph,
        nodeIndex: int,
        CPURange: List[int],
        latencyRange: List[int],
        storageRange: List[int]
        ) -> None:
    """
    The preferred function for adding a node to the graph. This collects all logic for creating nodes in a single function.
    """
    
    nodeCPUs = generateNUniformValues(1, CPURange)
    nodeCPU = np.rint(nodeCPUs).tolist()[0]

    nodeStorage = generateNUniformValues(1, storageRange)[0]

    nodeLatency = generateNUniformValues(1, latencyRange)[0]

    graph.add_node(nodeIndex, cpu=nodeCPU, storage=nodeStorage, latency=nodeLatency)


def padGraphToN(
        graph: nx.Graph,
        n: int
        ) -> None:
    zeroInterval = [0, 0]

    potentialNodeIndex = 0
    while graph.number_of_nodes() < n:
        if potentialNodeIndex not in graph:
            addSingleNode(graph, potentialNodeIndex, zeroInterval, zeroInterval, zeroInterval) #Pad graph with dummy nodes without resources nor links

        potentialNodeIndex += 1

def mapNodeToDomain(
        node: int,
        domains: List[List[int]] = [range(0, 10), range(10, 40), range(40, 100)]
        ) -> int:
    
    for domainIndex, domain in enumerate(domains):
        if node in domain:
            return domainIndex
        
    raise Exception(f"Supplied node '{node}' not in nested list of domains.")
    

def generateNUniformValues(
        n: int = 10,
        distributionEndPoints: List[int] = [50, 100] 
        ) -> List[float]:
    """
    Function that creates a list of length n of randomly distributed values
    """

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
                graph.add_edge(mainIndex, secondIndex, latency=50.0)


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
        graphs = generateToyGraph()
    elif args.graphType == "zhang":
        graphs = generateZhangGraph(args.seed)
    elif args.graphType == "enhanced":
        graphs = generateEnhancedGraph(args.seed, singleNetwork=True)
    elif args.graphType == "enhancedFederated":
        graphs = generateEnhancedGraph(args.seed)
    elif args.graphType == "proper":
        sys.exit()
    else:
        raise ValueError("Not valid graphtype!")

    if len(graphs) == 1:
        names = [""]
    elif len(graphs) == 4:
        names = ["", "_space", "_air", "_ground"]
    else:
        raise ValueError("Invalid amount of graphs!")

    for graphIndex, graph in enumerate(graphs):
        saveGraph(graph, f"{args.outputfile}{names[graphIndex]}")
