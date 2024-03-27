import operator
import functools
from collections import Counter
from networkx.exception import NetworkXNoPath

from typing import List

import networkx as nx

from environment.sc import ServiceChain

import pickle


class Network:
    def __init__(
            self,
            networkPath: str,
            costs: dict[str, float] = {"cpu": 0.2, "memory": 0.2, "bandwidth": 0.0006, "latency": 0.2}):

        self.overlay, properties = Network.checkOverlay(networkPath)
        self.numNodes = properties["num_nodes"]

        self.timestep = 0
        self.costs = costs

        self.scAllocation: dict[ServiceChain, List[int]] = dict()

    def checkTTL(
            self,
            sc: ServiceChain
            ) -> bool:

            return sc.arrivalTime + sc.ttl >= self.timestep

    def update(
            self,
            timeIncrement: int = 1
            ) -> None:

        self.timestep += timeIncrement

        scAllocation = {
            sc: nodes for sc, nodes in self.scAllocation.items() if self.checkTTL(sc)
        }

        self.scAllocation = scAllocation

    def allocateVNF(
            self,
            sc: ServiceChain,
            vnf: dict[str, float],
            node: int
            ) -> bool:
        
        if node >= self.numNodes or not self.checkIfAnyNodeCanAllocateVNF(vnf, node):
            return False

        if sc not in self.scAllocation:
            self.scAllocation[sc] = []

        self.scAllocation[sc].append(node)

        return True

    def calculateAllNodeResources(
            self, 
            remaining=True
            ) -> List[dict[str, float]]:

        resources = [
            {res: maxVal for res, maxVal in res.items()}
            for _, res in self.overlay.nodes(data=True)
        ]

        if remaining:
            for sc, nodes in self.scAllocation.items():
                for vnfIndex, nodeIndex in enumerate(nodes):
                    for resourceKey, resourceValue in sc.vnfs[vnfIndex].items():
                        #Generalized calculation enabled by the dictionarization of the vnf
                        resources[nodeIndex][resourceKey] -= resourceValue

        return resources
    
    def checkSCConstraints(
            self, 
            sc: ServiceChain
            ) -> bool:
        
        #TODO: Add bandwidth logic?

        if sc not in self.scAllocation:
            return True

        try:
            #TODO: Evaluate if this never checks the last allocated VNF
            latency = self.calculateCurrentSCLatency(sc)
            #print(f"{latency} vs {sc.maxCumulativeLatency}")
            latencyConstraint = latency <= sc.maxCumulativeLatency
        except NetworkXNoPath:
            latencyConstraint = False

        return latencyConstraint

    #NOTE: May be unneeded
    def calculateCurrentSCBW(
            self, 
            sc: ServiceChain
            ) -> float:
        
        bw = 0

        if sc in self.scAllocation:
            nodes = self.scAllocation[sc]

            bwList = []
            for nodeIndex in range(1, len(nodes)):
                bw = nx.dijkstra_path_length(self.overlay, nodes[nodeIndex-1], nodes[nodeIndex], weight="bandwidth")

                bwList.append(bw)

            bw = sum(bwList)

        return bw

    def calculateCurrentSCLatency(
            self, 
            sc: ServiceChain
            ) -> float:
        
        latency = 0

        if sc in self.scAllocation:
            nodes = self.scAllocation[sc]

            latencyList = [list(self.overlay.nodes(data=True))[nodes[0]][1]["latency"]]
            for nodeIndex in range(1, len(nodes)):
                latencyList.append(nx.dijkstra_path_length(self.overlay, nodes[nodeIndex-1], nodes[nodeIndex], weight="latency"))

                #Add the latency each node requires
                latencyList.append(list(self.overlay.nodes(data=True))[nodes[nodeIndex]][1]["latency"])

            latency = sum(latencyList)

        return latency
    
    def canAllocate(
            self, 
            sc: ServiceChain, 
            vnfOffset: int = 0
            ) -> bool:
        
        vnfs = sc.vnfs[vnfOffset:]

        #NOTE: Chaining does not explicitly happen here, as it is implied in "calculateCurrentSCLatency"

        nextVNFisOK = True #NOTE: This value defaults to true since it assumes that this is the last VNF until proven otherwise
        #NOTE: This checks specifically if the next VNF (if it exists) can be allocated
        if len(vnfs) > 0:
            nextVNF = next(iter(vnfs))
            nextVNFisOK = self.checkIfAnyNodeCanAllocateVNF(nextVNF)

        SCisOK = self.checkSCConstraints(sc)

        return nextVNFisOK and SCisOK

    def checkIfAnyNodeCanAllocateVNF(
            self, 
            vnf: dict[str, float], 
            node=None
            ) -> bool:
        
        resources = self.calculateAllNodeResources()
        if node is not None:
            resources = [resources[node]]

        nodes = set(num for num, res in enumerate(resources) if VNFConstraints(res, vnf))

        return bool(nodes) #TODO: Simplify


    def calculateUtilization(self) -> dict[str, float]:
        max_resources = self.calculateAllNodeResources(remaining=False)
        avail_resources = self.calculateAllNodeResources(remaining=True)

        max_resources = dict(
            functools.reduce(operator.add, map(Counter, max_resources))
        )
        avail_resources = dict(
            functools.reduce(operator.add, map(Counter, avail_resources))
        )

        utilization = {
            key: (max_resources[key] - avail_resources[key]) / max_resources[key]
            for key in max_resources
        }

        return utilization

    def getOperatingServers(self) -> set[int]:
        operatingServers = {
            server for sc in self.scAllocation for server in self.scAllocation[sc]
        }
        return operatingServers

    def calculateCosts(self) -> dict[str, float]:
        operatingServers = self.getOperatingServers()

        if not operatingServers:
            return {key: 0 for key in self.costs}

        # filter out non-operating servers
        resources = [
            res
            for idx, res in enumerate(self.calculateAllNodeResources(remaining=False))
            if idx in operatingServers
        ]

        #TODO: Figure this one out, very unfamiliar set operations
        resources = dict(functools.reduce(operator.add, map(Counter, resources)))

        cost = {res: resources[res] * self.costs[res] for res in resources}

        return cost

    @staticmethod
    def checkOverlay(overlay: str) -> tuple[nx.Graph, dict[str, float]]:
        with open(overlay, 'rb') as f:
            overlay = pickle.load(f)

        nodeAttributes = {"cpu": float}
        for _, data in overlay.nodes(data=True):
            assert all(
                [nattr in data for nattr, _ in nodeAttributes.items()]
            ), "Overlay must specify all required node attributes."
            assert all(
                [type(data[nattr]) == ntype for nattr, ntype in nodeAttributes.items()]
            ), "Overlay must specify the correct data types."

        edgeAttributes = {"latency": float}
        for _, _, data in overlay.edges(data=True):
            assert all(
                [eattr in data for eattr, _ in edgeAttributes.items()]
            ), "Overlay must specify all required edge attributes."
            assert all(
                [type(data[eattr]) == etype for eattr, etype in edgeAttributes.items()]
            ), "Overlay must specify the correct data types."

        properties = {}
        properties["num_nodes"] = overlay.number_of_nodes()
        _, resource = next(iter(overlay.nodes(data=True)))
        properties["num_node_resources"] = len(resource)

        return overlay, properties


def VNFConstraints(
        res: dict[str, float], 
        vnf: dict[str, float]
        ) -> bool:
    
    resourceSufficient = True

    for resourceKey, resourceValue in vnf.items():
        resourceSufficient = resourceSufficient and (res[resourceKey] >= resourceValue)

    return resourceSufficient