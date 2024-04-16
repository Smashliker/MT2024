import operator
import functools
from collections import Counter, defaultdict
from networkx.exception import NetworkXNoPath

from typing import List

import networkx as nx

from environment.sc import ServiceChain

import pickle

import math

RESTRAINTS = ["latency"]

class Network:
    def __init__(
            self,
            networkPath: str,
            costs: dict[str, float] = {"cpu": 0.2, "storage": 0.2, "bandwidth": 0.0006, "latency": 0.2}):

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
            newTime: int
            ) -> None:

        self.timestep = newTime

        scAllocation = {
            sc: nodes for sc, nodes in self.scAllocation.items() if self.checkTTL(sc)
        }
        #NOTE: That the resources are deallocated upon timeout since our view of the resources is in "calculateAllNodeResources", and that is based upon this dict ^

        self.scAllocation = scAllocation

    def allocateVNF(
            self,
            sc: ServiceChain,
            vnf: dict[str, float],
            virtualLinkRequirement: float,
            node: int
            ) -> bool:
        
        #Node requirements
        if node >= self.numNodes or not self.checkIfAnyNodeCanAllocateVNF(vnf, node):
            return False
        
        #Link requirements
        if virtualLinkRequirement > self.calculateMinLinkBW(sc, node):
            return False

        if sc not in self.scAllocation:
            self.scAllocation[sc] = []

        self.scAllocation[sc].append(node)

        return True

    def calculateAllNodeResources(
            self, 
            remaining=True
            ) -> List[dict[str, float]]:
        resources = []

        """
        resources = [
            {res: maxVal for res, maxVal in res.items()}
            for _, res in self.overlay.nodes(data=True)
        ]
        """

        for _, nodes in self.overlay.nodes(data=True):
            nodeDict = dict()

            for resourceType, maxVal in nodes.items():

                #Remove restraints from resources
                if resourceType not in RESTRAINTS:
                    nodeDict[resourceType] =  maxVal

            resources.append(nodeDict)

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
        
        #TODO: Add cumulative bandwidth logic?

        if sc not in self.scAllocation:
            return True

        try:
            #TODO 7: Evaluate if this never checks the last allocated VNF, it seems to always lag behind a step
            latency = self.calculateCurrentSCLatency(sc)
            latencyConstraint = latency <= sc.maxCumulativeLatency
        except NetworkXNoPath:
            latencyConstraint = False

        return latencyConstraint

    #NOTE: This needs to be called on self.network in the environment, as that one has a self.scAllocation that is up to date
    def calculateARC(
            self

    ) -> float:
        revenue = defaultdict(lambda: 0)
        cost = defaultdict(lambda: 0)

        returnARC = 0

        for sc, allocations in self.scAllocation.items():

            revenue[sc] += sum(sc.vnfs[0].values())
            cost[sc] += sum(sc.vnfs[0].values())

            assert len(allocations) == len(sc.vnfs) 
            for allocationIndex in range(1, len(allocations)):
                revenue[sc] += sum(sc.vnfs[allocationIndex].values())
                cost[sc] += sum(sc.vnfs[allocationIndex].values())
                
                allocationTuple = (allocations[allocationIndex - 1], allocations[allocationIndex])

                if allocationTuple[0] != allocationTuple[1]:
                    revenue[sc] += sc.virtualLinkRequirements[allocationIndex]

                    cost[sc] += sc.virtualLinkRequirements[allocationIndex] * nx.dijkstra_path_length(self.overlay, allocationTuple[0], allocationTuple[1])

        sumRevenue = sum(revenue.values())
        sumCost = sum(cost.values())

        if sumCost != 0:
            returnARC = sumRevenue / sumCost

        return returnARC

    #NOTE: Definition of link splitting from paper "Distributed DRL" by Chao Wang largely fits with the idea behind this method
    #TODO: Make the algorithm consume BW, take inspiration from the resource calculation in 
    def calculateMinLinkBW(
            self, 
            sc: ServiceChain,
            destinationNode: int
            ) -> float:
        """
        Calculates the minimum available BW between the last allocated node (found in sc) and the desinationNode.
        """
        
        bw = 0

        if sc in self.scAllocation:
            originNode = self.scAllocation[sc][-1]

            try:
                pathNodes = nx.dijkstra_path(self.overlay, originNode, destinationNode, weight="latency") #TODO: Calculating SP with latency as weight is a simplification ensuring that the SP is the same wherever it is calculated, but this should maybe change
            except nx.exception.NetworkXNoPath:
                return -1

            #No BW is required if the node is allocated to the same node as previously, set minimum BW excessively high
            if len(pathNodes) < 2:
                return math.inf

            bwList = []
            for nodeIndex in range(len(pathNodes) - 1):
                nodeTuple = (pathNodes[nodeIndex], pathNodes[nodeIndex + 1])

                bwList.append(self.overlay[nodeTuple[0]][nodeTuple[1]]["bandwidth"])

            bw = min(bwList)

        return bw

    def calculateCurrentSCLatency(
            self, 
            sc: ServiceChain
            ) -> float:
        
        latency = 0

        if sc in self.scAllocation:
            nodes = self.scAllocation[sc]

            #nodes.append(node: int) #TODO 7: May do something like this to avoid this function lagging behind?

            #Add the initial node latency
            latencyList = [list(self.overlay.nodes(data=True))[nodes[0]][1]["latency"]]

            #This essentially calculates the SP between each of the previous nodes allocated, and sums them (along with node latencies)
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

        assert bool(nodes) == (len(nodes) > 0) #TODO: Remove this line once sufficiently convinced these are equivalent
        return len(nodes) > 0


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

        nodeAttributes = {"cpu": float, "storage": float}
        for _, data in overlay.nodes(data=True):
            assert all(
                [nattr in data for nattr, _ in nodeAttributes.items()]
            ), "Overlay must specify all required node attributes."
            assert all(
                [type(data[nattr]) == ntype for nattr, ntype in nodeAttributes.items()]
            ), "Overlay must specify the correct data types."

        edgeAttributes = {"latency": float, "bandwidth": float}
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

        #Remove restraints from resources
        for key, _ in resource.items():
            if key in RESTRAINTS:
                properties["num_node_resources"] -= 1

        return overlay, properties


def VNFConstraints(
        res: dict[str, float], 
        vnf: dict[str, float]
        ) -> bool:
    
    resourceSufficient = True

    #NOTE: The assumption here that each item in the vnf is a resource, NOT a restraint (like latency).
    for resourceKey, resourceValue in vnf.items():
        resourceSufficient = resourceSufficient and (res[resourceKey] >= resourceValue)

    return resourceSufficient