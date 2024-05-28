import logging
from copy import deepcopy

import gym
from gym import spaces

import numpy as np
import numpy.typing as npt

from environment.network import Network
from environment.arrival import ArrivalProcess

from environment.sc import *

from graph_generator import mapNodeToDomain

import networkx as nx

from dataclasses import dataclass, field


class Env(gym.Env):
    def __init__(
            self,
            networkPath: str,
            arrivalConfig,
            takeStats: bool = False
            ):

        self.networkPath = networkPath
        self.arrivalConfig = arrivalConfig

        _, properties = Network.checkOverlay(self.networkPath)
        numNodes = properties["num_nodes"]
        resourcesPerNode = properties["num_node_resources"]

        # action "num_nodes" refers to volutarily rejecting the VNF embedding
        self.action_space = spaces.Discrete(numNodes + 1)

        observationDimension = numNodes * resourcesPerNode + resourcesPerNode + 3 #NOTE: This "constant" may change later
        #observationDimension += numNodes
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(observationDimension,), dtype=np.float16)

        self.reward = 0
        self.vnfIndex = 0

        self.nAccepted = 0
        self.nProcessed = 0

        if takeStats:
            self.statKeeper = StatKeeper()

        self.network = Network(self.networkPath)
        #Initialize some initial values so that the check_env may run in main.py, this will be overriden at runtime by .reset() shortly
        self.reset()

    def step(
            self,
            action: int
            ) -> tuple[npt.NDArray[any], float, bool, dict[str, bool]]:
        assert not self.episodeDone, "Episode was already finished, but step() was called"

        info = {"accepted": False, "rejected": False}
        logging.debug(f"Agent wants to allocate VNF {self.vnfIndex} on Node: {action}")

        vnf = self.sc.vnfs[self.vnfIndex]
        virtualLinkRequirement = self.sc.virtualLinkRequirements[self.vnfIndex]

        isValidSC = self.vnfBacktrack.allocateVNF(self.sc, vnf, virtualLinkRequirement, action)
        #Must calculate these separately so that we see the if the last VNF violates SC requirements
        isValidSC = isValidSC and self.vnfBacktrack.checkSCConstraints(self.sc) #self.vnfBacktrack.canAllocate(self.sc, self.vnfIndex + 1)
        
        logging.debug(
                "This SC allocation is: {}.".format(
                    "possible" if isValidSC else "impossible"
                )
            )

        isLastVNF = self.vnfIndex >= len(self.sc.vnfs) - 1

        if isValidSC and isLastVNF:
            if hasattr(self, 'statKeeper'):
                self.statKeeper.acceptedRejectedList[0].append(len(self.sc.vnfs))

            self.vnfBacktrack.calculateRevenueCost(self.sc) #NOTE: Deepcopy above creates a deepcopy of the SC as well, means we must call this on VNFBacktrack
            self.nAccepted += 1

            self.network = deepcopy(self.vnfBacktrack)

            info["accepted"] = True
            info["placements"] = self.vnfBacktrack.scAllocation[self.sc]
            info["sc"] = self.sc

        if not isValidSC:
            if hasattr(self, 'statKeeper'):
                self.statKeeper.acceptedRejectedList[1].append(len(self.sc.vnfs))

            self.vnfBacktrack = deepcopy(self.network)

            info["rejected"] = True
            info["placements"] = None
            info["sc"] = self.sc

        if isValidSC and not isLastVNF:
            self.vnfIndex += 1

        if not isValidSC or (isLastVNF and isValidSC):
            self.nProcessed += 1

            self.vnfIndex = 0
            self.episodeDone = self.progressNetworkTime()

        self.reward = self.computeReward(isLastVNF, isValidSC)
        logging.debug(f"Environment will attribute reward: {self.reward}")
                    
        """
        #Updates needed for monitor.py
        #resourceUtilization = self.vnfBacktrack.calculateUtilization()
        
        resourceCost = self.vnfBacktrack.calculateCosts()
        info.update(
            {res + "_utilization": val for res, val in resourceUtilization.items()}
        )
        info.update({res + "_cost": val for res, val in resourceCost.items()})
        """
        
        numOperating = len(self.vnfBacktrack.getOperatingServers())
        info.update({"operating_servers": numOperating})

        acr = self.nAccepted / self.nProcessed if self.nProcessed != 0 else None
        info.update({"acr": acr})

        arc = self.vnfBacktrack.revenue / self.vnfBacktrack.cost if self.vnfBacktrack.cost != 0 else None
        info.update({"arc": arc})

        lar = self.vnfBacktrack.revenue / self.vnfBacktrack.timestep if self.vnfBacktrack.timestep != 0 else None
        info.update({"lar": lar})

        if hasattr(self, 'statKeeper'):
            self.statKeeper.acrList[0].append(acr)
            self.statKeeper.arcList[0].append(arc)
            self.statKeeper.larList[0].append(lar)

            if not isValidSC or (isLastVNF and isValidSC):
                self.statKeeper.acrList[1].append(acr)
                self.statKeeper.arcList[1].append(arc)
                self.statKeeper.larList[1].append(lar)

        return self.compute_state(episodeDone=self.episodeDone), self.reward, self.episodeDone, info

    def reset(self) -> npt.NDArray[any]:
        self.episodeDone = False #NOTE: This use of the term "episode" is a bit iffy, but it should match with documentation, e.g: https://stable-baselines3.readthedocs.io/en/master/common/evaluation.html

        self.arrivalProcess = ArrivalProcess.factory(self.arrivalConfig)

        self.network = Network(self.networkPath)
        self.vnfBacktrack = deepcopy(self.network)

        self.nAccepted = 0
        self.nProcessed = 0

        self.vnfIndex = 0

        if hasattr(self, 'statKeeper'):
            self.statKeeper = StatKeeper()

        self.progressNetworkTime()

        return self.compute_state()

    def progressNetworkTime(self) -> bool:
        self.sc = None #NOTE: It is assumed here that this function is only called when the service chain is finished (rejected or accepted)

        try:
            while self.sc == None:
                nextSC = next(self.arrivalProcess)

                if nextSC != None:
                    self.sc = deepcopy(nextSC)
                    self.network.update(self.sc.arrivalTime) #Set the current network time to the time the next SC arrives

                    logging.debug("Time progressed, new SC arrived.")

            self.vnfBacktrack = deepcopy(self.network)
            return False

        except StopIteration:
            # The episode finishes when the arrivalProcess finishes
            return True
        
    def compute_state(
            self,
            episodeDone: bool = False
            ) -> npt.NDArray[any]:

        if episodeDone:
            return np.zeros(self.observation_space.shape)

        resourceKeyToIndex = {
            "cpu": 0,
            "storage": 1,
            "bandwidth": 2
        }

        networkResources = self.vnfBacktrack.calculateAllNodeResources(remaining=True)
        #networkResources = np.asarray(
        #    [list(nodeResources.values()) for nodeResources in networkResources], dtype=np.float16,
        #)

        #sortedNetworkResources = [0] * (len(resourceKeyToIndex) * len(networkResources))
        sortedNetworkResources = []
        for _ in range(len(networkResources)):
            sortedNetworkResources.append([0] * len(resourceKeyToIndex))

        for nodeIndex, nodeResourceDict in enumerate(networkResources):
            for key, value in nodeResourceDict.items():
                if key != "domain":
                    sortedNetworkResources[nodeIndex][resourceKeyToIndex[key]] = value

        sortedNetworkResources = np.array(sortedNetworkResources, dtype=np.float16)

        """
        nNodes = self.vnfBacktrack.overlay.number_of_nodes()
        dstppList = [0] * nNodes

        if self.sc in self.vnfBacktrack.scAllocation:
            for nodeIndex in range(nNodes):
                dstppList[nodeIndex] = len(nx.dijkstra_path(self.vnfBacktrack.overlay, self.vnfBacktrack.scAllocation[self.sc][-1], nodeIndex, weight="latency")) / nNodes  #Divide by nNodes as an approximation of maximum hops
        dstppArray = np.asarray(dstppList, dtype=np.float16)
        """
        
        maxResources = self.vnfBacktrack.calculateAllNodeResources(remaining=False)

        sortedMaxResources = []
        for _ in range(len(maxResources)):
            sortedMaxResources.append([0] * len(resourceKeyToIndex))

        for nodeIndex, nodeResourceDict in enumerate(maxResources):
            for key, value in nodeResourceDict.items():
                if key != "domain":
                    sortedMaxResources[nodeIndex][resourceKeyToIndex[key]] = value

        sortedMaxResources = np.array(sortedMaxResources, dtype=np.float16)

        maxResources = np.max(sortedMaxResources, axis=0) #NOTE: How this calculates the maximum for each resource along each column
        
        networkResources = sortedNetworkResources / maxResources
        
        networkResources = networkResources.reshape(-1)

        vnf = self.sc.vnfs[self.vnfIndex]

        listVNF = [0] * len(resourceKeyToIndex)
        VNFRESTRAINTS = ["candidate_domain"]
        for key, value in vnf.items():
            if key not in VNFRESTRAINTS:
                listVNF[resourceKeyToIndex[key]] = value

        normVNFResources = np.asarray(listVNF, dtype=np.float16)

        normVNFResources = normVNFResources / maxResources

        normVNFResources = np.append(normVNFResources, [vnf["candidate_domain"] / 2]) #Hardcoded division by 2 corresponding to max domain 2 (0, 1, 2)

        #normUndeployedVNFs = (len(self.sc.vnfs) - (self.vnfIndex + 1)) / 6 #NOTE: Hardcoded division by 6, the assumed maximum amount of VNFs in a single SC
        normCumLatency = self.sc.maxCumulativeLatency / 200 #NOTE: Hardcoded division by 200, the assumed maximum possible cumLatency
        normSClength = len(self.sc.vnfs) / 6

        observation = np.concatenate(
            (
                networkResources,

                #dstppArray,

                normVNFResources,

                normCumLatency,

                #normUndeployedVNFs,

                normSClength
            ),
            axis=None,
        )

        return observation

    def computeReward(
            self,
            isLastVNF: bool,
            isValidSC: bool
            ) -> float:
        
        reward = 0

        if isLastVNF and isValidSC:
            reward = self.vnfBacktrack.revenue / self.vnfBacktrack.cost if self.vnfBacktrack.cost != 0 else 0 #ARC
        elif isValidSC:
            reward = 0.1 #- len(nx.dijkstra_path(self.vnfBacktrack.overlay, self.vnfBacktrack.scAllocation[self.sc][-1], self.vnfBacktrack.scAllocation[self.sc][-2], weight="latency")) / (10 * self.vnfBacktrack.overlay.number_of_nodes())

        return reward
    
@dataclass
class StatKeeper:
    #https://stackoverflow.com/a/63231305
    acrList: List[List[float]] = field(default_factory=lambda: [[0], [0]]) #Two lists: 1 with x <=> timesteps, 1 with x <=> SCs. They start at zero.
    larList: List[List[float]] = field(default_factory=lambda: [[0], [0]])
    arcList: List[List[float]] = field(default_factory=lambda: [[0], [0]])

    acceptedRejectedList: List[List[int]] = field(default_factory=lambda: [[], []]) #Two lists: 1 for amount of accepted SCs of length n, and 1 for amount of rejected SCs of length n

    #arrivalTimeList: 

    

