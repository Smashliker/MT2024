from stable_baselines3 import PPO
from environment.federatedPPO import federatedPPO, stepPPO
from environment.env import Env
from pathlib import Path
import json

import pickle
from typing import Tuple
import numpy as np

import copy
import math

class GRC():
    def __init__(
            self,
            env: Env,
            d: float,
            th: float
            ) -> None:
        
        self.env = env
        self.d = d
        self.th = th

    def predict(
            self,
            obs: None = None, #Needed to fit with input 
            deterministic: bool = True,
            threeResources: bool = True
            ) -> Tuple[int, None]:
        networkResources = self.env.vnfBacktrack.calculateAllNodeResources(remaining=True)

        cpuVector = [0] * len(networkResources)
        storageVector = [0] * len(networkResources)
        bandwidthVector = [0] * len(networkResources)

        for nodeIndex, nodeResourceDict in enumerate(networkResources):
            cpuVector[nodeIndex] = nodeResourceDict["cpu"]
            storageVector[nodeIndex] = nodeResourceDict["storage"]
            bandwidthVector[nodeIndex] = nodeResourceDict["bandwidth"]

        sumCPUs = sum(cpuVector)
        sumStorage = sum(storageVector)
        sumBandwidth = sum(bandwidthVector)

        cpuVector = np.array(cpuVector, dtype=np.float16) / sumCPUs
        storageVector = np.array(storageVector, dtype=np.float16) / sumStorage
        bandwidthVector = np.array(bandwidthVector, dtype=np.float16) / sumBandwidth

        numNodes = self.env.vnfBacktrack.overlay.number_of_nodes()
        bigM = [[]] * numNodes
        #Divide each link BW by node BW
        for nodeI in self.env.vnfBacktrack.overlay:
            bigM[nodeI] = [0] * numNodes

            for nodeJ in self.env.vnfBacktrack.overlay.neighbors(nodeI):
                bigM[nodeI][nodeJ] = self.env.vnfBacktrack.overlay.get_edge_data(nodeI, nodeJ)["bandwidth"]
                bigM[nodeI][nodeJ] /= self.env.vnfBacktrack.overlay.nodes[nodeJ]["bandwidth"]

        bigM = np.array(bigM, dtype=np.float16)

        grcTuple = [copy.deepcopy(cpuVector), copy.deepcopy(cpuVector)]
        delta = math.inf

        while delta >= self.th:
            if threeResources:
                grcTuple[1] = (1 - self.d) * (cpuVector + storageVector + bandwidthVector) / 3 + self.d * bigM @ grcTuple[0]
            else:
                grcTuple[1] = (1 - self.d) * (cpuVector + storageVector) / 2 + self.d * bigM @ grcTuple[0].T
            delta = np.linalg.norm(grcTuple[1] - grcTuple[0])
            grcTuple[0] = grcTuple[1]

        return grcTuple[0].argmax(), None

