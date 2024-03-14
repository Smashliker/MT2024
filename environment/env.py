import logging
from copy import deepcopy

import gym
from gym import spaces

import numpy as np
import numpy.typing as npt

from environment.network import Network
from environment.arrival import ArrivalProcess



class Env(gym.Env):
    def __init__(
            self,
            networkPath: str,
            arrivalConfig):

        self.networkPath = networkPath
        self.arrivalConfig = arrivalConfig

        _, properties = Network.checkOverlay(self.networkPath)
        numNodes = properties["num_nodes"]
        resourcesPerNode = properties["num_node_resources"]

        # action "num_nodes" refers to volutarily rejecting the VNF embedding
        self.action_space = spaces.Discrete(numNodes + 1)

        observationDimension = numNodes * resourcesPerNode + resourcesPerNode + 3 #TODO find out why these values make sense. They should in theory...
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(observationDimension,), dtype=np.float16)

        self.reward = 0
        self.vnfIndex = 0

    def step(
            self,
            action: int,
            ) -> tuple[npt.NDArray[any], float, bool, dict[str, bool]]:
        assert not self.episodeDone, "Episode was already finished, but step() was called"

        info = {"accepted": False, "rejected": False}
        logging.debug(f"Agent wants to allocate VNF {self.vnfIndex} on Node: {action}")

        vnf = self.sc.vnfs[self.vnfIndex]

        isValidSC = self.vnfBacktrack.allocateVNF(self.sc, vnf, action) and self.vnfBacktrack.canAllocate(self.sc, self.vnfIndex + 1)
        
        logging.debug(
                "This SC allocation is: {}.".format(
                    "possible" if isValidSC else "impossible"
                )
            )

        isLastVNF = self.vnfIndex >= len(self.sc.vnfs) - 1

        if isValidSC and isLastVNF:
            self.network = deepcopy(self.vnfBacktrack)

            info["accepted"] = True
            info["placements"] = self.vnfBacktrack.scAllocation[self.sc]
            info["sc"] = self.sc

        if not isValidSC:
            self.vnfBacktrack = deepcopy(self.network)

            info["rejected"] = True
            info["placements"] = None
            info["sc"] = self.sc

        if isValidSC and not isLastVNF:
            self.vnfIndex += 1

        if not isValidSC or (isLastVNF and isValidSC):
            self.vnfIndex = 0
            self.episodeDone = self.progressNetworkTime()

        self.reward = self.computeReward(isLastVNF, isValidSC)
        logging.debug(f"Environment will attribute reward: {self.reward}")
                    

        resourceUtilization = self.vnfBacktrack.calculateUtilization()
        resourceCost = self.vnfBacktrack.calculateCosts()
        info.update(
            {res + "_utilization": val for res, val in resourceUtilization.items()}
        )
        info.update({res + "_cost": val for res, val in resourceCost.items()})

        return self.compute_state(episodeDone=self.episodeDone), self.reward, self.episodeDone, info

    def reset(self) -> npt.NDArray[any]:
        self.episodeDone = False

        self.arrivalProcess = ArrivalProcess.factory(self.arrivalConfig)

        self.network = Network(self.networkPath)
        self.vnfBacktrack = deepcopy(self.network)

        self.vnfIndex = 0

        self.progressNetworkTime()

        return self.compute_state()

    def progressNetworkTime(self) -> bool:
        self.sc = None #NOTE: It is assumed here that this function is only called when the service chain is finished (rejected or accepted)

        try:
            while self.sc == None:
                self.network.update()
                batch = next(self.arrivalProcess)

                if len(batch) > 0:
                    self.sc = deepcopy(batch[0])

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

        networkResources = self.vnfBacktrack.calculateAllNodeResources(remaining=True)
        networkResources = np.asarray(
            [list(nodeResources.values()) for nodeResources in networkResources], dtype=np.float16,
        )

        maxResources = self.vnfBacktrack.calculateAllNodeResources(remaining=False)
        maxResources = np.asarray(
            [list(nodeResources.values()) for nodeResources in maxResources], dtype=np.float16
        )
        maxResources = np.max(maxResources, axis=0) #NOTE: How this calculates the maximum for each resource
        networkResources = networkResources / maxResources
        
        networkResources = networkResources.reshape(-1)

        vnf = self.sc.vnfs[self.vnfIndex]

        normVNFResources = np.asarray([*vnf]) #TODO: May add SC requirements here, or as its own variable
        normVNFResources = list(normVNFResources / maxResources)

        normUndeployedVNFs = (len(self.sc.vnfs) - (self.vnfIndex + 1)) / 7 #NOTE: Hardcoded division by 7, the assumed maximum amount of VNFs in a single SC
        normTTL = self.sc.ttl / 1000 #NOTE: Hardcoded division by 1000, the assumed maximum TTL

        observation = np.concatenate(
            (
                networkResources,
                normVNFResources,
                normUndeployedVNFs,
                normTTL,
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
            reward = 1

        return reward
