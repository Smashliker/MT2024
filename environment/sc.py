from typing import List

class ServiceChain:
    def __init__(
        self,
        arrivalTime: float,
        ttl: float,
        vnfs: List[dict[str, float]],
        max_response_latency: float,
        virtualLinkRequirements: List[float]
    ):
        self.arrivalTime = arrivalTime
        self.ttl = ttl
        self.vnfs = vnfs
        self.maxCumulativeLatency = max_response_latency
        self.num_vnfs = len(self.vnfs)

        assert len(virtualLinkRequirements) == len(vnfs)
        self.virtualLinkRequirements = virtualLinkRequirements
        self.virtualLinkRequirements[0] = -1 #Add a dummy value to the first one, since the first VNF does not require a link
