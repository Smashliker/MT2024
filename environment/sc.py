from typing import List

class ServiceChain:
    def __init__(
        self,
        arrivalTime: float,
        ttl: float,
        vnfs: List[dict[str, float]],
        max_response_latency: float
    ):
        self.arrivalTime = arrivalTime
        self.ttl = ttl
        self.vnfs = vnfs
        self.maxCumulativeLatency = max_response_latency
        self.num_vnfs = len(self.vnfs)
