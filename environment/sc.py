from typing import List

class ServiceChain:
    def __init__(
        self,
        arrivalTime: float,
        ttl: float,
        vnfs: List[dict[str, float]] #Should eventually be dict to get access to keys in network.py
    ):
        self.arrivalTime = arrivalTime
        self.ttl = ttl
        self.vnfs = vnfs
        #E2E
        self.num_vnfs = len(self.vnfs)
