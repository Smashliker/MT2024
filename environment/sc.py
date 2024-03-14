from typing import List

class ServiceChain:
    def __init__(
        self,
        arrival_time: float,
        ttl: float,
        vnfs: List[List[float]] #Should eventually be dict to get access to keys in network.py
    ):
        self.arrival_time = arrival_time
        self.ttl = ttl
        self.vnfs = vnfs
        #E2E
        self.num_vnfs = len(self.vnfs)
