import json
import heapq
import random
from abc import abstractmethod
from collections.abc import Generator

import numpy as np

from environment.sc import ServiceChain

from copy import deepcopy
from typing import List


class ArrivalProcess(Generator):
    def __init__(self):
        self.timeslot = 1

        #This queue being empty denotes a finished episode
        self.requests = self.generateRequests()

        self.requests = [
            ((sc.arrivalTime, num), sc) for num, sc in enumerate(self.requests)
        ]
        heapq.heapify(self.requests)

    def send(
            self,
            dummy #This parameter is needed to implement the send method correctly. 
            ) -> ServiceChain:
        
        if len(self.requests) <= 0:
            self.throw()

        _, sc = heapq.heappop(self.requests)
        self.timeslot += 1
        return sc

    def throw(self):
        raise StopIteration

    @staticmethod
    def factory(config: str) -> Generator: #any class inheriting Generator (ArrivalProcess)
        #Used primarily for digesting requests.json
        if "type" not in config:
            arrival = JSONArrivalProcess(config)

        arrivalType = config["type"]

        params = {key: value for key, value in config.items() if key != "type"}

        if arrivalType == "poisson_arrival_uniform_load":
            arrival = PoissonArrivalUniformLoad(**params)

        else:
            raise ValueError("Unknown arrival process")

        return arrival

    @abstractmethod
    def generateRequests(self):
        raise NotImplementedError("Must be overwritten by an inheriting class")


class JSONArrivalProcess(ArrivalProcess):
    def __init__(
            self,
            request_path: str = '/data/requests.json'):
        
        assert request_path.endswith(".json")
        self.request_path = request_path
        super().__init__()

    def generateRequests(self) -> List[ServiceChain]:
        with open(self.request_path, "rb") as file:
            requests = json.load(file)

        req = []
        for sc in requests:
            vnfs = sc.pop("vnfs")
            sc = ServiceChain(vnfs=vnfs, **sc)

            req.append(sc)

        return req
    
    #TODO
    def parseVNFs(self, vnfs):
            return [tuple(vnf.values()) for vnf in vnfs]

class UniformLoadGenerator:
    def __init__(
        self,
        sc_length,
        num_vnfs,
        max_response_latency,
        bandwidth,
        cpus,
        memory,
        vnf_delays,
    ):
        self.sc_length = sc_length
        self.num_vnfs = num_vnfs
        self.bandwidth = bandwidth
        self.max_response_latency = max_response_latency

        self.cpus = cpus
        self.memory = memory
        self.vnf_delays = vnf_delays

    #NOTE: Here the difference between sc_length and num_vnfs is relevant: num_vnfs is to create some pre-defined VNFs which the new SC picks from, while sc_length is the actual length of the SC
    #TODO: Find out if this uniform generation is useful (creating samples instead of just creating all of them on-the-go)
    def nextSCLoad(self) -> any:

        vnfSamples = [
            {
                "cpu": float(random.randint(*self.cpus)), 
            #    "memory": random.uniform(*self.memory)
            }
            for _ in range(self.num_vnfs)
        ]
        """
        vnfSamples = [
            (random.randint(*self.cpus), random.uniform(*self.memory))
            for _ in range(self.num_vnfs)
        ]
        """
        delays = [random.uniform(*self.vnf_delays) for _ in range(self.num_vnfs)]

        while True:
            scParameters = {}

            numVNFsInSC = random.randint(*self.sc_length)
            vnfIndices = [
                random.randint(0, len(vnfSamples) - 1) for _ in range(numVNFsInSC)
            ]

            scParameters["vnfs"] = [vnfSamples[index] for index in vnfIndices]
            scParameters["processing_delays"] = [delays[index] for index in vnfIndices]
            scParameters["max_response_latency"] = random.uniform(
                *self.max_response_latency
            )
            scParameters["bandwidth_demand"] = random.uniform(*self.bandwidth)

            yield scParameters


class StochasticProcess(ArrivalProcess):
    """
    Abstract class for a stochastic process, meant to be inherited
    """
    def __init__(
            self,
            numRequests: int,
            loadGenerator: UniformLoadGenerator):
        
        self.numRequests = numRequests
        self.loadGenerator = loadGenerator

        super().__init__()

    def generateRequests(self) -> List[ServiceChain]:
        loadGen = self.loadGenerator.nextSCLoad()
        arrivalGen = self.nextArrival()

        req = []
        while len(req) < self.numRequests:
            arrivalTime, ttl = next(arrivalGen)
            sc_params = next(loadGen)
            sc = ServiceChain(arrivalTime=arrivalTime, ttl=ttl, vnfs=sc_params["vnfs"])
            req.append(sc)

        return req
    
    @abstractmethod
    def nextArrival(self):
        raise NotImplementedError("Must be overwritten by an inheriting class")


class PoissonArrivalUniformLoad(StochasticProcess):
    def __init__(
        self,
        num_timeslots, #Used to find arrival rate
        num_requests,
        service_rate, #The service rate is more or less synonymous with the TTL
        num_vnfs,
        sc_length,
        bandwidth,
        max_response_latency,
        cpus,
        memory,
        vnf_delays,
        seed=None,
        **kwargs #This line needs to exist to collect parameters not accounted for but still present in the passed arguments
    ):
        if seed is not None:
            random.seed(seed)

        self.numRequests = random.randint(*num_requests)
        self.numTimeslots = random.randint(*num_timeslots)

        self.meanArrivalRate = self.numRequests / self.numTimeslots
        self.meanTTL = random.randint(*service_rate)

        loadGenerator = UniformLoadGenerator(
            sc_length,
            num_vnfs,
            max_response_latency,
            bandwidth,
            cpus,
            memory,
            vnf_delays,
        )

        super().__init__(self.numRequests, loadGenerator)

    def nextArrival(self) -> any:
        arrivalTimes = [
            random.expovariate(self.meanArrivalRate) for _ in range(self.numRequests)
        ]
        arrivalTimes = np.ceil(np.cumsum(arrivalTimes))
        arrivalTimes = arrivalTimes.astype(int)

        ttls = [
            random.expovariate(1 / self.meanTTL)
            for _ in range(len(arrivalTimes))
        ]
        ttls = np.floor(ttls)
        ttls = ttls.astype(int)

        for arrivalTime, ttl in zip(arrivalTimes, ttls):
            yield arrivalTime, ttl

