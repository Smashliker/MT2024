import os
import json
import unittest
import pytest

import networkx as nx
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

import sys

#This line ensure that pytest is actually able to run in the main directory
sys.path.append('.')

#NOTE: Must be after the path append
from environment.network import Network
from environment.sc import ServiceChain

from test_graph_generator import generateTestFiles, deleteTestFiles

class Tests(unittest.TestCase):

    def test_network_exists(self):
        try:
            txtFile, pickleFile = generateTestFiles()

            network = Network(pickleFile)

            assert network is not None
            assert isinstance(network, Network)

        finally:
            deleteTestFiles(txtFile, pickleFile)

    def test_successfullSCAllocation(self):
        try:
            txtFile, pickleFile = generateTestFiles()

            network = Network(pickleFile)

            SCparams = {
                "arrivalTime": 0,
                "ttl": 1000,
                "vnfs": [{"cpu": 10, "storage": 10, "bandwidth": 10}, {"cpu": 10, "storage": 10, "bandwidth": 10}],
                "max_response_latency": 75, #Can't be much less than this
                "virtualLinkRequirements": [1, 1]
            }

            sc = ServiceChain(**SCparams)
            assert network.checkSCConstraints(sc) is True

            #Calling both of these methods mimics how the environment checks allocations
            firstAllocate = network.allocateVNF(sc, sc.vnfs[0], 0, 0)
            firstAllocate = firstAllocate and network.checkSCConstraints(sc)
            assert firstAllocate == True

            secondAllocate = network.allocateVNF(sc, sc.vnfs[1], 0, 1)
            secondAllocate = secondAllocate and network.checkSCConstraints(sc)
            assert firstAllocate == True

            assert sc in network.scAllocation

            #Timeskip
            network.update(1001)
            assert sc not in network.scAllocation

        finally:
            deleteTestFiles(txtFile, pickleFile)

    def test_unsuccessfullSCAllocation(self):
        try:
            txtFile, pickleFile = generateTestFiles()

            network = Network(pickleFile)

            SCparams = {
                "arrivalTime": 0,
                "ttl": 1000,
                "vnfs": [{"cpu": 38, "storage": 0, "bandwidth": 0}, {"cpu": 0, "storage": 73, "bandwidth": 0}, {"cpu": 0, "storage": 0, "bandwidth": 604}],
                "max_response_latency": 24,
                "virtualLinkRequirements": [1, 1, 1]
            }

            sc = ServiceChain(**SCparams)

            assert network.allocateVNF(sc, sc.vnfs[0], 0, 0) is False
            assert network.allocateVNF(sc, sc.vnfs[1], 0, 0) is False
            assert network.allocateVNF(sc, sc.vnfs[2], 0, 0) is False

            assert sc not in network.scAllocation

            #Allocate so that we may test SC reqiurements
            assert network.allocateVNF(sc, sc.vnfs[1], 0, 3) is True

            assert network.checkSCConstraints(sc) is False

        finally:
            deleteTestFiles(txtFile, pickleFile)

    """
    def test_ARC(self):
        try:
            txtFile, pickleFile = generateTestFiles()

            network = Network(pickleFile)

            SCparams = {
                "arrivalTime": 0,
                "ttl": 1000,
                "vnfs": [{"cpu": 10, "storage": 10, "bandwidth": 10}, {"cpu": 10, "storage": 10, "bandwidth": 10}],
                "max_response_latency": 75,
                "virtualLinkRequirements": [1, 1]
            }

            sc = ServiceChain(**SCparams)

            network.scAllocation[sc] = [0, 1]

            assert network.calculateARC() == 1

            network.scAllocation[sc] = [99, 97] #Two hops required => (resourceSUM + 1)/(resourceSUM + 1 * 2) (resourceSUM=10*6)

            assert network.calculateARC() == pytest.approx(0.98387, 0.00001)

        finally:
            deleteTestFiles(txtFile, pickleFile)
    """


    def test_deterministicGeneration(self):
        _, pickleFile1 = generateTestFiles()
        graph1, _ = Network.checkOverlay(pickleFile1)

        txtFile2, pickleFile2 = generateTestFiles()
        graph2, _ = Network.checkOverlay(pickleFile2)

        try:
            for nodeIndex in graph1:
                for resource, value in graph1[nodeIndex].items():
                    assert graph2[nodeIndex][resource] == value

        #Clean up
        finally:
            deleteTestFiles(txtFile2, pickleFile2)