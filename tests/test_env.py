import os
import json
import unittest
import pytest

from pathlib import Path

import networkx as nx
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

import sys

#This line ensure that pytest is actually able to run in the main directory
sys.path.append('.')

#NOTE: Must be after the path append
from environment.network import Network
from environment.sc import ServiceChain
from environment.env import Env

from test_graph_generator import generateTestFiles, deleteTestFiles

class Tests(unittest.TestCase):
    def test_deterministic_state(self):
        try:
            txtFile, pickleFile = generateTestFiles()

            with open(Path("./data/requests.json"), "r") as file:
                arrival_config = json.load(file)

            arrival_config["seed"] = 0

            env = Env(pickleFile, arrival_config)

            grandFatherState = env.compute_state()
            nTests = 1000

            for _ in range(nTests):
                testState = env.compute_state()

                for index in range(len(grandFatherState)):
                    assert testState[index] == grandFatherState[index]

        finally:
            deleteTestFiles(txtFile, pickleFile)