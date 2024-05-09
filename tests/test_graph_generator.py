import os
import json
import unittest

import networkx as nx
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

import sys

#This line ensure that pytest is actually able to run in the main directory
sys.path.append('.')

#NOTE: Must be after the path append
from graph_generator import *
from typing import Tuple

def generateTestFiles() -> Tuple[str, str]:
        baseString = r'./data/test'
        txtFile = f"{baseString}.txt"
        pickleFile = f"{baseString}.gpickle"

        graph = generateEnhancedGraph()[0]
        saveGraph(graph, baseString)

        return txtFile, pickleFile

def deleteTestFiles(
        txtFile: str,
        pickleFile: str
        ) -> None:
    
    os.remove(txtFile)
    os.remove(pickleFile)

class Tests(unittest.TestCase):

    def test_generateFiles(self):

        txtFile, pickleFile = generateTestFiles()

        try:
            assert os.path.isfile(txtFile)
            assert os.path.isfile(pickleFile)

            #Test that the .txt file is formatted as expected
            with open(txtFile, 'r') as f:
                assert len(f.readlines()) == 8

        #Clean up
        finally:
            deleteTestFiles(txtFile, pickleFile)

    def test_edgesReachable(self):

        txtFile, pickleFile = generateTestFiles()

        try:
            with open(pickleFile, 'rb') as f:
                graph = pickle.load(f)

                path = nx.dijkstra_path(graph, 0, 99)

                assert len(path) > 0

        #Clean up
        finally:
            deleteTestFiles(txtFile, pickleFile)


    


    

            

    


