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

class Tests(unittest.TestCase):

    def test_generateFiles(self):
        baseString = r'./test'
        txtFile = f"{baseString}.txt"
        pickleFile = f"{baseString}.gpickle"

        graph = generateToyGraph()[0]
        saveGraph(graph, baseString)

        try:
            assert os.path.isfile(txtFile)
            assert os.path.isfile(pickleFile)

            #Test that the .txt file is formatted as expected
            with open(txtFile, 'r') as f:
                assert len(f.readlines()) == 8

        #Clean up
        finally:
            os.remove(txtFile)
            os.remove(pickleFile)

    

            

    


