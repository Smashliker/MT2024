from stable_baselines3 import PPO
from environment.env import Env
from pathlib import Path
import json
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

import pickle

from main import DEFAULTARRIVALCONFIG, DEFAULTNETWORKPATH, DEFAULTSEED
from environment.env import StatKeeper
from typing import List

from collections import Counter

def plotStandardValues(statKeeper: StatKeeper) -> None:
    subplotDimensions = (2, 3)
    axes: List[Axes] = []

    toPlots = [
        statKeeper.acrList[0],
        statKeeper.arcList[0],
        statKeeper.larList[0],
        statKeeper.acrList[1],
        statKeeper.arcList[1],
        statKeeper.larList[1],
        ]
    
    xAxes = [
        "Timestep",
        "Timestep",
        "Timestep",
        "#SCs",
        "#SCs",
        "#SCs",
        ]
    
    yAxes = [
        "ACR",
        "ARC",
        "LAR",
        "ACR",
        "ARC",
        "LAR",
        ]

    for index, toPlot in enumerate(toPlots):
        axes.append(plt.subplot(subplotDimensions[0], subplotDimensions[1], len(axes) + 1))
        axes[-1].plot(toPlot)
        #axes[-1].legend()
        axes[-1].set_xlabel(xAxes[index])
        axes[-1].set_ylabel(yAxes[index])

    plt.show()

def acceptedRejectedBarPlot(statKeeper: StatKeeper) -> None:
    subplotDimensions = (1, 3)
    axes: List[Axes] = []

    counters = [
        Counter(statKeeper.acceptedRejectedList[0]),
        Counter(statKeeper.acceptedRejectedList[1]),
        Counter(statKeeper.acceptedRejectedList[0] + statKeeper.acceptedRejectedList[1]),
        ]

    barLabels = [[0] * len(counters[2])] * len(counters)
    barHeights = []

    for index, counter in enumerate(counters):
        barHeights.append([0] * len(counters[2]))

        for length, count in counter.items():
            barLabels[index][length - 1] = length

            barHeights[index][length - 1] = count

    xAxes = [
        "Length of Accepted SCs",
        "Length of Rejected SCs",
        "Lengt of SCs",
        ]
    
    yAxes = [
        "#Accepted",
        "#Rejected",
        "#SCs",
        ]

    for index, _ in enumerate(counters):
        axes.append(plt.subplot(subplotDimensions[0], subplotDimensions[1], len(axes) + 1))
        axes[-1].bar(barLabels[index], barHeights[index])

        axes[-1].set_xlabel(xAxes[index])
        axes[-1].set_ylabel(yAxes[index])

    plt.show()


if __name__ == "__main__":
    with open("./data/5MregularPolicyStatKeeper.gpickle", 'rb') as f:
        statKeeper: StatKeeper = pickle.load(f)

    plotStandardValues(statKeeper)

    acceptedRejectedBarPlot(statKeeper)

