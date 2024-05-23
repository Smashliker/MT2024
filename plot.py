from stable_baselines3 import PPO
from environment.env import Env
from pathlib import Path
import json
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

import pickle

from main import DEFAULTARRIVALCONFIG, DEFAULTNETWORKPATH, DEFAULTSEED
from environment.env import StatKeeper
from typing import List, Tuple

from collections import Counter

def plotStandardValues(statKeepers: List[StatKeeper], legends: List[str]) -> None:
    subplotDimensions = (2, 3)
    axes: List[Axes] = []

    manyToPlots: List[List[List[float]]] = []
    for statKeeper in statKeepers:
        manyToPlots.append(
            [
            statKeeper.acrList[0],
            statKeeper.arcList[0],
            statKeeper.larList[0],
            statKeeper.acrList[1],
            statKeeper.arcList[1],
            statKeeper.larList[1],
            ]
        )
    
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

    for i, oneToPlots in enumerate(manyToPlots):
        for j, toPlot in enumerate(oneToPlots):
            if i == 0:
                axes.append(plt.subplot(subplotDimensions[0], subplotDimensions[1], len(axes) + 1))

                axes[j].set_xlabel(xAxes[j])
                axes[j].set_ylabel(yAxes[j])

            axes[j].plot(toPlot, label=legends[i])
            #axes[-1].legend()

            if i == len(manyToPlots) - 1:
            #    axes[j].legend(legends)
                axes[j].legend()
            

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
    statKeeperTuple: Tuple[List, List] = (
    [
        "./data/bestRegularPolicyStatKeeper.gpickle",
        "./data/federatedPolicyStatKeeper.gpickle",
        "./data/grcStatKeeper.gpickle",
    ],

    [
        "PPO",
        "Federated",
        "GRC",
    ],
    )

    statKeepers: List[StatKeeper] = []
    for fileName in statKeeperTuple[0]:
        with open(fileName, 'rb') as f:
            statKeeper: StatKeeper = pickle.load(f)
            statKeepers.append(statKeeper)

    plotStandardValues(statKeepers, statKeeperTuple[1])

    for statKeeper in statKeepers:
        acceptedRejectedBarPlot(statKeeper)

