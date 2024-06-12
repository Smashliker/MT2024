from stable_baselines3 import PPO
from environment.env import Env
from pathlib import Path
import json
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib.ticker as mtick

import pickle

from main import DEFAULTARRIVALCONFIG, DEFAULTNETWORKPATH, DEFAULTSEED
from environment.env import StatKeeper
from typing import List, Tuple

from collections import Counter

import numpy as np

def plotStandardValues(
        statKeepers: List[StatKeeper], 
        legends: List[str], 
        specificPlot: int | None = None, #This is a value that maps to which plot type the user wants. 0 -> ACR as an example
        ) -> None:
    if specificPlot == None:
        subplotDimensions = (3, 1)
    else:
        subplotDimensions = (1, 1)

    manyToPlots: List[List[List[float]]] = []
    for statKeeper in statKeepers:
        manyToPlots.append(
            [
            #statKeeper.acrList[0],
            #statKeeper.arcList[0],
            #statKeeper.larList[0],
            statKeeper.acrList[1],
            statKeeper.arcList[1],
            statKeeper.larList[1],
            ]
        )
    
    xAxes = [
#        "Time",
#        "Time",
#        "Time",
        "Number of SCs",
        "Number of SCs",
        "Number of SCs",
        ]
    
    yAxes = [
#        "Acceptance Rate (ACR)",
#        "Average Revenue to Cost ratio (ARC)",
#        "Average Revenue (LAR)",
        "Acceptance Rate (ACR)",
        "Average Revenue to Cost ratio (ARC)",
        "Average Revenue (AR)",
        ]
    
    yLims = [
#        (5, 35),
#        (90, 105),
#        (50, 385),
        (5, 35),
        (90, 105),
        (50, 385),
        ]

    axes: List[Axes | None] = [None] * len(xAxes)
    for i, oneToPlots in enumerate(manyToPlots): #For each agent

        j = 0
        if specificPlot != None:
            oneToPlots = [oneToPlots[specificPlot]]
            j = specificPlot

        for _, toPlot in enumerate(oneToPlots): #For each subplot
            if i == 0:
                axes[j] = plt.subplot(subplotDimensions[0], subplotDimensions[1], sum(ax != None for ax in axes) + 1) #https://stackoverflow.com/a/29422718

                axes[j].set_xlabel(xAxes[j])
                axes[j].set_ylabel(yAxes[j])
                axes[j].yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))

                thisyLim = yLims[j]
                if thisyLim != None:
                    axes[j].set_ylim(thisyLim[0], thisyLim[1])

            axes[j].plot(toPlot, label=legends[i])
            #axes[-1].legend()

            if i == len(manyToPlots) - 1:
            #    axes[j].legend(legends)
                axes[j].legend()

            j += 1
            
    #regularParameters = plt.rcParams["figure.dpi"] 
    #plt.rcParams["figure.dpi"] = 1000
    #plt.savefig("plot.png", format="png")
    plt.show()
    #plt.rcParams["figure.dpi"] = regularParameters

def plotRejectionStats(
        statKeepers: List[StatKeeper], 
        legends: List[str],
        titles: List[str],
        ) -> None:
    subplotDimensions = (3, 1)

    manyToPlots: List[List[List[float]]] = []
    for statKeeper in statKeepers:
        manyToPlots.append(
            [
            statKeeper.vnfList,
            statKeeper.scList
            ]
        )
    
    xAxes = [
        "Number of SCs",
        "Number of SCs",
        "Number of SCs",
        ]
    
    yAxes = [
        "Number of Rejected SCs",
        "Number of Rejected SCs",
        "Number of Rejected SCs",
        ]
    
    """
    yLims = [
        (5, 35),
        (90, 105),
        (50, 385),
        ]
    """

    axes: List[Axes | None] = [None] * len(xAxes)
    for i, oneToPlots in enumerate(manyToPlots): #For each agent / subplot
        for j, toPlot in enumerate(oneToPlots): #For each graph within the subplot
            if j == 0:
                axes[i] = plt.subplot(subplotDimensions[0], subplotDimensions[1], sum(ax != None for ax in axes) + 1)

                axes[i].set_xlabel(xAxes[i])
                axes[i].set_ylabel(yAxes[i])

                axes[i].set_title(titles[i])

                """
                thisyLim = yLims[j]
                if thisyLim != None:
                    axes[j].set_ylim(thisyLim[0], thisyLim[1])
                """

            axes[i].plot(toPlot, label=legends[j])

            if j == len(oneToPlots) - 1:
                axes[i].legend()

    plt.show()


def acceptedRejectedBarPlot(statKeeper: StatKeeper, title: str ="") -> None:
    subplotDimensions = (3, 1)
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
        "Number of Accepted SCs",
        "Number of Rejected SCs",
        "Number of SCs",
        ]

    for index, _ in enumerate(counters):
        axes.append(plt.subplot(subplotDimensions[0], subplotDimensions[1], len(axes) + 1))
        axes[-1].bar(barLabels[index], barHeights[index])

        axes[-1].set_xlabel(xAxes[index])
        axes[-1].set_ylabel(yAxes[index])

    plt.suptitle(title)
    plt.show()

def arrivalTimesBarPlot(statKeeper: StatKeeper) -> None:
    subplotDimensions = (1, 1)
    axes: List[Axes] = []

    toPlot = [
        statKeeper.arrivalTimeList
        ]

    xAxes = [
        "Arrival Time [units]",
        ]
    
    yAxes = [
        "Density",
        ]
    
    x = np.linspace(0, 140, 1000)
    param = 1/25

    for index, data in enumerate(toPlot):
        axes.append(plt.subplot(subplotDimensions[0], subplotDimensions[1], len(axes) + 1))
        axes[-1].hist(data, density=True, bins=50, label="Share of SCs")

        axes[-1].set_xlabel(xAxes[index])
        axes[-1].set_ylabel(yAxes[index])

        axes[-1].plot(x, param * np.e**(-param * x), label="Exp(1/25)")

        axes[-1].legend()

    

    plt.show()

def domainBarPlot(
        statKeepers: List[StatKeeper],
        titles: List[str],
        sansFailures: bool = True,
        ) -> None:
    subplotDimensions = (3, 1)
    axes: List[Axes] = []

    """
    manyCounters: List[List[Counter]]
    for statKeeper in statKeepers:
        manyCounters.append([
            Counter(statKeeper.acceptedRejectedList[0]),
            Counter(statKeeper.acceptedRejectedList[1]),
            Counter(statKeeper.acceptedRejectedList[0] + statKeeper.acceptedRejectedList[1]),
            ])
    """

    counters: List[Counter] = []
    for statKeeper in statKeepers:
        counters.append(
            Counter(statKeeper.domainLists[1]) if sansFailures else Counter(statKeeper.domainLists[0])
            )

    barLabels = [0] * len(counters[0])
    barHeights: List[List[int]] = []

    domainDict = {
        0: "Space",
        1: "Air",
        2: "Ground"
    }

    for index, counter in enumerate(counters):
        barHeights.append([0] * len(counters[0]))

        for domain, count in counter.items():
            barLabels[domain] = domainDict[domain]

            barHeights[index][domain] = count / len(statKeepers[index].domainLists[1]) if sansFailures else count / len(statKeepers[index].domainLists[0])

    """
    manyToPlots: List[List[int]] = []
    for statKeeper in statKeepers:
        manyToPlots.append(
            statKeeper.domainLists[1]
        )
    """

    xAxes = ["Domain of Accepted VNF"] * 3 if sansFailures else ["Domain of attempted allocation"] * 3

    
    yAxes = [
        "Normalized Share of VNFs",
        "Normalized Share of VNFs",
        "Normalized Share of VNFs",
        ]

    for index, _ in enumerate(counters):
        axes.append(plt.subplot(subplotDimensions[0], subplotDimensions[1], len(axes) + 1))
        axes[-1].bar(barLabels, barHeights[index])

        axes[-1].set_xlabel(xAxes[index])
        axes[-1].set_ylabel(yAxes[index])

        axes[-1].set_title(titles[index])

    plt.show()


if __name__ == "__main__":
    statKeeperTuple: Tuple[List, List] = (
    [
        "./data/bandwidthRegularPolicyStatKeeper.gpickle",
        "./data/bandwidthFederatedPolicyStatKeeper.gpickle",
        "./data/grcStatKeeper.gpickle",

        #"./data/d=0.85GrcStatKeeper.gpickle",
        #"./data/d=0.15GrcStatKeeper.gpickle",
        #"./data/d=0.85FalseGrcStatKeeper.gpickle",
        #"./data/d=0.15FalseGrcStatKeeper.gpickle",

        #"./data/th=0.1GrcStatKeeper.gpickle",
        #"./data/th=100GrcStatKeeper.gpickle",

        #"./data/1250NewFederatedPolicyStatKeeper.gpickle",
        #"./data/2500NewFederatedPolicyStatKeeper.gpickle",
        #"./data/5000NewFederatedPolicyStatKeeper.gpickle",
        #"./data/10000NewFederatedPolicyStatKeeper.gpickle",
        #"./data/20000NewFederatedPolicyStatKeeper.gpickle",

        #"./data/2500NewFederatedPolicyStatKeeper.gpickle",
        #"./data/rewardFederatedPolicyStatKeeper.gpickle",
    ],

    [
        "PPOA",
        "PPOFA",
        "GRCA",

        #"three-GRC d=0.85",
        #"three-GRC d=0.15",
        #"two-GRC d=0.85",
        #"two-GRC d=0.15",

        #"GRC th: base",
        #"GRC th=0.1",
        #"GRC th=100",

        #"Federated upload=1250",
        #"Federated upload=2500",
        #"Federated upload=5000",
        #"Federated upload=10000",
        #"Federated upload=20000",

        #"Federated static weight",
        #"Federated reward weight"
    ],
    )

    statKeepers: List[StatKeeper] = []
    for fileName in statKeeperTuple[0]:
        with open(fileName, 'rb') as f:
            statKeeper: StatKeeper = pickle.load(f)
            statKeepers.append(statKeeper)

    #plotStandardValues(statKeepers, statKeeperTuple[1], 2)

    for i, statKeeper in enumerate(statKeepers):
        #acceptedRejectedBarPlot(statKeeper, statKeeperTuple[1][i])
        continue

    #arrivalTimesBarPlot(statKeepers[0])

    plotRejectionStats(statKeepers, ["VNF rejects", "SC rejects"], statKeeperTuple[1])

    #domainBarPlot(statKeepers, statKeeperTuple[1], True)

