[![CI](https://github.com/Smashliker/MT2024/actions/workflows/python-package.yml/badge.svg)](https://github.com/Smashliker/MT2024/actions/workflows/python-package.yml)

# Master Thesis
This is the folder for the code in Benjamin Mydland's Master Thesis written at the University of Stavanger (UiS). Ali Gohar was the supervisor.


## Code Sources
The code is heavily inspired by two other repositories, each with their own MIT license
1. https://github.com/CN-UPB/NFVdeep/tree/main - LICENSE
2. https://github.com/DLR-RM/stable-baselines3 - LICENSE.txt 

The policy aggregation for the federated learning, a vital part of this thesis, was also inspired by a third repository, though no code was borrowed: https://github.com/pietromosca1994/DeRL_Golem

## Setup and Execution
The project is designed for Python 3.10.12

All dependencies need be installed:  
```pip install .```  

The network topology is created by editing the function ```generate_graph()``` in "graph_generator.py", and then executing the script: ```python3 graph_generator.py```. The network will then be generated as "/data/network.gpickle".
  
Then it is just a matter of executing the main function, with or without optional user arguments:  
```python3 main.py```

There are a number of arguments that can be put into ```main.py``` when it runs. These are:

- --notFederated: Runs the PPOA instead of the PPOFA
- --totalTimesteps: Sets the amount of env steps that will occur before the learning finishes
- --networkPath: The path of the .gpickle file representing the SAGIN created via ```graph_generator.py```
- --nEvalEpisodes: Sets n_eval_episodes in the evalCallback equal to this
- --evalFreq: Sets eval_freq in the evalCallback equal to this
- --output: Sets the output folder for the tensorboard logs
- --debug: If set to true, makes the environment print some degug messages
- --arivalConfig: json-file which decides the parameters used by the generated SCs
- --seed: Sets a seed for the arrival config, making the generated SCs deterministic each time they are generated
- --verbose: Creates the environment slightly different if not zero.

All of these arguments, besides --notFederated, were not altered when performing the training and evaluations in the thesis.

## File Overview

A brief explanation of the central files is warranted.

- ```main.py```: The file that does a full training of either the PPOA or the PPOFA
- ```eval.py```: The evaluation of the different agents
- ```plot.py```: The plotting of the evaluation data
- ```graph_generator.py```: The creation of the ```.gpickle``` file representing the SAGIN
- ```./tests```: All of the Pytests that are implemented
- ```./data/requests.json```: The specification for the parameters of the arriving SCs
- ```./environment```:
  - ```arrival.py```: The creation of the SCs from the ```.json``` file ```requests.json```
  - ```env.py```: The gym environment representing the environment
  - ```federatedPPO.py```: A rewrite of how the PPO class from SB3 works to make it work federated
  - ```grc.py```: An implementation of the GRCA
  - ```monitor.py```: The collection of custom statistics for the tensorboard log
  - ```network.py```: The representation of the SAGIN and all logic associated with it, like allocation restrictions
  - ```sc.py```: The representation of the Service Chain

## Docker
The most useful way to execute the code in this project is to create a docker image out of it.  

This can be achieved by first building the image:
```docker build .```  
If one wants to change arguments in ```main.py```, then one may edit the ```CMD``` line in the Dockerfile directly.

Then the image can be run as a container. After the container finishes execution, the ```tfevents```file required for analyzing the result with tensorboard will be available to the user as a Docker volume:  
```docker volume ls```  

After downloading this file, one may start tensorboard by running the following command from the directory of the file:
```tensorboard --logdir .```
