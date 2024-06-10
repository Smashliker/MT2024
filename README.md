[![CI](https://github.com/Smashliker/MT2024/actions/workflows/python-package.yml/badge.svg)](https://github.com/Smashliker/MT2024/actions/workflows/python-package.yml)

# Master Thesis
This is the folder for the code in Benjamin Mydland's Master Thesis. It is built on top of the following repository: https://github.com/CN-UPB/NFVdeep/tree/main  

## Setup and Execution
All dependencies need be installed:  
```pip install .```  

The network topology is created by editing the function ```generate_graph()``` in "graph_generator.py", and then executing the script: ```python3 graph_generator.py```. The network will then be generated as "/data/network.gpickle".
  
Then it is just a matter of executing the main function, with or without optional user arguments:  
```python3 main.py```

## Docker
The most useful way to execute the code in this project is to create a docker image out of it.  

This can be achieved by first building the image:
```docker build .```  
If one wants to change arguments in ```main.py```, then one may edit the ```CMD``` line in the Dockerfile directly.

Then the image can be run as a container. After the container finishes execution, the ```tfevents```file required for analyzing the result with tensorboard will be available to the user as a Docker volume:  
```docker volume ls```  

After downloading this file, one may start tensorboard by running the following command from the directory of the file:
```tensorboard --logdir .```
