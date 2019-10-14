#!/bin/bash
# Basic range in for loop
python main_simulation_pyro.py 1 &
sleep .1 
python main_simulation_pyro.py 2 &
sleep .1 
python main_simulation_pyro.py 3 &
sleep .1 
python main_simulation_pyro.py 4
wait