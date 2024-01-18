import gym
import torch 
import numpy as np 
import matplotlib.pyplot as plt
from enum import Enum

# TODO: Grid world simulation 
# TODO: Firm actions -- firms can set prices and quantity
# TODO: Consumers actions -- Move around, buy product, harvest resources 


class Cell: 
    def __init__(self):
        self.contents = 0


class World:
    def __init__(self, width, height):
        self.width = width 
        self.height = height 
        self.grid = [[Cell() for _ in range(width)] for _ in range(height)]