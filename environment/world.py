import gym
import torch 
import numpy as np 
import matplotlib.pyplot as plt
from enum import Enum 
from typing import Dict 

from .entity import Entity

class Cell: 
    def __init__(self, x, y):
        self.contents = 0
        self.x = x 
        self.y = y

    def __str__(self) -> str:
        return str(self.contents)


class World:
    """
    Arguments

    width - the width of the world

    height - the height of the world 

    _skills - the number of initial skills that is recognized in the current world.
    """
    def __init__(self, width: int, height: int , _skills: int = 10):
        self.width = width 
        self.height = height 
        self.grid = [[Cell(x, y) for x in range(width)] for y in range(height)]

        self.current_time = 0
        self._skills = _skills 

        self.entities : Dict[str, Entity]= {}

    def add_entity(self, count: int):
        for _ in range(count):
            self.entities[len(self.entities)] = Entity(len(self.entities), self)

    def reset(self):
        self._current_time = 0 
        for entity in self.entities.values():
            entity.reset()


    def update(self):
        self._current_time += 1

        for entity in self.entities.values():
            entity.update()

    def render(self):
        for row in self.grid:
            r = ""
            for cell in row: 
                r += str(cell) + " "
            print(r)

    def generate_report(self):
        report = { 
            "time": self._current_time
        }

        agents = {}
        for x in self.entities.values():
            agents[x.id] = x.report()

        report["agents"] = agents 
        return report 