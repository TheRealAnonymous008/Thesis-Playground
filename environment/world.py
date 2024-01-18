import gym
import torch 
import numpy as np 
import matplotlib.pyplot as plt
from enum import Enum 
from typing import Dict 

from .firm import Firm
from .entity import Entity

# TODO: Grid world simulation 
# TODO: Consumers actions -- Move around, buy product, harvest resources 


class Cell: 
    def __init__(self, x, y):
        self.contents = 0
        self.x = x 
        self.y = y

    def __str__(self) -> str:
        return str(self.contents)


class World:
    def __init__(self, width, height):
        self.width = width 
        self.height = height 
        self.grid = [[Cell(x, y) for x in range(width)] for y in range(height)]

        self.current_time = 0
        self.firms : Dict[str, Firm] = {}
        self.entities : Dict[str, Entity]= {}

        self._current_time = 0

    def add_firm(self, firm : Firm):
        self.firms[firm.id] = firm 

        for firm in self.firms.values(): 
            firm.update()

        for entity in self.entities.values():
            entity.update()

    def add_entity(self, entity : Entity):
        self.entities[entity.id] = entity 

    def reset(self):
        self._current_time = 0 
        for firm in self.firms.values(): 
            firm.reset()

        for entity in self.entities.values():
            entity.reset()


    def update(self):
        self._current_time += 1
        for firm in self.firms.values(): 
            firm.update()

        for entity in self.entities.values():
            entity.update()

    def render(self):
        for row in self.grid:
            r = ""
            for cell in row: 
                r += str(cell) + " "
            print(r)

    def generate_report(self):
        print("===============================")
        print("Time Step ", self._current_time)
        print("Firm Report")
        for firm in self.firms.values(): 
            firm.report()

        print("-------------------------------")
        print("Entity Report")
        for entity in self.entities.values():
            entity.report()

        print("===============================")