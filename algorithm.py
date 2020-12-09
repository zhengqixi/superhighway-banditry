from environment import Highway
from enum import Enum
import numpy as np


class GridType(Enum):
    minimax = 1
    geometric = 2
    arithmetic = 3


class Algorithm:
    def __init__(self, T: int, M: int, grid_type: GridType, environment: Highway) -> None:
        self._T = T
        self._M = M
        self._grid = self._generate_grid(T=T, M=M, grid_type=grid_type)
        self._environment = environment

    def _generate_grid(self, T: int, M: int, grid_type: GridType):
        if grid_type == GridType.minimax:
            a = T**(1/(2-2**(1-M)))
            grid = np.floor(np.power(np.ones(M) * a, 2-1 /
                                     np.power(np.ones(M) * 2, (np.arange(M)))))
            grid = np.insert(grid, 0, 0)
            grid[M] = T
            return grid
        elif grid_type == GridType.geometric:
            b = T**(1/M)
            grid = np.floor(np.power(np.ones(M+1) * b, np.arange(M+1)))
            grid[M] = T
            grid[0] = 0
            return grid
        elif grid_type == GridType.arithmetic:
            return np.floor(np.linspace(start=0, stop=T, num=M+1))
        else:
            raise Exception("Unknown grid type")

    def run(self) -> float:
        raise NotImplementedError("Not implemented in base class")
