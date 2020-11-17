from typing import List, Tuple, Type, TypeVar
from itertools import groupby
from numpy.random import default_rng
import json

Highway = TypeVar('Highway', bound='Highway')


class Highway():

    class Arm:
        def __init__(self, id: str, mean: float, variance: float) -> None:
            self.__id = id
            self.__mean = mean
            self.__variance = variance

        @property
        def id(self) -> str:
            return self.__id

        @property
        def mean(self) -> float:
            return self.__mean

        @property
        def variance(self) -> float:
            return self.__variance

    class Node:
        def __init__(self, id: str, failure: float) -> None:
            self.__id = id
            self.__failure = failure

        @property
        def id(self) -> str:
            return self.__id

        @property
        def failure_probability(self) -> float:
            return self.__failure

    def __init__(self, arms: List[Tuple[str, float, float]], nodes: List[Tuple[str, float]], mapping: List[Tuple[str, str]], failure_penalty: float = 0, seed: int = None) -> None:
        self.__failure_penalty = failure_penalty

        self.__arms = [self.Arm(x[0], x[1], x[2]) for x in arms]
        self.__arms_dict = {x.id: x for x in self.__arms}
        if len(self.__arms_dict) != len(self.__arms):
            raise Exception("Duplicate arm ids detected")

        if len(list(filter(lambda x: x[1] >= 0 and x[1] <= 1, nodes))) != len(nodes):
            raise Exception("Nodes with invalid failure probability detected")
        self.__nodes = [self.Node(x[0], x[1]) for x in nodes]
        self.__nodes_dict = {x.id: x for x in self.__nodes}
        if len(self.__nodes) != len(self.__nodes_dict):
            raise Exception("Duplicate node ids detected")

        if len(mapping) != len(list(filter(lambda x: x[0] in self.__arms_dict, mapping))):
            raise Exception("Mapping has arms not in arms list")
        grouped_mapping = groupby(mapping, lambda x: x[0])
        self.__arms_to_nodes = {x[0]: [self.__nodes_dict[y[1]]
                                       for y in x[1]] for x in grouped_mapping}
        if seed is not None:
            self.__generator = default_rng(seed)
        else:
            self.__generator = default_rng()

    def step(self, arm: str) -> Tuple[float, bool]:
        if arm not in self.__arms_dict:
            raise Exception("{} is not a valid arm to pull".format(arm))
        arm_obj = self.__arms_dict[arm]
        arm_reward = self.__generator.normal(
            loc=arm_obj.mean, scale=arm_obj.variance)
        penalty_applied = False
        if arm in self.__arms_to_nodes:
            failure_threshold = self.__generator.uniform()
            for node in self.__arms_to_nodes[arm]:
                if node.failure_probability >= failure_threshold:
                    arm_reward -= self.__failure_penalty
                    penalty_applied = True
        return [arm_reward, penalty_applied]

    @property
    def arms(self):
        return [x.id for x in self.__arms]

    @property
    def nodes(self):
        return [x.id for x in self.__nodes]

    @property
    def arms_to_nodes(self):
        return {x[0]: [y.id for y in x[1]] for x in self.__arms_to_nodes.items()}

    @property
    def failure_penalty(self):
        return self.__failure_penalty

    @classmethod
    def from_json_file(cls: Type[Highway], filename: str) -> Highway:
        with open(filename, 'r') as json_file:
            json_dump = json.load(json_file)
            arms = json_dump["arms"]
            nodes = json_dump["nodes"]
            penalty = 0
            if "failure_penalty" in json_dump:
                penalty = json_dump["failure_penalty"]
            mapping = []
            if "mapping" in json_dump:
                mapping = json_dump["mapping"]
            highway = cls(arms=arms, nodes=nodes,
                          mapping=mapping, failure_penalty=penalty)
            seed = None
            if "seed" in json_dump:
                seed = json_dump["seed"]
            highway = cls(arms=arms, nodes=nodes,
                          mapping=mapping, failure_penalty=penalty, seed=seed)
            return highway
