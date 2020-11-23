from typing import Dict, List, Tuple, Type, TypeVar
from itertools import groupby
from numpy.random import default_rng
import math
import json

Highway = TypeVar('Highway', bound='Highway')


class Highway():

    class Arm:
        def __init__(self, id: str, mean: float, variance: float) -> None:
            self._id = id
            self._mean = mean
            self._variance = variance

        @property
        def id(self) -> str:
            return self._id

        @property
        def mean(self) -> float:
            return self._mean

        @property
        def variance(self) -> float:
            return self._variance

    class Node:
        def __init__(self, id: str, failure: float) -> None:
            self._id = id
            self._failure = failure

        @property
        def id(self) -> str:
            return self._id

        @property
        def failure_probability(self) -> float:
            return self._failure

    def __init__(self, arms: List[Tuple[str, float, float]], nodes: List[Tuple[str, float]], mapping: List[Tuple[str, str]], failure_penalty: float = 0, seed: int = None) -> None:
        self._failure_penalty = failure_penalty

        self._arms = [self.Arm(x[0], x[1], x[2]) for x in arms]
        self._arms_dict = {x.id: x for x in self._arms}
        if len(self._arms_dict) != len(self._arms):
            raise Exception("Duplicate arm ids detected")

        if len(list(filter(lambda x: x[1] >= 0 and x[1] <= 1, nodes))) != len(nodes):
            raise Exception("Nodes with invalid failure probability detected")
        self._nodes = [self.Node(x[0], x[1]) for x in nodes]
        self._nodes_dict = {x.id: x for x in self._nodes}
        if len(self._nodes) != len(self._nodes_dict):
            raise Exception("Duplicate node ids detected")

        if len(mapping) != len(list(filter(lambda x: x[0] in self._arms_dict, mapping))):
            raise Exception("Mapping has arms not in arms list")
        grouped_mapping = groupby(mapping, lambda x: x[0])
        self._arms_to_nodes = {x[0]: [self._nodes_dict[y[1]]
                                      for y in x[1]] for x in grouped_mapping}
        if seed is not None:
            self._generator = default_rng(seed)
        else:
            self._generator = default_rng()

    def pull_arm(self, arm: str) -> Tuple[float, bool]:
        if arm not in self._arms_dict:
            raise Exception("{} is not a valid arm to pull".format(arm))
        arm_obj = self._arms_dict[arm]
        arm_reward = self._generator.normal(
            loc=arm_obj.mean, scale=arm_obj.variance)
        penalty_applied = False
        if arm in self._arms_to_nodes:
            failure_threshold = self._generator.uniform()
            for node in self._arms_to_nodes[arm]:
                if node.failure_probability >= failure_threshold:
                    arm_reward -= self._failure_penalty
                    penalty_applied = True
                    break
        return [arm_reward, penalty_applied]

    @property
    def arms(self):
        return [x.id for x in self._arms]

    @property
    def nodes(self):
        return [x.id for x in self._nodes]

    @property
    def arms_to_nodes(self):
        return {x[0]: [y.id for y in x[1]] for x in self._arms_to_nodes.items()}

    @property
    def failure_penalty(self):
        return self._failure_penalty

    @property
    def best_arm(self) -> Tuple[str, float]:
        '''
        The best arm will be the one that has the highest expected value.
        In our case, the expected value will be:
        probability that no node fails * arm's mean + probability that at least one node fails * (arm's mean - penalty)
        '''
        best_so_far = ("", -math.inf)
        for arm in self._arms:
            arm_expected_value = best_so_far[1]
            if arm.id not in self._arms_to_nodes:
                arm_expected_value = arm.mean
            else:
                linked_nodes = self._arms_to_nodes[arm.id]
                no_failure_probability = math.prod([
                    1 - x.failure_probability for x in linked_nodes])
                at_least_one_failure_probability = 1 - no_failure_probability
                arm_expected_value = no_failure_probability*arm.mean + \
                    at_least_one_failure_probability * \
                    (arm.mean - self._failure_penalty)
            if arm_expected_value > best_so_far[1]:
                best_so_far = (arm.id, arm_expected_value)
        return best_so_far

    @classmethod
    def from_json_file(cls: Type[Highway], filename: str) -> Highway:
        with open(filename, 'r') as json_file:
            json_dump = json.load(json_file)
            return Highway.from_dict(json_dump)

    @classmethod
    def from_dict(cls: Type[Highway], builder: Dict) -> Highway:
        arms = builder["arms"]
        nodes = builder["nodes"]
        penalty = 0
        if "failure_penalty" in builder:
            penalty = builder["failure_penalty"]
        mapping = []
        if "mapping" in builder:
            mapping = builder["mapping"]
        highway = cls(arms=arms, nodes=nodes,
                      mapping=mapping, failure_penalty=penalty)
        seed = None
        if "seed" in builder:
            seed = builder["seed"]
        highway = cls(arms=arms, nodes=nodes,
                      mapping=mapping, failure_penalty=penalty, seed=seed)
        return highway
