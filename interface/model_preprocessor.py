import pickle
import numpy as np
from typing import List, Tuple, Dict, Union, cast, Generic, TypeVar
from abc import ABC, abstractmethod

from ..cass import cass
from .cass_manager import CASS


T = TypeVar('T')


class ModelPreprocessor(ABC, Generic[T]):
    @abstractmethod
    def preprocess_cass(self, cass: CASS) -> T:
        pass

    @abstractmethod
    def preprocess_casses_combined(self, casses: List[CASS]) -> T:
        """
        Convert multiple CASSes into one instace of model input.
        """
        pass

    @abstractmethod
    def preprocess_casses_seperated(self, casses: List[CASS]) -> List[T]:
        """
        Convert each CASS into one instace of model input.
        """
        pass


class GNNPreprocessor(ModelPreprocessor[Tuple[np.ndarray, np.ndarray]]):
    def __init__(self, vocab: Union[str, Dict[str, int]]) -> None:
        if isinstance(vocab, str):
            with open(vocab, 'rb') as f:
                self.vocab: Dict[str, int] = pickle.load(f)
        else:
            self.vocab = vocab

    def preprocess_cass(self, cass: CASS) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build a graph from one CASS.
        """
        nodes = []
        edges = []
        build_graph(cass.root, nodes, edges)
        if cass.fun_sig_node is not None and cass.fun_sig_node.n is not None:
            nodes.append(cass.fun_sig_node.n)
        return np.array([self.vocab.get(t, 0) for t in nodes]), np.array(edges).T

    def preprocess_casses_combined(self, casses: List[CASS]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build a graph from a list of CASSes.
        """
        nodes = []
        edges = []
        for cass in casses:
            build_graph(cass.root, nodes, edges)
            if cass.fun_sig_node is not None and cass.fun_sig_node.n is not None:
                nodes.append(cass.fun_sig_node.n)
        return np.array([self.vocab.get(t, 0) for t in nodes]), np.array(edges).T

    def preprocess_casses_seperated(self, casses: List[CASS]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Build a list of graphs from a list of CASSes.
        """
        return [self.preprocess_cass(c) for c in casses]


def build_graph(node: cass.CassNode, nodes: List[str], edges: List[Tuple[int, int]]):
    node_id = len(nodes)
    nodes.append(cast(str, node.n))
    last_id = node_id
    for c in node.children:
        edges.append((node_id, last_id + 1))
        last_id = build_graph(c, nodes, edges)
    return last_id
