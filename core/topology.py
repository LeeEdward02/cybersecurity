"""
topology.py
------------
负责网络拓扑的生成与管理，支持多种网络类型：
- lattice（格点网络）
- smallworld（小世界）
- scalefree（无标度）
- random（随机网络）
"""

import networkx as nx


class NetworkTopology:
    """
    网络拓扑类
    ----------
    用于创建和维护节点间的连接关系。
    """

    def __init__(self, topology='lattice', N=1600, params=None):
        """
        Args:
            topology (str): 网络类型，可选值 'lattice'/'smallworld'/'scalefree'/'random'
            N (int): 节点数
            params (dict): 网络参数，如重连概率p或连接度m
        """
        self.topology = topology
        self.N = N
        self.params = params or {}
        self.graph = self._build_network()

    def _build_network(self):
        """
        根据拓扑类型生成网络。
        Returns:
            networkx.Graph: 网络图对象
        """
        if self.topology == 'lattice':
            L = int(self.N ** 0.5)
            return nx.grid_2d_graph(L, L, periodic=True)
        elif self.topology == 'smallworld':
            return nx.watts_strogatz_graph(self.N, k=6, p=self.params.get('rewire_p', 0.08))
        elif self.topology == 'scalefree':
            return nx.barabasi_albert_graph(self.N, m=self.params.get('m', 3))
        elif self.topology == 'random':
            return nx.erdos_renyi_graph(self.N, p=self.params.get('p', 0.01))
        else:
            raise ValueError("不支持的网络类型")

    def get_neighbors(self, node):
        """
        获取某节点的所有邻居节点ID。
        Args:
            node (int): 节点编号
        Returns:
            list[int]: 邻居节点列表
        """
        return list(self.graph.neighbors(node))
