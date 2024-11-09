import random
import pickle
from pathlib import Path
from copy import deepcopy
from typing import Optional, Union, List, Tuple, Dict, Any, TYPE_CHECKING

import torch
import scipy.spatial

from dhg.structure import BaseHypergraph
from dhg.visualization.structure.draw import draw_hypergraph
from dhg.utils.sparse import sparse_dropout

if TYPE_CHECKING:
    from ..graphs import Graph, BiGraph


class Hypergraph(BaseHypergraph):
    r"""The ``Hypergraph`` class is developed for hypergraph structures.

    Args:
        ``num_v`` (``int``): The number of vertices in the hypergraph.
        ``e_list`` (``Union[List[int], List[List[int]]]``, optional): A list of hyperedges describes how the vertices point to the hyperedges. Defaults to ``None``.
        ``e_weight`` (``Union[float, List[float]]``, optional): A list of weights for hyperedges. If set to ``None``, the value ``1`` is used for all hyperedges. Defaults to ``None``.
        ``v_weight`` (``Union[List[float]]``, optional): A list of weights for vertices. If set to ``None``, the value ``1`` is used for all vertices. Defaults to ``None``.
        ``merge_op`` (``str``): The operation to merge those conflicting hyperedges in the same hyperedge group, which can be ``'mean'``, ``'sum'`` or ``'max'``. Defaults to ``'mean'``.
        ``device`` (``torch.device``, optional): The deivce to store the hypergraph. Defaults to ``torch.device('cpu')``.
    """

    def __init__(
        self,
        num_v: int,
        e_list: Optional[Union[List[int], List[List[int]]]] = None,
        e_weight: Optional[Union[float, List[float]]] = None,
        v_weight: Optional[List[float]] = None,
        merge_op: str = "mean",
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(num_v, device=device)
        # init vertex weight
        if v_weight is None:
            self._v_weight = [1.0] * self.num_v
        else:
            assert len(v_weight) == self.num_v, "The length of vertex weight is not equal to the number of vertices."
            self._v_weight = v_weight
        # init hyperedges
        if e_list is not None:
            self.add_hyperedges(e_list, e_weight, merge_op=merge_op)
        '''
        这是一个 Python 代码段，包含了一个类的构造函数 __init__ 的实现。
        这个类的作用不确定，因为代码缺少上下文，但是可以看出这个类涉及到图或者超图的相关操作。
        具体来说，构造函数的参数包括顶点数量 num_v、超边的列表 e_list、超边的权重 e_weight、
        顶点权重 v_weight、合并操作 merge_op 和设备类型 device。在构造函数中，
        根据传入的参数初始化了类的一些属性，包括顶点权重和超边。
        '''
    def __repr__(self) -> str:
        r"""Print the hypergraph information.
        """
        return f"Hypergraph(num_v={self.num_v}, num_e={self.num_e})"
        '''
        这是一个 Python 代码段，包含了一个类的方法 __repr__ 的实现。在 Python 中
        ，__repr__ 是一个魔法方法（magic method），用于返回一个对象的字符串表示形式，
        通常用于调试和日志输出。在这个类中，__repr__ 方法返回一个字符串，表示该超图对象的基本信息，
        包括顶点数和超边数。当打印该超图对象时，该方法将被调用，并返回相应的字符串表示形式。
        '''
    @property
    def state_dict(self) -> Dict[str, Any]:
        r"""Get the state dict of the hypergraph.
        """
        return {"num_v": self.num_v, "raw_groups": self._raw_groups}
        '''
        这是一个 Python 代码段，包含了一个类的方法 state_dict 的实现。该方法返回一个字典，
        其中包含超图的状态信息。具体来说，字典包括两个键值对，分别是 "num_v" 和 "raw_groups"。
        其中，"num_v" 表示超图的顶点数，"raw_groups" 则是一个列表，包含超图的所有超边。
        超边通过一个列表的形式存储，其中每个元素是一个列表，表示这条超边所连接的顶点集合。
        当需要保存超图的状态信息时，可以调用该方法获取超图的状态字典，并将其保存到文件或数据库中。
        '''
    def save(self, file_path: Union[str, Path]):
        r"""Save the DHG's hypergraph structure a file.

        Args:
            ``file_path`` (``Union[str, Path]``): The file path to store the DHG's hypergraph structure.
        """
        file_path = Path(file_path)
        assert file_path.parent.exists(), "The directory does not exist."
        data = {
            "class": "Hypergraph",
            "state_dict": self.state_dict,
        }
        with open(file_path, "wb") as fp:
            pickle.dump(data, fp)
        '''
        这是一个 Python 代码段，包含了一个类的方法 save 的实现。
        该方法用于将 DHG（Dynamic Hypergraph）对象的超图结构保存到文件中，
        以便之后可以快速加载和使用。在方法实现中，首先将传入的文件路径转换为 Path 类型，
        并检查其父目录是否存在。接着，将超图对象的状态信息保存到一个字典中，
        字典包含两个键值对，分别是 "class" 和 "state_dict"。
        其中，"class" 表示超图对象的类名，即 "Hypergraph"，
        "state_dict" 则是超图对象的状态字典。最后，使用 pickle 库将字典对象序列化并保存到文件中。
        '''
    @staticmethod
    def load(file_path: Union[str, Path]):
        r"""Load the DHG's hypergraph structure from a file.

        Args:
            ``file_path`` (``Union[str, Path]``): The file path to load the DHG's hypergraph structure.
        """
        file_path = Path(file_path)
        assert file_path.exists(), "The file does not exist."
        with open(file_path, "rb") as fp:
            data = pickle.load(fp)
        assert data["class"] == "Hypergraph", "The file is not a DHG's hypergraph file."
        return Hypergraph.from_state_dict(data["state_dict"])
        '''
        这是一个 Python 代码段，包含了一个静态方法 load 的实现。
        该方法用于从文件中加载 DHG（Dynamic Hypergraph）对象的超图结构，
        以便可以快速使用。在方法实现中，首先将传入的文件路径转换为 Path 类型，
        并检查文件是否存在。接着，使用 pickle 库从文件中反序列化数据，并将其保存到一个字典中。
        字典中包含两个键值对，分别是 "class" 和 "state_dict"。
        其中，"class" 表示序列化数据的类名，即 "Hypergraph"，
        用于检查文件是否为 DHG 的超图文件。最后，调用 Hypergraph.from_state_dict 方法创建并返回 DHG 超图对象。
        '''
    def draw(
        self,
        e_style: str = "circle",
        v_label: Optional[List[str]] = None,
        v_size: Union[float, list] = 1.0,
        v_color: Union[str, list] = "r",
        v_line_width: Union[str, list] = 1.0,
        e_color: Union[str, list] = "gray",
        e_fill_color: Union[str, list] = "whitesmoke",
        e_line_width: Union[str, list] = 1.0,
        font_size: float = 1.0,
        font_family: str = "sans-serif",
        push_v_strength: float = 1.0,
        push_e_strength: float = 1.0,
        pull_e_strength: float = 1.0,
        pull_center_strength: float = 1.0,
    ):
        r"""Draw the hypergraph structure.

        Args:
            ``e_style`` (``str``): The style of hyperedges. The available styles are only ``'circle'``. Defaults to ``'circle'``.
            ``v_label`` (``list``): The labels of vertices. Defaults to ``None``.
            ``v_size`` (``float`` or ``list``): The size of vertices. Defaults to ``1.0``.
            ``v_color`` (``str`` or ``list``): The `color <https://matplotlib.org/stable/gallery/color/named_colors.html>`_ of vertices. Defaults to ``'r'``.
            ``v_line_width`` (``float`` or ``list``): The line width of vertices. Defaults to ``1.0``.
            ``e_color`` (``str`` or ``list``): The `color <https://matplotlib.org/stable/gallery/color/named_colors.html>`_ of hyperedges. Defaults to ``'gray'``.
            ``e_fill_color`` (``str`` or ``list``): The fill `color <https://matplotlib.org/stable/gallery/color/named_colors.html>`_ of hyperedges. Defaults to ``'whitesmoke'``.
            ``e_line_width`` (``float`` or ``list``): The line width of hyperedges. Defaults to ``1.0``.
            ``font_size`` (``float``): The font size of labels. Defaults to ``1.0``.
            ``font_family`` (``str``): The font family of labels. Defaults to ``'sans-serif'``.
            ``push_v_strength`` (``float``): The strength of pushing vertices. Defaults to ``1.0``.
            ``push_e_strength`` (``float``): The strength of pushing hyperedges. Defaults to ``1.0``.
            ``pull_e_strength`` (``float``): The strength of pulling hyperedges. Defaults to ``1.0``.
            ``pull_center_strength`` (``float``): The strength of pulling vertices to the center. Defaults to ``1.0``.
        """
        draw_hypergraph(
            self,
            e_style,
            v_label,
            v_size,
            v_color,
            v_line_width,
            e_color,
            e_fill_color,
            e_line_width,
            font_size,
            font_family,
            push_v_strength,
            push_e_strength,
            pull_e_strength,
            pull_center_strength,
        )
        '''
        draw()方法是一个绘制超图结构的函数，用于将超图的结构以可视化的方式展示出来。
        具体来说，它可以接受多个参数，用于指定绘图的细节，比如顶点的标签、大小和颜色，边的样式和颜色等等。
        这些参数都是可选的，如果不指定的话，将会使用默认值。
        draw()方法的具体实现是通过调用一个名为draw_hypergraph()的辅助函数实现的。
        这个函数使用了matplotlib库来绘制超图。具体来说，它首先将超图转换成一个由点和边组成的图形，
        然后按照指定的参数对各个元素进行绘制。例如，如果设置了顶点的标签，那么它将在相应的顶点处显示出来。
        需要注意的是，draw()方法只是一个用于展示超图结构的函数，它并没有改变超图本身的结构或状态。
        如果需要对超图进行修改或操作，需要使用其他方法。
        '''
    def clear(self):
        r"""Clear all hyperedges and caches from the hypergraph.
        """
        return super().clear()
        '''
        这个函数是继承自超类BaseGraph的clear方法。
        它的作用是清除该超图中的所有超边以及相关的缓存信息，使该超图变为空图。
        具体而言，它将超图中所有的顶点和超边信息全部清除，并将邻接矩阵等相关的缓存信息全部清空。
        这个函数没有返回值，执行完之后，该超图对象就变成了一个空图对象。       
        '''
    def clone(self) -> "Hypergraph":
        r"""Return a copy of the hypergraph.
        """
        hg = Hypergraph(self.num_v, device=self.device)
        hg._raw_groups = deepcopy(self._raw_groups)
        hg.cache = deepcopy(self.cache)
        hg.group_cache = deepcopy(self.group_cache)
        return hg
        '''
        这个方法返回一个超图的副本。副本的创建过程中，首先创建一个空的超图对象，
        然后将原始超图中的所有顶点和超边复制到副本中，同时也复制了超图的缓存和组缓存。
        因为深度拷贝(deepcopy)的使用，原始超图和副本中的这些数据结构是相互独立的，
        互不影响。最后返回这个副本超图对象。这个方法对于需要在多个任务之间共享超图结构的场景非常有用。
        '''
    def to(self, device: torch.device):
        r"""Move the hypergraph to the specified device.

        Args:
            ``device`` (``torch.device``): The target device.
        """
        return super().to(device)

    # =====================================================================================
    # some construction functions
    @staticmethod
    def from_state_dict(state_dict: dict):
        r"""Load the hypergraph from the state dict.

        Args:
            ``state_dict`` (``dict``): The state dict to load the hypergraph.
        """
        _hg = Hypergraph(state_dict["num_v"])
        _hg._raw_groups = deepcopy(state_dict["raw_groups"])
        return _hg
        '''
        这是Hypergraph类的一个类方法。它从一个状态字典中加载超图对象。具体来说，它接受一个状态字典作为输入，返回一个超图对象。
        该方法首先创建一个Hypergraph对象 _hg，然后将状态字典中的 num_v 属性的值设置为 _hg 的 num_v 属性的值。
        然后，将状态字典中的 raw_groups 属性的深层副本设置为 _hg 的 _raw_groups 属性。最后，返回 _hg 对象。
        '''
    @staticmethod
    def _e_list_from_feature_kNN(features: torch.Tensor, k: int):
        r"""Construct hyperedges from the feature matrix. Each hyperedge in the hypergraph is constructed by the central vertex ans its :math:`k-1` neighbor vertices.

        Args:
            ``features`` (``torch.Tensor``): The feature matrix.
            ``k`` (``int``): The number of nearest neighbors.
        """

        features = features.cpu().numpy()
        assert features.ndim == 2, "The feature matrix should be 2-D."
        assert (
            k <= features.shape[0]
        ), "The number of nearest neighbors should be less than or equal to the number of vertices."

        tree = scipy.spatial.cKDTree(features)
        _, nbr_array = tree.query(features, k=k)
        return nbr_array.tolist()
        '''
        该函数是 Hypergraph 类的一个静态方法，用于从特征矩阵中构造超图的超边。
        每个超边由中心节点和其k-1个最近邻节点构成。输入参数为特征矩阵和最近邻节点数目k。
        函数首先将输入的特征矩阵转换为numpy数组，并检查其维度是否为2。
        然后使用scipy中的cKDTree构建KDTree，以便在特征空间中快速搜索最近邻点。
        对于每个节点，函数找到其k个最近邻点，并将这些点的索引存储为一个列表，
        并返回这些列表的列表作为输出，即超图的超边列表。
        '''
    @staticmethod
    def from_feature_kNN(features: torch.Tensor, k: int, device: torch.device = torch.device("cpu")):
        r"""Construct the hypergraph from the feature matrix. Each hyperedge in the hypergraph is constructed by the central vertex ans its :math:`k-1` neighbor vertices.

        .. note::
            The constructed hypergraph is a k-uniform hypergraph. If the feature matrix has the size :math:`N \times C`, the number of vertices and hyperedges of the constructed hypergraph are both :math:`N`.

        Args:
            ``features`` (``torch.Tensor``): The feature matrix.
            ``k`` (``int``): The number of nearest neighbors.
            ``device`` (``torch.device``, optional): The device to store the hypergraph. Defaults to ``torch.device('cpu')``.
        """

        e_list = Hypergraph._e_list_from_feature_kNN(features, k)
        hg = Hypergraph(features.shape[0], e_list, device=device)
        return hg
        '''
        这个函数是用于从给定的特征矩阵中构建一个 kNN 超图。其中，每个超边都是由中心节点及其 k-1 个最近邻节点构成的。
        构建的超图是 k-uniform 超图，即如果特征矩阵的大小为 N×C，则构建的超图的节点数和超边数都为 N。
        函数首先使用 scipy 库中的 cKDTree 方法计算特征矩阵中每个节点的 k 个最近邻节点。
        然后，它使用这些最近邻节点构建超边列表，并使用列表初始化一个新的超图对象，最后返回该对象。
        函数还允许指定所构建超图存储的设备。
        '''
    @staticmethod
    def from_graph(graph: "Graph", device: torch.device = torch.device("cpu")) -> "Hypergraph":
        r"""Construct the hypergraph from the graph. Each edge in the graph is treated as a hyperedge in the constructed hypergraph.

        .. note::
            The construsted hypergraph is a 2-uniform hypergraph, and has the same number of vertices and edges/hyperedges as the graph.

        Args:
            ``graph`` (``Graph``): The graph to construct the hypergraph.
            ``device`` (``torch.device``, optional): The device to store the hypergraph. Defaults to ``torch.device('cpu')``.
        """
        e_list, e_weight = graph.e
        hg = Hypergraph(graph.num_v, e_list, e_weight=e_weight, device=device)
        return hg
        '''
        这个函数是一个类方法，用于从给定的图 graph 构建一个对应的超图。
        其中，原始图中的每条边都会被视为新构建的超图中的一条超边。
        在超图中，每个超边的节点集合是原始图边所连接的所有节点。这个函数返回构建的超图对象。

        注意，构建的超图是一个2-uniform超图，即每个超边的节点数都是2。
        另外，构建的超图和原始图具有相同的节点数和边数/超边数。

        这个函数有两个参数：

        graph：原始的图对象，用于构建超图。
        device（可选）：超图所存储的设备，默认为 torch.device('cpu')。
        这个函数首先通过 graph 对象的边列表 e_list 和边权重 e_weight 创建一个新的超图对象 hg，
        并将原始图的节点数 num_v 作为超图节点数。然后，该函数返回新的超图对象。
        '''
    @staticmethod
    def _e_list_from_graph_kHop(graph: "Graph", k: int, only_kHop: bool = False,) -> List[tuple]:
        r"""Construct the hyperedge list from the graph by k-Hop neighbors. Each hyperedge in the hypergraph is constructed by the central vertex and its :math:`k`-Hop neighbor vertices.

        .. note::
            If the graph have :math:`|\mathcal{V}|` vertices, the constructed hypergraph will have :math:`|\mathcal{V}|` vertices and equal to or less than :math:`|\mathcal{V}|` hyperedges.

        Args:
            ``graph`` (``Graph``): The graph to construct the hypergraph.
            ``k`` (``int``): The number of hop neighbors.
            ``only_kHop`` (``bool``, optional): If set to ``True``, only the central vertex and its :math:`k`-th Hop neighbors are used to construct the hyperedges. By default, the constructed hyperedge will include the central vertex and its [ :math:`1`-th, :math:`2`-th, :math:`\cdots`, :math:`k`-th ] Hop neighbors. Defaults to ``False``.
        """
        assert k >= 1, "The number of hop neighbors should be larger than or equal to 1."
        A_1, A_k = graph.A.clone(), graph.A.clone()
        A_history = []
        for _ in range(k - 1):
            A_k = torch.sparse.mm(A_k, A_1)
            if not only_kHop:
                A_history.append(A_k.clone())
        if not only_kHop:
            A_k = A_1
            for A_ in A_history:
                A_k = A_k + A_
        e_list = [
            tuple(set([v_idx] + A_k[v_idx]._indices().cpu().squeeze(0).tolist())) for v_idx in range(graph.num_v)
        ]
        return e_list
        '''
        这个函数用于从图中构建一个超图，其中每个超边由中心顶点和其k-hop邻居顶点构成。其中参数k指定了k-hop邻居的数量。
        如果图有|V|个顶点，那么构建的超图将具有|V|个顶点和等于或小于|V|的超边。
        该函数首先复制了输入图的邻接矩阵A_1，并在每个迭代步骤中执行了一个稀疏矩阵乘法
        A_k = torch.sparse.mm(A_k, A_1)。如果only_kHop为False，函数会保存所有A_k的历史值，
        并将它们相加来构建完整的k-hop邻居集合，最终构造超边列表e_list。
        该函数返回超边列表e_list。
        '''
    @staticmethod
    def from_graph_kHop(
        graph: "Graph", k: int, only_kHop: bool = False, device: torch.device = torch.device("cpu"),
    ) -> "Hypergraph":
        r"""Construct the hypergraph from the graph by k-Hop neighbors. Each hyperedge in the hypergraph is constructed by the central vertex and its :math:`k`-Hop neighbor vertices.

        .. note::
            If the graph have :math:`|\mathcal{V}|` vertices, the constructed hypergraph will have :math:`|\mathcal{V}|` vertices and equal to or less than :math:`|\mathcal{V}|` hyperedges.

        Args:
            ``graph`` (``Graph``): The graph to construct the hypergraph.
            ``k`` (``int``): The number of hop neighbors.
            ``only_kHop`` (``bool``): If set to ``True``, only the central vertex and its :math:`k`-th Hop neighbors are used to construct the hyperedges. By default, the constructed hyperedge will include the central vertex and its [ :math:`1`-th, :math:`2`-th, :math:`\cdots`, :math:`k`-th ] Hop neighbors. Defaults to ``False``.
            ``device`` (``torch.device``, optional): The device to store the hypergraph. Defaults to ``torch.device('cpu')``.
        """
        e_list = Hypergraph._e_list_from_graph_kHop(graph, k, only_kHop)
        hg = Hypergraph(graph.num_v, e_list, device=device)
        return hg
        '''
        这是一个从一个图中构建 k-Hop 超图的函数。k-Hop 超图是一种将中心节点和其 k-Hop 邻居节点作为超边构成的超图，
        其中 k 是一个预先定义的参数，表示跳数。
        该函数使用 _e_list_from_graph_kHop 函数生成一个超边列表，
        然后调用 Hypergraph 类的构造函数生成超图。
        函数的输入参数包括：
        graph：一个 Graph 对象，表示要从中构建超图的图。
        k：一个整数，表示要使用的 k 值，即跳数。
        only_kHop：一个布尔值，表示是否只使用中心节点和其 k-Hop 邻居节点来构造超边。
        默认为 False，表示使用中心节点和其 [ 1st，2nd，...，kth ] Hop 邻居来构造超边。
        device：一个 torch.device 对象，表示要在哪个设备上存储超图。默认为 torch.device('cpu')。
        函数的输出是一个 Hypergraph 对象，表示从图中构建的超图。
        '''
    @staticmethod
    def _e_list_from_bigraph(bigraph: "BiGraph", U_as_vertex: bool = True) -> List[tuple]:
        r"""Construct hyperedges from the bipartite graph.

        Args:
            ``bigraph`` (``BiGraph``): The bipartite graph to construct the hypergraph.
            ``U_as_vertex`` (``bool``, optional): If set to ``True``, vertices in set :math:`\mathcal{U}` and set :math:`\mathcal{V}`
             will be treated as vertices and hyperedges in the constructed hypergraph, respectively.
             If set to ``False``, vertices in set :math:`\mathcal{U}` and set :math:`\mathcal{V}`
             will be treated as hyperedges and vertices in the constructed hypergraph, respectively. Defaults to ``True``.
        """
        e_list = []
        if U_as_vertex:
            for v in range(bigraph.num_v):
                u_list = bigraph.nbr_u(v)
                if len(u_list) > 0:
                    e_list.append(u_list)
        else:
            for u in range(bigraph.num_u):
                v_list = bigraph.nbr_v(u)
                if len(v_list) > 0:
                    e_list.append(v_list)
        return e_list
        '''
        这段代码是用于从二分图中构建超图的。二分图是一类特殊的图，
        它的顶点可以分成两个不相交的集合，集合内的顶点之间没有边相连，只有集合间的顶点之间有边相连。

        构建超图的目的是将多个顶点看作一个超顶点，这些顶点之间的关系被称为超边。
        在这段代码中，我们可以看到有两个参数：

        bigraph：需要转化为超图的二分图。
        U_as_vertex：布尔值，如果为True，则在超图中，将集合U中的顶点视为节点，
        集合V中的顶点视为超边；如果为False，则反之。
        在函数的实现中，我们通过遍历集合V中的每个节点，将与之相连的集合U中的节点构成一个超边。
        如果U_as_vertex为False，则我们会遍历集合U中的每个节点，并将与之相连的集合V中的节点构成一个超边。

        最终返回值是一个包含所有超边的列表。
        '''
    @staticmethod
    def from_bigraph(
        bigraph: "BiGraph", U_as_vertex: bool = True, device: torch.device = torch.device("cpu")
    ) -> "Hypergraph":
        r"""Construct the hypergraph from the bipartite graph.

        Args:
            ``bigraph`` (``BiGraph``): The bipartite graph to construct the hypergraph.
            ``U_as_vertex`` (``bool``, optional): If set to ``True``, vertices in set :math:`\mathcal{U}` and set :math:`\mathcal{V}`
             will be treated as vertices and hyperedges in the constructed hypergraph, respectively.
             If set to ``False``, vertices in set :math:`\mathcal{U}` and set :math:`\mathcal{V}`
             will be treated as hyperedges and vertices in the constructed hypergraph, respectively. Defaults to ``True``.
            ``device`` (``torch.device``, optional): The device to store the hypergraph. Defaults to ``torch.device('cpu')``.
        """
        e_list = Hypergraph._e_list_from_bigraph(bigraph, U_as_vertex)
        if U_as_vertex:
            hg = Hypergraph(bigraph.num_u, e_list, device=device)
        else:
            hg = Hypergraph(bigraph.num_v, e_list, device=device)
        return hg
        '''
        这个函数从一个二分图构建一个超图。如果 U_as_vertex 被设置为 True，
        则二分图的 U 集合中的节点将成为超图中的顶点，V 集合中的节点将成为超图中的超边；
        否则 U 集合中的节点将成为超图中的超边，V 集合中的节点将成为超图中的顶点。
        超图的边集是从 Hypergraph._e_list_from_bigraph 函数获取的。
        根据 U_as_vertex，超图的顶点数量将是二分图的 U 或 V 集合的大小。
        返回的超图将存储在指定的设备上，如果没有指定，则默认存储在 CPU 上。
        '''
    # =====================================================================================
    # some structure modification functions
    def add_hyperedges(
        self,
        e_list: Union[List[int], List[List[int]]],
        e_weight: Optional[Union[float, List[float]]] = None,
        merge_op: str = "mean",
        group_name: str = "main",
    ):
        r"""Add hyperedges to the hypergraph. If the ``group_name`` is not specified, the hyperedges will be added to the default ``main`` hyperedge group.

        Args:
            ``num_v`` (``int``): The number of vertices in the hypergraph.
            ``e_list`` (``Union[List[int], List[List[int]]]``): A list of hyperedges describes how the vertices point to the hyperedges.
            ``e_weight`` (``Union[float, List[float]]``, optional): A list of weights for hyperedges. If set to ``None``, the value ``1`` is used for all hyperedges. Defaults to ``None``.
            ``merge_op`` (``str``): The merge operation for the conflicting hyperedges. The possible values are ``"mean"``, ``"sum"``, and ``"max"``. Defaults to ``"mean"``.
            ``group_name`` (``str``, optional): The target hyperedge group to add these hyperedges. Defaults to the ``main`` hyperedge group.
        """
        e_list = self._format_e_list(e_list)
        if e_weight is None:
            e_weight = [1.0] * len(e_list)
        elif type(e_weight) in (int, float):
            e_weight = [e_weight]
        elif type(e_weight) is list:
            pass
        else:
            raise TypeError(f"The type of e_weight should be float or list, but got {type(e_weight)}")
        assert len(e_list) == len(e_weight), "The number of hyperedges and the number of weights are not equal."

        for _idx in range(len(e_list)):
            self._add_hyperedge(
                self._hyperedge_code(e_list[_idx], e_list[_idx]), {"w_e": float(e_weight[_idx])}, merge_op, group_name,
            )
        self._clear_cache(group_name)
        '''
        这是一个Hypergraph类的方法，用于向hypergraph对象添加超边(hyperedge)。
        一个超边是一个链接到一个或多个节点的点集。超边可以具有权重。
        函数参数：
        e_list：描述超边连接节点的列表。可以是包含int值的一维列表，也可以是包含int值的二维列表。
        如果是一维列表，那么这个超边就只与一个节点相连。如果是二维列表，那么这个超边就与多个节点相连。
        e_weight：可选参数，表示超边的权重。如果没有指定，所有超边的权重都为1.0。
        merge_op：可选参数，表示处理冲突超边的方法。如果多个超边连接到同一组节点，
        那么需要将它们合并为一个超边。可选值包括："mean"表示取平均值，"sum"表示求和，"max"表示取最大值。默认为"mean"。
        group_name：可选参数，表示超边所属的超边组的名称。如果没有指定，就属于默认的"main"组。
        函数实现：
        首先，函数将e_list转换为一个标准的二维列表，使得列表的每个元素都是一个一维列表。
        如果e_weight是None，则将所有超边的权重设为1.0；如果e_weight是整数或浮点数，
        则将其转换为只包含一个元素的列表；如果e_weight是一个列表，那么保持不变。
        接下来，对于每个超边，使用_add_hyperedge方法添加一个新的超边到hypergraph对象中。
        超边代码是使用_hyperedge_code方法生成的，它接受一个由节点组成的一维列表，
        然后返回一个标识符(字符串)来唯一标识这个超边。然后，函数将超边的权重设置为
        {"w_e": float(e_weight[_idx])}并使用指定的merge_op方法将超边添加到指定的超边组中。最后，清除缓存。
        '''
    def add_hyperedges_from_feature_kNN(self, feature: torch.Tensor, k: int, group_name: str = "main"):
        r"""Add hyperedges from the feature matrix by k-NN. Each hyperedge is constructed by the central vertex and its :math:`k`-Nearest Neighbor vertices.

        Args:
            ``features`` (``torch.Tensor``): The feature matrix.
            ``k`` (``int``): The number of nearest neighbors.
            ``group_name`` (``str``, optional): The target hyperedge group to add these hyperedges. Defaults to the ``main`` hyperedge group.
        """
        assert (
            feature.shape[0] == self.num_v
        ), "The number of vertices in the feature matrix is not equal to the number of vertices in the hypergraph."
        e_list = Hypergraph._e_list_from_feature_kNN(feature, k)
        self.add_hyperedges(e_list, group_name=group_name)
        '''
        这段代码实现了通过输入特征矩阵和k值构建超图的功能。其中，特征矩阵包含了每个节点的特征向量
        ，k值表示每个节点与其k个最近邻节点构成一个超边。具体地，该函数首先检查输入的特征矩阵是否与超图中节点数目相等，
        如果不相等则会抛出异常。然后，使用Hypergraph类中定义的静态方法_e_list_from_feature_kNN
        生成超图的边列表e_list，该方法根据输入的特征矩阵和k值，计算出每个节点的k个最近邻节点，
        并将它们与该节点一起构成一条边。最后，使用add_hyperedges方法将这些边加入到超图的指定超边组（默认为main组）中
        '''
    def add_hyperedges_from_graph(self, graph: "Graph", group_name: str = "main"):
        r"""Add hyperedges from edges in the graph. Each edge in the graph is treated as a hyperedge.

        Args:
            ``graph`` (``Graph``): The graph to join the hypergraph.
            ``group_name`` (``str``, optional): The target hyperedge group to add these hyperedges. Defaults to the ``main`` hyperedge group.
        """
        assert self.num_v == graph.num_v, "The number of vertices in the hypergraph and the graph are not equal."
        e_list, e_weight = graph.e
        self.add_hyperedges(e_list, e_weight=e_weight, group_name=group_name)
        '''
        这段代码实现了从一个Graph对象中提取边信息，将每条边都看作是一个超边添加到Hypergraph中的功能。
        其中，参数graph是一个Graph对象，参数group_name是可选的目标超边组名称，
        用于指定将超边添加到Hypergraph哪个组中。函数首先确保Hypergraph中的节点数与Graph中的节点数相等，
        然后将Graph中的边提取出来作为超边列表e_list和边权重列表e_weight，
        最后调用add_hyperedges方法将这些超边添加到Hypergraph对象中。
        '''
    def add_hyperedges_from_graph_kHop(
        self, graph: "Graph", k: int, only_kHop: bool = False, group_name: str = "main"
    ):
        r"""Add hyperedges from vertices and its k-Hop neighbors in the graph. Each hyperedge in the hypergraph is constructed by the central vertex and its :math:`k`-Hop neighbor vertices.

        .. note::
            If the graph have :math:`|\mathcal{V}|` vertices, the constructed hypergraph will have :math:`|\mathcal{V}|` vertices and equal to or less than :math:`|\mathcal{V}|` hyperedges.

        Args:
            ``graph`` (``Graph``): The graph to join the hypergraph.
            ``k`` (``int``): The number of hop neighbors.
            ``only_kHop`` (``bool``): If set to ``True``, only the central vertex and its :math:`k`-th Hop neighbors are used to construct the hyperedges. By default, the constructed hyperedge will include the central vertex and its [ :math:`1`-th, :math:`2`-th, :math:`\cdots`, :math:`k`-th ] Hop neighbors. Defaults to ``False``.
            ``group_name`` (``str``, optional): The target hyperedge group to add these hyperedges. Defaults to the ``main`` hyperedge group.
        """
        assert self.num_v == graph.num_v, "The number of vertices in the hypergraph and the graph are not equal."
        e_list = Hypergraph._e_list_from_graph_kHop(graph, k, only_kHop=only_kHop)
        self.add_hyperedges(e_list, group_name=group_name)
        '''
        这个函数用于根据图中的顶点和其 k-Hop 邻居添加超边到超图中。每个超边由中心顶点及其 k-Hop 邻居顶点构成。
        如果图中有 |V| 个顶点，则构造的超图将有 |V| 个顶点，且不超过 |V| 条超边。
        参数中，graph 是待加入超图的图，k 是跳数，only_kHop 表示是否只包含中心顶点和它的 k-Hop 邻居，
        缺省为 False，即构造的超边将包括中心顶点及其 [1、2、…、k]-th Hop 邻居。
        group_name 表示待添加的超边所在的超边组，默认为“main”超边组。
        '''
    def add_hyperedges_from_bigraph(self, bigraph: "BiGraph", U_as_vertex: bool = False, group_name: str = "main"):
        r"""Add hyperedges from the bipartite graph.

        Args:
            ``bigraph`` (``BiGraph``): The bigraph to join the hypergraph.
            ``U_as_vertex`` (``bool``): If set to ``True``, vertices in set :math:`\mathcal{U}` and set :math:`\mathcal{V}`
             will be treated as vertices and hyperedges in the constructed hypergraph, respectively.
             If set to ``False``, vertices in set :math:`\mathcal{U}` and set :math:`\mathcal{V}`
             will be treated as hyperedges and vertices in the constructed hypergraph, respectively. Defaults to ``True``.
            ``group_name`` (``str``, optional): The target hyperedge group to add these hyperedges. Defaults to the ``main`` hyperedge group.
        """
        if U_as_vertex:
            assert (
                self.num_v == bigraph.num_u
            ), "The number of vertices in the hypergraph and the number of vertices in set U of the bipartite graph are not equal."
        else:
            assert (
                self.num_v == bigraph.num_v
            ), "The number of vertices in the hypergraph and the number of vertices in set V of the bipartite graph are not equal."
        e_list = Hypergraph._e_list_from_bigraph(bigraph, U_as_vertex=U_as_vertex)
        self.add_hyperedges(e_list, group_name=group_name)
        '''
        这段代码是为超图对象添加来自二分图的超边。传入一个二分图对象，如果参数 U_as_vertex 为 True，
        则会将二分图中的集合 U 中的顶点作为超图的顶点，集合 V 中的顶点作为超图的超边；
        如果为 False，则将集合 V 中的顶点作为超图的顶点，集合 U 中的顶点作为超图的超边。
        函数内部调用 _e_list_from_bigraph 方法，将二分图中每个顶点作为超边的顶点，
        每个超边包含的顶点作为超边的元素，构成一个超边列表。
        最后将超边列表传给超图对象的 add_hyperedges 方法，添加到超图中。
        '''
    def remove_hyperedges(
        self, e_list: Union[List[int], List[List[int]]], group_name: Optional[str] = None,
    ):
        r"""Remove the specified hyperedges from the hypergraph.

        Args:
            ``e_list`` (``Union[List[int], List[List[int]]]``): A list of hyperedges describes how the vertices point to the hyperedges.
            ``group_name`` (``str``, optional): Remove these hyperedges from the specified hyperedge group. If not specified, the function will
                remove those hyperedges from all hyperedge groups. Defaults to the ``None``.
        """
        assert (
            group_name is None or group_name in self.group_names
        ), "The specified group_name is not in existing hyperedge groups."
        e_list = self._format_e_list(e_list)
        if group_name is None:
            for _idx in range(len(e_list)):
                e_code = self._hyperedge_code(e_list[_idx], e_list[_idx])
                for name in self.group_names:
                    self._raw_groups[name].pop(e_code, None)
        else:
            for _idx in range(len(e_list)):
                e_code = self._hyperedge_code(e_list[_idx], e_list[_idx])
                self._raw_groups[group_name].pop(e_code, None)
        self._clear_cache(group_name)
        '''
        该函数实现从超图中删除指定的超边。其中参数e_list为要删除的超边列表，
        可以是单个超边的ID或超边ID的列表；group_name为指定要从哪个超边组中删除这些超边，
        如果未指定，则将其从所有超边组中删除。函数首先判断指定的group_name是否存在于超图中，
        然后将超边从指定的组或所有超边组中删除，并清除缓存。
        '''
    def remove_group(self, group_name: str):
        r"""Remove the specified hyperedge group from the hypergraph.

        Args:
            ``group_name`` (``str``): The name of the hyperedge group to remove.
        """
        self._raw_groups.pop(group_name, None)
        self._clear_cache(group_name)
        '''
        这段代码定义了一个方法用于从hypergraph中移除指定的hyperedge group。
        它接受一个字符串参数group_name，表示要删除的hyperedge group的名称。
        函数会使用该名称来访问存储在hypergraph中的raw hyperedge group，然后从中删除对应的hyperedges。
        最后，函数会调用_clear_cache方法来清除与这个group相关的任何缓存数据。
        '''
    def drop_hyperedges(self, drop_rate: float, ord="uniform"):
        r"""Randomly drop hyperedges from the hypergraph. This function will return a new hypergraph with non-dropped hyperedges.

        Args:
            ``drop_rate`` (``float``): The drop rate of hyperedges.
            ``ord`` (``str``): The order of dropping edges. Currently, only ``'uniform'`` is supported. Defaults to ``uniform``.
        """
        if ord == "uniform":
            _raw_groups = {}
            for name in self.group_names:
                _raw_groups[name] = {k: v for k, v in self._raw_groups[name].items() if random.random() > drop_rate}
            state_dict = {
                "num_v": self.num_v,
                "raw_groups": _raw_groups,
            }
            _hg = Hypergraph.from_state_dict(state_dict)
            _hg = _hg.to(self.device)
        else:
            raise ValueError(f"Unkonwn drop order: {ord}.")
        return _hg
        '''
        这个函数是一个超图的方法，用于随机删除一部分超边，并返回一个不含被删除超边的新的超图。
        该函数的参数包括：
        drop_rate: 超边被删除的概率。
        ord: 删除超边的顺序，目前只支持 uniform （均匀删除）。
        当 ord 参数为 uniform 时，函数会在每个超边上以相同的概率进行随机采样，
        如果采样结果小于 drop_rate，则该超边被删除。函数最终返回一个新的超图，包括原来超图中未被删除的超边。
        值得注意的是，该函数返回的新的超图是使用 from_state_dict() 方法构造的，
        并拥有与原超图相同的节点数和节点特征，但是超边数和边集不同。
        同时，该函数支持设备转移，并可以将新的超图放置在指定设备上（默认为原来的设备）。
        '''
    def drop_hyperedges_of_group(self, group_name: str, drop_rate: float, ord="uniform"):
        r"""Randomly drop hyperedges from the specified hyperedge group. This function will return a new hypergraph with non-dropped hyperedges.

        Args:
            ``group_name`` (``str``): The name of the hyperedge group.
            ``drop_rate`` (``float``): The drop rate of hyperedges.
            ``ord`` (``str``): The order of dropping edges. Currently, only ``'uniform'`` is supported. Defaults to ``uniform``.
        """
        if ord == "uniform":
            _raw_groups = {}
            for name in self.group_names:
                if name == group_name:
                    _raw_groups[name] = {
                        k: v for k, v in self._raw_groups[name].items() if random.random() > drop_rate
                    }
                else:
                    _raw_groups[name] = self._raw_groups[name]
            state_dict = {
                "num_v": self.num_v,
                "raw_groups": _raw_groups,
            }
            _hg = Hypergraph.from_state_dict(state_dict)
            _hg = _hg.to(self.device)
        else:
            raise ValueError(f"Unkonwn drop order: {ord}.")
        return _hg
        '''
        drop_hyperedges_of_group 函数的功能是从指定的超边组中随机删除一定比例的超边，
        并返回一个新的超图，该超图不包含被删除的超边。该函数的输入参数包括：
        group_name: 需要从中删除超边的超边组的名称。
        drop_rate: 随机删除超边的比例。
        ord: 超边删除顺序，当前仅支持 "uniform"。默认为 "uniform"。
        该函数的实现方式与 drop_hyperedges 函数类似，不同之处在于只在指定的超边组中删除超边。
        首先，函数根据输入参数 group_name 确定要删除的超边所在的超边组。接下来，函数遍历每个超边组，
        如果当前超边组是指定的超边组，则随机删除该超边组中的一部分超边，否则保留所有超边。
        最后，函数根据更新后的超边集合构建一个新的超图，并返回该超图。
        需要注意的是，函数只支持一次从一个超边组中删除超边，如果需要删除多个超边组中的超边，则需要多次调用该函数。
        '''
    # =====================================================================================
    # properties for representation
    @property
    def v(self) -> List[int]:
        r"""Return the list of vertices.
        """
        return super().v
    '''
    这个函数是Hypergraph类的方法，它继承自父类GraphBase中的同名方法。它的作用是返回当前超图中的所有节点的列表。
    该方法没有参数，直接调用即可返回一个由整数组成的列表，每个整数代表超图中的一个节点。
    由于超图的节点不一定是连续的整数，所以返回的节点列表也不一定连续。
    '''
    @property
    def v_weight(self) -> List[float]:
        r"""Return the list of vertex weights.
        """
        return self._v_weight
    '''
    这个函数返回一个包含图中所有顶点权重的列表，列表中的每个元素对应于图中一个顶点的权重。
    '''
    @property
    def e(self) -> Tuple[List[List[int]], List[float]]:
        r"""Return all hyperedges and weights in the hypergraph.
        """
        if self.cache.get("e", None) is None:
            e_list, e_weight = [], []
            for name in self.group_names:
                _e = self.e_of_group(name)
                e_list.extend(_e[0])
                e_weight.extend(_e[1])
            self.cache["e"] = (e_list, e_weight)
        return self.cache["e"]
        '''
        这是一个返回超图中所有超边及其权重的方法。如果在缓存中存在，它会返回缓存的结果，
        否则它会遍历超图的所有超边，并将其存储在缓存中以便以后访问。
        这个方法返回一个二元组，第一个元素是超边的列表，每个超边表示为一个整数列表，第二个元素是与超边对应的权重的列表。
        '''
    def e_of_group(self, group_name: str) -> Tuple[List[List[int]], List[float]]:
        r"""Return all hyperedges and weights of the specified hyperedge group.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("e", None) is None:
            e_list = [e_code[0] for e_code in self._raw_groups[group_name].keys()]
            e_weight = [e_content["w_e"] for e_content in self._raw_groups[group_name].values()]
            self.group_cache[group_name]["e"] = (e_list, e_weight)
        return self.group_cache[group_name]["e"]
        '''
        这个函数是一个返回指定超边组中的所有超边及其权重的方法。参数 group_name 是指定的超边组的名称。
        如果缓存中没有相应的超边组信息，则会从 _raw_groups 中获取超边组信息，并将其存储在 group_cache 缓存中。
        然后将超边列表和超边权重列表返回。函数的返回值是一个包含两个列表的元组。
        第一个列表包含超边列表，其中每个超边表示为一个由节点编号组成的列表。第二个列表包含相应的权重。
        '''
    @property
    def num_v(self) -> int:
        r"""Return the number of vertices in the hypergraph.
        """
        return super().num_v
    '''
    这个函数返回该超图中的顶点数。其中，超图继承了图（Graph）类的成员函数和属性，
    因此 num_v 函数直接调用了 Graph 类中的 num_v 属性，返回该属性值。
    '''
    @property
    def num_e(self) -> int:
        r"""Return the number of hyperedges in the hypergraph.
        """
        return super().num_e
    '''
    这个函数实现了返回超图中的超边数量，即 num_e。
    '''
    def num_e_of_group(self, group_name: str) -> int:
        r"""Return the number of hyperedges of the specified hyperedge group.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        return super().num_e_of_group(group_name)
    '''
    这个函数的作用是返回指定的超边组中的超边数量。传入参数包括：
    group_name：指定的超边组的名称。
    函数的返回值是一个整数，表示指定超边组中的超边数量。
    这个函数在内部调用了 super().num_e_of_group(group_name)，
    也就是调用了父类 HypergraphBase 中的同名函数，因此它们的功能是完全一样的。
    它们都通过 group_name 参数来指定超边组，并返回该超边组中的超边数量。
    '''
    @property
    def deg_v(self) -> List[int]:
        r"""Return the degree list of each vertex.
        """
        return self.D_v._values().cpu().view(-1).numpy().tolist()
    '''
    这个函数返回一个列表，其中包含每个顶点的度数。顶点的度数是指与该顶点相关联的超边数量。
    在该函数中，它调用了一个叫做D_v的成员变量，它是一个字典类型，存储了每个顶点的度数。
    调用_values()方法可以获得字典中的值，它是一个PyTorch张量。
    接着，它调用了PyTorch的cpu()方法将张量移动到CPU上，再调用view(-1)将其展平为一维数组。
    最后，调用numpy().tolist()方法将该一维数组转换为Python列表并返回。
    '''
    def deg_v_of_group(self, group_name: str) -> List[int]:
        r"""Return the degree list of each vertex of the specified hyperedge group.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        return self.D_v_of_group(group_name)._values().cpu().view(-1).numpy().tolist()
        '''
        这个函数是返回指定超边组中每个顶点的度数列表。
        首先，它通过检查指定的组名是否在超图中来确保指定的超边组存在。
        然后，它调用 D_v_of_group 函数，该函数返回一个大小为 $|V|$ 的张量，
        其中第 $i$ 个元素是第 $i$ 个顶点在指定超边组中的度数。
        最后，它将张量转换为 Numpy 数组并将其转换为 Python 列表，以便于用户使用。
        '''
    @property
    def deg_e(self) -> List[int]:
        r"""Return the degree list of each hyperedge.
        """
        return self.D_e._values().cpu().view(-1).numpy().tolist()
    '''
    该函数返回每个超边的度数列表。超边的度数是指它包含的顶点数量。
    具体来说，对于超图 $G=(V,E)$，每个超边 $e\in E$ 的度数定义为 $deg(e)=|e|$，
    其中 $|e|$ 表示 $e$ 中顶点的数量。因此，该函数返回一个列表，其中第 $i$ 个元素为超图中第 $i$ 个超边的度数。
    '''
    def deg_e_of_group(self, group_name: str) -> List[int]:
        r"""Return the degree list of each hyperedge of the specified hyperedge group.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        return self.D_e_of_group(group_name)._values().cpu().view(-1).numpy().tolist()
    '''
    该函数返回指定超边组中每个超边的度数列表。
    其中，度数指该超边在超图中出现的次数（即包含该超边的超边组数）。
    该函数接受一个参数group_name，表示要查询度数列表的超边组名称。
    如果指定的超边组不存在于超图中，将会抛出一个异常。函数返回一个整数列表，列表中的每个元素表示对应超边的度数。
    '''
    def nbr_e(self, v_idx: int) -> List[int]:
        r"""Return the neighbor hyperedge list of the specified vertex.

        Args:
            ``v_idx`` (``int``): The index of the vertex.
        """
        return self.N_e(v_idx).cpu().numpy().tolist()
    '''
    这个函数是返回指定顶点的相邻超边列表，输入参数是顶点的索引编号，返回值是一个包含相邻超边索引编号的列表。
    '''
    def nbr_e_of_group(self, v_idx: int, group_name: str) -> List[int]:
        r"""Return the neighbor hyperedge list of the specified vertex of the specified hyperedge group.

        Args:
            ``v_idx`` (``int``): The index of the vertex.
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        return self.N_e_of_group(v_idx, group_name).cpu().numpy().tolist()
    '''
    这个函数返回特定超边组中指定顶点的邻接超边列表。 
    它需要两个参数，一个是顶点的索引，另一个是超边组的名称。
    它首先检查超边组的名称是否在图中，如果不存在，则引发异常。 
    然后调用内部方法“N_e_of_group”，返回相应的邻接超边列表，并将其转换为python列表格式。最终将其返回。
    '''
    def nbr_v(self, e_idx: int) -> List[int]:
        r"""Return the neighbor vertex list of the specified hyperedge.

        Args:
            ``e_idx`` (``int``): The index of the hyperedge.
        """
        return self.N_v(e_idx).cpu().numpy().tolist()
    '''
    这个函数是返回指定超边的邻居顶点列表。输入参数是超边的索引，输出是包含所有邻居顶点索引的列表。
    它利用了超图的邻接表来实现这个功能，即在超图中对于每个超边存储了与它相连的所有顶点的列表，
    这个列表就是邻接表的一个元素。因此，对于给定的超边索引，可以直接获取邻接表中对应的列表，从而得到这个超边的所有邻居顶点。
    '''
    def nbr_v_of_group(self, e_idx: int, group_name: str) -> List[int]:
        r"""Return the neighbor vertex list of the specified hyperedge of the specified hyperedge group.

        Args:
            ``e_idx`` (``int``): The index of the hyperedge.
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        return self.N_v_of_group(e_idx, group_name).cpu().numpy().tolist()
    '''
    该方法返回指定超边在指定超边组中的相邻顶点列表。
    它接受两个参数，即超边的索引和超边组的名称，如果指定的超边组不在当前超图的超边组中，
    则会引发一个 AssertionError。函数返回一个列表，其中包含超边相邻的顶点索引
    '''
    @property
    def num_groups(self) -> int:
        r"""Return the number of hyperedge groups in the hypergraph.
        """
        return super().num_groups
    '''
    这个函数返回超图中存在的超边组数量。超边组是根据给定的超边特征创建的。
    在这个超图中，每个超边都属于一组，因此可以使用这个函数来查看超图中有多少种不同的超边特征
    '''
    @property
    def group_names(self) -> List[str]:
        r"""Return the names of all hyperedge groups in the hypergraph.
        """
        return super().group_names
    '''
    这个函数返回一个列表，包含该超图中所有超边组的名称。
    '''
    # =====================================================================================
    # properties for deep learning
    @property
    def vars_for_DL(self) -> List[str]:
        r"""Return a name list of available variables for deep learning in the hypergraph including

        Sparse Matrices:

        .. math::
            \mathbf{H}, \mathbf{H}^\top, \mathcal{L}_{sym}, \mathcal{L}_{rw} \mathcal{L}_{HGNN},

        Sparse Diagnal Matrices:

        .. math::
            \mathbf{W}_v, \mathbf{W}_e, \mathbf{D}_v, \mathbf{D}_v^{-1}, \mathbf{D}_v^{-\frac{1}{2}}, \mathbf{D}_e, \mathbf{D}_e^{-1},

        Vectors:

        .. math::
            \overrightarrow{v2e}_{src}, \overrightarrow{v2e}_{dst}, \overrightarrow{v2e}_{weight},\\
            \overrightarrow{e2v}_{src}, \overrightarrow{e2v}_{dst}, \overrightarrow{e2v}_{weight}

        """
        return [
            "H",
            "H_T",
            "L_sym",
            "L_rw",
            "L_HGNN",
            "W_v",
            "W_e",
            "D_v",
            "D_v_neg_1",
            "D_v_neg_1_2",
            "D_e",
            "D_e_neg_1",
            "v2e_src",
            "v2e_dst",
            "v2e_weight" "e2v_src",
            "e2v_dst",
            "e2v_weight",
        ]

    @property
    def v2e_src(self) -> torch.Tensor:
        r"""Return the source vertex index vector :math:`\overrightarrow{v2e}_{src}` of the connections (vertices point to hyperedges) in the hypergraph.
        """
        return self.H_T._indices()[1].clone()
    '''
    这个函数返回超图中每个连接（边）的源顶点索引，它是一个torch张量对象。
    这个张量的形状是(1, E) ，其中 E 是超图中的边数。张量的每个元素存储的是一个从 0 开始的整数值，
    代表超图中一个顶点的索引。每个元素的位置代表相应连接的索引，即在超图的超边集合中的位置。
    '''

    def v2e_src_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the source vertex index vector :math:`\overrightarrow{v2e}_{src}` of the connections (vertices point to hyperedges) in the specified hyperedge group.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        return self.H_T_of_group(group_name)._indices()[1].clone()
    '''
    这个方法用于获取指定超边组中顶点指向超边的连接中的源顶点索引向量 :math:\overrightarrow{v2e}_{src}。
    它首先检查指定的超边组是否存在于超图中，然后调用 H_T_of_group 方法获取指定超边组的转置超边关系矩阵，
    再从中提取源顶点索引向量。最后，它返回一个张量表示源顶点索引向量。
    '''
    @property
    def v2e_dst(self) -> torch.Tensor:
        r"""Return the destination hyperedge index vector :math:`\overrightarrow{v2e}_{dst}` of the connections (vertices point to hyperedges) in the hypergraph.
        """
        return self.H_T._indices()[0].clone()
    '''
    这个函数返回了超图中顶点指向边的目标超边的索引向量 :math:\overrightarrow{v2e}_{dst} 。
    在这个向量中，索引向量的第 i 个元素是指向超边的索引，即超边的 ID ，
    对应于 :math:\overrightarrow{v2e}_{src} 中第 i 个元素的源节点的 ID。
    '''
    def v2e_dst_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the destination hyperedge index vector :math:`\overrightarrow{v2e}_{dst}` of the connections (vertices point to hyperedges) in the specified hyperedge group.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        return self.H_T_of_group(group_name)._indices()[0].clone()
    '''
    这个方法返回指定超边组中的源顶点连接到的超边的目标超边的索引向量 :math:\overrightarrow{v2e}_{dst}。
    它使用的是超图稀疏矩阵的转置 :math:\mathbf{H}^{\top}，其中第一行包含目标超边的索引，
    第二行包含源顶点的索引。对于给定的超边组名，它首先确保指定的超边组存在，
    然后返回由 :math:\mathbf{H}^{\top} 得到的目标超边索引向量。该方法返回的张量是PyTorch张量。
    '''
    @property
    def v2e_weight(self) -> torch.Tensor:
        r"""Return the weight vector :math:`\overrightarrow{v2e}_{weight}` of the connections (vertices point to hyperedges) in the hypergraph.
        """
        return self.H_T._values().clone()
    '''
    这段代码实现了获取连接（从顶点到超边）的权重向量 $\overrightarrow{v2e}_{weight}$，
    其中顶点到超边的连接被表示为一个稀疏矩阵 $\mathbf{H}^\top$ 的非零值。
    在这个稀疏矩阵中，每一列对应于一个超边，每一行对应于一个顶点，非零值对应于从该顶点到该超边的连接的权重。

    这个方法直接调用了超图对象的 _values() 方法，返回稀疏矩阵 $\mathbf{H}^\top$ 中所有非零值的张量。
    由于这个方法返回一个张量对象，所以可以方便地用于深度学习框架中的张量运算。
    '''

    def v2e_weight_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the weight vector :math:`\overrightarrow{v2e}_{weight}` of the connections (vertices point to hyperedges) in the specified hyperedge group.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        return self.H_T_of_group(group_name)._values().clone()
    '''
    这是一个Hypergraph类中的方法，用于获取指定超边组中连接(指向)超边的边界点的权重向量，
    也就是 :math:\overrightarrow{v2e}_{weight}。
    首先，该方法会检查传入的超边组名是否在已有的超边组名列表中，如果不存在则会抛出异常。
    然后，方法会通过调用H_T_of_group方法获取到指定超边组的转置超边张量 H_T。最后，返回 H_T 张量中的权重向量。
    需要注意的是，如果在 Hypergraph 对象创建时未指定超边权重，则默认权重为 1。
    '''
    @property
    def e2v_src(self) -> torch.Tensor:
        r"""Return the source hyperedge index vector :math:`\overrightarrow{e2v}_{src}` of the connections (hyperedges point to vertices) in the hypergraph.
        """
        return self.H._indices()[1].clone()
    '''
    这个函数返回超图中每个连接的源超边索引向量，这些连接是超边到顶点的连接。
    这个源超边索引向量是通过访问转置超图中的目标顶点索引向量得到的。
    返回的张量是PyTorch张量的克隆版本，这样就不会影响到超图本身。
    '''
    def e2v_src_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the source hyperedge index vector :math:`\overrightarrow{e2v}_{src}` of the connections (hyperedges point to vertices) in the specified hyperedge group.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        return self.H_of_group(group_name)._indices()[1].clone()
    '''
    这个方法返回连接（超边指向顶点）在超图中的源超边索引向量 $\overrightarrow{e2v}_{src}$。
    其中，源超边指向的是超边在超图中的顺序编号。
    这个方法使用了 self.H 属性，它是一个大小为 $2\times|\mathcal{E}|$ 的稀疏张量，
    其中第一行是每个超边连接到的顶点在 $V$ 中的索引，第二行是每个超边在 $\mathcal{E}$ 中的索引。
    这个方法首先通过 self.H._indices() 获取到超图的超边连接矩阵，
    然后通过索引 [1] 选择第二行即超边索引，最后通过 clone() 方法返回一个与该张量值相同但梯度不同的张量。
    '''
    @property
    def e2v_dst(self) -> torch.Tensor:
        r"""Return the destination vertex index vector :math:`\overrightarrow{e2v}_{dst}` of the connections (hyperedges point to vertices) in the hypergraph.
        """
        return self.H._indices()[0].clone()
    '''
    这是一个HyperGraph类中用于获取与连接( hyperedge 指向 vertices )相关的信息的方法。
    其中，e2v_src() 方法返回连接源的超边索引向量，即 $\overrightarrow{e2v}{src}$，
    而e2v_dst() 方法返回连接目标的顶点索引向量，即 $\overrightarrow{e2v}{dst}$。
    这两个方法返回的向量均为PyTorch张量类型，可以用于深度学习中的计算。
    由于HyperGraph类的数据结构特殊，其中的超边和顶点都以索引形式存储，
    因此这些方法返回的是超边和顶点的索引，而不是它们的具体数值。
    '''
    def e2v_dst_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the destination vertex index vector :math:`\overrightarrow{e2v}_{dst}` of the connections (hyperedges point to vertices) in the specified hyperedge group.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        return self.H_of_group(group_name)._indices()[0].clone()
    '''
    这个函数与之前提到的 e2v_src_of_group 函数类似，只不过返回的是指定超边组中超边连接的目标顶点的索引向量。
    该函数首先检查 group_name 是否是当前超图实例的一个超边组的名称，如果不是，则抛出一个断言错误。
    然后，它使用 H_of_group 方法获取指定的超边组的超边张量，并返回其中第一维度的索引向量，它表示超边连接的目标顶点
    '''
    @property
    def e2v_weight(self) -> torch.Tensor:
        r"""Return the weight vector :math:`\overrightarrow{e2v}_{weight}` of the connections (hyperedges point to vertices) in the hypergraph.
        """
        return self.H._values().clone()
    '''
    这个函数返回的是超图中所有连接（超边指向顶点）的权重向量 $\overrightarrow{e2v}_{weight}$，
    其中每个元素表示一个连接的权重。这个函数的实现很简单，直接调用了存储超图连接的张量 self.H 的 _values() 方法，
    并使用 clone() 方法复制一份返回，以防对原张量进行不必要的修改。
    '''
    def e2v_weight_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the weight vector :math:`\overrightarrow{e2v}_{weight}` of the connections (hyperedges point to vertices) in the specified hyperedge group.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        return self.H_of_group(group_name)._values().clone()
    '''
    该函数返回指定超边组内连接（超边指向顶点）的权重向量。它接受一个参数“group_name”，
    表示所需超边组的名称。函数首先使用assert语句检查指定的组名是否在现有超边组中，
    如果不在则返回一条错误消息。接下来，函数将调用H_of_group方法获取指定组内的超边张量，
    然后返回该张量的值部分的克隆。换句话说，这个函数返回了指定组内超边连接到顶点的权重值。
    '''
    @property
    def H(self) -> torch.Tensor:
        r"""Return the hypergraph incidence matrix :math:`\mathbf{H}` with ``torch.sparse_coo_tensor`` format.
        """
        if self.cache.get("H") is None:
            self.cache["H"] = self.H_v2e
        return self.cache["H"]
    '''
    这是一个类中的方法，名称为 H，作用是返回一个稀疏的 torch 张量，
    代表了该超图的关联矩阵（即超图的关系表达方式），其中每一行代表了一条超边（即超图中的一个元素），
    每一列代表了一个节点，对于位置 (i, j) 上的值，如果第 $i$ 个超边连接了第 $j$ 个节点，
    则该位置上的值为该超边连接该节点的权重；如果第 $i$ 个超边没有连接第 $j$ 个节点，则该位置上的值为 0。
    该方法首先检查一个名为 cache 的字典中是否已经存储了 H，如果有，
    则直接返回存储在 cache 中的值，否则返回 H_v2e，H_v2e 是该超图的顶点到超边的关联矩阵。
    '''
    def H_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the hypergraph incidence matrix :math:`\mathbf{H}` of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("H") is None:
            self.group_cache[group_name]["H"] = self.H_v2e_of_group(group_name)
        return self.group_cache[group_name]["H"]
    '''
    这个函数的作用是返回指定超边组的超图关联矩阵 $\mathbf{H}$，以稀疏张量的格式存储。
    首先，它会检查指定的超边组是否在超图的超边组名称列表中，如果不在，就会抛出一个断言错误。
    接着，如果指定超边组的缓存中不存在超图关联矩阵，则会调用 H_v2e_of_group() 函数计算，
    并将结果存储在指定超边组的缓存中。最后，函数返回该超图关联矩阵。如果已经存在于缓存中，则直接返回缓存中的结果。
    '''
    @property
    def H_T(self) -> torch.Tensor:
        r"""Return the transpose of the hypergraph incidence matrix :math:`\mathbf{H}^\top` with ``torch.sparse_coo_tensor`` format.
        """
        if self.cache.get("H_T") is None:
            self.cache["H_T"] = self.H.t()
        return self.cache["H_T"]
    '''
    这个函数返回超图关联矩阵 $\mathbf{H}$ 的转置 $\mathbf{H}^\top$，以稀疏张量的格式存储。
    如果缓存中存在，则直接返回缓存中的结果。
    在函数内部，首先检查缓存中是否存在超图关联矩阵的转置。
    如果不存在，则调用 PyTorch 的 t() 函数计算超图关联矩阵的转置，并将结果存储在缓存中。
    最后，返回计算出的超图关联矩阵的转置。如果已经存在，则直接返回缓存中的结果    
    '''
    def H_T_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the transpose of the hypergraph incidence matrix :math:`\mathbf{H}^\top` of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("H_T") is None:
            self.group_cache[group_name]["H_T"] = self.H_of_group(group_name).t()
        return self.group_cache[group_name]["H_T"]
    '''
    这个函数返回指定超边组的超图关联矩阵 $\mathbf{H}^\top$ 的转置，以稀疏张量的格式存储。
    如果缓存中存在，则直接返回缓存中的结果。
    在函数内部，首先检查指定的超边组是否在超图的超边组名称列表中。
    如果不存在，则会抛出一个断言错误。否则，如果指定超边组的缓存中不存在超图关联矩阵的转置，
    则调用 H_of_group 函数获取指定超边组的超图关联矩阵，并计算其转置并存储在指定超边组的缓存中，
    最后返回该转置矩阵。如果已经存在，则直接返回缓存中的结果。   
    '''
    @property
    def W_v(self) -> torch.Tensor:
        r"""Return the weight matrix :math:`\mathbf{W}_v` of vertices with ``torch.sparse_coo_tensor`` format.
        """
        if self.cache.get("W_v") is None:
            _tmp = torch.Tensor(self.v_weight)
            _num_v = _tmp.size(0)
            self.cache["W_v"] = torch.sparse_coo_tensor(
                torch.arange(0, _num_v, device=self.device).view(1, -1).repeat(2, 1),
                _tmp,
                torch.Size([_num_v, _num_v]),
                device=self.device,
            ).coalesce()
        return self.cache["W_v"]
    '''
    这个函数返回顶点的权重矩阵 $\mathbf{W}_v$，以稀疏张量的格式存储。
    如果缓存中存在，则直接返回缓存中的结果。
    在函数内部，首先检查缓存中是否存在顶点权重矩阵。
    如果不存在，则创建一个形状为 $n_v \times n_v$ 的稀疏张量，其中 $n_v$ 是顶点数。
    稀疏张量的下标是一个二维张量，第一维是 $0 \sim n_v-1$ 的连续整数，
    第二维也是 $0 \sim n_v-1$ 的连续整数，表示矩阵中每个元素的下标。
    权重是输入的顶点权重的张量，每个元素对应于矩阵中相应的位置。
    然后将这个稀疏张量缓存起来，并返回。如果已经存在，则直接返回缓存中的结果。
    '''
    @property
    def W_e(self) -> torch.Tensor:
        r"""Return the weight matrix :math:`\mathbf{W}_e` of hyperedges with ``torch.sparse_coo_tensor`` format.
        """
        if self.cache.get("W_e") is None:
            _tmp = [self.W_e_of_group(name)._values().clone() for name in self.group_names]
            _tmp = torch.cat(_tmp, dim=0).view(-1)
            _num_e = _tmp.size(0)
            self.cache["W_e"] = torch.sparse_coo_tensor(
                torch.arange(0, _num_e, device=self.device).view(1, -1).repeat(2, 1),
                _tmp,
                torch.Size([_num_e, _num_e]),
                device=self.device,
            ).coalesce()
        return self.cache["W_e"]
    '''
    这个函数返回超边权重矩阵 $\mathbf{W}_e$，以稀疏张量的格式存储。如果缓存中存在，则直接返回缓存中的结果。
    在函数内部，如果缓存中不存在，则将每个超边组的超边权重矩阵取出来，连接起来成为一个张量。
    然后，使用 torch.sparse_coo_tensor 函数将张量转换为稀疏张量，并存储在缓存中。最后返回稀疏张量。
    '''
    def W_e_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the weight matrix :math:`\mathbf{W}_e` of hyperedges of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("W_e") is None:
            _tmp = self._fetch_W_of_group(group_name).view(-1)
            _num_e = _tmp.size(0)
            self.group_cache[group_name]["W_e"] = torch.sparse_coo_tensor(
                torch.arange(0, _num_e, device=self.device).view(1, -1).repeat(2, 1),
                _tmp,
                torch.Size([_num_e, _num_e]),
                device=self.device,
            ).coalesce()
        return self.group_cache[group_name]["W_e"]
    '''
    这个函数返回的是指定超边组的超图中的边权矩阵 $\mathbf{W_e}$，以稀疏张量的格式存储。
    如果缓存中存在，则直接返回缓存中的结果。
    在函数内部，首先检查指定的超边组是否在超图的超边组名称列表中。
    如果不存在，则会抛出一个断言错误。否则，如果指定超边组的缓存中不存在边权矩阵，
    则调用 _fetch_W_of_group 函数计算并存储在指定超边组的缓存中，最后返回该矩阵。如果已经存在，则直接返回缓存中的结果。
    '''
    @property
    def D_v(self) -> torch.Tensor:
        r"""Return the vertex degree matrix :math:`\mathbf{D}_v` with ``torch.sparse_coo_tensor`` format.
        """
        if self.cache.get("D_v") is None:
            _tmp = [self.D_v_of_group(name)._values().clone() for name in self.group_names]
            _tmp = torch.vstack(_tmp).sum(dim=0).view(-1)
            self.cache["D_v"] = torch.sparse_coo_tensor(
                torch.arange(0, self.num_v, device=self.device).view(1, -1).repeat(2, 1),
                _tmp,
                torch.Size([self.num_v, self.num_v]),
                device=self.device,
            ).coalesce()
        return self.cache["D_v"]
    '''
    该方法实现返回顶点度矩阵，即每个顶点的度数，其使用稀疏张量格式返回结果。
    如果缓存中已有该结果，则直接返回缓存结果，否则，对于每个超边组，调用 D_v_of_group 方法获得每个顶点度数，
    再将结果沿着行方向堆叠，计算总的度数，最后构建稀疏张量表示的度矩阵。返回构建的稀疏张量表示的顶点度矩阵。
    '''
    def D_v_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the vertex degree matrix :math:`\mathbf{D}_v` of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("D_v") is None:
            H = self.H_of_group(group_name).clone()
            w_e = self.W_e_of_group(group_name)._values().clone()
            val = w_e[H._indices()[1]] * H._values()
            H_ = torch.sparse_coo_tensor(H._indices(), val, size=H.shape, device=self.device).coalesce()
            _tmp = torch.sparse.sum(H_, dim=1).to_dense().clone().view(-1)
            _num_v = _tmp.size(0)
            self.group_cache[group_name]["D_v"] = torch.sparse_coo_tensor(
                torch.arange(0, _num_v, device=self.device).view(1, -1).repeat(2, 1),
                _tmp,
                torch.Size([_num_v, _num_v]),
                device=self.device,
            ).coalesce()
        return self.group_cache[group_name]["D_v"]
    '''
    这个方法用于计算指定超边组的顶点度矩阵 :math:\mathbf{D}_v。
    顶点度矩阵是一个对角线矩阵，其对角线元素为每个顶点的度数，即与之相连的超边的数量。
    它可以通过超图的邻接矩阵和边权重计算得到。
    方法首先检查缓存，如果不存在则计算。首先获取指定超边组的邻接矩阵 H 和权重矩阵 W_e。
    接下来，将权重矩阵中每个元素乘以邻接矩阵中对应位置的值，得到一个新的邻接矩阵 H_。
    然后，计算 H_ 每行的和，这个和就是对应顶点的度数。最后，构造对角线矩阵并将其存储在缓存中。
    如果该超边组的顶点度矩阵已经存在于缓存中，则直接返回缓存中的值。
    需要注意的是，这里计算的顶点度矩阵是一个稀疏矩阵，
    使用了 PyTorch 的稀疏矩阵表示方式 torch.sparse_coo_tensor，
    并且使用了 PyTorch 提供的稀疏矩阵运算函数 torch.sparse.sum 来计算每行的和。
    '''
    @property
    def D_v_neg_1(self) -> torch.Tensor:
        r"""Return the vertex degree matrix :math:`\mathbf{D}_v^{-1}` with ``torch.sparse_coo_tensor`` format.
        """
        if self.cache.get("D_v_neg_1") is None:
            _mat = self.D_v.clone()
            _val = _mat._values() ** -1
            _val[torch.isinf(_val)] = 0
            self.cache["D_v_neg_1"] = torch.sparse_coo_tensor(
                _mat._indices(), _val, _mat.size(), device=self.device
            ).coalesce()
        return self.cache["D_v_neg_1"]
    '''
    这个函数返回的是图的顶点度矩阵的逆矩阵。首先会检查这个矩阵是否在缓存中，
    如果不在则会将顶点度矩阵的每个元素取倒数得到一个新的稀疏张量。
    同时，对于原矩阵中的值为0或inf的元素，会将其赋值为0。最后会将这个新的稀疏张量存入缓存并返回。
    '''
    def D_v_neg_1_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the vertex degree matrix :math:`\mathbf{D}_v^{-1}` of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("D_v_neg_1") is None:
            _mat = self.D_v_of_group(group_name).clone()
            _val = _mat._values() ** -1
            _val[torch.isinf(_val)] = 0
            self.group_cache[group_name]["D_v_neg_1"] = torch.sparse_coo_tensor(
                _mat._indices(), _val, _mat.size(), device=self.device
            ).coalesce()
        return self.group_cache[group_name]["D_v_neg_1"]
    '''
    这个函数的作用是计算指定超边组的顶点度矩阵的逆矩阵 :math:\mathbf{D}_v^{-1}。
    其中，顶点度矩阵 :math:\mathbf{D}_v 是一个对角矩阵，其对角线上的元素是对应顶点的度数。
    对于每个顶点，其度数等于它所在的所有超边的度数之和。
    该函数首先会判断当前缓存中是否已经存在该超边组的顶点度矩阵的逆矩阵，
    如果不存在，则会先调用 D_v_of_group 函数来计算该超边组的顶点度矩阵，
    然后将其取每个元素的倒数来得到逆矩阵。注意，如果某个顶点的度数为 0，则它在逆矩阵中对应的元素会被设为 0。
    最后，该函数会返回一个稀疏的 COO 格式的矩阵，其中非零元素的值是逆矩阵的相应元素值，其余元素的值均为 0。
    '''
    @property
    def D_v_neg_1_2(self) -> torch.Tensor:
        r"""Return the vertex degree matrix :math:`\mathbf{D}_v^{-\frac{1}{2}}` with ``torch.sparse_coo_tensor`` format.
        """
        if self.cache.get("D_v_neg_1_2") is None:
            _mat = self.D_v.clone()
            _val = _mat._values() ** -0.5
            _val[torch.isinf(_val)] = 0
            self.cache["D_v_neg_1_2"] = torch.sparse_coo_tensor(
                _mat._indices(), _val, _mat.size(), device=self.device
            ).coalesce()
        return self.cache["D_v_neg_1_2"]
    '''
    这是一个Python函数，计算一个图的顶点度数矩阵的“负半次幂”（即 :math:\mathbf{D}_v^{-\frac{1}{2}}）。
    这个函数使用 PyTorch 库的稀疏张量数据结构，并利用了缓存技术来避免重复计算。

    具体地，它的实现方式是先从缓存中查找是否已经计算过这个矩阵，如果没有，则复制顶点度数矩阵
    ，对其所有元素取负半次幂，然后用稀疏张量表示这个矩阵，并将其压缩。最后将压缩后的稀疏张量返回。

    这个函数的返回值是一个稀疏张量，表示顶点度数矩阵的负半次幂。
    这个稀疏张量的形状与顶点度数矩阵相同，但是其中的值被替换为相应的负半次幂。
    这个稀疏张量中的每个非零元素表示一个边的权重，它被赋值为边所连接的两个顶点的度数的倒数的平方。
    '''
    def D_v_neg_1_2_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the vertex degree matrix :math:`\mathbf{D}_v^{-\frac{1}{2}}` of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("D_v_neg_1_2") is None:
            _mat = self.D_v_of_group(group_name).clone()
            _val = _mat._values() ** -0.5
            _val[torch.isinf(_val)] = 0
            self.group_cache[group_name]["D_v_neg_1_2"] = torch.sparse_coo_tensor(
                _mat._indices(), _val, _mat.size(), device=self.device
            ).coalesce()
        return self.group_cache[group_name]["D_v_neg_1_2"]
    '''
    这段代码实现了计算指定超边组的顶点度数矩阵的两个变换：1) 逆矩阵的计算；2) 逆平方根矩阵的计算。

    其中，第一个变换所得的结果是逆矩阵 $\mathbf{D}_v^{-1}$，
    其定义为矩阵 $\mathbf{D}_v$ 中每个元素除以对应的行或列的和，然后取其倒数。
    在此代码实现中，会先从已有的顶点度数矩阵 $\mathbf{D}_v$ 开始，对其对角线上的每个元素求倒数，
    然后构建出一个新的稀疏矩阵。具体地，先克隆出一个与原矩阵形状相同的新矩阵 _mat，然后用 _mat._values() 
    获取其非零元素构成的向量，再对其进行取倒数的操作。为了避免出现无穷大的情况，
    还需要对取倒数之后为无穷大的元素进行处理，此处是将其赋为 $0$。
    最后，将取倒数后的 _val 以及 _mat 的行列坐标信息输入到 torch.sparse_coo_tensor 中，构建出新的稀疏矩阵即可。

    第二个变换所得的结果是逆平方根矩阵 $\mathbf{D}_v^{-\frac{1}{2}}$，
    其定义为矩阵 $\mathbf{D}_v$ 中每个元素的平方根的倒数。这里的实现方式和第一个变换类似，
    只需要将 _val 的求倒数改为求平方根的倒数即可。
    '''
    @property
    def D_e(self) -> torch.Tensor:
        r"""Return the hyperedge degree matrix :math:`\mathbf{D}_e` with ``torch.sparse_coo_tensor`` format.
        """
        if self.cache.get("D_e") is None:
            _tmp = [self.D_e_of_group(name)._values().clone() for name in self.group_names]
            _tmp = torch.cat(_tmp, dim=0).view(-1)
            _num_e = _tmp.size(0)
            self.cache["D_e"] = torch.sparse_coo_tensor(
                torch.arange(0, _num_e, device=self.device).view(1, -1).repeat(2, 1),
                _tmp,
                torch.Size([_num_e, _num_e]),
                device=self.device,
            ).coalesce()
        return self.cache["D_e"]
    '''
    这段代码实现了计算超边度矩阵D_e的功能，返回一个torch.sparse_coo_tensor格式的矩阵。
    如果缓存中没有保存过D_e，则从每个超边组中取出对应的D_e_of_group，并将所有的值拼接起来得到一个一维向量，
    然后将该向量设置为D_e的非零元素，并通过torch.sparse_coo_tensor函数创建D_e稀疏矩阵。
    最后调用coalesce函数将矩阵压缩成紧凑的格式。如果缓存中已经保存了D_e，则直接返回缓存中的D_e。
    '''
    def D_e_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the hyperedge degree matrix :math:`\mathbf{D}_e` of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("D_e") is None:
            _tmp = torch.sparse.sum(self.H_T_of_group(group_name), dim=1).to_dense().clone().view(-1)
            _num_e = _tmp.size(0)
            self.group_cache[group_name]["D_e"] = torch.sparse_coo_tensor(
                torch.arange(0, _num_e, device=self.device).view(1, -1).repeat(2, 1),
                _tmp,
                torch.Size([_num_e, _num_e]),
                device=self.device,
            ).coalesce()
        return self.group_cache[group_name]["D_e"]
    '''
    这段代码定义了一个名为D_e_of_group的函数，它接受一个字符串类型的参数group_name，表示指定的超边组。
    函数的目的是返回该超边组的超边度矩阵，即$D_e$。返回值的格式为稀疏张量（torch.sparse_coo_tensor）。
    函数首先使用断言（assert）检查group_name是否在现有的超边组中，如果不在则抛出异常。
    然后，它检查该超边组是否已经缓存在group_cache字典中的D_e键下，如果没有，
    则计算$D_e$并将结果存储在缓存中。计算$D_e$的方法是先计算指定超边组的超边-节点关系矩阵$H$的转置$H^T$，
    然后对其进行按行求和，得到每个超边的度数，最后将这些度数组成一个对角矩阵，就是$D_e$。
    '''
    @property
    def D_e_neg_1(self) -> torch.Tensor:
        r"""Return the hyperedge degree matrix :math:`\mathbf{D}_e^{-1}` with ``torch.sparse_coo_tensor`` format.
        """
        if self.cache.get("D_e_neg_1") is None:
            _mat = self.D_e.clone()
            _val = _mat._values() ** -1
            _val[torch.isinf(_val)] = 0
            self.cache["D_e_neg_1"] = torch.sparse_coo_tensor(
                _mat._indices(), _val, _mat.size(), device=self.device
            ).coalesce()
        return self.cache["D_e_neg_1"]
    '''
    该函数实现了计算超边度数矩阵的逆矩阵的功能。具体来说，输入为一个超图对象，
    输出为一个 torch 稀疏张量（sparse tensor），表示超边度数矩阵的逆矩阵。
    在计算过程中，需要使用超边度数矩阵（即所有超边的度数构成的对角矩阵）作为输入，
    通过对其对角线上的元素取逆来得到逆矩阵。如果某个元素为零，则将逆矩阵对应位置的元素也设为零
    '''
    def D_e_neg_1_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the hyperedge degree matrix :math:`\mathbf{D}_e^{-1}` of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("D_e_neg_1") is None:
            _mat = self.D_e_of_group(group_name).clone()
            _val = _mat._values() ** -1
            _val[torch.isinf(_val)] = 0
            self.group_cache[group_name]["D_e_neg_1"] = torch.sparse_coo_tensor(
                _mat._indices(), _val, _mat.size(), device=self.device
            ).coalesce()
        return self.group_cache[group_name]["D_e_neg_1"]
    '''
    这段代码定义了一个名为 D_e_neg_1_of_group 的方法，该方法用于计算指定超边组的超边度数矩阵的倒数。
    其返回一个 torch 稀疏张量（sparse tensor）对象，表示超边度数矩阵的倒数。

    这个方法与 D_e_of_group 方法类似，不同之处在于 D_e_neg_1_of_group 方法返回的是超边度数矩阵的倒数，
    而不是超边度数矩阵本身。在计算超图的拉普拉斯矩阵时，超边度数矩阵的倒数常常被用于归一化超边的权重。
    '''
    def N_e(self, v_idx: int) -> torch.Tensor:
        r"""Return the neighbor hyperedges of the specified vertex with ``torch.Tensor`` format.

        .. note::
            The ``v_idx`` must be in the range of [0, :attr:`num_v`).

        Args:
            ``v_idx`` (``int``): The index of the vertex.
        """
        assert v_idx < self.num_v
        _tmp, e_bias = [], 0
        for name in self.group_names:
            _tmp.append(self.N_e_of_group(v_idx, name) + e_bias)
            e_bias += self.num_e_of_group(name)
        return torch.cat(_tmp, dim=0)
    '''
    这段代码定义了一个方法 N_e，用于返回指定顶点的邻居超边。
    该方法将遍历每个超边组，调用 N_e_of_group 方法计算每个组内的邻居超边，
    并使用 torch.cat 方法将结果连接成一个张量并返回。这个方法的参数为 v_idx，表示指定顶点的索引。
    该索引必须在 [0, num_v) 的范围内，否则会抛出 AssertionError。
    '''
    def N_e_of_group(self, v_idx: int, group_name: str) -> torch.Tensor:
        r"""Return the neighbor hyperedges of the specified vertex of the specified hyperedge group with ``torch.Tensor`` format.

        .. note::
            The ``v_idx`` must be in the range of [0, :attr:`num_v`).

        Args:
            ``v_idx`` (``int``): The index of the vertex.
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        assert v_idx < self.num_v
        e_indices = self.H_of_group(group_name)[v_idx]._indices()[0]
        return e_indices.clone()
    '''
    这个函数是用来返回指定超边组的指定节点的邻居超边的索引列表。
    它首先检查所提供的超边组名称是否存在，然后检查提供的节点索引是否在范围内，
    然后从该组的超边张量中提取出指定节点的所有邻居超边索引。它返回一个由这些超边索引组成的张量。
    '''
    def N_v(self, e_idx: int) -> torch.Tensor:
        r"""Return the neighbor vertices of the specified hyperedge with ``torch.Tensor`` format.

        .. note::
            The ``e_idx`` must be in the range of [0, :attr:`num_e`).

        Args:
            ``e_idx`` (``int``): The index of the hyperedge.
        """
        assert e_idx < self.num_e
        for name in self.group_names:
            if e_idx < self.num_e_of_group(name):
                return self.N_v_of_group(e_idx, name)
            else:
                e_idx -= self.num_e_of_group(name)
    '''
    这个方法实现的功能是返回指定超边的邻居顶点列表。方法参数为指定超边的索引。
    该方法遍历每个超边组来查找指定超边所属的超边组。如果指定超边的索引小于该超边组的超边数量，
    则调用 N_v_of_group() 方法来返回邻居顶点列表；否则减去该超边组的超边数量后继续遍历下一个超边组。
    '''
    def N_v_of_group(self, e_idx: int, group_name: str) -> torch.Tensor:
        r"""Return the neighbor vertices of the specified hyperedge of the specified hyperedge group with ``torch.Tensor`` format.

        .. note::
            The ``e_idx`` must be in the range of [0, :func:`num_e_of_group`).

        Args:
            ``e_idx`` (``int``): The index of the hyperedge.
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        assert e_idx < self.num_e_of_group(group_name)
        v_indices = self.H_T_of_group(group_name)[e_idx]._indices()[0]
        return v_indices.clone()
    '''
    这是一个Graph HyperNetworks模型的类中的一个方法。
    给定一个超边组名称和超边的索引，返回超边的所有相邻顶点的索引列表。
    '''
    # =====================================================================================
    # spectral-based convolution/smoothing
    def smoothing(self, X: torch.Tensor, L: torch.Tensor, lamb: float) -> torch.Tensor:
        return super().smoothing(X, L, lamb)
    '''
    输入是节点特征矩阵 X，拉普拉斯矩阵 L，以及一个平滑因子 lamb，返回平滑后的特征矩阵。
    平滑过程中会用到拉普拉斯矩阵的特征分解。
    '''
    @property
    def L_sym(self) -> torch.Tensor:
        r"""Return the symmetric Laplacian matrix :math:`\mathcal{L}_{sym}` of the hypergraph with ``torch.sparse_coo_tensor`` format.

        .. math::
            \mathcal{L}_{sym} = \mathbf{I} - \mathbf{D}_v^{-\frac{1}{2}} \mathbf{H} \mathbf{W}_e \mathbf{D}_e^{-1} \mathbf{H}^\top \mathbf{D}_v^{-\frac{1}{2}}
        """
        if self.cache.get("L_sym") is None:
            L_HGNN = self.L_HGNN.clone()
            self.cache["L_sym"] = torch.sparse_coo_tensor(
                torch.hstack([torch.arange(0, self.num_v, device=self.device).view(1, -1).repeat(2, 1), L_HGNN._indices(),]),
                torch.hstack([torch.ones(self.num_v, device=self.device), -L_HGNN._values()]),
                torch.Size([self.num_v, self.num_v]),
                device=self.device,
            ).coalesce()
        return self.cache["L_sym"]
    '''
    这个函数返回一个超图的对称拉普拉斯矩阵，矩阵以稀疏张量的形式表示。
    对称拉普拉斯矩阵是基于超图的拉普拉斯矩阵的一种形式，其中节点度量和边权重都考虑了。
    拉普拉斯矩阵表示图结构的拓扑信息，它的性质与图的性质密切相关。
    这个函数中使用的算法基于超图神经网络 (HGNN)，通过计算超图的拉普拉斯矩阵得到对称拉普拉斯矩阵，
    其中每个节点和每个超边都被视为矩阵的一维，张量中的每个值表示相应节点和超边之间的连接强度。
    '''
    def L_sym_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the symmetric Laplacian matrix :math:`\mathcal{L}_{sym}` of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        .. math::
            \mathcal{L}_{sym} = \mathbf{I} - \mathbf{D}_v^{-\frac{1}{2}} \mathbf{H} \mathbf{W}_e \mathbf{D}_e^{-1} \mathbf{H}^\top \mathbf{D}_v^{-\frac{1}{2}}

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("L_sym") is None:
            L_HGNN = self.L_HGNN_of_group(group_name).clone()
            self.group_cache[group_name]["L_sym"] = torch.sparse_coo_tensor(
                torch.hstack([torch.arange(0, self.num_v, device=self.device).view(1, -1).repeat(2, 1), L_HGNN._indices(),]),
                torch.hstack([torch.ones(self.num_v, device=self.device), -L_HGNN._values()]),
                torch.Size([self.num_v, self.num_v]),
                device=self.device,
            ).coalesce()
        return self.group_cache[group_name]["L_sym"]
    '''
    这段代码实现了计算一个超图的对称拉普拉斯矩阵（symmetric Laplacian matrix），
    并且也实现了计算超图中特定超边组的对称拉普拉斯矩阵。对称拉普拉斯矩阵是图论中的一种常见矩阵，
    用于描述节点之间的关系和相似度。这个矩阵的计算需要用到超图的节点、超边、以及它们之间的邻接关系，
    因此需要用到一些超图的基本属性和算法，例如超边的邻居节点、度数、权重、以及超图的邻接矩阵。
    这个代码实现了一个缓存机制来加速计算，避免重复计算和占用过多的计算资源。
    '''
    @property
    def L_rw(self) -> torch.Tensor:
        r"""Return the random walk Laplacian matrix :math:`\mathcal{L}_{rw}` of the hypergraph with ``torch.sparse_coo_tensor`` format.

        .. math::
            \mathcal{L}_{rw} = \mathbf{I} - \mathbf{D}_v^{-1} \mathbf{H} \mathbf{W}_e \mathbf{D}_e^{-1} \mathbf{H}^\top
        """
        if self.cache.get("L_rw") is None:
            _tmp = self.D_v_neg_1.mm(self.H).mm(self.W_e).mm(self.D_e_neg_1).mm(self.H_T)
            self.cache["L_rw"] = (
                torch.sparse_coo_tensor(
                    torch.hstack([torch.arange(0, self.num_v, device=self.device).view(1, -1).repeat(2, 1), _tmp._indices(),]),
                    torch.hstack([torch.ones(self.num_v, device=self.device), -_tmp._values()]),
                    torch.Size([self.num_v, self.num_v]),
                    device=self.device,
                )
                .coalesce()
                .clone()
            )
        return self.cache["L_rw"]
    '''
    这是一个Python函数，名为L_rw，用于计算一个超图的随机游走拉普拉斯矩阵（random walk Laplacian matrix），
    返回结果是一个torch.sparse_coo_tensor类型的稀疏张量。以下是具体实现细节：

    如果该超图的随机游走拉普拉斯矩阵还没有被计算出来（即缓存中没有L_rw），则进行如下计算：
    首先，将超图的度矩阵的负1/2次幂乘以超图的关联矩阵，得到一个稀疏矩阵；
    接着，将上述稀疏矩阵乘以超图的超边权重矩阵的负1次幂，再乘以上述稀疏矩阵的转置，得到一个稀疏矩阵；
    最后，用单位矩阵减去上述稀疏矩阵，即为该超图的随机游走拉普拉斯矩阵，将结果保存到缓存中。
    返回缓存中的随机游走拉普拉斯矩阵L_rw。
    '''
    def L_rw_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the random walk Laplacian matrix :math:`\mathcal{L}_{rw}` of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        .. math::
            \mathcal{L}_{rw} = \mathbf{I} - \mathbf{D}_v^{-1} \mathbf{H} \mathbf{W}_e \mathbf{D}_e^{-1} \mathbf{H}^\top

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("L_rw") is None:
            _tmp = (
                self.D_v_neg_1_of_group(group_name)
                .mm(self.H_of_group(group_name))
                .mm(self.W_e_of_group(group_name),)
                .mm(self.D_e_neg_1_of_group(group_name),)
                .mm(self.H_T_of_group(group_name),)
            )
            self.group_cache[group_name]["L_rw"] = (
                torch.sparse_coo_tensor(
                    torch.hstack([torch.arange(0, self.num_v, device=self.device).view(1, -1).repeat(2, 1), _tmp._indices(),]),
                    torch.hstack([torch.ones(self.num_v, device=self.device), -_tmp._values()]),
                    torch.Size([self.num_v, self.num_v]),
                    device=self.device,
                )
                .coalesce()
                .clone()
            )
        return self.group_cache[group_name]["L_rw"]
    '''
    这是一个类中的方法，用于计算指定超边组的随机游走拉普拉斯矩阵。
    它的计算方式与计算整个超图的随机游走拉普拉斯矩阵方法相同，即通过多次矩阵乘法来计算。
    该方法首先检查缓存中是否有计算结果，如果没有则计算，并将结果存入缓存中以便下次使用。


    '''
    ## HGNN Laplacian smoothing
    @property
    def L_HGNN(self) -> torch.Tensor:
        r"""Return the HGNN Laplacian matrix :math:`\mathcal{L}_{HGNN}` of the hypergraph with ``torch.sparse_coo_tensor`` format.

        .. math::
            \mathcal{L}_{HGNN} = \mathbf{D}_v^{-\frac{1}{2}} \mathbf{H} \mathbf{W}_e \mathbf{D}_e^{-1} \mathbf{H}^\top \mathbf{D}_v^{-\frac{1}{2}}
        """
        if self.cache.get("L_HGNN") is None:
            _tmp = self.D_v_neg_1_2.mm(self.H).mm(self.W_e).mm(self.D_e_neg_1).mm(self.H_T,).mm(self.D_v_neg_1_2)
            self.cache["L_HGNN"] = _tmp.coalesce()
        return self.cache["L_HGNN"]
    '''
    这个函数计算的是超图的 HGNN Laplacian 矩阵，使用的是稀疏张量表示形式。计算公式如下：
    其中 $\mathbf{D}_v$ 是顶点度数矩阵（对角矩阵），$\mathbf{D}_e$ 是超边度数矩阵（对角矩阵），
    $\mathbf{W}_e$ 是超边权重矩阵（对角矩阵），$\mathbf{H}$ 是超图中的超边-顶点邻接矩阵。

    具体而言，该函数会从 cache 缓存中获取 HGNN Laplacian 矩阵。如果缓存中不存在，将计算矩阵，
    然后将其存入缓存中，并返回。其中计算步骤如下：

    计算并用稀疏矩阵表示；
    将稀疏矩阵进行一次压缩（使用 coalesce() 方法）以减少内存使用；
    将压缩后的稀疏矩阵存入缓存中，并返回。
    '''
    def L_HGNN_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the HGNN Laplacian matrix :math:`\mathcal{L}_{HGNN}` of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        .. math::
            \mathcal{L}_{HGNN} = \mathbf{D}_v^{-\frac{1}{2}} \mathbf{H} \mathbf{W}_e \mathbf{D}_e^{-1} \mathbf{H}^\top \mathbf{D}_v^{-\frac{1}{2}}

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("L_HGNN") is None:
            _tmp = (
                self.D_v_neg_1_2_of_group(group_name)
                .mm(self.H_of_group(group_name))
                .mm(self.W_e_of_group(group_name))
                .mm(self.D_e_neg_1_of_group(group_name),)
                .mm(self.H_T_of_group(group_name),)
                .mm(self.D_v_neg_1_2_of_group(group_name),)
            )
            self.group_cache[group_name]["L_HGNN"] = _tmp.coalesce()
        return self.group_cache[group_name]["L_HGNN"]
    '''
    这个函数的作用是计算指定超边组的 HGNN 拉普拉斯矩阵。
    HGNN 拉普拉斯矩阵是基于超图的一种图拉普拉斯矩阵，用于表示超图中的节点之间的关系。
    具体来说，它是由超图中的超边权重构成的邻接矩阵计算得到的，同时考虑到超图的结构，
    使用度矩阵对邻接矩阵进行了归一化。该函数先检查指定的超边组是否存在，
    如果存在且尚未计算 HGNN 拉普拉斯矩阵，则使用对应的度矩阵、邻接矩阵、超边权重矩阵计算 HGNN 拉普拉斯矩阵，
    并将结果缓存以便下次调用。最后返回计算得到的 HGNN 拉普拉斯矩阵。
    '''
    def smoothing_with_HGNN(self, X: torch.Tensor, drop_rate: float = 0.0) -> torch.Tensor:
        r"""Return the smoothed feature matrix with the HGNN Laplacian matrix :math:`\mathcal{L}_{HGNN}`.

            .. math::
                \mathbf{X} = \mathbf{D}_v^{-\frac{1}{2}} \mathbf{H} \mathbf{W}_e \mathbf{D}_e^{-1} \mathbf{H}^\top \mathbf{D}_v^{-\frac{1}{2}} \mathbf{X}

        Args:
            ``X`` (``torch.Tensor``): The feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
    """
        if self.device != X.device:
            X = X.to(self.device)
        if drop_rate > 0.0:
            L_HGNN = sparse_dropout(self.L_HGNN, drop_rate)
        else:
            L_HGNN = self.L_HGNN
        return L_HGNN.mm(X)
    '''
    这个函数是在给定节点特征矩阵的情况下使用HGNN Laplacian矩阵来平滑（smooth）节点特征。
    具体来说，这个函数首先将节点特征矩阵乘以HGNN Laplacian矩阵，然后返回结果。
    这个过程等价于对节点特征进行了一个由HGNN Laplacian矩阵所定义的线性变换。
    如果给定了dropout rate，函数会以一定的概率随机dropout HGNN Laplacian矩阵中的某些元素。
    最终返回的矩阵与输入矩阵的形状相同。
    '''
    def smoothing_with_HGNN_of_group(self, group_name: str, X: torch.Tensor, drop_rate: float = 0.0) -> torch.Tensor:
        r"""Return the smoothed feature matrix with the HGNN Laplacian matrix :math:`\mathcal{L}_{HGNN}`.

            .. math::
                \mathbf{X} = \mathbf{D}_v^{-\frac{1}{2}} \mathbf{H} \mathbf{W}_e \mathbf{D}_e^{-1} \mathbf{H}^\top \mathbf{D}_v^{-\frac{1}{2}} \mathbf{X}

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
            ``X`` (``torch.Tensor``): The feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.device != X.device:
            X = X.to(self.device)
        if drop_rate > 0.0:
            L_HGNN = sparse_dropout(self.L_HGNN_of_group(group_name), drop_rate)
        else:
            L_HGNN = self.L_HGNN_of_group(group_name)
        return L_HGNN.mm(X)
    '''
    这是一个基于Hypergraph的图神经网络模型中的方法，用于对给定的特征矩阵进行平滑处理。
    该方法利用HGNN Laplacian矩阵来实现平滑处理，具体的计算公式如下：
    
    其中，$\mathbf{X}$是特征矩阵，$\mathbf{H}$是超边-节点关系的关联矩阵，
    $\mathbf{W}_e$是超边权重矩阵，$\mathbf{D}_e$和$\mathbf{D}_v$分别是超边度数矩阵和节点度数矩阵。
    这个方法需要输入一个指定的超边组名称，会返回该组超边的HGNN Laplacian矩阵作用于特征矩阵之后的结果。
    另外，可以选择在计算中加入dropout操作以增强模型的鲁棒性。
    '''
    # =====================================================================================
    # spatial-based convolution/message-passing
    ## general message passing functions
    def v2e_aggregation(
        self, X: torch.Tensor, aggr: str = "mean", v2e_weight: Optional[torch.Tensor] = None, drop_rate: float = 0.0
    ):
        r"""Message aggretation step of ``vertices to hyperedges``.

        Args:
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``v2e_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (vertices point to hyepredges). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        assert aggr in ["mean", "sum", "softmax_then_sum"]
        if self.device != X.device:
            self.to(X.device)
        if v2e_weight is None:
            if drop_rate > 0.0:
                P = sparse_dropout(self.H_T, drop_rate)
            else:
                P = self.H_T
            if aggr == "mean":
                X = torch.sparse.mm(P, X)
                X = torch.sparse.mm(self.D_e_neg_1, X)
            elif aggr == "sum":
                X = torch.sparse.mm(P, X)
            elif aggr == "softmax_then_sum":
                P = torch.sparse.softmax(P, dim=1)
                X = torch.sparse.mm(P, X)
            else:
                raise ValueError(f"Unknown aggregation method {aggr}.")
        else:
            # init message path
            assert (
                v2e_weight.shape[0] == self.v2e_weight.shape[0]
            ), "The size of v2e_weight must be equal to the size of self.v2e_weight."
            P = torch.sparse_coo_tensor(self.H_T._indices(), v2e_weight, self.H_T.shape, device=self.device)
            if drop_rate > 0.0:
                P = sparse_dropout(P, drop_rate)
            # message passing
            if aggr == "mean":
                X = torch.sparse.mm(P, X)
                D_e_neg_1 = torch.sparse.sum(P, dim=1).to_dense().view(-1, 1)
                D_e_neg_1[torch.isinf(D_e_neg_1)] = 0
                X = D_e_neg_1 * X
            elif aggr == "sum":
                X = torch.sparse.mm(P, X)
            elif aggr == "softmax_then_sum":
                P = torch.sparse.softmax(P, dim=1)
                X = torch.sparse.mm(P, X)
            else:
                raise ValueError(f"Unknown aggregation method {aggr}.")
        return X
    '''
    这个函数实现的是“顶点到超边”的信息聚合步骤。它将给定的顶点特征矩阵聚合为超边特征矩阵。
    具体来说，函数中的 X 是输入的顶点特征矩阵，大小为 $(|\mathcal{V}|, C)$，
    其中 $|\mathcal{V}|$ 是顶点数量，$C$ 是特征向量的维数。函数的 aggr 参数指定了信息聚合的方式，
    可以是 mean（求均值）、sum（求和）或 softmax_then_sum（进行 softmax 然后求和）。
    如果 v2e_weight 参数没有被指定，函数将使用在构建超图时指定的权重进行信息聚合；
    否则，函数将使用 v2e_weight 权重进行信息聚合。
    在信息聚合时，函数先将权重矩阵和输入特征矩阵相乘，然后应用超边度数矩阵的负一次幂，最后返回聚合后的超边特征矩阵。
    如果指定了 drop_rate 参数，则在相乘时会对权重矩阵应用 dropout。
    '''
    def v2e_aggregation_of_group(
        self,
        group_name: str,
        X: torch.Tensor,
        aggr: str = "mean",
        v2e_weight: Optional[torch.Tensor] = None,
        drop_rate: float = 0.0,
    ):
        r"""Message aggregation step of ``vertices to hyperedges`` in specified hyperedge group.

        Args:
            ``group_name`` (``str``): The specified hyperedge group.
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``v2e_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (vertices point to hyepredges). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        assert aggr in ["mean", "sum", "softmax_then_sum"]
        if self.device != X.device:
            self.to(X.device)
        if v2e_weight is None:
            if drop_rate > 0.0:
                P = sparse_dropout(self.H_T_of_group(group_name), drop_rate)
            else:
                P = self.H_T_of_group(group_name)
            if aggr == "mean":
                X = torch.sparse.mm(P, X)
                X = torch.sparse.mm(self.D_e_neg_1_of_group(group_name), X)
            elif aggr == "sum":
                X = torch.sparse.mm(P, X)
            elif aggr == "softmax_then_sum":
                P = torch.sparse.softmax(P, dim=1)
                X = torch.sparse.mm(P, X)
            else:
                raise ValueError(f"Unknown aggregation method {aggr}.")
        else:
            # init message path
            assert (
                v2e_weight.shape[0] == self.v2e_weight_of_group(group_name).shape[0]
            ), f"The size of v2e_weight must be equal to the size of self.v2e_weight_of_group('{group_name}')."
            P = torch.sparse_coo_tensor(
                self.H_T_of_group(group_name)._indices(),
                v2e_weight,
                self.H_T_of_group(group_name).shape,
                device=self.device,
            )
            if drop_rate > 0.0:
                P = sparse_dropout(P, drop_rate)
            # message passing
            if aggr == "mean":
                X = torch.sparse.mm(P, X)
                D_e_neg_1 = torch.sparse.sum(P, dim=1).to_dense().view(-1, 1)
                D_e_neg_1[torch.isinf(D_e_neg_1)] = 0
                X = D_e_neg_1 * X
            elif aggr == "sum":
                X = torch.sparse.mm(P, X)
            elif aggr == "softmax_then_sum":
                P = torch.sparse.softmax(P, dim=1)
                X = torch.sparse.mm(P, X)
            else:
                raise ValueError(f"Unknown aggregation method {aggr}.")
        return X
    '''
    这是一个使用PyTorch实现的函数，用于在指定的超边组中执行“顶点到超边”的信息聚合操作。
    函数输入包括指定的超边组名字、顶点特征矩阵、聚合方法、连接权重等。
    函数会根据指定的超边组名字找到对应的超边组，然后根据输入的参数执行不同的聚合操作，并返回聚合后的特征矩阵。
    具体实现中，如果没有指定连接权重，则会使用超图构造中指定的权重。
    函数支持三种聚合方法，分别为平均、求和和softmax之后求和。如果指定了连接权重，则会根据输入的权重进行聚合。
    在聚合过程中，还支持随机丢弃部分连接的dropout操作，以防止过拟合。
    函数最后返回聚合后的特征矩阵。
    '''
    def v2e_update(self, X: torch.Tensor, e_weight: Optional[torch.Tensor] = None):
        r"""Message update step of ``vertices to hyperedges``.

        Args:
            ``X`` (``torch.Tensor``): Hyperedge feature matrix. Size :math:`(|\mathcal{E}|, C)`.
            ``e_weight`` (``torch.Tensor``, optional): The hyperedge weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
        """
        if self.device != X.device:
            self.to(X.device)
        if e_weight is None:
            X = torch.sparse.mm(self.W_e, X)
        else:
            e_weight = e_weight.view(-1, 1)
            assert e_weight.shape[0] == self.num_e, "The size of e_weight must be equal to the size of self.num_e."
            X = e_weight * X
        return X
    '''
    这个函数是一个Hypergraph对象的方法，用于在vertices to hyperedges中进行信息更新，
    即将来自顶点的信息聚合到相应的超边中。具体来说：
    X是一个大小为(|V|, C)的张量，表示顶点特征矩阵。
    e_weight是一个大小为(|E|,)的张量，表示超边权重向量，用于加权顶点信息的聚合。
    函数执行的操作如下：
    如果没有指定e_weight，则将超图中定义的超边权重矩阵W_e和X做矩阵乘法，得到新的超边特征矩阵。
    如果指定了e_weight，则先将其转换为大小为(|E|, 1)的张量，并检查其形状是否与超图中定义的超边数相等。
    然后，将e_weight和X逐元素相乘，得到新的超边特征矩阵。
    最后，函数返回更新后的超边特征矩阵。
    '''
    def v2e_update_of_group(self, group_name: str, X: torch.Tensor, e_weight: Optional[torch.Tensor] = None):
        r"""Message update step of ``vertices to hyperedges`` in specified hyperedge group.

        Args:
            ``group_name`` (``str``): The specified hyperedge group.
            ``X`` (``torch.Tensor``): Hyperedge feature matrix. Size :math:`(|\mathcal{E}|, C)`.
            ``e_weight`` (``torch.Tensor``, optional): The hyperedge weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.device != X.device:
            self.to(X.device)
        if e_weight is None:
            X = torch.sparse.mm(self.W_e_of_group(group_name), X)
        else:
            e_weight = e_weight.view(-1, 1)
            assert e_weight.shape[0] == self.num_e_of_group(
                group_name
            ), f"The size of e_weight must be equal to the size of self.num_e_of_group('{group_name}')."
            X = e_weight * X
        return X
    '''
    这是一个用于在指定的超边组中执行“顶点到超边”的消息更新步骤的函数。接受三个参数：
    group_name (str)：指定超边组的名称。
    X (torch.Tensor)：超边特征矩阵，大小为(|E|, C)。
    e_weight (torch.Tensor, 可选)：超边权重向量。如果未指定，
    则函数将使用超图构建中指定的权重。默认值为None。
    函数首先检查指定的超边组是否存在，然后检查输入张量的设备是否与模型的设备匹配。
    如果未指定超边权重，则使用超图构建中指定的权重通过矩阵乘法更新输入张量。
    如果指定了超边权重，则将其重新形状为(-1,1)，并检查其大小是否与指定的超边组中的边数相同。
    最后，通过将超边权重乘以输入张量来更新张量。函数返回更新后的张量。
    '''
    def v2e(
        self,
        X: torch.Tensor,
        aggr: str = "mean",
        v2e_weight: Optional[torch.Tensor] = None,
        e_weight: Optional[torch.Tensor] = None,
        drop_rate: float = 0.0,
    ):
        r"""Message passing of ``vertices to hyperedges``. The combination of ``v2e_aggregation`` and ``v2e_update``.

        Args:
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``v2e_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (vertices point to hyepredges). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``e_weight`` (``torch.Tensor``, optional): The hyperedge weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        X = self.v2e_aggregation(X, aggr, v2e_weight, drop_rate=drop_rate)
        X = self.v2e_update(X, e_weight)
        return X
    '''
    这是一个 PyTorch 中的方法，用于实现超图上的“顶点到超边”的消息传递。
    该方法将“聚合”（aggregation）和“更新”（update）两个步骤组合在一起。
    具体而言，该方法接受顶点特征矩阵 X，执行顶点到超边的聚合操作，然后将结果传递给超边的更新函数进行更新。
    可选参数包括聚合方法（aggr）、连接权重（v2e_weight）、超边权重（e_weight）和随机断开率（drop_rate）。
    '''
    def v2e_of_group(
        self,
        group_name: str,
        X: torch.Tensor,
        aggr: str = "mean",
        v2e_weight: Optional[torch.Tensor] = None,
        e_weight: Optional[torch.Tensor] = None,
        drop_rate: float = 0.0,
    ):
        r"""Message passing of ``vertices to hyperedges`` in specified hyperedge group. The combination of ``e2v_aggregation_of_group`` and ``e2v_update_of_group``.

        Args:
            ``group_name`` (``str``): The specified hyperedge group.
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``v2e_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (vertices point to hyepredges). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``e_weight`` (``torch.Tensor``, optional): The hyperedge weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        X = self.v2e_aggregation_of_group(group_name, X, aggr, v2e_weight, drop_rate=drop_rate)
        X = self.v2e_update_of_group(group_name, X, e_weight)
        return X
    '''
    这是一个类中的方法，用于对指定超边组中的顶点到超边的信息传递。
    它结合了“v2e_aggregation_of_group”和“v2e_update_of_group”两个方法。
    传入的参数包括超边组名称、顶点特征矩阵X、聚合方法aggr、指定连接的权重矩阵v2e_weight、超边权重矩阵e_weight，
    以及可选的随机失活率drop_rate。方法会先使用“v2e_aggregation_of_group”方法对X进行聚合操作，
    然后使用“v2e_update_of_group”方法对结果进行更新。如果指定的超边组不存在，则会引发一个AssertionError。
    最后返回更新后的结果X。
    '''
    def e2v_aggregation(
        self, X: torch.Tensor, aggr: str = "mean", e2v_weight: Optional[torch.Tensor] = None, drop_rate: float = 0.0
    ):
        r"""Message aggregation step of ``hyperedges to vertices``.

        Args:
            ``X`` (``torch.Tensor``): Hyperedge feature matrix. Size :math:`(|\mathcal{E}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``e2v_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (hyperedges point to vertices). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        assert aggr in ["mean", "sum", "softmax_then_sum"]
        if self.device != X.device:
            self.to(X.device)
        if e2v_weight is None:
            if drop_rate > 0.0:
                P = sparse_dropout(self.H, drop_rate)
            else:
                P = self.H
            if aggr == "mean":
                X = torch.sparse.mm(P, X)
                X = torch.sparse.mm(self.D_v_neg_1, X)
            elif aggr == "sum":
                X = torch.sparse.mm(P, X)
            elif aggr == "softmax_then_sum":
                P = torch.sparse.softmax(P, dim=1)
                X = torch.sparse.mm(P, X)
            else:
                raise ValueError(f"Unknown aggregation method: {aggr}")
        else:
            # init message path
            assert (
                e2v_weight.shape[0] == self.e2v_weight.shape[0]
            ), "The size of e2v_weight must be equal to the size of self.e2v_weight."
            P = torch.sparse_coo_tensor(self.H._indices(), e2v_weight, self.H.shape, device=self.device)
            if drop_rate > 0.0:
                P = sparse_dropout(P, drop_rate)
            # message passing
            if aggr == "mean":
                X = torch.sparse.mm(P, X)
                D_v_neg_1 = torch.sparse.sum(P, dim=1).to_dense().view(-1, 1)
                D_v_neg_1[torch.isinf(D_v_neg_1)] = 0
                X = D_v_neg_1 * X
            elif aggr == "sum":
                X = torch.sparse.mm(P, X)
            elif aggr == "softmax_then_sum":
                P = torch.sparse.softmax(P, dim=1)
                X = torch.sparse.mm(P, X)
            else:
                raise ValueError(f"Unknown aggregation method: {aggr}")
        return X
    '''
    这是一个用于超图上的"超边到顶点"信息聚合函数。函数将超边特征矩阵与关联权重矩阵相乘并聚合，生成顶点特征矩阵。
    聚合方法可以是"mean"，"sum"和"softmax_then_sum"。如果没有指定关联权重，函数将使用在超图构造中指定的关联权重。
    如果指定了关联权重，函数将使用指定的关联权重。在信息聚合之前，函数还可以选择将关联矩阵中的连接随机删除。
    '''
    def e2v_aggregation_of_group(
        self,
        group_name: str,
        X: torch.Tensor,
        aggr: str = "mean",
        e2v_weight: Optional[torch.Tensor] = None,
        drop_rate: float = 0.0,
    ):
        r"""Message aggregation step of ``hyperedges to vertices`` in specified hyperedge group.

        Args:
            ``group_name`` (``str``): The specified hyperedge group.
            ``X`` (``torch.Tensor``): Hyperedge feature matrix. Size :math:`(|\mathcal{E}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``e2v_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (hyperedges point to vertices). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        assert aggr in ["mean", "sum", "softmax_then_sum"]
        if self.device != X.device:
            self.to(X.device)
        if e2v_weight is None:
            if drop_rate > 0.0:
                P = sparse_dropout(self.H_of_group(group_name), drop_rate)
            else:
                P = self.H_of_group(group_name)
            if aggr == "mean":
                X = torch.sparse.mm(P, X)
                X = torch.sparse.mm(self.D_v_neg_1_of_group(group_name), X)
            elif aggr == "sum":
                X = torch.sparse.mm(P, X)
            elif aggr == "softmax_then_sum":
                P = torch.sparse.softmax(P, dim=1)
                X = torch.sparse.mm(P, X)
            else:
                raise ValueError(f"Unknown aggregation method: {aggr}")
        else:
            # init message path
            assert (
                e2v_weight.shape[0] == self.e2v_weight_of_group(group_name).shape[0]
            ), f"The size of e2v_weight must be equal to the size of self.e2v_weight_of_group('{group_name}')."
            P = torch.sparse_coo_tensor(
                self.H_of_group(group_name)._indices(),
                e2v_weight,
                self.H_of_group(group_name).shape,
                device=self.device,
            )
            if drop_rate > 0.0:
                P = sparse_dropout(P, drop_rate)
            # message passing
            if aggr == "mean":
                X = torch.sparse.mm(P, X)
                D_v_neg_1 = torch.sparse.sum(P, dim=1).to_dense().view(-1, 1)
                D_v_neg_1[torch.isinf(D_v_neg_1)] = 0
                X = D_v_neg_1 * X
            elif aggr == "sum":
                X = torch.sparse.mm(P, X)
            elif aggr == "softmax_then_sum":
                P = torch.sparse.softmax(P, dim=1)
                X = torch.sparse.mm(P, X)
            else:
                raise ValueError(f"Unknown aggregation method: {aggr}")
        return X
    '''
    这段代码实现了在指定的超边组中进行从超边到顶点的信息聚合步骤。
    它包括一个参数group_name，用于指定超边组；一个参数X，表示超边的特征矩阵；一个参数aggr，表示聚合方法，
    可以是“mean”、“sum”或“softmax_then_sum”；一个参数e2v_weight，表示连接（从超边到顶点）上的权重向量，
    如果不指定，则函数将使用超图构建中指定的权重；最后一个参数drop_rate，表示dropout的概率。函数的输出是聚合后的特征矩阵。
    
    这个函数是实现“超边到顶点”（hyperedge-to-vertex, E2V）的聚合过程，即将超图中的超边信息聚合到对应的顶点上。
    具体来说，该函数将输入的超边特征矩阵 X 和一个超边组 group_name 作为输入，
    然后按照指定的聚合方式 aggr 对超边信息进行聚合，并返回聚合后的顶点特征矩阵。

    聚合方式 aggr 可以是三种之一：平均聚合（"mean"）、求和聚合（"sum"）和 softmax 加权求和聚合（"softmax_then_sum"）。具体实现中，对于每种聚合方式，都会先根据超边组 group_name 获得该组的超边-顶点连接矩阵 H 和顶点度数矩阵 D_v_neg_1（D_v_neg_1 是对顶点度数矩阵 D_v 取负倒数后得到的矩阵）。

    然后，对于平均聚合，会先将 H 与输入特征矩阵 X 相乘得到消息矩阵，再将该消息矩阵与 D_v_neg_1 相乘，
    得到最终的聚合结果。对于求和聚合，直接将 H 与输入特征矩阵 X 相乘即可。对于 softmax 加权求和聚合，
    需要先对连接矩阵 H 沿着第二维（即顶点维度）进行 softmax 操作，得到超边-顶点的权重，
    再将该权重矩阵与输入特征矩阵 X 相乘，得到最终的聚合结果。

    如果用户指定了超边-顶点连接矩阵的权重 e2v_weight，则会使用该权重矩阵进行聚合。
    此时，会先将 e2v_weight 转化成稀疏张量 P，然后按照与上述相同的方法进行聚合。

    最后，如果用户指定了 dropout 率 drop_rate，则会随机将超边-顶点连接矩阵中的一些连接（即边）概率地置零，
    以达到模型的正则化效果。
    '''
    def e2v_update(self, X: torch.Tensor):
        r"""Message update step of ``hyperedges to vertices``.

        Args:
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
        """
        if self.device != X.device:
            self.to(X.device)
        return X
    '''
    这个函数是 hypergraph neural network 中的一个空函数，即它没有实际的操作。
    在 e2v_aggregation_of_group 函数中进行了消息聚合后，
    将聚合后的消息传递给此函数进行更新操作。但是此函数不执行任何操作，因为更新操作通常是在超图神经网络的下一层中执行的。
    所以这个函数可以看作是一个占位符，为下一步操作做准备。
    '''

    def e2v_update_of_group(self, group_name: str, X: torch.Tensor):
        r"""Message update step of ``hyperedges to vertices`` in specified hyperedge group.

        Args:
            ``group_name`` (``str``): The specified hyperedge group.
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.device != X.device:
            self.to(X.device)
        return X

    '''
    这个函数是指定超边组进行“超边到顶点”的消息更新步骤，更新顶点特征矩阵X。
    其中，需要指定超边组的名称，如果指定的名称不在已有的超边组中，则会引发异常。
    如果顶点特征矩阵X与超图对象不在同一个设备上，则会将其移到超图对象所在的设备上。最后，该函数返回更新后的顶点特征矩阵X。
    '''
    def e2v(
        self, X: torch.Tensor, aggr: str = "mean", e2v_weight: Optional[torch.Tensor] = None, drop_rate: float = 0.0,
    ):
        r"""Message passing of ``hyperedges to vertices``. The combination of ``e2v_aggregation`` and ``e2v_update``.

        Args:
            ``X`` (``torch.Tensor``): Hyperedge feature matrix. Size :math:`(|\mathcal{E}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``e2v_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (hyperedges point to vertices). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        X = self.e2v_aggregation(X, aggr, e2v_weight, drop_rate=drop_rate)
        X = self.e2v_update(X)
        return X
    '''
    这是一个实现“超边到顶点”信息传递的方法。
    在该方法中，输入的参数包括超边特征矩阵 X，信息聚合的方式 aggr（可以是“mean”、“sum”或“softmax_then_sum”），
    连接权重 e2v_weight（默认情况下使用超图构建中指定的权重），以及 dropout 率 drop_rate。
    方法中的第一步是对超边特征矩阵进行聚合操作，第二步是对聚合后的结果进行信息更新。最后将更新后的结果返回。
    '''
    def e2v_of_group(
        self,
        group_name: str,
        X: torch.Tensor,
        aggr: str = "mean",
        e2v_weight: Optional[torch.Tensor] = None,
        drop_rate: float = 0.0,
    ):
        r"""Message passing of ``hyperedges to vertices`` in specified hyperedge group. The combination of ``e2v_aggregation_of_group`` and ``e2v_update_of_group``.

        Args:
            ``group_name`` (``str``): The specified hyperedge group.
            ``X`` (``torch.Tensor``): Hyperedge feature matrix. Size :math:`(|\mathcal{E}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``e2v_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (hyperedges point to vertices). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        X = self.e2v_aggregation_of_group(group_name, X, aggr, e2v_weight, drop_rate=drop_rate)
        X = self.e2v_update_of_group(group_name, X)
        return X
    '''
    这是一个实现了超图到顶点的信息传递过程的方法，其中使用了两个函数：e2v_aggregation 和 e2v_update。
    超图是一种广义的图形，其中连接可以同时连接多个节点，称为超边或超连接。
    超图到顶点的信息传递过程是将超边的特征聚合到连接的顶点上，并对顶点特征进行更新的过程。
    在这个方法中，通过指定group_name可以对特定的超边组进行信息传递。
    函数的参数包括超边特征矩阵X、聚合方法aggr、超边到顶点的权重矩阵e2v_weight和连接随机dropout的概率drop_rate。
    '''
    def v2v(
        self,
        X: torch.Tensor,
        aggr: str = "mean",
        drop_rate: float = 0.0,
        v2e_aggr: Optional[str] = None,
        v2e_weight: Optional[torch.Tensor] = None,
        v2e_drop_rate: Optional[float] = None,
        e_weight: Optional[torch.Tensor] = None,
        e2v_aggr: Optional[str] = None,
        e2v_weight: Optional[torch.Tensor] = None,
        e2v_drop_rate: Optional[float] = None,
    ):
        r"""Message passing of ``vertices to vertices``. The combination of ``v2e`` and ``e2v``.

        Args:
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``. If specified, this ``aggr`` will be used to both ``v2e`` and ``e2v``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
            ``v2e_aggr`` (``str``, optional): The aggregation method for hyperedges to vertices. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``. If specified, it will override the ``aggr`` in ``e2v``.
            ``v2e_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (vertices point to hyepredges). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``v2e_drop_rate`` (``float``, optional): Dropout rate for hyperedges to vertices. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. If specified, it will override the ``drop_rate`` in ``e2v``. Default: ``None``.
            ``e_weight`` (``torch.Tensor``, optional): The hyperedge weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``e2v_aggr`` (``str``, optional): The aggregation method for vertices to hyperedges. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``. If specified, it will override the ``aggr`` in ``v2e``.
            ``e2v_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (hyperedges point to vertices). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``e2v_drop_rate`` (``float``, optional): Dropout rate for vertices to hyperedges. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. If specified, it will override the ``drop_rate`` in ``v2e``. Default: ``None``.
        """
        if v2e_aggr is None:
            v2e_aggr = aggr
        if e2v_aggr is None:
            e2v_aggr = aggr
        if v2e_drop_rate is None:
            v2e_drop_rate = drop_rate
        if e2v_drop_rate is None:
            e2v_drop_rate = drop_rate
        X = self.v2e(X, v2e_aggr, v2e_weight, e_weight, drop_rate=v2e_drop_rate)
        X = self.e2v(X, e2v_aggr, e2v_weight, drop_rate=e2v_drop_rate)
        return X
    '''
    这是一个用于超图上消息传递的方法。它将 v2e 和 e2v 结合起来，完成了从顶点到顶点的消息传递。
    输入参数包括：
    X: 顶点特征矩阵，大小为 :math:(|\mathcal{V}|, C)。
    aggr: 聚合方法，可以是 'mean'，'sum' 和 'softmax_then_sum'。如果指定，该 aggr 将用于 v2e 和 e2v。
    drop_rate: Dropout 率。随机将与接合矩阵中的连接在概率 drop_rate 下丢弃。默认值为 0.0。
    v2e_aggr: 从超边到顶点的聚合方法。可以是 'mean'，'sum' 和 'softmax_then_sum'。如果指定，它将覆盖 e2v 中的 aggr。
    v2e_weight: 连接（从顶点指向超边）附加的权重向量。如果未指定，函数将使用超图构造中指定的权重。默认值为 None。
    v2e_drop_rate: 从超边到顶点的 Dropout 率。随机将与接合矩阵中的连接在概率 drop_rate 下丢弃。如果指定，它将覆盖 e2v 中的 drop_rate。默认值为 None。
    e_weight: 超边权重向量。如果未指定，函数将使用超图构造中指定的权重。默认值为 None。
    e2v_aggr: 从顶点到超边的聚合方法。可以是 'mean'，'sum' 和 'softmax_then_sum'。如果指定，它将覆盖 v2e 中的 aggr。
    e2v_weight: 连接（从超边指向顶点）附加的权重向量。如果未指定，函数将使用超图构造中指定的权重。默认值为 None。
    e2v_drop_rate: 从顶点到超边的 Dropout 率。随机将与接合矩阵中的连接在概率 drop_rate 下丢弃。如果指定，它将覆盖 v2e 中的 drop_rate。默认值为 None。
    函数首先根据输入的参数，对 v2e_aggr，e2v_aggr，v2e_drop_rate 和 e2v_drop_rate 进行了默认值处理。然后它先调用 v2e 方法，再调用 e2v 方法，最后返回结果。
    '''
    def v2v_of_group(
        self,
        group_name: str,
        X: torch.Tensor,
        aggr: str = "mean",
        drop_rate: float = 0.0,
        v2e_aggr: Optional[str] = None,
        v2e_weight: Optional[torch.Tensor] = None,
        v2e_drop_rate: Optional[float] = None,
        e_weight: Optional[torch.Tensor] = None,
        e2v_aggr: Optional[str] = None,
        e2v_weight: Optional[torch.Tensor] = None,
        e2v_drop_rate: Optional[float] = None,
    ):
        r"""Message passing of ``vertices to vertices`` in specified hyperedge group. The combination of ``v2e_of_group`` and ``e2v_of_group``.

        Args:
            ``group_name`` (``str``): The specified hyperedge group.
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``. If specified, this ``aggr`` will be used to both ``v2e_of_group`` and ``e2v_of_group``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
            ``v2e_aggr`` (``str``, optional): The aggregation method for hyperedges to vertices. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``. If specified, it will override the ``aggr`` in ``e2v_of_group``.
            ``v2e_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (vertices point to hyepredges). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``v2e_drop_rate`` (``float``, optional): Dropout rate for hyperedges to vertices. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. If specified, it will override the ``drop_rate`` in ``e2v_of_group``. Default: ``None``.
            ``e_weight`` (``torch.Tensor``, optional): The hyperedge weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``e2v_aggr`` (``str``, optional): The aggregation method for vertices to hyperedges. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``. If specified, it will override the ``aggr`` in ``v2e_of_group``.
            ``e2v_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (hyperedges point to vertices). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``e2v_drop_rate`` (``float``, optional): Dropout rate for vertices to hyperedges. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. If specified, it will override the ``drop_rate`` in ``v2e_of_group``. Default: ``None``.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if v2e_aggr is None:
            v2e_aggr = aggr
        if e2v_aggr is None:
            e2v_aggr = aggr
        if v2e_drop_rate is None:
            v2e_drop_rate = drop_rate
        if e2v_drop_rate is None:
            e2v_drop_rate = drop_rate
        X = self.v2e_of_group(group_name, X, v2e_aggr, v2e_weight, e_weight, drop_rate=v2e_drop_rate)
        X = self.e2v_of_group(group_name, X, e2v_aggr, e2v_weight, drop_rate=e2v_drop_rate)
        return X
    '''
    这是一个使用 PyTorch 实现的超图神经网络中的一个函数，用于在特定的超边组内执行“点到点”消息传递。该函数结合了“点到边”的消息传递和“边到点”的消息传递。
    函数的参数解释如下：
    group_name：超边组的名称；
    X：顶点特征矩阵。大小为 :math:(|\mathcal{V}|, C)；
    aggr：聚合方法。可以是“mean”、“sum”或“softmax_then_sum”。如果指定，该“aggr”将用于“v2e_of_group”和“e2v_of_group”；
    drop_rate：Dropout 比率。以概率“drop_rate”随机丢弃关联矩阵中的连接。默认为“0.0”；
    v2e_aggr：超边到顶点的聚合方法。可以是“mean”、“sum”或“softmax_then_sum”。如果指定，它将覆盖“e2v_of_group”中的“aggr”；
    v2e_weight：连接（顶点指向超边）的权重向量。如果未指定，函数将使用超图构造中指定的权重。默认为“None”；
    v2e_drop_rate：超边到顶点的 Dropout 比率。以概率“drop_rate”随机丢弃关联矩阵中的连接。如果指定，它将覆盖“e2v_of_group”中的“drop_rate”。默认为“None”；
    e_weight：超边权重向量。如果未指定，函数将使用超图构造中指定的权重。默认为“None”；
    e2v_aggr：顶点到超边的聚合方法。可以是“mean”、“sum”或“softmax_then_sum”。如果指定，它将覆盖“v2e_of_group”中的“aggr”；
    e2v_weight：连接（超边指向顶点）的权重向量。如果未指定，函数将使用超图构造中指定的权重。默认为“None”；
    e2v_drop_rate：顶点到超边的 Dropout 比率。以概率“drop_rate”随机丢弃关联矩阵中的连接。如果指定，它将覆盖“v2e_of_group”中的“drop_rate”。默认为“None”。
    函数首先检查超边组是否存在，然后根据参数调用“v2e_of_group”和“e2v_of_group”函数执行消息传递，并返回结果。
    '''