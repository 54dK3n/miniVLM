from typing import List, Tuple

from torch import Tensor

LayerKVCache = Tuple[Tensor, Tensor]
KVCache = List[LayerKVCache]
