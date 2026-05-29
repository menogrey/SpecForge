try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    import torch
    torch.cuda.get_device_capability = lambda *args, **kwargs: (9, 0)
except ImportError:
    pass

from .core import *  # noqa
from .modeling import *  # noqa

__all__ = ["modeling", "core"]
