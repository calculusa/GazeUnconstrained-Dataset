# gaze360_dataset.py
from unifiedGazeDataset import UnifiedGazeDataset

class Gaze360Dataset(UnifiedGazeDataset):
    """纯别名；如需特定预处理可覆写 __getitem__。"""
    pass
