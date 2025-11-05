"""
NURBS (Non-Uniform Rational B-Splines) 工具箱
从 MATLAB NURBS Toolbox 转换为 Python

主要功能:
- B-Spline基础函数
- NURBS曲线、曲面和体的构造和求值
- NURBS操作(节点插入、次数提升等)
"""

from .bspline import findspan, basisfun, basisfunder, numbasisfun
from .nrb_core import nrbmak, nrbeval, nrbbasisfun, nrbbasisfunder, nrbnumbasisfun
from .nrb_ops import nrbdegelev, nrbkntins, bspkntins, bspdegelev

__all__ = [
    'findspan', 'basisfun', 'basisfunder', 'numbasisfun',
    'nrbmak', 'nrbeval', 'nrbbasisfun', 'nrbbasisfunder', 'nrbnumbasisfun',
    'nrbdegelev', 'nrbkntins', 'bspkntins', 'bspdegelev'
]

