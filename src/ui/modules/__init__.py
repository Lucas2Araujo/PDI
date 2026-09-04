"""
Pacote src.ui.modules — Módulos didáticos de Processamento Digital de Imagens.
"""

from src.ui.modules.base_module import BasePDIModule
from src.ui.modules.quantize_module import QuantizeModule
from src.ui.modules.grayscale_module import GrayscaleModule
from src.ui.modules.binary_ops_module import BinaryOpsModule, BinaryOpType

__all__ = [
    "BasePDIModule",
    "QuantizeModule",
    "GrayscaleModule",
    "BinaryOpsModule",
    "BinaryOpType",
]

