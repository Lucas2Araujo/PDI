"""
src.ui.components — Componentes visuais modulares e responsivos da interface PDI.
"""

from src.ui.components.batch_queue import BatchQueue, BatchQueueItem
from src.ui.components.histogram_chart import NativeHistogramChart
from src.ui.components.image_canvas import DisplayMode, ImageCanvas
from src.ui.components.input_slot import InputSlot
from src.ui.components.telemetry_panel import TelemetryPanel

__all__ = [
    "InputSlot",
    "ImageCanvas",
    "DisplayMode",
    "TelemetryPanel",
    "BatchQueue",
    "BatchQueueItem",
    "NativeHistogramChart",
]
