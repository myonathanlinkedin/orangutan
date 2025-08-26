"""
ORANGUTAN: Optimal Resource Allocation via Navigational Grouping, 
Utilization, Territorial Adaptation, and Negotiation

A GPU scheduling simulator for modern GPU architectures.
"""

__version__ = "0.1.0"
__author__ = "ORANGUTAN Team"

from . import env
from . import simulator
from . import telemetry
from . import verification
from . import scheduling
from . import baselines

__all__ = [
    "env",
    "simulator",
    "telemetry",
    "verification",
    "scheduling",
    "baselines",
]
