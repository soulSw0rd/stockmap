"""
Module contenant les modèles de données de l'application.
"""

from .data_models import SalesData
from .events import Event, EventManager

__all__ = ['SalesData', 'Event', 'EventManager']