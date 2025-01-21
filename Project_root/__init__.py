"""
Stock Management System - Application de gestion des stocks avec prédictions sous license (en cours de débat)
Version 1.0.0
"""

from src.data.data_loader import CSVDataLoader
from src.models.data_models import SalesData
from src.models.events import Event, EventManager
from src.analysis.stock_analyzer import StockAnalyzer
from src.analysis.prediction_engine import PredictionEngine
from src.ui.pages import StockAnalysisPage, MapPage
from src.ui.components import (
    UIComponent,
    SalesChart,
    PredictionChart,
    StockMetricsTable,
    EventForm,
    EventList
)

__version__ = '1.0.0'
__author__ = 'SoulSw0rd'
__github__ = 'https://github.com/soulSw0rd'
__licence__ = 'might be a gnu 3 like but we don\'t know yet...'

# Exports principaux
__all__ = [
    'CSVDataLoader',
    'SalesData',
    'Event',
    'EventManager',
    'StockAnalyzer',
    'PredictionEngine',
    'StockAnalysisPage',
    'MapPage',
    'UIComponent',
    'SalesChart',
    'PredictionChart',
    'StockMetricsTable',
    'EventForm',
    'EventList'
]

# Configuration de base de l'application
APP_CONFIG = {
    'title': 'Système de Gestion des Stocks',
    'description': 'Application de gestion des stocks avec prédictions',
    'default_service_level': 0.95,  # 95% niveau de service par défaut
    'default_lead_time': 7,  # 7 jours de délai par défaut
    'min_data_points': 3,  # Minimum de points de données pour l'analyse
    'max_prediction_years': 10  # Maximum d'années pour les prédictions
}