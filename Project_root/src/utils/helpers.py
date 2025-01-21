"""
Utilitaires généraux pour l'application de gestion des stocks.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import json

def calculate_growth_rate(values: List[float]) -> float:
    """
    Calcule le taux de croissance entre la première et la dernière valeur
    """
    if len(values) < 2:
        return 0.0
    start_value = values[0]
    end_value = values[-1]
    if start_value == 0:
        return float('inf') if end_value > 0 else 0.0
    return (end_value - start_value) / start_value

def moving_average(values: List[float], window: int = 3) -> List[float]:
    """
    Calcule la moyenne mobile sur une fenêtre donnée
    """
    result = []
    for i in range(len(values)):
        start_idx = max(0, i - window + 1)
        window_values = values[start_idx:i + 1]
        result.append(sum(window_values) / len(window_values))
    return result

def format_currency(value: float) -> str:
    """
    Formate un nombre en devise
    """
    return f"€{value:,.2f}"

def calculate_date_range(start_date: datetime, end_date: datetime) -> List[datetime]:
    """
    Génère une liste de dates entre deux dates données
    """
    date_list = []
    current_date = start_date
    while current_date <= end_date:
        date_list.append(current_date)
        current_date += timedelta(days=1)
    return date_list

def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """
    Division sécurisée évitant les divisions par zéro
    """
    try:
        return a / b if b != 0 else default
    except:
        return default

def validate_json(json_string: str) -> bool:
    """
    Vérifie si une chaîne est un JSON valide
    """
    try:
        json.loads(json_string)
        return True
    except:
        return False

def normalize_data(data: List[float], min_val: float = 0, max_val: float = 1) -> List[float]:
    """
    Normalise une liste de valeurs entre min_val et max_val
    """
    if not data:
        return []
    data_min = min(data)
    data_max = max(data)
    if data_max == data_min:
        return [min_val] * len(data)
    return [min_val + (x - data_min) * (max_val - min_val) / (data_max - data_min) for x in data]

def format_large_number(value: Union[int, float]) -> str:
    """
    Formate les grands nombres avec K, M, B
    """
    if value < 1000:
        return str(value)
    elif value < 1000000:
        return f"{value/1000:.1f}K"
    elif value < 1000000000:
        return f"{value/1000000:.1f}M"
    return f"{value/1000000000:.1f}B"

def calculate_confidence_interval(mean: float, std: float, 
                               confidence: float = 0.95) -> Dict[str, float]:
    """
    Calcule l'intervalle de confiance pour une distribution normale
    """
    z_score = {
        0.90: 1.645,
        0.95: 1.96,
        0.99: 2.576
    }.get(confidence, 1.96)
    
    margin = z_score * std
    return {
        'lower': mean - margin,
        'upper': mean + margin,
        'confidence': confidence
    }

def detect_outliers(data: List[float], threshold: float = 1.5) -> List[bool]:
    """
    Détecte les valeurs aberrantes en utilisant la méthode IQR
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    return [x < lower_bound or x > upper_bound for x in data]

def create_date_periods(dates: List[datetime]) -> Dict[str, List[datetime]]:
    """
    Groupe les dates en périodes (jour, semaine, mois, trimestre)
    """
    df = pd.DataFrame({'date': dates})
    return {
        'daily': dates,
        'weekly': df.resample('W', on='date').groups,
        'monthly': df.resample('M', on='date').groups,
        'quarterly': df.resample('Q', on='date').groups
    }