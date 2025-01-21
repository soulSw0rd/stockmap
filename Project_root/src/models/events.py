# src/models/events.py

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List
import pandas as pd

@dataclass
class Event:
    """
    Structure de données pour représenter un événement (entrepôt ou campagne marketing)
    """
    event_type: str  # 'warehouse' ou 'marketing'
    start_date: datetime
    end_date: datetime
    description: str
    impact_factor: float  # ex: 1.2 pour +20% d'impact
    location: Optional[str] = None
    radius_km: Optional[float] = None
    capacity: Optional[int] = None

    def is_active(self, date: datetime) -> bool:
        """Vérifie si l'événement est actif à une date donnée"""
        return self.start_date <= date <= self.end_date

    def get_impact_description(self) -> str:
        """Retourne une description de l'impact de l'événement"""
        impact_percent = (self.impact_factor - 1) * 100
        return f"{impact_percent:+.1f}% d'impact sur les ventes"

class EventManager:
    """
    Gestionnaire d'événements pour suivre et analyser l'impact des entrepôts et campagnes
    """
    def __init__(self):
        self.events: List[Event] = []

    def add_event(self, event: Event) -> bool:
        """Ajoute un nouvel événement"""
        if self._validate_event(event):
            self.events.append(event)
            return True
        return False

    def remove_event(self, event: Event) -> bool:
        """Supprime un événement"""
        if event in self.events:
            self.events.remove(event)
            return True
        return False

    def get_events_for_period(self, start_date: datetime, end_date: datetime) -> List[Event]:
        """Récupère tous les événements actifs sur une période donnée"""
        return [
            event for event in self.events
            if not (event.end_date < start_date or event.start_date > end_date)
        ]

    def calculate_impact_factor(self, date: datetime, location: Optional[str] = None) -> float:
        """Calcule l'impact cumulé de tous les événements actifs à une date donnée"""
        total_impact = 1.0
        active_events = [e for e in self.events if e.is_active(date)]
        
        for event in active_events:
            if event.event_type == 'warehouse':
                if location and event.location:
                    if self._is_location_in_radius(location, event.location, event.radius_km):
                        total_impact *= event.impact_factor
            else:  # marketing
                total_impact *= event.impact_factor
        
        return total_impact

    def get_active_events_summary(self) -> pd.DataFrame:
        """Retourne un résumé des événements actifs"""
        events_data = [{
            'Type': e.event_type,
            'Début': e.start_date.strftime('%Y-%m-%d'),
            'Fin': e.end_date.strftime('%Y-%m-%d'),
            'Description': e.description,
            'Impact': e.get_impact_description(),
            'Location': e.location if e.location else 'N/A'
        } for e in self.events]
        
        return pd.DataFrame(events_data)

    def _validate_event(self, event: Event) -> bool:
        """Valide les données d'un événement"""
        if event.start_date > event.end_date:
            return False
        if event.impact_factor <= 0:
            return False
        if event.event_type not in ['warehouse', 'marketing']:
            return False
        if event.event_type == 'warehouse' and not event.location:
            return False
        return True

    def _is_location_in_radius(self, loc1: str, loc2: str, radius_km: float) -> bool:
        """
        Vérifie si deux locations sont dans un rayon donné
        Note: Cette implémentation est simplifiée et devrait être améliorée
        avec un vrai calcul de distance géographique
        """
        # Simplification - à remplacer par un vrai calcul de distance
        return loc1.lower() == loc2.lower()