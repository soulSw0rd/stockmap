# src/analysis/prediction_engine.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
from math import sqrt
import streamlit as st
from src.models.events import EventManager

class PredictionEngine:
    """Moteur de prédiction intégrant les événements et l'analyse temporelle"""
    
    def __init__(self, event_manager: EventManager):
        self.event_manager = event_manager
        self.knowledge_base = {
            'growth_patterns': {
                'strong_growth': {
                    'description': 'Croissance continue sur les derniers trimestres',
                    'uncertainty_factor': 1.3
                },
                'stable': {
                    'description': 'Ventes stables avec variations saisonnières',
                    'uncertainty_factor': 1.0
                },
                'declining': {
                    'description': 'Tendance à la baisse sur la période',
                    'uncertainty_factor': 1.4
                },
                'volatile': {
                    'description': 'Fortes variations imprévisibles',
                    'uncertainty_factor': 1.6
                }
            },
            'event_impacts': {
                'warehouse': {
                    'local_boost': 1.2,
                    'uncertainty_reduction': 0.8,
                    'ramp_up_period': 30,
                    'max_capacity_factor': 1.5
                },
                'marketing': {
                    'campaign_boost': 1.3,
                    'uncertainty_increase': 1.2,
                    'aftereffect_period': 60,
                    'aftereffect_decay': 0.9
                }
            }
        }

    def analyze_product_pattern(self, historical_data: pd.DataFrame, product_code: str) -> dict:
        """Analyse le comportement historique d'un produit"""
        try:
            product_data = historical_data[historical_data['stock_code'] == product_code].copy()
            
            if len(product_data) < 2:
                return self._get_default_pattern()
            
            # Convertir quantity en float si nécessaire
            product_data['quantity'] = pd.to_numeric(product_data['quantity'], errors='coerce')
            
            # Calculer les métriques de base
            growth_rate = self._calculate_growth_rate(product_data)
            seasonality = self._calculate_seasonality(product_data)
            volatility = self._calculate_volatility(product_data)
            
            # Déterminer le type de pattern
            pattern_type = self._determine_pattern_type(growth_rate, seasonality, volatility)
            
            return {
                'growth_rate': float(growth_rate),
                'seasonality': {k: float(v) for k, v in seasonality.items()},
                'volatility': float(volatility),
                'pattern_type': pattern_type
            }
        except Exception as e:
            st.warning(f"Erreur dans l'analyse du pattern pour {product_code}: {str(e)}")
            return self._get_default_pattern()

    def predict_future_demand(self, historical_data: pd.DataFrame, product_code: str, 
                            months_ahead: int = 120, location: str = None) -> Tuple[pd.DataFrame, float]:
        """Prédit la demande future avec niveau d'incertitude"""
        try:
            # Analyser le pattern du produit
            analysis = self.analyze_product_pattern(historical_data, product_code)
            product_data = historical_data[historical_data['stock_code'] == product_code].copy()
            
            if len(product_data) == 0:
                return pd.DataFrame(), 1.0
            
            # Convertir en numérique si nécessaire
            product_data['quantity'] = pd.to_numeric(product_data['quantity'], errors='coerce')
            
            # Calculer la tendance de base
            base_trend = self._calculate_base_trend(product_data)
            
            # Générer les prédictions
            predictions = []
            current_date = product_data['date'].max()
            
            for i in range(months_ahead):
                next_date = current_date + pd.DateOffset(months=i+1)
                month = next_date.month
                
                # Calcul de la prédiction
                prediction = self._calculate_base_prediction(base_trend, analysis, i, month)
                event_impact = self._calculate_event_impact(next_date, location, prediction)
                uncertainty = self._calculate_prediction_uncertainty(analysis, i, next_date)
                
                predictions.append({
                    'date': next_date,
                    'stock_code': product_code,
                    'predicted_demand': float(prediction * event_impact),
                    'uncertainty': float(uncertainty * 100),
                    'event_impact': float(event_impact)
                })
            
            predictions_df = pd.DataFrame(predictions)
            overall_uncertainty = self._calculate_overall_uncertainty(analysis)
            
            return predictions_df, overall_uncertainty
            
        except Exception as e:
            st.error(f"Erreur dans les prédictions pour {product_code}: {str(e)}")
            return pd.DataFrame(), 1.0

    def _calculate_growth_rate(self, data: pd.DataFrame) -> float:
        """Calcule le taux de croissance moyen"""
        try:
            monthly_sales = data.groupby('date')['quantity'].sum()
            if len(monthly_sales) < 2:
                return 0.0
            
            # Utiliser les valeurs non-nulles uniquement
            monthly_sales = monthly_sales[monthly_sales.notna()]
            if len(monthly_sales) < 2:
                return 0.0
                
            total_growth = (monthly_sales.iloc[-1] / monthly_sales.iloc[0]) - 1
            years = (data['date'].max() - data['date'].min()).days / 365
            
            if years < 1:
                return float(total_growth)
            return float((1 + total_growth) ** (1/years) - 1)
        except:
            return 0.0

    def _calculate_seasonality(self, data: pd.DataFrame) -> dict:
        """Calcule les facteurs saisonniers"""
        try:
            monthly_avg = data.groupby('month')['quantity'].mean()
            overall_avg = monthly_avg.mean()
            
            if pd.isna(overall_avg) or overall_avg == 0:
                return {month: 1.0 for month in range(1, 13)}
            
            return {
                month: float(avg/overall_avg) if pd.notna(avg) and avg > 0 else 1.0 
                for month, avg in monthly_avg.items()
            }
        except:
            return {month: 1.0 for month in range(1, 13)}

    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calcule la volatilité des ventes"""
        try:
            if len(data) < 2:
                return 1.0
            qty_mean = data['quantity'].mean()
            if pd.isna(qty_mean) or qty_mean == 0:
                return 1.0
            return float(data['quantity'].std() / qty_mean)
        except:
            return 1.0

    def _determine_pattern_type(self, growth_rate: float, 
                              seasonality: dict, volatility: float) -> str:
        """Détermine le type de pattern du produit"""
        try:
            if volatility > 0.5:
                return 'volatile'
            if abs(growth_rate) > 0.2:
                return 'strong_growth' if growth_rate > 0 else 'declining'
            seasonal_variation = max(seasonality.values()) - min(seasonality.values())
            if seasonal_variation > 0.5:
                return 'seasonal'
            return 'stable'
        except:
            return 'stable'
    def _calculate_base_trend(self, data: pd.DataFrame) -> float:
        """Calcule la tendance de base des ventes"""
        try:
            monthly_sales = data.groupby('date')['quantity'].sum()
            
            if len(monthly_sales) == 0:
                return 0.0
                
            # Utiliser une moyenne pondérée
            weights = np.exp(np.linspace(-1, 0, len(monthly_sales)))
            weights = weights / weights.sum()
            
            return float(np.average(monthly_sales, weights=weights))
        except:
            return 0.0

    def _calculate_base_prediction(self, base_trend: float, analysis: dict, 
                                 months_ahead: int, month: int) -> float:
        """Calcule la prédiction de base pour un mois donné"""
        growth_factor = (1 + analysis['growth_rate']) ** (months_ahead/12)
        seasonal_factor = analysis['seasonality'].get(month, 1.0)
        return float(base_trend * growth_factor * seasonal_factor)

    def _calculate_event_impact(self, date: datetime, location: str, 
                              base_prediction: float) -> float:
        """Calcule l'impact des événements sur la prédiction"""
        impact = self.event_manager.calculate_impact_factor(date, location)
        
        # Limiter l'impact maximum
        max_impact = self.knowledge_base['event_impacts']['warehouse']['max_capacity_factor']
        min_impact = 0.5  # Limite inférieure à -50%
        
        return float(np.clip(impact, min_impact, max_impact))

    def _calculate_prediction_uncertainty(self, analysis: dict, months_ahead: int, 
                                       date: datetime) -> float:
        """Calcule l'incertitude de la prédiction"""
        try:
            # Incertitude de base basée sur le pattern
            base_uncertainty = self.knowledge_base['growth_patterns'][
                analysis['pattern_type']
            ]['uncertainty_factor']
            
            # Facteur temporel
            time_factor = np.log1p(months_ahead) / 10
            
            # Impact des événements sur l'incertitude
            events = self.event_manager.get_events_for_period(date, date)
            event_uncertainty = 1.0
            for event in events:
                if event.event_type == 'warehouse':
                    event_uncertainty *= self.knowledge_base['event_impacts']['warehouse']['uncertainty_reduction']
                else:
                    event_uncertainty *= self.knowledge_base['event_impacts']['marketing']['uncertainty_increase']
            
            # Calculer l'incertitude finale
            final_uncertainty = base_uncertainty * (1 + time_factor) * event_uncertainty
            
            return float(np.clip(final_uncertainty, 0.05, 0.95))
        except:
            return 0.5

    def _calculate_overall_uncertainty(self, analysis: dict) -> float:
        """Calcule l'incertitude globale des prédictions"""
        try:
            pattern_uncertainty = {
                'stable': 0.2,
                'seasonal': 0.3,
                'strong_growth': 0.4,
                'declining': 0.4,
                'volatile': 0.6
            }
            
            base_uncertainty = pattern_uncertainty.get(analysis['pattern_type'], 0.5)
            volatility_factor = analysis['volatility'] * 0.3
            
            return float(min(0.95, base_uncertainty + volatility_factor))
        except:
            return 0.5
    def _get_default_pattern(self) -> dict:
        """Retourne un pattern par défaut"""
        return {
            'growth_rate': 0.0,
            'seasonality': {m: 1.0 for m in range(1, 13)},
            'volatility': 1.0,
            'pattern_type': 'stable'
        }

    def get_prediction_summary(self, predictions_df: pd.DataFrame) -> Dict[str, float]:
        """Génère un résumé des prédictions"""
        if len(predictions_df) == 0:
            return {
                'moyenne_demande': 0.0,
                'max_demande': 0.0,
                'min_demande': 0.0,
                'incertitude_moyenne': 0.0,
                'impact_evenements_moyen': 1.0
            }
            
        return {
            'moyenne_demande': float(predictions_df['predicted_demand'].mean()),
            'max_demande': float(predictions_df['predicted_demand'].max()),
            'min_demande': float(predictions_df['predicted_demand'].min()),
            'incertitude_moyenne': float(predictions_df['uncertainty'].mean()),
            'impact_evenements_moyen': float(predictions_df['event_impact'].mean())
        }