# src/ui/components.py

from abc import ABC, abstractmethod
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Optional, Any, Callable
import pandas as pd
from datetime import datetime

class UIComponent(ABC):
    """Classe de base abstraite pour tous les composants UI"""
    @abstractmethod
    def render(self):
        """Méthode abstraite pour le rendu du composant"""
        pass

class SalesChart(UIComponent):
    """Composant pour afficher les graphiques de ventes"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def render(self):
        try:
            # Convertir les valeurs en numérique si nécessaire
            if not pd.api.types.is_numeric_dtype(self.data['quantity']):
                self.data['quantity'] = pd.to_numeric(self.data['quantity'], errors='coerce')

            fig = px.line(
                self.data,
                x='date',
                y='quantity',
                title='Évolution des Ventes',
                labels={'quantity': 'Quantité', 'date': 'Date'}
            )
            
            fig.update_layout(
                showlegend=True,
                yaxis_title="Quantité vendue",
                xaxis_title="Date",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Erreur lors de l'affichage du graphique des ventes: {str(e)}")

class PredictionChart(UIComponent):
    """Composant pour afficher les prédictions"""
    
    def __init__(self, predictions: pd.DataFrame):
        self.predictions = predictions

    def render(self):
        try:
            fig = go.Figure()

            # Ligne principale des prédictions
            fig.add_trace(go.Scatter(
                x=self.predictions['date'],
                y=self.predictions['predicted_demand'],
                name='Prédiction',
                line=dict(color='rgb(31, 119, 180)')
            ))

            # Zones d'incertitude
            upper_bound = self.predictions['predicted_demand'] * (1 + self.predictions['uncertainty']/100)
            lower_bound = self.predictions['predicted_demand'] * (1 - self.predictions['uncertainty']/100)

            fig.add_trace(go.Scatter(
                x=self.predictions['date'],
                y=upper_bound,
                fill=None,
                mode='lines',
                line_color='rgba(31, 119, 180, 0.1)',
                name='Limite supérieure'
            ))

            fig.add_trace(go.Scatter(
                x=self.predictions['date'],
                y=lower_bound,
                fill='tonexty',
                mode='lines',
                line_color='rgba(31, 119, 180, 0.1)',
                name='Limite inférieure'
            ))

            fig.update_layout(
                title='Prévisions de Demande avec Incertitude',
                xaxis_title='Date',
                yaxis_title='Demande Prévue',
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)
            
            # Afficher les statistiques clés
            self._display_key_metrics()
            
        except Exception as e:
            st.error(f"Erreur lors de l'affichage des prédictions: {str(e)}")

    def _display_key_metrics(self):
        """Affiche les métriques clés des prédictions"""
        try:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Demande moyenne",
                    f"{self.predictions['predicted_demand'].mean():.1f}"
                )
            with col2:
                st.metric(
                    "Incertitude moyenne",
                    f"{self.predictions['uncertainty'].mean():.1f}%"
                )
            with col3:
                st.metric(
                    "Impact événements",
                    f"{(self.predictions['event_impact'].mean()-1)*100:+.1f}%"
                )
        except Exception as e:
            st.warning(f"Impossible d'afficher certaines métriques: {str(e)}")

class StockMetricsTable(UIComponent):
    """Composant pour afficher les métriques de stock"""
    
    def __init__(self, metrics: pd.DataFrame):
        self.metrics = metrics

    def render(self):
        try:
            # Formatter les colonnes numériques
            display_metrics = self.metrics.copy()
            numeric_columns = [
                'avg_monthly_sales', 'safety_stock', 'reorder_point',
                'max_monthly_sales', 'seasonal_factor', 'variability_index'
            ]
            
            for col in numeric_columns:
                if col in display_metrics.columns:
                    display_metrics[col] = pd.to_numeric(
                        display_metrics[col], errors='coerce'
                    ).apply(lambda x: f"{float(x):.1f}" if pd.notna(x) else "-")
            
            # Configuration des colonnes pour l'affichage
            st.dataframe(
                display_metrics,
                column_config={
                    'stock_code': 'Code Produit',
                    'description': 'Description',
                    'avg_monthly_sales': 'Ventes Moyennes',
                    'safety_stock': 'Stock de Sécurité',
                    'reorder_point': 'Point de Réapprovisionnement',
                    'max_monthly_sales': 'Ventes Max',
                    'seasonal_factor': 'Facteur Saisonnier',
                    'variability_index': 'Indice de Variabilité'
                },
                hide_index=True
            )
        except Exception as e:
            st.error(f"Erreur lors de l'affichage des métriques: {str(e)}")

class EventForm(UIComponent):
    """Formulaire pour ajouter et gérer les événements"""
    
    def __init__(self, on_submit: Callable):
        self.on_submit = on_submit

    def render(self):
        try:
            with st.form("event_form"):
                event_type = st.selectbox(
                    "Type d'événement",
                    ['warehouse', 'marketing'],
                    help="Sélectionnez le type d'événement à ajouter"
                )

                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input(
                        "Date de début",
                        help="Date de début de l'événement"
                    )
                with col2:
                    end_date = st.date_input(
                        "Date de fin",
                        help="Date de fin de l'événement"
                    )

                description = st.text_input(
                    "Description",
                    help="Description détaillée de l'événement"
                )

                impact = st.slider(
                    "Impact sur les ventes (%)",
                    min_value=-50,
                    max_value=100,
                    value=0,
                    help="Impact estimé sur les ventes en pourcentage"
                )

                if event_type == 'warehouse':
                    location = st.text_input(
                        "Localisation",
                        help="Emplacement de l'entrepôt"
                    )
                    radius = st.number_input(
                        "Rayon d'impact (km)",
                        min_value=0.0,
                        help="Rayon d'influence de l'entrepôt"
                    )
                    capacity = st.number_input(
                        "Capacité de stockage",
                        min_value=0.0,
                        help="Capacité maximale de stockage"
                    )
                else:
                    location = None
                    radius = None
                    capacity = None

                submitted = st.form_submit_button("Ajouter l'événement")
                if submitted:
                    self.on_submit({
                        'event_type': event_type,
                        'start_date': pd.Timestamp(start_date),
                        'end_date': pd.Timestamp(end_date),
                        'description': description,
                        'impact_factor': 1 + impact/100,
                        'location': location,
                        'radius_km': float(radius) if radius is not None else None,
                        'capacity': float(capacity) if capacity is not None else None
                    })
        except Exception as e:
            st.error(f"Erreur dans le formulaire d'événement: {str(e)}")

class EventList(UIComponent):
    """Liste des événements avec possibilité de filtrage"""
    
    def __init__(self, events: List[Dict], on_delete: Optional[Callable] = None):
        self.events = events
        self.on_delete = on_delete

    def render(self):
        try:
            if not self.events:
                st.info("Aucun événement planifié")
                return

            events_df = pd.DataFrame([{
                'Type': e.event_type,
                'Début': e.start_date.strftime('%Y-%m-%d'),
                'Fin': e.end_date.strftime('%Y-%m-%d'),
                'Description': e.description,
                'Impact': f"{(e.impact_factor-1)*100:+.1f}%",
                'Location': e.location if e.location is not None else 'N/A'
            } for e in self.events])

            st.dataframe(
                events_df,
                column_config={
                    'Type': "Type d'événement",
                    'Début': 'Date de début',
                    'Fin': 'Date de fin',
                    'Description': 'Description',
                    'Impact': 'Impact (%)',
                    'Location': 'Localisation'
                },
                hide_index=True
            )

            if self.on_delete and len(self.events) > 0:
                if st.button("Supprimer les événements sélectionnés"):
                    self.on_delete()
                    
        except Exception as e:
            st.error(f"Erreur dans l'affichage des événements: {str(e)}")