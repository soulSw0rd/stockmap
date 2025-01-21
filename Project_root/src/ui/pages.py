# src/ui/pages.py

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import plotly.express as px
import plotly.graph_objects as go
from src.analysis.stock_analyzer import StockAnalyzer
from src.analysis.prediction_engine import PredictionEngine
from src.models.events import EventManager, Event
from .components import (
    UIComponent, SalesChart, PredictionChart, 
    StockMetricsTable, EventForm, EventList
)

class StockAnalysisPage(UIComponent):
    """Page principale d'analyse des stocks"""
    
    def __init__(self, analyzer: StockAnalyzer, event_manager: EventManager):
        self.analyzer = analyzer
        self.event_manager = event_manager
        self.prediction_engine = PredictionEngine(event_manager)

    def render(self):
        st.header("Analyse et Recommandations de Stock")
        
        # Panneau latéral pour les filtres et configurations
        with st.sidebar:
            self._render_sidebar()
        
        # Onglets principaux
        tab1, tab2, tab3 = st.tabs([
            "Prédictions et Stock", 
            "Gestion des Événements",
            "Analyse Détaillée"
        ])
        
        with tab1:
            self._render_predictions_tab()
        
        with tab2:
            self._render_events_tab()
            
        with tab3:
            self._render_analysis_tab()

    def _render_sidebar(self):
        """Rendu du panneau latéral"""
        try:
            st.sidebar.header("Configuration")
            
            # Sélection du produit
            analysis_results = self.analyzer.analyze_patterns()
            stock_metrics = analysis_results['stock_metrics']
            
            st.session_state.selected_product = st.sidebar.selectbox(
                "Sélectionner un produit",
                options=stock_metrics['stock_code'].unique(),
                format_func=lambda x: f"{x} - {stock_metrics[stock_metrics['stock_code']==x]['description'].iloc[0]}"
            )
            
            # Configuration des prévisions
            st.sidebar.subheader("Configuration des Prévisions")
            st.session_state.prediction_years = st.sidebar.slider(
                "Années de prévision",
                min_value=1,
                max_value=10,
                value=5
            )
            
            # Paramètres avancés
            with st.sidebar.expander("Paramètres Avancés"):
                st.session_state.service_level = st.slider(
                    "Niveau de Service (%)",
                    min_value=80,
                    max_value=99,
                    value=95
                )
                
                st.session_state.lead_time = st.number_input(
                    "Délai de Réapprovisionnement (jours)",
                    min_value=1,
                    max_value=90,
                    value=7
                )
        except Exception as e:
            st.sidebar.error(f"Erreur de chargement du panneau latéral: {str(e)}")

    def _render_predictions_tab(self):
        """Rendu de l'onglet des prédictions"""
        try:
            if not hasattr(st.session_state, 'selected_product'):
                st.warning("Veuillez sélectionner un produit dans le panneau latéral")
                return
                
            analysis_results = self.analyzer.analyze_patterns()
            monthly_trends = analysis_results['monthly_trends']
            
            # Génération des prédictions
            predictions_df, uncertainty = self.prediction_engine.predict_future_demand(
                monthly_trends,
                st.session_state.selected_product,
                months_ahead=st.session_state.prediction_years * 12
            )
            
            # Métriques principales
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Demande Moyenne Prévue",
                    f"{predictions_df['predicted_demand'].mean():.1f}"
                )
            with col2:
                st.metric(
                    "Incertitude Moyenne",
                    f"{predictions_df['uncertainty'].mean():.1f}%"
                )
            with col3:
                st.metric(
                    "Impact des Événements",
                    f"{(predictions_df['event_impact'].mean()-1)*100:+.1f}%"
                )
            
            # Graphique des prédictions
            st.subheader("Prévisions de Demande")
            fig = self._create_prediction_chart(predictions_df)
            st.plotly_chart(fig, use_container_width=True)
            
            # Téléchargement des données
            if st.button("Télécharger les prédictions"):
                self._download_predictions(predictions_df)

        except Exception as e:
            st.error(f"Erreur dans l'affichage des prédictions: {str(e)}")

    def _render_events_tab(self):
        """Rendu de l'onglet de gestion des événements"""
        try:
            col1, col2 = st.columns([2, 3])
            
            with col1:
                st.subheader("Ajouter un Événement")
                EventForm(self._handle_event_submit).render()
            
            with col2:
                st.subheader("Événements Planifiés")
                EventList(
                    self.event_manager.events,
                    on_delete=self._handle_event_delete
                ).render()

        except Exception as e:
            st.error(f"Erreur dans la gestion des événements: {str(e)}")

    def _render_analysis_tab(self):
        """Rendu de l'onglet d'analyse détaillée"""
        try:
            if not hasattr(st.session_state, 'selected_product'):
                st.warning("Veuillez sélectionner un produit dans le panneau latéral")
                return
                
            analysis_results = self.analyzer.analyze_patterns()
            
            # Métriques de stock
            st.subheader("Métriques de Stock")
            metrics_df = analysis_results['stock_metrics']
            
            # Formater les colonnes numériques
            display_metrics = metrics_df.copy()
            numeric_columns = ['avg_monthly_sales', 'safety_stock', 'reorder_point', 
                             'max_monthly_sales', 'seasonal_factor']
            for col in numeric_columns:
                display_metrics[col] = display_metrics[col].apply(lambda x: f"{float(x):.1f}")
            
            st.dataframe(display_metrics)
            
            # Graphiques d'analyse
            col1, col2 = st.columns(2)
            
            with col1:
                self._render_seasonal_analysis(analysis_results)
            
            with col2:
                self._render_trend_analysis(analysis_results)

        except Exception as e:
            st.error(f"Erreur dans l'affichage de l'analyse: {str(e)}")

    def _create_prediction_chart(self, predictions_df: pd.DataFrame) -> go.Figure:
        """Crée le graphique des prédictions"""
        fig = go.Figure()

        # Ligne principale
        fig.add_trace(go.Scatter(
            x=predictions_df['date'],
            y=predictions_df['predicted_demand'],
            name='Prédiction',
            line=dict(color='rgb(31, 119, 180)')
        ))

        # Zones d'incertitude
        upper_bound = predictions_df['predicted_demand'] * (1 + predictions_df['uncertainty']/100)
        lower_bound = predictions_df['predicted_demand'] * (1 - predictions_df['uncertainty']/100)

        fig.add_trace(go.Scatter(
            x=predictions_df['date'],
            y=upper_bound,
            fill=None,
            mode='lines',
            line_color='rgba(31, 119, 180, 0.1)',
            name='Limite supérieure'
        ))

        fig.add_trace(go.Scatter(
            x=predictions_df['date'],
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

        return fig

    def _render_seasonal_analysis(self, analysis_results: Dict):
        """Affiche l'analyse saisonnière"""
        st.subheader("Analyse Saisonnière")
        seasonal_df = analysis_results['seasonal_factors']
        product_data = seasonal_df[
            seasonal_df['stock_code'] == st.session_state.selected_product
        ]
        
        fig = px.line(
            product_data,
            x='month',
            y='quantity',
            title='Facteurs Saisonniers'
        )
        st.plotly_chart(fig)

    def _render_trend_analysis(self, analysis_results: Dict):
        """Affiche l'analyse des tendances"""
        st.subheader("Tendances Historiques")
        trends_df = analysis_results['monthly_trends']
        product_data = trends_df[
            trends_df['stock_code'] == st.session_state.selected_product
        ]
        
        fig = px.line(
            product_data,
            x='date',
            y='quantity',
            title='Évolution des Ventes'
        )
        st.plotly_chart(fig)

    def _handle_event_submit(self, event_data: Dict):
        """Gestion de la soumission d'un événement"""
        try:
            event = Event(**event_data)
            if self.event_manager.add_event(event):
                st.success("Événement ajouté avec succès!")
                st.experimental_rerun()
            else:
                st.error("Erreur lors de l'ajout de l'événement")
        except Exception as e:
            st.error(f"Erreur: {str(e)}")

    def _handle_event_delete(self):
        """Gestion de la suppression d'événements"""
        try:
            self.event_manager.events.clear()
            st.success("Événements supprimés avec succès!")
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Erreur lors de la suppression: {str(e)}")

    def _download_predictions(self, predictions_df: pd.DataFrame):
        """Prépare et permet le téléchargement des prédictions"""
        try:
            csv = predictions_df.to_csv(index=False)
            st.download_button(
                "Télécharger les prédictions (CSV)",
                csv,
                f"predictions_{st.session_state.selected_product}.csv",
                "text/csv",
                key='download-predictions'
            )
        except Exception as e:
            st.error(f"Erreur lors de la préparation du téléchargement: {str(e)}")

class MapPage(UIComponent):
    """Page de visualisation de la carte des commandes"""
    
    def __init__(self, analyzer: StockAnalyzer):
        self.analyzer = analyzer

    def render(self):
        st.header("Carte des Commandes")
        
        # Sélection de la période
        col1, col2 = st.columns(2)
        with col1:
            selected_year = st.selectbox(
                "Année",
                options=sorted(self.analyzer.df['year'].unique())
            )
        with col2:
            selected_month = st.selectbox(
                "Mois",
                options=sorted(self.analyzer.df['month'].unique())
            )
        
        # Filtrage des données
        mask = (
            (self.analyzer.df['year'] == selected_year) & 
            (self.analyzer.df['month'] == selected_month)
        )
        filtered_df = self.analyzer.df[mask]
        
        # Agrégation par pays
        country_stats = filtered_df.groupby('country').agg({
            'invoice_no': 'count',
            'quantity': 'sum',
            'unit_price': lambda x: (x * filtered_df.loc[x.index, 'quantity']).sum()
        }).reset_index()
        
        # Calcul des pourcentages
        total_orders = country_stats['invoice_no'].sum()
        country_stats['orders_percentage'] = (
            country_stats['invoice_no'] / total_orders * 100
        ).round(2)
        
        # Création de la carte
        fig = px.choropleth(
            country_stats,
            locations='country',
            locationmode='country names',
            color='orders_percentage',
            hover_data={
                'country': True,
                'orders_percentage': ':.2f',
                'invoice_no': ':.0f',
                'quantity': ':.0f',
                'unit_price': ':.2f'
            },
            color_continuous_scale='RdYlBu_r',
            title='Distribution des Commandes par Pays'
        )
        
        fig.update_layout(
            coloraxis_colorbar_title="% des Commandes",
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type='equirectangular'
            ),
            width=1000,
            height=600
        )
        
        st.plotly_chart(fig)
        
        # Tableaux des statistiques
        st.subheader("Statistiques par Pays")
        st.dataframe(country_stats.sort_values('orders_percentage', ascending=False))