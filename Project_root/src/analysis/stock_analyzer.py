# src/analysis/stock_analyzer.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
from math import sqrt
import streamlit as st
from src.models.data_models import SalesData

class StockAnalyzer:
    """Classe pour analyser les données de stock et générer des recommandations"""
    
    def __init__(self, sales_data: List[SalesData]):
        self.sales_data = sales_data
        self.df = None

    def to_dataframe(self) -> pd.DataFrame:
        """Convertit les données de vente en DataFrame"""
        if self.df is None:
            self.df = pd.DataFrame([vars(sale) for sale in self.sales_data])
        return self.df

    def analyze_patterns(self) -> Dict[str, pd.DataFrame]:
        """Analyse complète des patterns de stock"""
        try:
            df = self.to_dataframe()
            
            # Calculer les facteurs saisonniers
            seasonal_factors = df.groupby(['stock_code', 'month'])['quantity'].mean()
            yearly_avg = df.groupby('stock_code')['quantity'].mean()
            seasonal_factors = (seasonal_factors / yearly_avg).reset_index()
            
            # Statistiques mensuelles
            monthly_stats = df.groupby(['stock_code', 'description', 'year', 'month']).agg({
                'quantity': ['sum', 'mean', 'std', 'count']
            }).reset_index()
            
            monthly_stats.columns = [
                'stock_code', 'description', 'year', 'month',
                'total_sales', 'avg_sales', 'std_sales', 'transaction_count'
            ]
            
            # Tendances temporelles
            trend = self.calculate_trends(df)
            
            # Métriques de stock
            stock_metrics = self.calculate_stock_metrics(monthly_stats, seasonal_factors)
            
            return {
                'stock_metrics': stock_metrics,
                'seasonal_factors': seasonal_factors,
                'monthly_trends': trend
            }
            
        except Exception as e:
            st.error(f"Erreur lors de l'analyse des patterns: {str(e)}")
            raise

    def calculate_trends(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule les tendances temporelles des ventes"""
        trend = df.groupby(['stock_code', 'year', 'month']).agg({
            'quantity': 'sum'
        }).reset_index()
        
        # Créer la colonne date
        trend['date'] = pd.to_datetime(trend[['year', 'month']].assign(day=1))
        trend = trend.sort_values('date')
        
        # Calculer la moyenne mobile
        trend['moving_avg'] = trend.groupby('stock_code')['quantity'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        
        return trend

    def calculate_stock_metrics(self, monthly_stats: pd.DataFrame, 
                              seasonal_factors: pd.DataFrame) -> pd.DataFrame:
        """Calcule les métriques de stock pour chaque produit"""
        stock_metrics = pd.DataFrame()
        
        for stock_code in self.df['stock_code'].unique():
            try:
                product_data = monthly_stats[monthly_stats['stock_code'] == stock_code]
                
                if len(product_data) >= 3:
                    # Utiliser float au lieu de int pour tous les calculs
                    recent_avg = product_data['avg_sales'].tail(3).mean()
                    recent_std = product_data['std_sales'].tail(3).mean()
                    max_sales = product_data['total_sales'].max()
                    
                    seasonal_factor = seasonal_factors[
                        seasonal_factors['stock_code'] == stock_code
                    ]['quantity'].max()
                    
                    # Calculs de base
                    lead_time = 7  # jours
                    service_level_z = 1.96  # 95% niveau de service
                    
                    daily_avg = recent_avg / 30 if pd.notna(recent_avg) else 0.0
                    daily_std = recent_std / sqrt(30) if pd.notna(recent_std) else 0.0
                    
                    # Garder les résultats en float
                    safety_stock = service_level_z * daily_std * sqrt(lead_time)
                    reorder_point = (daily_avg * lead_time) + safety_stock
                    variability_index = recent_std / recent_avg if recent_avg > 0 else 0.0
                    
                    # Création du dictionnaire de métriques
                    metrics_dict = {
                        'stock_code': [stock_code],
                        'description': [str(product_data['description'].iloc[0])],
                        'avg_monthly_sales': [float(daily_avg * 30)],
                        'safety_stock': [float(safety_stock)],
                        'reorder_point': [float(reorder_point)],
                        'max_monthly_sales': [float(max_sales)],
                        'seasonal_factor': [float(seasonal_factor) if pd.notna(seasonal_factor) else 1.0],
                        'variability_index': [float(variability_index)]
                    }
                    
                    # Vérification des valeurs avant l'ajout
                    if all(pd.notna(val[0]) for val in metrics_dict.values()):
                        stock_metrics = pd.concat([stock_metrics, pd.DataFrame(metrics_dict)])

            except Exception as e:
                st.warning(f"Information incomplète pour le produit {stock_code}")
                continue
        
        if stock_metrics.empty:
            raise ValueError("Aucune métrique de stock n'a pu être calculée")
        
        return stock_metrics.reset_index(drop=True)

    def get_product_details(self, stock_code: str) -> Dict:
        """Récupère les détails d'un produit spécifique"""
        df = self.to_dataframe()
        product_data = df[df['stock_code'] == stock_code]
        
        if product_data.empty:
            return None
            
        return {
            'stock_code': stock_code,
            'description': product_data['description'].iloc[0],
            'total_quantity': float(product_data['quantity'].sum()),
            'average_price': float(product_data['unit_price'].mean()),
            'total_revenue': float((product_data['quantity'] * product_data['unit_price']).sum()),
            'distinct_customers': len(product_data['customer_id'].unique()),
            'countries': list(product_data['country'].unique())
        }

    def get_stock_health(self, stock_code: str) -> Dict:
        """Évalue la santé du stock pour un produit donné"""
        analysis_results = self.analyze_patterns()
        stock_metrics = analysis_results['stock_metrics']
        
        product_metrics = stock_metrics[stock_metrics['stock_code'] == stock_code]
        
        if product_metrics.empty:
            return None
            
        # Calculer les indicateurs de santé
        metrics = product_metrics.iloc[0]
        current_stock = metrics['safety_stock']  # À remplacer par le stock réel si disponible
        
        health_status = {
            'status': 'Bon' if current_stock >= metrics['safety_stock'] else 'Critique',
            'stock_level': float(current_stock),
            'days_until_reorder': float((current_stock - metrics['reorder_point']) / 
                                      (metrics['avg_monthly_sales'] / 30))
            if metrics['avg_monthly_sales'] > 0 else float('inf'),
            'risk_level': 'Élevé' if metrics['variability_index'] > 0.5 else 'Normal'
        }
        
        return health_status