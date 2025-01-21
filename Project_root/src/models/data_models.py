# src/models/data_models.py

from dataclasses import dataclass
from datetime import datetime
import pandas as pd
from typing import Optional

@dataclass
class SalesData:
    """Structure de données pour représenter une vente"""
    invoice_no: str
    stock_code: str
    description: str
    quantity: float  # Changé de int à float
    invoice_date: datetime
    unit_price: float
    customer_id: str
    country: str
    year: int
    month: int

    @classmethod
    def from_row(cls, row) -> Optional['SalesData']:
        """
        Crée une instance SalesData à partir d'une ligne de DataFrame
        avec gestion des valeurs manquantes
        """
        try:
            # Conversion et validation des valeurs numériques
            quantity = pd.to_numeric(row['quantity'], errors='coerce')
            unit_price = pd.to_numeric(row['unit_price'], errors='coerce')
            
            # Vérification des valeurs requises
            if pd.isna(quantity) or pd.isna(unit_price):
                return None
                
            # Gestion du CustomerID
            customer_id = str(row['customer_id']) if pd.notna(row['customer_id']) else 'Unknown'
            
            # Vérification et nettoyage des chaînes
            if not row['stock_code'] or not row['description'] or not row['country']:
                return None

            return cls(
                invoice_no=str(row['invoice_no']).strip(),
                stock_code=str(row['stock_code']).strip(),
                description=str(row['description']).strip(),
                quantity=float(quantity),  # Conversion en float
                invoice_date=row['invoice_date'],
                unit_price=float(unit_price),
                customer_id=customer_id,
                country=str(row['country']).strip(),
                year=int(row['year']),
                month=int(row['month'])
            )
        except (ValueError, TypeError, AttributeError) as e:
            return None

    def is_valid(self) -> bool:
        """Vérifie si les données sont valides"""
        try:
            return all([
                # Vérification des valeurs numériques
                isinstance(self.quantity, (int, float)) and self.quantity > 0,
                isinstance(self.unit_price, (int, float)) and self.unit_price >= 0,
                
                # Vérification des chaînes
                bool(str(self.stock_code).strip()),
                bool(str(self.description).strip()),
                bool(str(self.country).strip()),
                
                # Vérification de la date
                isinstance(self.invoice_date, datetime),
                
                # Vérification année/mois
                isinstance(self.year, int) and self.year > 0,
                isinstance(self.month, int) and 1 <= self.month <= 12
            ])
        except:
            return False

    def total_price(self) -> float:
        """Calcule le prix total de la vente"""
        try:
            return float(self.quantity * self.unit_price)
        except:
            return 0.0

    def to_dict(self) -> dict:
        """Convertit l'instance en dictionnaire"""
        return {
            'invoice_no': self.invoice_no,
            'stock_code': self.stock_code,
            'description': self.description,
            'quantity': float(self.quantity),
            'invoice_date': self.invoice_date,
            'unit_price': float(self.unit_price),
            'customer_id': self.customer_id,
            'country': self.country,
            'year': int(self.year),
            'month': int(self.month),
            'total_price': self.total_price()
        }

    @property
    def formatted_price(self) -> str:
        """Retourne le prix formaté"""
        return f"€{self.unit_price:.2f}"

    @property
    def formatted_total(self) -> str:
        """Retourne le total formaté"""
        return f"€{self.total_price():.2f}"

    @property
    def formatted_date(self) -> str:
        """Retourne la date formatée"""
        return self.invoice_date.strftime("%d/%m/%Y %H:%M")