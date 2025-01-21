# src/data/data_loader.py

from abc import ABC, abstractmethod
import pandas as pd
from typing import List
import streamlit as st
from src.models.data_models import SalesData
import io
import chardet

class DataLoader(ABC):
    @abstractmethod
    def load_data(self, source) -> List[SalesData]:
        pass

class CSVDataLoader(DataLoader):
    def clean_special_chars(self, text: str) -> str:
        """Nettoie les caractères spéciaux tout en préservant ceux qui sont valides"""
        if pd.isna(text):
            return ""
        valid_chars = set(' "\'!?/:.,-+&()_*abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        return ''.join(c if c in valid_chars else ' ' for c in str(text))

    def detect_encoding(self, file_content: bytes) -> str:
        """Détecte l'encodage du fichier"""
        try:
            encodings = ['cp1252', 'latin1', 'utf-8', 'iso-8859-1']
            for encoding in encodings:
                try:
                    file_content.decode(encoding)
                    return encoding
                except UnicodeDecodeError:
                    continue
            
            result = chardet.detect(file_content)
            return result['encoding'] if result['encoding'] else 'cp1252'
        except Exception:
            return 'cp1252'

    def load_data(self, file_path) -> List[SalesData]:
        try:
            # Lecture du contenu brut du fichier
            if hasattr(file_path, 'read'):
                file_content = file_path.getvalue()
            else:
                with open(file_path, 'rb') as f:
                    file_content = f.read()

            # Détection de l'encodage
            encoding = self.detect_encoding(file_content)
            st.write(f"Encodage détecté : {encoding}")

            # Lecture du CSV avec gestion des virgules dans les champs
            df = pd.read_csv(
                io.BytesIO(file_content),
                encoding=encoding,
                sep=',',
                quoting=1,
                quotechar='"',
                escapechar='\\'
            )

            # Debug information
            st.write("Colonnes trouvées:", df.columns.tolist())
            st.write("Nombre de lignes chargées:", len(df))

            # Normaliser les noms de colonnes (tout en minuscules)
            df.columns = df.columns.str.lower()

            # Vérification des colonnes requises
            required_columns = {
                'invoiceno', 'stockcode', 'description', 'quantity',
                'invoicedate', 'unitprice', 'customerid', 'country'
            }
            
            missing_columns = required_columns - set(df.columns)
            if missing_columns:
                raise ValueError(f"Colonnes manquantes: {', '.join(missing_columns)}")

            # Nettoyage des chaînes de caractères
            for col in ['description', 'stockcode', 'country']:
                if col in df.columns:
                    df[col] = df[col].apply(self.clean_special_chars)
            
            # Nettoyage des données numériques
            df = df.dropna(subset=['invoiceno', 'stockcode'])
            df['customerid'] = df['customerid'].fillna('Unknown')
            
            # Conversion des types
            df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
            df['unitprice'] = pd.to_numeric(df['unitprice'], errors='coerce')
            
            # Élimination des lignes avec des valeurs invalides
            df = df[df['quantity'].notna() & df['unitprice'].notna()]
            
            # Gestion des dates
            try:
                df['invoicedate'] = pd.to_datetime(df['invoicedate'], format='%d/%m/%Y %H:%M')
            except Exception:
                try:
                    df['invoicedate'] = pd.to_datetime(df['invoicedate'])
                except Exception as e:
                    st.error(f"Erreur de conversion des dates: {str(e)}")
                    raise

            df['year'] = df['invoicedate'].dt.year
            df['month'] = df['invoicedate'].dt.month

            # Création des objets SalesData
            sales_data_list = []
            for _, row in df.iterrows():
                try:
                    sales_data = SalesData(
                        invoice_no=str(row['invoiceno']),
                        stock_code=str(row['stockcode']),
                        description=str(row['description']),
                        quantity=float(row['quantity']),
                        invoice_date=row['invoicedate'],
                        unit_price=float(row['unitprice']),
                        customer_id=str(row['customerid']),
                        country=str(row['country']),
                        year=int(row['year']),
                        month=int(row['month'])
                    )
                    if sales_data.is_valid():
                        sales_data_list.append(sales_data)
                except Exception as e:
                    st.warning(f"Ligne ignorée: {str(e)}")
                    continue

            if not sales_data_list:
                raise ValueError("Aucune donnée valide n'a été trouvée")

            st.success(f"Données chargées avec succès: {len(sales_data_list)} lignes valides")
            return sales_data_list

        except Exception as e:
            raise ValueError(f"Erreur lors du traitement des données: {str(e)}")

    def validate_file_format(self, file_path) -> bool:
        """Vérifie si le fichier a le bon format"""
        try:
            if hasattr(file_path, 'read'):
                file_content = file_path.getvalue()
            else:
                with open(file_path, 'rb') as f:
                    file_content = f.read()

            encoding = self.detect_encoding(file_content)
            
            df = pd.read_csv(
                io.BytesIO(file_content), 
                encoding=encoding,
                sep=',',
                quoting=1,
                quotechar='"',
                escapechar='\\',
                nrows=1
            )
            
            # Convertir les noms de colonnes en minuscules pour la vérification
            df.columns = df.columns.str.lower()
            
            required_columns = {
                'invoiceno', 'stockcode', 'description', 'quantity',
                'invoicedate', 'unitprice', 'customerid', 'country'
            }
            
            missing_columns = required_columns - set(df.columns)
            if missing_columns:
                st.error(f"Colonnes manquantes: {', '.join(missing_columns)}")
                return False
                
            return True
            
        except Exception as e:
            st.error(f"Erreur de validation: {str(e)}")
            return False