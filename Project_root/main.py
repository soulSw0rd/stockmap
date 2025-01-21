# main.py

import streamlit as st
from src.data.data_loader import CSVDataLoader
from src.models.events import EventManager
from src.analysis.stock_analyzer import StockAnalyzer
from src.ui.pages import StockAnalysisPage, MapPage
import pandas as pd

class StockManagementApp:
    """Application principale de gestion des stocks"""
    
    def __init__(self):
        self.data_loader = CSVDataLoader()
        self.event_manager = EventManager()
        
        # Initialisation de l'état de session Streamlit
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False

    def initialize_session(self):
        """Initialise les variables de session"""
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "Analyse des Stocks"

    def render_navigation(self):
        """Affiche la navigation principale"""
        st.sidebar.title("Navigation")
        st.session_state.current_page = st.sidebar.radio(
            "Aller à",
            ["Analyse des Stocks", "Carte des Commandes"]
        )

    def load_data(self, file):
        """Charge les données depuis un fichier"""
        try:
            if not self.data_loader.validate_file_format(file):
                st.error("Format de fichier invalide. Veuillez vérifier les colonnes requises.")
                return False
            
            sales_data = self.data_loader.load_data(file)
            st.session_state.analyzer = StockAnalyzer(sales_data)
            st.session_state.data_loaded = True
            return True
        except Exception as e:
            st.error(f"Erreur lors du chargement des données: {str(e)}")
            return False

    def render_upload_section(self):
        """Affiche la section de chargement de fichier"""
        st.write("### Charger les données")
        uploaded_file = st.file_uploader(
            "Choisir un fichier CSV",
            type=['csv'],
            help="Le fichier doit contenir les colonnes: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country"
        )
        
        if uploaded_file is not None:
            if self.load_data(uploaded_file):
                st.success("Données chargées avec succès!")
                st.rerun()

    def render_page_content(self):
        """Affiche le contenu de la page en fonction de la sélection"""
        if not st.session_state.data_loaded:
            self.render_upload_section()
            return

        if st.session_state.current_page == "Analyse des Stocks":
            StockAnalysisPage(
                st.session_state.analyzer,
                self.event_manager
            ).render()
        else:
            MapPage(st.session_state.analyzer).render()

    def run(self):
        """Point d'entrée principal de l'application"""
        st.set_page_config(
            page_title="Gestion des Stocks",
            page_icon="📊",
            layout="wide"
        )
        
        self.initialize_session()
        
        # Titre principal
        st.title("Système de Gestion des Stocks")
        
        # Navigation
        self.render_navigation()
        
        # Contenu principal
        self.render_page_content()
        
        # Footer
        st.sidebar.markdown("---")
        st.sidebar.info(
            "Développé avec Streamlit et Python. "
            "Pour plus d'informations, consultez la documentation."
        )

def main():
    """Fonction principale"""
    try:
        app = StockManagementApp()
        app.run()
    except Exception as e:
        st.error(f"Une erreur inattendue s'est produite: {str(e)}")
        if st.button("Réinitialiser l'application"):
            st.session_state.clear()
            st.rerun()

if __name__ == "__main__":
    main()