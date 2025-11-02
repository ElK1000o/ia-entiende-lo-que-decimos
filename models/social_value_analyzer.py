
"""
Analizador de valor social usando un modelo de clasificación afinado localmente.
"""
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import streamlit as st
from typing import Dict
import torch

class SocialValueAnalyzer:
    """Analiza el valor social percibido del texto usando un modelo local afinado."""
    
    def __init__(self):
        """Inicializa la ruta al modelo y lo carga."""
        self.model_path = "models/social_value_model"
        self._load_model()
    
    @st.cache_resource
    def _load_model(_self):
        """
        Carga el modelo y tokenizador afinados desde una ruta local
        y crea un pipeline de clasificación de texto.
        
        Returns:
            Pipeline de clasificación de texto.
        """
        try:
            model = AutoModelForSequenceClassification.from_pretrained(_self.model_path)
            tokenizer = AutoTokenizer.from_pretrained(_self.model_path)
            
            classifier = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                device=-1  # CPU (usa 0 para GPU)
            )
            print("Modelo de valor social afinado cargado exitosamente.")
            return classifier
        except Exception as e:
            st.error(f"Error al cargar el modelo de valor social local: {e}")
            return None
    
    def analyze(self, text: str) -> Dict[str, any]:
        """
        Analiza el valor social del texto usando el modelo local.
        
        Args:
            text: Texto a analizar.
            
        Returns:
            Diccionario con el valor social detectado y su score.
        """
        classifier = self._load_model()
        
        if classifier is None:
            return {
                "social_value": "error",
                "score": 0.0,
                "all_scores": {}
            }
        
        try:
            # Realizar clasificación con el pipeline
            results = classifier(text, top_k=None, truncation=True)
            
            # Obtener valor social principal
            main_result = results[0]
            social_value = main_result['label']
            score = main_result['score']
            
            # Crear diccionario con todas las puntuaciones
            all_scores = {res['label']: res['score'] for res in results}
            
            # El modelo devuelve "positivo", "negativo", "neutral". Lo adaptamos al formato anterior.
            social_value_formatted = f"{social_value} para la sociedad"

            return {
                "social_value": social_value_formatted,
                "score": float(score),
                "all_scores": all_scores
            }
        
        except Exception as e:
            st.warning(f"Error en análisis de valor social con modelo local: {e}")
            return {
                "social_value": "neutral para la sociedad",
                "score": 0.0,
                "all_scores": {}
            }
    
    def get_social_value_icon(self, social_value: str) -> str:
        """
        Retorna un icono representativo del valor social.
        
        Args:
            social_value: Valor social.
            
        Returns:
            Icono correspondiente.
        """
        icon_map = {
            "positivo para la sociedad": "🌟",
            "neutral para la sociedad": "⚖️",
            "negativo para la sociedad": "⚠️"
        }
        return icon_map.get(social_value, "❓")

