
"""
Analizador de veracidad percibida usando un modelo de clasificación afinado localmente.
"""
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import streamlit as st
from typing import Dict
import torch

class VeracityAnalyzer:
    """Analiza la veracidad percibida del texto usando un modelo local afinado."""
    
    def __init__(self):
        """Inicializa la ruta al modelo y lo carga."""
        self.model_path = "models/veracity_model"
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
            print("Modelo de veracidad afinado cargado exitosamente.")
            return classifier
        except Exception as e:
            st.error(f"Error al cargar el modelo de veracidad local: {e}")
            return None
    
    def analyze(self, text: str) -> Dict[str, any]:
        """
        Analiza la veracidad percibida del texto usando el modelo local.
        
        Args:
            text: Texto a analizar.
            
        Returns:
            Diccionario con la veracidad detectada y su score.
        """
        classifier = self._load_model()
        
        if classifier is None:
            return {
                "veracity": "error",
                "score": 0.0,
                "all_scores": {}
            }
        
        try:
            # Realizar clasificación con el pipeline
            results = classifier(text, top_k=None, truncation=True)
            
            # Obtener veracidad principal
            main_result = results[0]
            veracity = main_result['label']
            score = main_result['score']
            
            # Crear diccionario con todas las puntuaciones
            all_scores = {res['label']: res['score'] for res in results}
            
            return {
                "veracity": veracity,
                "score": float(score),
                "all_scores": all_scores
            }
        
        except Exception as e:
            st.warning(f"Error en análisis de veracidad con modelo local: {e}")
            return {
                "veracity": "dudoso",
                "score": 0.0,
                "all_scores": {}
            }
    
    def get_veracity_icon(self, veracity: str) -> str:
        """
        Retorna un icono representativo de la veracidad.
        
        Args:
            veracity: Nivel de veracidad.
            
        Returns:
            Icono correspondiente.
        """
        icon_map = {
            "verdadero": "✅",
            "falso": "❌",
            "dudoso": "⚠️"
        }
        return icon_map.get(veracity, "❓")

