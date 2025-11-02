
"""
Analizador de emociones usando un modelo de clasificación afinado localmente.
"""
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import streamlit as st
from typing import Dict
import torch

class EmotionAnalyzer:
    """Analiza emociones en texto usando un modelo local afinado."""
    
    def __init__(self):
        """Inicializa la ruta al modelo y lo carga."""
        # Ruta a la carpeta donde guardamos el modelo afinado
        self.model_path = "ElK1000o/emotion-model"
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
            print("Modelo de emoción afinado cargado exitosamente.")
            return classifier
        except Exception as e:
            st.error(f"Error al cargar el modelo de emociones local: {e}")
            return None
    
    def analyze(self, text: str) -> Dict[str, any]:
        """
        Analiza la emoción predominante en el texto usando el modelo local.
        
        Args:
            text: Texto a analizar.
            
        Returns:
            Diccionario con la emoción detectada y su score.
        """
        classifier = self._load_model()
        
        if classifier is None:
            return {
                "emotion": "error",
                "score": 0.0,
                "all_scores": {}
            }
        
        try:
            # Realizar clasificación con el pipeline
            # top_k=None y truncation=True para asegurar que procese bien
            results = classifier(text, top_k=None, truncation=True)
            
            # El resultado es una lista de diccionarios, uno por cada etiqueta
            # Ejemplo: [{'label': 'alegría', 'score': 0.9}, {'label': 'tristeza', 'score': 0.05}]
            
            # Obtener emoción principal (la primera de la lista, que tiene el score más alto)
            main_result = results[0]
            emotion = main_result['label']
            score = main_result['score']
            
            # Crear diccionario con todas las puntuaciones
            all_scores = {res['label']: res['score'] for res in results}
            
            return {
                "emotion": emotion,
                "score": float(score),
                "all_scores": all_scores
            }
        
        except Exception as e:
            st.warning(f"Error en análisis de emoción con modelo local: {e}")
            return {
                "emotion": "neutral",
                "score": 0.0,
                "all_scores": {}
            }
    
    def get_emotion_emoji(self, emotion: str) -> str:
        """
        Retorna un emoji representativo de la emoción.
        
        Args:
            emotion: Nombre de la emoción.
            
        Returns:
            Emoji correspondiente.
        """
        emoji_map = {
            "alegría": "😊",
            "tristeza": "😢",
            "enojo": "😠",
            "miedo": "😨",
            "amor": "❤️",
            "sorpresa": "😲",
            "asco": "🤢", # Esta etiqueta no está en nuestro modelo, pero la mantenemos por si acaso
            "neutral": "😐"
        }
        return emoji_map.get(emotion, "🤔")

