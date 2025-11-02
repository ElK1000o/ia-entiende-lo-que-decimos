"""
Extractor de palabras clave usando YAKE
"""
import yake
import streamlit as st
from typing import List, Tuple
from config.config import YAKE_CONFIG

class KeywordExtractor:
    """Extrae palabras clave relevantes del texto"""
    
    def __init__(self):
        """Inicializa el extractor YAKE"""
        self.config = YAKE_CONFIG
        self._load_extractor()
    
    @st.cache_resource
    def _load_extractor(_self):
        """
        Carga el extractor YAKE (cached)
        
        Returns:
            Extractor YAKE configurado
        """
        try:
            extractor = yake.KeywordExtractor(
                lan=_self.config["lan"],
                n=_self.config["n"],
                dedupLim=_self.config["dedupLim"],
                top=_self.config["top"]
            )
            return extractor
        except Exception as e:
            st.error(f"Error al cargar YAKE: {e}")
            return None
    
    def extract(self, text: str) -> List[str]:
        """
        Extrae palabras clave del texto
        
        Args:
            text: Texto del cual extraer keywords
            
        Returns:
            Lista de palabras clave ordenadas por relevancia
        """
        extractor = self._load_extractor()
        
        if extractor is None or not text.strip():
            return []
        
        try:
            # Extraer keywords con sus scores
            keywords_with_scores = extractor.extract_keywords(text)
            
            # Retornar solo las palabras (sin scores)
            keywords = [kw[0] for kw in keywords_with_scores]
            
            return keywords
        
        except Exception as e:
            st.warning(f"Error en extracción de keywords: {e}")
            return []
    
    def extract_with_scores(self, text: str) -> List[Tuple[str, float]]:
        """
        Extrae palabras clave con sus scores de relevancia
        
        Args:
            text: Texto del cual extraer keywords
            
        Returns:
            Lista de tuplas (keyword, score)
        """
        extractor = self._load_extractor()
        
        if extractor is None or not text.strip():
            return []
        
        try:
            keywords_with_scores = extractor.extract_keywords(text)
            return keywords_with_scores
        
        except Exception as e:
            st.warning(f"Error en extracción de keywords: {e}")
            return []
