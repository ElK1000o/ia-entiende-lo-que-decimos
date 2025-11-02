"""
Procesamiento y limpieza de texto
"""
import re
from typing import str

class TextProcessor:
    """Utilidades para procesar y limpiar texto"""
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normaliza el texto para análisis
        
        Args:
            text: Texto a normalizar
            
        Returns:
            Texto normalizado
        """
        if not text:
            return ""
        
        # Normalizar espacios en blanco
        text = ' '.join(text.split())
        
        # Eliminar múltiples signos de puntuación consecutivos
        text = re.sub(r'([.!?])\1+', r'\1', text)
        
        return text.strip()
    
    @staticmethod
    def remove_urls(text: str) -> str:
        """
        Elimina URLs del texto
        
        Args:
            text: Texto con posibles URLs
            
        Returns:
            Texto sin URLs
        """
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.sub(url_pattern, '', text)
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 500) -> str:
        """
        Trunca el texto a una longitud máxima
        
        Args:
            text: Texto a truncar
            max_length: Longitud máxima
            
        Returns:
            Texto truncado
        """
        if len(text) <= max_length:
            return text
        
        # Truncar en el último espacio antes del límite
        truncated = text[:max_length]
        last_space = truncated.rfind(' ')
        
        if last_space > 0:
            truncated = truncated[:last_space]
        
        return truncated + "..."
    
    @staticmethod
    def count_words(text: str) -> int:
        """
        Cuenta las palabras en el texto
        
        Args:
            text: Texto a analizar
            
        Returns:
            Número de palabras
        """
        return len(text.split())
    
    @staticmethod
    def get_text_stats(text: str) -> dict:
        """
        Obtiene estadísticas del texto
        
        Args:
            text: Texto a analizar
            
        Returns:
            Diccionario con estadísticas
        """
        words = text.split()
        
        return {
            'characters': len(text),
            'words': len(words),
            'sentences': text.count('.') + text.count('!') + text.count('?'),
            'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0
        }
