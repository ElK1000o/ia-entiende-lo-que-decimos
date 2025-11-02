"""
Configuración general de la aplicación
"""
import os
from pathlib import Path

# Rutas del proyecto
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "database"

# Base de datos
DB_PATH = DB_DIR / "phrases.db"

# Modelos de Hugging Face
MODELS = {
    "emotion": "joeddav/xlm-roberta-large-xnli",  # Zero-shot multilingüe
    "base": "xlm-roberta-base"  # Modelo base para embeddings
}

# Categorías de análisis
EMOTIONS = [
    "alegría", "tristeza", "enojo", "miedo", "amor", 
    "sorpresa", "asco", "neutral"
]

VERACITY_LABELS = [
    "verdadero", "falso", "dudoso"
]

SOCIAL_VALUES = [
    "positivo para la sociedad", 
    "neutral para la sociedad", 
    "negativo para la sociedad"
]

# Configuración de YAKE
YAKE_CONFIG = {
    "lan": "es",
    "n": 2,  # Tamaño máximo de n-gramas
    "dedupLim": 0.7,
    "top": 5  # Top 5 keywords
}

# Límites de seguridad
MAX_TEXT_LENGTH = 500  # Máximo de caracteres por frase
MAX_BATCH_SIZE = 100   # Máximo de frases a procesar simultáneamente
RATE_LIMIT_SECONDS = 1  # Tiempo mínimo entre análisis

# Configuración de UI
APP_TITLE = "🤖 La IA que entiende lo que decimos"
APP_DESCRIPTION = """
Bienvenido a esta herramienta educativa que te permite explorar cómo 
la Inteligencia Artificial puede analizar el significado de nuestras palabras.
"""

# Estilos de emociones (para visualización)
EMOTION_COLORS = {
    "alegría": "#FFD700",
    "tristeza": "#4169E1",
    "enojo": "#DC143C",
    "miedo": "#9370DB",
    "amor": "#FF69B4",
    "sorpresa": "#FF8C00",
    "asco": "#8B4513",
    "neutral": "#808080"
}

VERACITY_COLORS = {
    "verdadero": "#2ECC71",
    "falso": "#E74C3C",
    "dudoso": "#F39C12"
}

SOCIAL_COLORS = {
    "positivo para la sociedad": "#27AE60",
    "neutral para la sociedad": "#95A5A6",
    "negativo para la sociedad": "#C0392B"
}
