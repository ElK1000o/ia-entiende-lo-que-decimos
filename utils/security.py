"""
Funciones de seguridad y validación
"""
import re
import bleach
from typing import Tuple
from config.config import MAX_TEXT_LENGTH

class SecurityValidator:
    """Validador de seguridad para entradas de usuario"""
    
    # Patrones sospechosos (SQL injection, XSS, etc.)
    SUSPICIOUS_PATTERNS = [
        r"(<script|javascript:|onerror=|onload=)",  # XSS
        r"(union\s+select|drop\s+table|insert\s+into)",  # SQL injection
        r"(\.\.\/|\.\.\\)",  # Path traversal
        r"(<iframe|<embed|<object)",  # Embedding malicioso
    ]
    
    @staticmethod
    def sanitize_text(text: str) -> str:
        """
        Limpia el texto de entrada de caracteres peligrosos
        
        Args:
            text: Texto a sanitizar
            
        Returns:
            Texto limpio y seguro
        """
        if not text:
            return ""
        
        # Eliminar etiquetas HTML
        text = bleach.clean(text, tags=[], strip=True)
        
        # Limitar caracteres especiales excesivos
        text = re.sub(r'[^\w\s\.\,\!\?\-áéíóúñÁÉÍÓÚÑ]', '', text)
        
        # Normalizar espacios
        text = ' '.join(text.split())
        
        return text.strip()
    
    @staticmethod
    def validate_text(text: str) -> Tuple[bool, str]:
        """
        Valida que el texto sea seguro y apropiado
        
        Args:
            text: Texto a validar
            
        Returns:
            Tupla (es_válido, mensaje_error)
        """
        if not text or not text.strip():
            return False, "⚠️ El texto no puede estar vacío"
        
        if len(text) > MAX_TEXT_LENGTH:
            return False, f"⚠️ El texto no puede superar {MAX_TEXT_LENGTH} caracteres"
        
        if len(text.strip()) < 3:
            return False, "⚠️ El texto debe tener al menos 3 caracteres"
        
        # Verificar patrones sospechosos
        text_lower = text.lower()
        for pattern in SecurityValidator.SUSPICIOUS_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return False, "⚠️ El texto contiene caracteres o patrones no permitidos"
        
        # Verificar que no sea solo números o caracteres especiales
        if not re.search(r'[a-zA-ZáéíóúñÁÉÍÓÚÑ]', text):
            return False, "⚠️ El texto debe contener al menos algunas letras"
        
        return True, ""
    
    @staticmethod
    def validate_batch_size(size: int) -> bool:
        """
        Valida que el tamaño del lote sea apropiado
        
        Args:
            size: Tamaño del lote
            
        Returns:
            True si es válido, False en caso contrario
        """
        from config.config import MAX_BATCH_SIZE
        return 0 < size <= MAX_BATCH_SIZE
