"""
Gestor de base de datos SQLite
"""
import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
from config.config import DB_PATH, DB_DIR

class DatabaseManager:
    """Administrador de la base de datos de frases analizadas"""
    
    def __init__(self, db_path: Path = DB_PATH):
        """
        Inicializa el gestor de base de datos
        
        Args:
            db_path: Ruta a la base de datos SQLite
        """
        self.db_path = db_path
        # Crear directorio si no existe
        DB_DIR.mkdir(exist_ok=True)
        self._initialize_db()
    
    def _initialize_db(self):
        """Crea las tablas si no existen"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Tabla principal de frases
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS phrases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    emotion TEXT NOT NULL,
                    emotion_score REAL NOT NULL,
                    veracity TEXT NOT NULL,
                    veracity_score REAL NOT NULL,
                    social_value TEXT NOT NULL,
                    social_value_score REAL NOT NULL,
                    keywords TEXT,
                    user_session TEXT
                )
            """)
            
            # Índices para optimizar consultas
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON phrases(timestamp)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_emotion 
                ON phrases(emotion)
            """)
            
            conn.commit()
    
    def insert_phrase(self, 
                     text: str,
                     emotion: str,
                     emotion_score: float,
                     veracity: str,
                     veracity_score: float,
                     social_value: str,
                     social_value_score: float,
                     keywords: List[str],
                     user_session: str = "default") -> int:
        """
        Inserta una nueva frase analizada
        
        Args:
            text: Texto de la frase
            emotion: Emoción detectada
            emotion_score: Confianza de la emoción
            veracity: Veracidad detectada
            veracity_score: Confianza de la veracidad
            social_value: Valor social detectado
            social_value_score: Confianza del valor social
            keywords: Lista de palabras clave
            user_session: ID de sesión del usuario
            
        Returns:
            ID de la frase insertada
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            keywords_json = json.dumps(keywords, ensure_ascii=False)
            
            cursor.execute("""
                INSERT INTO phrases (
                    text, emotion, emotion_score, 
                    veracity, veracity_score,
                    social_value, social_value_score,
                    keywords, user_session
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (text, emotion, emotion_score, 
                  veracity, veracity_score,
                  social_value, social_value_score,
                  keywords_json, user_session))
            
            conn.commit()
            return cursor.lastrowid
    
    def get_all_phrases(self, limit: int = 100) -> List[Dict]:
        """
        Obtiene todas las frases analizadas
        
        Args:
            limit: Número máximo de frases a retornar
            
        Returns:
            Lista de diccionarios con los datos de las frases
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM phrases 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            
            phrases = []
            for row in rows:
                phrase = dict(row)
                phrase['keywords'] = json.loads(phrase['keywords'])
                phrases.append(phrase)
            
            return phrases
    
    def get_statistics(self) -> Dict:
        """
        Obtiene estadísticas generales
        
        Returns:
            Diccionario con estadísticas
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total de frases
            cursor.execute("SELECT COUNT(*) FROM phrases")
            total = cursor.fetchone()[0]
            
            # Distribución de emociones
            cursor.execute("""
                SELECT emotion, COUNT(*) as count 
                FROM phrases 
                GROUP BY emotion
            """)
            emotions = dict(cursor.fetchall())
            
            # Distribución de veracidad
            cursor.execute("""
                SELECT veracity, COUNT(*) as count 
                FROM phrases 
                GROUP BY veracity
            """)
            veracity = dict(cursor.fetchall())
            
            # Distribución de valor social
            cursor.execute("""
                SELECT social_value, COUNT(*) as count 
                FROM phrases 
                GROUP BY social_value
            """)
            social_values = dict(cursor.fetchall())
            
            return {
                "total": total,
                "emotions": emotions,
                "veracity": veracity,
                "social_values": social_values
            }
    
    def clear_database(self):
        """Elimina todas las frases (útil para testing)"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM phrases")
            conn.commit()
    
    def search_phrases(self, keyword: str, limit: int = 50) -> List[Dict]:
        """
        Busca frases que contengan una palabra clave
        
        Args:
            keyword: Palabra a buscar
            limit: Número máximo de resultados
            
        Returns:
            Lista de frases que coinciden
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM phrases 
                WHERE text LIKE ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (f"%{keyword}%", limit))
            
            rows = cursor.fetchall()
            
            phrases = []
            for row in rows:
                phrase = dict(row)
                phrase['keywords'] = json.loads(phrase['keywords'])
                phrases.append(phrase)
            
            return phrases
