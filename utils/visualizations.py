"""
Funciones para crear visualizaciones con Altair y Plotly
"""
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, List
from config.config import EMOTION_COLORS, VERACITY_COLORS, SOCIAL_COLORS

class Visualizer:
    """Crea visualizaciones interactivas de los análisis"""
    
    @staticmethod
    def create_emotion_chart(emotions_data: Dict[str, int]) -> alt.Chart:
        """
        Crea gráfico de barras para distribución de emociones
        
        Args:
            emotions_data: Diccionario {emoción: cantidad}
            
        Returns:
            Gráfico Altair
        """
        if not emotions_data:
            return None
        
        df = pd.DataFrame([
            {"Emoción": k, "Cantidad": v, "Color": EMOTION_COLORS.get(k, "#808080")}
            for k, v in emotions_data.items()
        ])
        
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('Emoción:N', sort='-y', axis=alt.Axis(labelAngle=-45)),
            y=alt.Y('Cantidad:Q', title='Número de frases'),
            color=alt.Color('Color:N', scale=None),
            tooltip=['Emoción', 'Cantidad']
        ).properties(
            title='Distribución de Emociones',
            height=300
        ).configure_axis(
            labelFontSize=12,
            titleFontSize=14
        )
        
        return chart
    
    @staticmethod
    def create_veracity_pie(veracity_data: Dict[str, int]) -> go.Figure:
        """
        Crea gráfico de torta para veracidad
        
        Args:
            veracity_data: Diccionario {veracidad: cantidad}
            
        Returns:
            Figura Plotly
        """
        if not veracity_data:
            return None
        
        labels = list(veracity_data.keys())
        values = list(veracity_data.values())
        colors = [VERACITY_COLORS.get(label, "#808080") for label in labels]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colors),
            hole=0.3
        )])
        
        fig.update_layout(
            title='Distribución de Veracidad',
            height=350
        )
        
        return fig
    
    @staticmethod
    def create_social_value_chart(social_data: Dict[str, int]) -> alt.Chart:
        """
        Crea gráfico horizontal para valor social
        
        Args:
            social_data: Diccionario {valor_social: cantidad}
            
        Returns:
            Gráfico Altair
        """
        if not social_data:
            return None
        
        # Simplificar nombres para mejor visualización
        simplified_names = {
            "positivo para la sociedad": "Positivo",
            "neutral para la sociedad": "Neutral",
            "negativo para la sociedad": "Negativo"
        }
        
        df = pd.DataFrame([
            {
                "Valor Social": simplified_names.get(k, k),
                "Cantidad": v,
                "Color": SOCIAL_COLORS.get(k, "#808080")
            }
            for k, v in social_data.items()
        ])
        
        chart = alt.Chart(df).mark_bar().encode(
            y=alt.Y('Valor Social:N', sort='-x'),
            x=alt.X('Cantidad:Q', title='Número de frases'),
            color=alt.Color('Color:N', scale=None),
            tooltip=['Valor Social', 'Cantidad']
        ).properties(
            title='Distribución de Valor Social',
            height=200
        ).configure_axis(
            labelFontSize=12,
            titleFontSize=14
        )
        
        return chart
    
    @staticmethod
    def create_timeline_chart(phrases: List[Dict]) -> alt.Chart:
        """
        Crea gráfico de línea temporal de análisis
        
        Args:
            phrases: Lista de frases con timestamps
            
        Returns:
            Gráfico Altair
        """
        if not phrases:
            return None
        
        df = pd.DataFrame(phrases)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Contar frases por hora
        df['hour'] = df['timestamp'].dt.floor('H')
        hourly_counts = df.groupby('hour').size().reset_index(name='count')
        
        chart = alt.Chart(hourly_counts).mark_line(
            point=True,
            strokeWidth=3
        ).encode(
            x=alt.X('hour:T', title='Hora'),
            y=alt.Y('count:Q', title='Frases analizadas'),
            tooltip=['hour:T', 'count:Q']
        ).properties(
            title='Análisis en el Tiempo',
            height=250
        )
        
        return chart
    
    @staticmethod
    def create_scores_distribution(all_scores: Dict[str, List[float]], 
                                   category: str) -> alt.Chart:
        """
        Crea histograma de distribución de scores
        
        Args:
            all_scores: Diccionario con scores por categoría
            category: Nombre de la categoría (emotion, veracity, social_value)
            
        Returns:
            Gráfico Altair
        """
        if not all_scores:
            return None
        
        data = []
        for label, scores in all_scores.items():
            for score in scores:
                data.append({"Categoría": label, "Confianza": score})
        
        df = pd.DataFrame(data)
        
        chart = alt.Chart(df).mark_boxplot().encode(
            x=alt.X('Categoría:N', axis=alt.Axis(labelAngle=-45)),
            y=alt.Y('Confianza:Q', scale=alt.Scale(domain=[0, 1])),
            color='Categoría:N'
        ).properties(
            title=f'Distribución de Confianza - {category}',
            height=300
        )
        
        return chart
