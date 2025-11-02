
"""
Componentes reutilizables de la interfaz
"""
import streamlit as st
from typing import Dict

class UIComponents:
    """Componentes de interfaz reutilizables"""
    
    @staticmethod
    def display_analysis_result(result: Dict, analyzer_type: str):
        """
        Muestra el resultado de un análisis de forma visual y añade un desglose.
        
        Args:
            result: Diccionario con resultado del análisis
            analyzer_type: Tipo de análisis (emotion, veracity, social_value)
        """
        # --- Lógica para mostrar el resultado principal ---
        if analyzer_type == "emotion":
            from models.emotion_analyzer import EmotionAnalyzer
            analyzer = EmotionAnalyzer()
            key_label = result.get('emotion', 'error')
            score = result.get('score', 0.0)
            emoji = analyzer.get_emotion_emoji(key_label)
            st.markdown(f"### {emoji} Emoción: **{key_label.title()}**")
            st.progress(score)
            st.caption(f"Confianza: {score:.1%}")
            
        elif analyzer_type == "veracity":
            from models.veracity_analyzer import VeracityAnalyzer
            analyzer = VeracityAnalyzer()
            key_label = result.get('veracity', 'error')
            score = result.get('score', 0.0)
            icon = analyzer.get_veracity_icon(key_label)
            st.markdown(f"### {icon} Veracidad: **{key_label.title()}**")
            st.progress(score)
            st.caption(f"Confianza: {score:.1%}")
            
        elif analyzer_type == "social_value":
            from models.social_value_analyzer import SocialValueAnalyzer
            analyzer = SocialValueAnalyzer()
            key_label = result.get('social_value', 'error')
            score = result.get('score', 0.0)
            icon = analyzer.get_social_value_icon(key_label)
            value = key_label.replace(' para la sociedad', '')
            st.markdown(f"### {icon} Valor Social: **{value.title()}**")
            st.progress(score)
            st.caption(f"Confianza: {score:.1%}")

        # --- NUEVO: Expander para mostrar el desglose de todas las puntuaciones ---
        all_scores = result.get("all_scores", {})
        if all_scores:
            with st.expander("Ver desglose de predicciones"):
                # Ordenar scores de mayor a menor y tomar los 3 primeros
                sorted_scores = sorted(all_scores.items(), key=lambda item: item[1], reverse=True)[:3]
                
                for label, score_val in sorted_scores:
                    # Formatear el nombre de la etiqueta para que sea legible
                    formatted_label = label.replace(' para la sociedad', '').title()
                    st.text(f"{formatted_label}: {score_val:.2%}")

    
    @staticmethod
    def display_keywords(keywords: list):
        """
        Muestra las palabras clave de forma visual
        
        Args:
            keywords: Lista de palabras clave
        """
        if keywords:
            st.markdown("### 🔑 Palabras Clave")
            # Crear badges para las keywords
            keywords_html = " ".join([
                f'<span style="background-color: #e3f2fd; padding: 5px 10px; '
                f'border-radius: 15px; margin: 3px; display: inline-block;">'
                f'{kw}</span>'
                for kw in keywords
            ])
            st.markdown(keywords_html, unsafe_allow_html=True)
        else:
            st.info("No se encontraron palabras clave relevantes")
    
    @staticmethod
    def create_stats_card(title: str, value: any, icon: str = "📊"):
        """
        Crea una tarjeta de estadística
        
        Args:
            title: Título de la estadística
            value: Valor a mostrar
            icon: Emoji/icono
        """
        st.markdown(f"""
        <div style="
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        ">
            <h2 style="margin: 0; color: #262730;">{icon}</h2>
            <h3 style="margin: 10px 0; color: #262730;">{value}</h3>
            <p style="margin: 0; color: #808495;">{title}</p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def show_info_box(message: str, box_type: str = "info"):
        """
        Muestra un cuadro de información
        
        Args:
            message: Mensaje a mostrar
            box_type: Tipo (info, warning, success, error)
        """
        if box_type == "info":
            st.info(message)
        elif box_type == "warning":
            st.warning(message)
        elif box_type == "success":
            st.success(message)
        elif box_type == "error":
            st.error(message)
    
    @staticmethod
    def create_sidebar_info():
        """Crea información en el sidebar"""
        with st.sidebar:
            st.markdown("## 📚 ¿Cómo funciona?")
            st.markdown("""
            Esta aplicación usa **Inteligencia Artificial** para:
            
            1. 🔍 **Analizar** lo que escribes
            2. 🎭 **Detectar** emociones en tu texto
            3. ✅ **Evaluar** si parece verdadero o falso
            4. 🌍 **Medir** su impacto social
            5. 🔑 **Extraer** las palabras más importantes
            
            ---
            
            ### 💡 Consejos:
            - Escribe frases completas
            - Sé claro en tu mensaje
            - ¡Experimenta con diferentes estilos!
            
            ---
            
            ### 🔒 Privacidad:
            Tus frases se guardan **localmente** 
            para mostrar estadísticas grupales.
            """)
    
    @staticmethod
    def display_recent_phrases(phrases: list, limit: int = 5):
        """
        Muestra las frases recientes analizadas
        
        Args:
            phrases: Lista de frases
            limit: Cantidad a mostrar
        """
        st.markdown("### 📝 Análisis Recientes")
        
        if not phrases:
            st.info("Aún no hay frases analizadas")
            return
        
        for phrase in phrases[:limit]:
            with st.expander(f"💬 {phrase['text'][:50]}..."):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"**Emoción:** {phrase['emotion']}")
                with col2:
                    st.markdown(f"**Veracidad:** {phrase['veracity']}")
                with col3:
                    simplified = phrase['social_value'].replace(' para la sociedad', '')
                    st.markdown(f"**Valor:** {simplified}")
                
                if phrase.get('keywords'):
                    st.markdown(f"**Keywords:** {', '.join(phrase['keywords'])}")
