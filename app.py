"""
Aplicación Principal: La IA que entiende lo que decimos
Aplicación educativa para análisis semántico de texto en tiempo real
"""

import streamlit as st
import time
import json
from pathlib import Path
import os

# os.system('pip install -r requirements.txt')

# Configuración de la página
st.set_page_config(
    page_title="La IA que entiende lo que decimos",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Imports de módulos propios
from config.config import APP_TITLE, APP_DESCRIPTION, MAX_TEXT_LENGTH
from models.emotion_analyzer import EmotionAnalyzer
from models.veracity_analyzer import VeracityAnalyzer
from models.social_value_analyzer import SocialValueAnalyzer
from models.keyword_extractor import KeywordExtractor
from database.db_manager import DatabaseManager
from utils.security import SecurityValidator
from utils.visualizations import Visualizer
from ui.components import UIComponents
from ui.styles import get_custom_css, get_loading_animation

# Aplicar estilos CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Inicializar componentes (con caché para evitar recargas)
@st.cache_resource
def initialize_analyzers():
    """Inicializa todos los analizadores"""
    return {
        'emotion': EmotionAnalyzer(),
        'veracity': VeracityAnalyzer(),
        'social_value': SocialValueAnalyzer(),
        'keywords': KeywordExtractor()
    }

@st.cache_resource
def initialize_database():
    """Inicializa el gestor de base de datos"""
    return DatabaseManager()

# Cargar componentes
analyzers = initialize_analyzers()
db = initialize_database()
validator = SecurityValidator()
visualizer = Visualizer()
ui = UIComponents()

# Inicializar session_state
if 'analyzed_count' not in st.session_state:
    st.session_state.analyzed_count = 0
if 'last_analysis_time' not in st.session_state:
    st.session_state.last_analysis_time = 0
if 'session_id' not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())

# =========================
# HEADER
# =========================
st.title(APP_TITLE)
st.markdown(APP_DESCRIPTION)
st.divider()

# =========================
# SIDEBAR
# =========================
ui.create_sidebar_info()

with st.sidebar:
    st.divider()
    
    # Estadísticas rápidas
    stats = db.get_statistics()
    st.markdown("### 📊 Estadísticas Globales")
    st.metric("Total de frases analizadas", stats['total'])
    st.metric("En esta sesión", st.session_state.analyzed_count)
    
    st.divider()
    
    # Opción para cargar frases de ejemplo
    if st.button("📚 Cargar Frases de Ejemplo"):
        try:
            sample_file = Path("data/sample_phrases.json")
            if sample_file.exists():
                with open(sample_file, 'r', encoding='utf-8') as f:
                    samples = json.load(f)
                st.session_state.sample_phrases = samples
                st.success(f"✅ {len(samples)} frases de ejemplo cargadas")
            else:
                st.warning("No se encontró el archivo de ejemplos")
        except Exception as e:
            st.error(f"Error al cargar ejemplos: {e}")
    
    # Botón para limpiar base de datos (solo para testing)
    with st.expander("⚙️ Opciones Avanzadas"):
        if st.button("🗑️ Limpiar Base de Datos", type="secondary"):
            if st.checkbox("¿Estás seguro?"):
                db.clear_database()
                st.success("Base de datos limpiada")
                st.rerun()

# =========================
# MAIN CONTENT
# =========================

# Crear tabs principales
tab1, tab2, tab3 = st.tabs(["🔍 Analizar Texto", "📊 Estadísticas", "📝 Historial"])

# =========================
# TAB 1: ANALIZAR TEXTO
# =========================
with tab1:
    st.markdown("### ✍️ Escribe tu frase para analizar")

    # --- Manejo de texto inicial desde ejemplo ---
    if 'selected_sample' in st.session_state:
        user_text = st.session_state.selected_sample
        del st.session_state.selected_sample  # Limpiar para evitar múltiples ejecuciones
        auto_analyze = True
    else:
        user_text = ""
        auto_analyze = False

    # --- Área de texto ---
    user_text = st.text_area(
        "Ingresa tu texto aquí:",
        value=user_text,
        height=150,
        max_chars=MAX_TEXT_LENGTH,
        placeholder="Ejemplo: Me siento muy feliz hoy porque aprendí algo nuevo...",
        help=f"Máximo {MAX_TEXT_LENGTH} caracteres"
    )

    # --- Contador de caracteres ---
    if user_text:
        char_count = len(user_text)
        color = "green" if char_count < MAX_TEXT_LENGTH * 0.8 else "orange" if char_count < MAX_TEXT_LENGTH else "red"
        st.markdown(f"<p style='color: {color}; text-align: right;'>Caracteres: {char_count}/{MAX_TEXT_LENGTH}</p>", 
                    unsafe_allow_html=True)

    # --- Botón de análisis manual ---
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button("🚀 Analizar Texto", type="primary", use_container_width=True)

    # --- Ejecutar análisis (manual o automático) ---
    if (analyze_button or auto_analyze) and user_text:
        # Validar texto
        is_valid, error_msg = validator.validate_text(user_text)

        if not is_valid:
            st.error(error_msg)
        else:
            # Rate limiting básico
            current_time = time.time()
            if current_time - st.session_state.last_analysis_time < 1:
                st.warning("⏳ Por favor espera un momento antes de analizar otra frase")
            else:
                # Sanitizar texto
                clean_text = validator.sanitize_text(user_text)

                # Mostrar animación de carga
                with st.spinner("🤖 La IA está analizando tu texto..."):
                    try:
                        # Realizar análisis
                        emotion_result = analyzers['emotion'].analyze(clean_text)
                        veracity_result = analyzers['veracity'].analyze(clean_text)
                        social_result = analyzers['social_value'].analyze(clean_text)
                        keywords = analyzers['keywords'].extract(clean_text)

                        # Actualizar session state
                        st.session_state.last_analysis_time = current_time
                        st.session_state.analyzed_count += 1

                        # Guardar en base de datos
                        db.insert_phrase(
                            text=clean_text,
                            emotion=emotion_result['emotion'],
                            emotion_score=emotion_result['score'],
                            veracity=veracity_result['veracity'],
                            veracity_score=veracity_result['score'],
                            social_value=social_result['social_value'],
                            social_value_score=social_result['score'],
                            keywords=keywords,
                            user_session=st.session_state.session_id
                        )

                        # Mostrar resultados
                        st.success("✅ ¡Análisis completado!")
                        st.divider()

                        # Mostrar resultados en columnas
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.markdown("#### ❤️ Análisis Emocional")
                            ui.display_analysis_result(emotion_result, "emotion")

                        with col2:
                            st.markdown("#### 🧠 Análisis de Veracidad")
                            ui.display_analysis_result(veracity_result, "veracity")

                        with col3:
                            st.markdown("#### 🌟 Valor Social")
                            ui.display_analysis_result(social_result, "social_value")

                        st.divider()

                        # Mostrar keywords
                        ui.display_keywords(keywords)

                        # Expander con detalles técnicos
                        with st.expander("🔬 Ver Análisis Detallado"):
                            st.json({
                                "texto_analizado": clean_text,
                                "emocion": {
                                    "principal": emotion_result['emotion'],
                                    "confianza": f"{emotion_result['score']:.2%}",
                                    "todas_las_emociones": emotion_result['all_scores']
                                },
                                "veracidad": {
                                    "clasificacion": veracity_result['veracity'],
                                    "confianza": f"{veracity_result['score']:.2%}"
                                },
                                "valor_social": {
                                    "clasificacion": social_result['social_value'],
                                    "confianza": f"{social_result['score']:.2%}"
                                },
                                "palabras_clave": keywords
                            })

                    except Exception as e:
                        st.error(f"❌ Error durante el análisis: {e}")
                        st.info("Por favor, intenta con otro texto o recarga la página")

    # --- Frases de ejemplo ---
    if 'sample_phrases' in st.session_state:
        st.divider()
        st.markdown("### 📚 Frases de Ejemplo")
        st.markdown("Haz clic en una frase para analizarla:")

        cols = st.columns(3)
        for idx, sample in enumerate(st.session_state.sample_phrases[:6]):
            with cols[idx % 3]:
                if st.button(sample['text'][:50] + "...", key=f"sample_{idx}"):
                    st.session_state.selected_sample = sample['text']
                    st.rerun()

# =========================
# TAB 2: ESTADÍSTICAS
# =========================
with tab2:
    st.markdown("### 📊 Visualización de Datos Colectivos")
    
    stats = db.get_statistics()
    
    if stats['total'] == 0:
        st.info("👋 Aún no hay datos para mostrar. ¡Analiza algunas frases primero!")
    else:
        # Métricas principales
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ui.create_stats_card("Total de Frases", stats['total'], "📝")
        
        with col2:
            most_common_emotion = max(stats['emotions'].items(), key=lambda x: x[1])[0] if stats['emotions'] else "N/A"
            ui.create_stats_card("Emoción Más Común", most_common_emotion.title(), "❤️")
        
        with col3:
            most_common_veracity = max(stats['veracity'].items(), key=lambda x: x[1])[0] if stats['veracity'] else "N/A"
            ui.create_stats_card("Veracidad Predominante", most_common_veracity.title(), "🧠")
        
        st.divider()
        
        # Gráficos
        col1, col2 = st.columns(2)
        
        with col1:
            if stats['emotions']:
                emotion_chart = visualizer.create_emotion_chart(stats['emotions'])
                if emotion_chart:
                    st.altair_chart(emotion_chart, use_container_width=True)
        
        with col2:
            if stats['veracity']:
                veracity_pie = visualizer.create_veracity_pie(stats['veracity'])
                if veracity_pie:
                    st.plotly_chart(veracity_pie, use_container_width=True)
        
        st.divider()
        
        # Gráfico de valor social
        if stats['social_values']:
            social_chart = visualizer.create_social_value_chart(stats['social_values'])
            if social_chart:
                st.altair_chart(social_chart, use_container_width=True)

# =========================
# TAB 3: HISTORIAL
# =========================
with tab3:
    st.markdown("### 📝 Historial de Análisis")
    
    # Opciones de filtrado
    col1, col2 = st.columns([3, 1])
    with col1:
        search_term = st.text_input("🔍 Buscar en el historial:", placeholder="Escribe una palabra clave...")
    with col2:
        limit = st.selectbox("Mostrar:", [10, 25, 50, 100], index=1)
    
    # Obtener frases
    if search_term:
        phrases = db.search_phrases(search_term, limit)
    else:
        phrases = db.get_all_phrases(limit)
    
    if not phrases:
        st.info("No se encontraron frases en el historial")
    else:
        st.markdown(f"**Mostrando {len(phrases)} frases:**")
        
        # Mostrar frases
        for idx, phrase in enumerate(phrases):
            with st.expander(f"🔹 {phrase['text'][:80]}{'...' if len(phrase['text']) > 80 else ''}"):
                st.markdown(f"**Texto completo:** {phrase['text']}")
                st.caption(f"📅 Fecha: {phrase['timestamp']}")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"**❤️ Emoción:** {phrase['emotion'].title()}")
                    st.progress(phrase['emotion_score'])
                
                with col2:
                    st.markdown(f"**🧠 Veracidad:** {phrase['veracity'].title()}")
                    st.progress(phrase['veracity_score'])
                
                with col3:
                    simplified = phrase['social_value'].replace(' para la sociedad', '')
                    st.markdown(f"**🌟 Valor Social:** {simplified.title()}")
                    st.progress(phrase['social_value_score'])
                
                if phrase['keywords']:
                    st.markdown(f"**🔑 Keywords:** {', '.join(phrase['keywords'])}")

# =========================
# FOOTER
# =========================
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🤖 <strong>La IA que entiende lo que decimos</strong> - Herramienta Educativa</p>
    <p style='font-size: 12px;'>Desarrollada con ❤️ para estudiantes de 8º básico</p>
    <p style='font-size: 12px;'>Powered by Streamlit + Transformers + YAKE</p>
</div>
""", unsafe_allow_html=True)
