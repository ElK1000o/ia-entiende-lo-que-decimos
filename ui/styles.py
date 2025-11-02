"""
Estilos CSS personalizados para la aplicación
"""

def get_custom_css() -> str:
    """
    Retorna el CSS personalizado para la aplicación
    
    Returns:
        String con código CSS
    """
    return """
    <style>
    /* Estilo principal */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Título principal */
    h1 {
        color: #1f77b4;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        text-align: center;
        padding: 20px 0;
    }
    
    /* Tarjetas de análisis */
    .analysis-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    
    /* Botón de análisis personalizado */
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 10px;
        padding: 15px 32px;
        border: none;
        transition: all 0.3s;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    
    /* Input de texto */
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        font-size: 16px;
        padding: 10px;
    }
    
    .stTextArea textarea:focus {
        border-color: #4CAF50;
        box-shadow: 0 0 5px rgba(76, 175, 80, 0.3);
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #f0f2f6;
        border-radius: 8px;
        font-weight: bold;
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background-color: #4CAF50;
    }
    
    /* Métricas */
    [data-testid="stMetricValue"] {
        font-size: 2em;
        font-weight: bold;
        color: #1f77b4;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #e3f2fd;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: bold;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
    
    /* Alertas personalizadas */
    .element-container div[data-testid="stAlert"] {
        border-radius: 10px;
        border-left: 5px solid;
    }
    
    /* Footer */
    footer {
        text-align: center;
        padding: 20px;
        color: #666;
        font-size: 14px;
    }
    
    /* Animación para los resultados */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .analysis-result {
        animation: fadeIn 0.5s ease-in;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .stButton > button {
            font-size: 16px;
            padding: 12px 24px;
        }
        
        h1 {
            font-size: 24px;
        }
    }
    </style>
    """

def get_loading_animation() -> str:
    """
    Retorna HTML/CSS para animación de carga
    
    Returns:
        String con HTML de animación
    """
    return """
    <div style="text-align: center; padding: 20px;">
        <div class="loading-spinner" style="
            border: 4px solid #f3f3f3;
            border-top: 4px solid #4CAF50;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        "></div>
        <p style="margin-top: 15px; color: #666;">Analizando tu texto...</p>
    </div>
    <style>
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
    """
