# 🤖 La IA que entiende lo que decimos

Aplicación educativa interactiva para análisis semántico de texto en tiempo real, diseñada para estudiantes de 8º básico.

## 📋 Descripción

Esta aplicación utiliza Inteligencia Artificial para analizar frases escritas por usuarios y evaluar tres dimensiones semánticas:

1. **❤️ Emoción predominante**: alegría, tristeza, enojo, miedo, amor, sorpresa, asco, neutral
2. **🧠 Veracidad percibida**: verdadero, falso, dudoso
3. **🌟 Valor social**: positivo, neutral o negativo para la sociedad

Además, extrae palabras clave relevantes y presenta visualizaciones en tiempo real de los análisis acumulados.

## ✨ Características

- ✅ Análisis de texto con modelos zero-shot multilingües (XLM-RoBERTa)
- ✅ Extracción de keywords con YAKE
- ✅ Almacenamiento en base de datos SQLite local
- ✅ Visualizaciones interactivas con Altair y Plotly
- ✅ Interfaz amigable y educativa con Streamlit
- ✅ Validaciones de seguridad contra inyecciones
- ✅ Caché de modelos para mejor rendimiento
- ✅ Historial de análisis con búsqueda

## 🚀 Instalación

### Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- 4GB de RAM mínimo (recomendado 8GB)

### Paso 1: Clonar o descargar el proyecto

```bash
# Si usas Git
git clone <url-del-repositorio>
cd ia-entiende-texto

# O simplemente descomprime el archivo ZIP en una carpeta
```

### Paso 2: Crear entorno virtual

**En Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**En macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Paso 3: Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Nota:** La primera instalación puede tardar varios minutos debido a la descarga de PyTorch y transformers.

### Paso 4: Crear estructura de carpetas

Si descargaste los archivos individualmente, asegúrate de tener esta estructura:

```
ia-entiende-texto/
├── app.py
├── requirements.txt
├── config/
│   └── config.py
├── models/
│   ├── __init__.py
│   ├── emotion_analyzer.py
│   ├── veracity_analyzer.py
│   ├── social_value_analyzer.py
│   └── keyword_extractor.py
├── database/
│   ├── __init__.py
│   └── db_manager.py
├── utils/
│   ├── __init__.py
│   ├── security.py
│   └── visualizations.py
├── ui/
│   ├── __init__.py
│   ├── components.py
│   └── styles.py
└── data/
    └── sample_phrases.json
```

Crea archivos `__init__.py` vacíos en las carpetas que los necesiten:

```bash
# En Windows
type nul > models/__init__.py
type nul > database/__init__.py
type nul > utils/__init__.py
type nul > ui/__init__.py

# En macOS/Linux
touch models/__init__.py
touch database/__init__.py
touch utils/__init__.py
touch ui/__init__.py
```

## 🎯 Uso

### Iniciar la aplicación

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

### Primera Ejecución

⚠️ **IMPORTANTE:** En la primera ejecución, la aplicación descargará automáticamente los modelos de Hugging Face (~2GB). Este proceso puede tardar 5-15 minutos dependiendo de tu conexión a internet.

**Durante la descarga verás:**
- Mensajes en la terminal sobre descarga de modelos
- La aplicación puede parecer "congelada" - esto es normal
- Una vez descargados, los modelos se cachean localmente

### Uso de la Aplicación

1. **Analizar Texto:**
   - Escribe una frase en el área de texto
   - Haz clic en "🚀 Analizar Texto"
   - Espera 2-5 segundos mientras la IA procesa
   - Revisa los resultados en las tres dimensiones

2. **Ver Estadísticas:**
   - Cambia a la pestaña "📊 Estadísticas"
   - Explora gráficos de distribución
   - Observa tendencias colectivas

3. **Revisar Historial:**
   - Cambia a la pestaña "📝 Historial"
   - Busca frases específicas
   - Revisa análisis anteriores

## 🛠️ Configuración Avanzada

### Ajustar Parámetros

Edita `config/config.py` para modificar:

- Límite de caracteres por frase
- Modelos de Hugging Face a utilizar
- Categorías de emociones
- Configuración de YAKE
- Colores de visualización

### Usar GPU (opcional)

Si tienes una GPU NVIDIA con CUDA:

1. Instala PyTorch con soporte CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

2. Modifica los archivos de modelos, cambiando:
```python
device=-1  # CPU
```
por:
```python
device=0  # GPU
```

## 🔒 Seguridad

La aplicación implementa múltiples capas de seguridad:

- ✅ Sanitización de entrada con `bleach`
- ✅ Validación contra patrones maliciosos (XSS, SQL injection)
- ✅ Límite de longitud de texto
- ✅ Rate limiting básico
- ✅ Uso de parámetros preparados en SQL

## 🧪 Testing

### Probar con frases de ejemplo

1. En el sidebar, haz clic en "📚 Cargar Frases de Ejemplo"
2. Selecciona una frase de ejemplo para analizar rápidamente

### Limpiar base de datos

1. En el sidebar, abre "⚙️ Opciones Avanzadas"
2. Haz clic en "🗑️ Limpiar Base de Datos"
3. Confirma la acción

## 📊 Tecnologías Utilizadas

- **Streamlit**: Framework de aplicaciones web
- **Transformers (Hugging Face)**: Modelos de NLP
- **XLM-RoBERTa**: Modelo multilingüe zero-shot
- **YAKE**: Extracción de keywords
- **SQLite**: Base de datos local
- **Altair & Plotly**: Visualizaciones interactivas
- **Bleach**: Sanitización de texto

## 🐛 Solución de Problemas

### Error: "ModuleNotFoundError"

```bash
# Asegúrate de tener el entorno virtual activado
# y reinstala las dependencias
pip install -r requirements.txt
```

### Error: "No module named 'config'"

```bash
# Verifica que estés en la carpeta raíz del proyecto
cd ia-entiende-texto
streamlit run app.py
```

### La aplicación es muy lenta

- **Primera ejecución:** Normal, está descargando modelos
- **Ejecuciones posteriores:** Considera usar GPU o reducir el tamaño del modelo en `config.py`

### Error de memoria

Si obtienes errores de memoria:
1. Cierra otras aplicaciones
2. Considera usar un modelo más pequeño
3. Reduce el batch size en la configuración

## 📚 Recursos Educativos

### Para Estudiantes

- Explora diferentes tipos de frases (positivas, negativas, neutras)
- Observa cómo la IA interpreta emociones
- Compara tus percepciones con las de la IA
- Analiza noticias y discursos

### Para Profesores

- Usa la app para enseñar análisis de texto
- Discute sesgos en IA con las predicciones
- Analiza diferentes estilos de escritura
- Genera debates sobre veracidad y fake news

## 🤝 Contribuciones

Este es un proyecto educativo. Si deseas contribuir:

1. Reporta bugs o sugerencias
2. Propón nuevas funcionalidades
3. Mejora la documentación
4. Comparte casos de uso educativos

## 📝 Licencia

Este proyecto es de código abierto para fines educativos.

## 👥 Créditos

- Desarrollado como herramienta educativa
- Modelos: Hugging Face
- Frameworks: Streamlit, PyTorch
- Inspirado en la necesidad de alfabetización digital

## 📧 Contacto

Para preguntas o soporte, consulta la documentación o contacta al desarrollador.

---

**¡Disfruta explorando cómo la IA entiende nuestras palabras!** 🚀
