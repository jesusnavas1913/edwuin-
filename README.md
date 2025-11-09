# 🎓 Simulador ICFES Pro

Aplicación web para generar y practicar preguntas tipo ICFES usando inteligencia artificial.

## 🚀 Características

- Generación de preguntas tipo ICFES con IA (Google Gemini)
- Múltiples niveles de dificultad (Fácil, Medio, Difícil)
- Sistema de evaluación con retroalimentación inmediata
- Estadísticas de desempeño
- Interfaz moderna y responsive

## 📋 Requisitos

- Python 3.8+
- API Key de Google Gemini

## 🔧 Instalación

1. Instala las dependencias:
```bash
pip install -r requirements.txt
```

2. Crea un archivo `.env` con tu API key:
```
GEMINI_API_KEY=tu_api_key_aqui
```

3. Ejecuta la aplicación:
```bash
python app.py
```

4. Abre tu navegador en: `http://localhost:5000`

## 📁 Estructura del Proyecto

- `app.py` - Backend Flask con la API
- `templates/index.html` - Interfaz web
- `static/style.css` - Estilos CSS
- `requirements.txt` - Dependencias Python

## 🎯 Endpoints API

- `POST /api/pregunta` - Genera una pregunta
- `POST /api/preguntas-multiples` - Genera múltiples preguntas
- `POST /api/retroalimentacion` - Obtiene feedback personalizado
- `GET /health` - Estado del servidor

## 📝 Licencia

Este proyecto es de código abierto.

