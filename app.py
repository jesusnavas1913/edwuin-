from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from dotenv import load_dotenv
import os
import json
import re
import logging
import time
import base64
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURACIÓN
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuración de API
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
PORT = int(os.getenv("PORT", 5000))

if not API_KEY:
    logger.error("❌ GEMINI_API_KEY no configurada en .env")
    raise ValueError("Se requiere GEMINI_API_KEY en archivo .env")

try:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash-exp")
    logger.info("✅ Gemini API configurada correctamente")
except Exception as e:
    logger.error(f"Error configurando Gemini: {e}")
    raise

# ============================================================
# UTILIDADES
# ============================================================
def extraer_json(texto):
    """Extrae y limpia JSON de la respuesta de IA"""
    texto = texto.strip()

    if texto.startswith('```json'):
        texto = texto[7:].strip()
    elif texto.startswith('```'):
        texto = texto[3:].strip()

    if texto.endswith('```'):
        texto = texto[:-3].strip()

    match = re.search(r'\{.*\}', texto, re.DOTALL)
    if match:
        json_text = match.group(0)
        json_text = re.sub(r',\s*([}\]])', r'\1', json_text)
        return json_text

    return texto

def retry_with_backoff(func, max_retries=3, base_delay=1):
    """Retry function con exponential backoff para rate limits"""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if "429" in str(e) or "Resource exhausted" in str(e):
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Rate limit, reintentando en {delay}s (intento {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"Max retries alcanzado: {e}")
                    raise e
            else:
                raise e

# ============================================================
# ENDPOINTS PRINCIPALES
# ============================================================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analysis')
def analysis():
    return render_template('generado.html')

@app.route('/api/pregunta', methods=['POST'])
def generar_pregunta():
    """Genera una pregunta tipo ICFES individual"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Datos JSON requeridos"}), 400

        tema = data.get("tema", "").strip()
        dificultad = data.get("dificultad", "medio").lower()
        
        if not tema:
            return jsonify({"error": "El tema es requerido"}), 400

        # Configurar nivel de dificultad
        niveles = {
            "facil": "básico, conceptos fundamentales",
            "medio": "intermedio, aplicación de conceptos",
            "dificil": "avanzado, análisis crítico y síntesis"
        }
        nivel_desc = niveles.get(dificultad, niveles["medio"])

        logger.info(f"Generando pregunta: tema='{tema}', dificultad='{dificultad}'")

        prompt = f"""Eres un experto en educación y diseño de evaluaciones ICFES colombianas.
Genera UNA pregunta tipo ICFES de nivel {nivel_desc} sobre el tema: {tema}

FORMATO JSON EXACTO (sin bloques de código markdown):
{{
    "pregunta": "Texto completo de la pregunta",
    "opciones": [
        "A) Primera opción",
        "B) Segunda opción", 
        "C) Tercera opción",
        "D) Cuarta opción"
    ],
    "respuesta_correcta": "A",
    "explicacion": "Explicación detallada de por qué es correcta esta respuesta y por qué las otras son incorrectas"
}}

REGLAS:
1. La pregunta debe ser clara y sin ambigüedades
2. Las 4 opciones deben ser distintas y plausibles
3. Solo una opción es correcta
4. La explicación debe ser pedagógica y constructiva
5. Responde SOLO con el JSON, sin texto adicional"""

        def generate_single():
            return model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=1500
                )
            )

        response = retry_with_backoff(generate_single)

        if not response or not response.text:
            logger.error("Respuesta vacía de Gemini")
            return jsonify({"error": "No se pudo generar la pregunta"}), 500

        # Extraer y parsear JSON
        json_text = extraer_json(response.text)
        data_pregunta = json.loads(json_text)

        # Validar estructura
        required = ["pregunta", "opciones", "respuesta_correcta", "explicacion"]
        if not all(key in data_pregunta for key in required):
            logger.error(f"JSON inválido: faltan campos requeridos")
            return jsonify({"error": "Estructura de respuesta inválida"}), 500

        if len(data_pregunta["opciones"]) != 4:
            logger.error(f"Número incorrecto de opciones: {len(data_pregunta['opciones'])}")
            return jsonify({"error": "Deben ser exactamente 4 opciones"}), 500

        logger.info("✅ Pregunta generada exitosamente")
        return jsonify(data_pregunta), 200

    except json.JSONDecodeError as e:
        logger.error(f"Error decodificando JSON: {e}")
        return jsonify({"error": "Error procesando respuesta de IA"}), 500
    except Exception as e:
        logger.error(f"Error inesperado: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/preguntas-multiples', methods=['POST'])
def generar_preguntas_multiples():
    """Genera múltiples preguntas tipo ICFES"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Datos JSON requeridos"}), 400

        tema = data.get("tema", "").strip()
        cantidad = int(data.get("cantidad", 5))
        dificultad = data.get("dificultad", "medio").lower()

        if not tema:
            return jsonify({"error": "El tema es requerido"}), 400

        if cantidad < 1 or cantidad > 10:
            return jsonify({"error": "La cantidad debe estar entre 1 y 10"}), 400

        niveles = {
            "facil": "básico, conceptos fundamentales",
            "medio": "intermedio, aplicación de conceptos",
            "dificil": "avanzado, análisis crítico y síntesis"
        }
        nivel_desc = niveles.get(dificultad, niveles["medio"])

        logger.info(f"Generando {cantidad} preguntas: tema='{tema}', dificultad='{dificultad}'")

        prompt = f"""Eres un experto en educación y diseño de evaluaciones ICFES colombianas.
Genera EXACTAMENTE {cantidad} preguntas tipo ICFES de nivel {nivel_desc} sobre: {tema}

FORMATO JSON EXACTO (sin bloques de código markdown):
{{
    "preguntas": [
        {{
            "numero": 1,
            "pregunta": "Texto de la pregunta 1",
            "opciones": ["A) ...", "B) ...", "C) ...", "D) ..."],
            "respuesta_correcta": "A",
            "explicacion": "Explicación detallada"
        }},
        {{
            "numero": 2,
            "pregunta": "Texto de la pregunta 2",
            "opciones": ["A) ...", "B) ...", "C) ...", "D) ..."],
            "respuesta_correcta": "B",
            "explicacion": "Explicación detallada"
        }}
    ]
}}

REGLAS CRÍTICAS:
1. Genera EXACTAMENTE {cantidad} preguntas completas
2. Cada pregunta debe tener 4 opciones distintas (A, B, C, D)
3. Las preguntas deben ser variadas y sobre diferentes aspectos del tema
4. Numera las preguntas desde 1 hasta {cantidad}
5. Responde SOLO con el JSON válido, sin texto adicional"""

        def generate_multiple():
            return model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=4000
                )
            )

        response = retry_with_backoff(generate_multiple)

        if not response or not response.text:
            return jsonify({"error": "No se pudo generar las preguntas"}), 500

        json_text = extraer_json(response.text)
        data_preguntas = json.loads(json_text)

        if "preguntas" not in data_preguntas:
            return jsonify({"error": "Formato de respuesta inválido"}), 500

        preguntas = data_preguntas["preguntas"]
        
        if len(preguntas) != cantidad:
            logger.warning(f"Se generaron {len(preguntas)} preguntas en lugar de {cantidad}")

        logger.info(f"✅ {len(preguntas)} preguntas generadas exitosamente")
        return jsonify({
            "preguntas": preguntas,
            "total": len(preguntas),
            "tema": tema,
            "dificultad": dificultad
        }), 200

    except json.JSONDecodeError as e:
        logger.error(f"Error decodificando JSON: {e}")
        return jsonify({"error": "Error procesando respuesta de IA"}), 500
    except Exception as e:
        logger.error(f"Error generando preguntas: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/retroalimentacion', methods=['POST'])
def obtener_retroalimentacion():
    """Obtiene retroalimentación personalizada sobre una respuesta"""
    try:
        data = request.get_json()
        pregunta = data.get("pregunta", "")
        respuesta_usuario = data.get("respuesta_usuario", "")
        respuesta_correcta = data.get("respuesta_correcta", "")

        if not all([pregunta, respuesta_usuario, respuesta_correcta]):
            return jsonify({"error": "Faltan datos requeridos"}), 400

        es_correcta = respuesta_usuario == respuesta_correcta

        prompt = f"""Eres un tutor experto y motivador. Proporciona retroalimentación constructiva.

PREGUNTA: {pregunta}
RESPUESTA DEL ESTUDIANTE: {respuesta_usuario}
RESPUESTA CORRECTA: {respuesta_correcta}
RESULTADO: {'CORRECTA' if es_correcta else 'INCORRECTA'}

Proporciona retroalimentación en formato JSON:
{{
    "mensaje": "Mensaje motivador",
    "analisis": "Análisis del razonamiento del estudiante",
    "explicacion": "Explicación clara de la respuesta correcta",
    "consejos": ["Consejo 1", "Consejo 2", "Consejo 3"]
}}

Sé positivo, constructivo y pedagógico."""

        def generate_feedback():
            return model.generate_content(prompt)

        response = retry_with_backoff(generate_feedback)
        json_text = extraer_json(response.text)
        feedback = json.loads(json_text)

        return jsonify(feedback), 200

    except Exception as e:
        logger.error(f"Error generando retroalimentación: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de salud del servidor"""
    try:
        def test_ai():
            return model.generate_content(
                "Responde solo 'OK'",
                generation_config=genai.types.GenerationConfig(max_output_tokens=10)
            )

        test_response = retry_with_backoff(test_ai)
        ai_ok = "OK" in test_response.text.upper()

        return jsonify({
            "status": "healthy" if ai_ok else "degraded",
            "server": "running",
            "ai_api": "connected" if ai_ok else "error",
            "model": "gemini-2.0-flash-exp"
        })
    except Exception as e:
        return jsonify({
            "status": "degraded",
            "server": "running",
            "ai_api": "error",
            "error": str(e)
        }), 200

@app.route('/analyze-document', methods=['POST'])
def analyze_document():
    """Analiza un documento de examen subido"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No se subió ningún archivo"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No se seleccionó ningún archivo"}), 400

        # Validar tipo de archivo
        allowed_extensions = {'pdf', 'doc', 'docx'}
        file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        if file_ext not in allowed_extensions:
            return jsonify({"error": "Tipo de archivo no permitido. Solo PDF y Word"}), 400

        # Leer contenido del archivo
        text_content = ""
        if file_ext == 'pdf':
            import pdfplumber
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    text_content += page.extract_text() or ""
        elif file_ext in {'doc', 'docx'}:
            from docx import Document
            doc = Document(file)
            for para in doc.paragraphs:
                text_content += para.text + "\n"

        if not text_content.strip():
            return jsonify({"error": "No se pudo extraer texto del archivo"}), 400

        # Prompt para analizar el examen con Gemini
        prompt = f"""Eres un experto en análisis de exámenes ICFES. Analiza el siguiente contenido de un examen subido y extrae:

1. Cada pregunta numerada
2. Las opciones (A, B, C, D)
3. La respuesta elegida por el estudiante (si se indica)
4. La respuesta correcta (si se indica)
5. Genera retroalimentación detallada para cada pregunta

Contenido del examen:
{text_content[:4000]}  # Limitar para token limit

Responde en formato JSON exacto:
{{
    "success": true,
    "metadata": {{
        "total_items": número_de_preguntas,
        "file_type": "{file_ext}",
        "analysis_date": "fecha_actual"
    }},
    "analysis": [
        {{
            "numero": 1,
            "pregunta": "texto_completo_pregunta",
            "opciones": ["A) opción", "B) opción", "C) opción", "D) opción"],
            "respuesta_elegida": "A",
            "respuesta_correcta": "A",
            "retroalimentacion": "Análisis detallado si es correcta o no, explicación",
            "errores_comunes": "Errores típicos en esta pregunta",
            "sugerencias_mejora": "Consejos para mejorar",
            "conceptos_clave": "Conceptos importantes involucrados"
        }}
        // ... más preguntas
    ]
}}

Si no puedes parsear el formato, haz tu mejor esfuerzo para estructurar la información disponible. Responde SOLO con JSON válido."""

        def analyze_exam():
            return model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=4000
                )
            )

        response = retry_with_backoff(analyze_exam)
        json_text = extraer_json(response.text)
        analysis_data = json.loads(json_text)

        if "analysis" not in analysis_data:
            return jsonify({"error": "No se pudo analizar el documento correctamente"}), 500

        logger.info(f"✅ Análisis completado: {len(analysis_data['analysis'])} preguntas procesadas")
        return jsonify(analysis_data), 200

    except ImportError as e:
        logger.error(f"Dependencia faltante: {e}")
        return jsonify({"error": "Dependencias faltantes. Instala pdfplumber y python-docx con: pip install pdfplumber python-docx"}), 500
    except Exception as e:
        logger.error(f"Error analizando documento: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/generate-visual', methods=['POST'])
def generate_visual():
    """Genera un gráfico visual del análisis"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Datos JSON requeridos"}), 400

        analysis_data = data.get("analysis_data", "")
        chart_type = data.get("chart_type", "bar")
        title = data.get("title", "Análisis de Rendimiento")
        xlabel = data.get("xlabel", "Categorías")
        ylabel = data.get("ylabel", "Número de Preguntas")

        # Parsear datos simples del analysis_data
        lines = analysis_data.split('\n')
        total = 0
        correct = 0
        for line in lines:
            if 'Total de preguntas analizadas:' in line:
                total = int(line.split(':')[1].strip())
            elif 'Respuestas correctas:' in line:
                correct = int(line.split(':')[1].split('(')[0].strip())

        incorrect = total - correct
        success_rate = (correct / total * 100) if total > 0 else 0

        # Crear gráfico con matplotlib
        fig, ax = plt.subplots(figsize=(10, 6))
        categories = ['Correctas', 'Incorrectas']
        values = [correct, incorrect]
        colors = ['#10b981', '#ef4444']
        bars = ax.bar(categories, values, color=colors)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)

        # Añadir valores en las barras
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')

        # Guardar como base64
        output = io.BytesIO()
        canvas = FigureCanvas(fig)
        canvas.print_png(output)
        image_base64 = base64.b64encode(output.getvalue()).decode('utf-8')
        plt.close(fig)

        return jsonify({
            "success": True,
            "image": f"data:image/png;base64,{image_base64}",
            "success_rate": f"{success_rate:.1f}%"
        }), 200

    except Exception as e:
        logger.error(f"Error generando visual: {e}")
        return jsonify({"error": str(e)}), 500

# ============================================================
# INICIO
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("🚀 ICFES PRO - BACKEND GENERADOR DE PREGUNTAS")
    print("=" * 60)
    print(f"✅ Gemini API: Configurada")
    print(f"🌐 Servidor: http://localhost:{PORT}")
    print("=" * 60)
    print("\n📋 ENDPOINTS DISPONIBLES:")
    print("   GET  /                    - Página principal (index.html)")
    print("   GET  /analysis            - Página de análisis de documentos")
    print("   POST /api/pregunta        - Generar 1 pregunta")
    print("   POST /api/preguntas-multiples - Generar múltiples preguntas")
    print("   POST /api/retroalimentacion - Obtener feedback personalizado")
    print("   POST /analyze-document    - Analizar documento subido")
    print("   POST /generate-visual     - Generar gráfico")
    print("   GET  /health              - Estado del servidor")
    print("=" * 60)
    
    try:
        app.run(host='0.0.0.0', port=PORT, debug=True)
    except Exception as e:
        logger.error(f"❌ Error iniciando servidor: {e}")
        raise
