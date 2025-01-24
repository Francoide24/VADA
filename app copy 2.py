import streamlit as st
import google.generativeai as genai
import os
import time
import json
import httpx  # Para descargar archivo desde Drive u otra URL si es necesario
from google.oauth2 import service_account
from google.cloud import bigquery
from google.api_core.exceptions import NotFound
from dotenv import load_dotenv
import logging


# Configuración de Logging
logging.basicConfig(level=logging.ERROR, filename="app.log", filemode="w", 
                    format='%(asctime)s - %(levelname)s - %(message)s')

#############################
# 1. CONFIGURACIÓN INICIAL
#############################

# Cargar variables de entorno (si las tienes en .env)
load_dotenv("a.env")

# Credenciales de servicio para BigQuery
service_account_info = {
    "type": "service_account",
    "project_id": "bbrands1",
    "private_key_id": "0dc0cbea9eed291eb2c0004a82d852a110b1aa7e",
    "private_key": """-----BEGIN PRIVATE KEY-----
MIIEvwIBADANBgkqhkiG9w0BAQEFAASCBKkwggSlAgEAAoIBAQDqSqeEXLdnXaOy
apqqRjD54aPM7ZRbU3yZpFu5DXJF+EROcZ5Ypd/0TQPdJOQPou2OpqfAj71clUkg
H9/N8dBsmvuj5GDtv1+xW06rY4vejDbFWJ+kg+dwwdArNg9OaiCcAEDWHdkx2hTg
ftSRvVs9CMYoKaL4bgvwc5U2C1GgigckOfhp7xZoDNN3HDiHJYG/l9yDu9/jDpxD
NF32h/5GGUzkbtL4nuY46IfirMXEXK5qEj/4/P66rxtLI82ykWchrQx6Sm8LGvFG
W4V39p+P5/5AT6W52GLdtYKYbCOmgoi+MbS8NJnybjEecx2c3bFtjuC0Y+bC5wiI
s0s5dU6nAgMBAAECggEAEbR3iUSaZXjJjIuKyZBJVjDLeIqBmg8qjM/DUK4n8wqq
WsUyQp+yV1tUjesiQt48lnmYlrAmDq+HWaKe/oimB/cESiPRgVXjbNsqDECXCsfM
wcgLNFr1a+txiDprGLFjanaIb8XMqnxA9KAQ/zxwfyHBG7rdwmlKhK6vWWisj/WJ
RbsBS92YQieuXKoXbGHlHWPTqQbDSzRZwkiDnBQuHBSHx6cH2Y7X6rJ5W+X8h4+l
/KLq9fvC5gCdo5vtv5W7WBe6JHSyBJjMX4vRUBtW2rTcBmYtxFLQNkePCFL8ODHe
RLVV0ih5EhsQjC9uzOdIRxz80Gl8mqjjGoSvW8o6ZQKBgQD2UTHoB7bFRH+ighWR
pYNlgU9JSVZgYBoZU65xYDMhzsl15CoigDAA+8mLUYSAAUfR4vbMdKbmudaagnJZ
ZSTvwuiznKV1jkbhFUMEWR2C3iW+GIxsx3xcmbC5RoQoIkW8/8N1FLu83tvpX1fe
vf1pkOp6KBP6e8QhMtv9qR/2VQKBgQDzgHDQhqr2GRZTplrGSPkbRbRFVlbnl9tQ
X/aDSaid6Ar3/xA5CJ2MRr3AtyHkc35wHeu8QFnfkaG0BiZ9gLPHHNtw3elJIOEk
g7AsWd8VnD6HoXVrN1ZI8T9Pdt1HucjuqD/1MYyzfKcAHZlE5q1bC29KsRnGlmew
25pDH53VCwKBgQCyM+/9RIdwlKwasC5Wnv4/E5x/EvXQ5/Y52JbeI5EapnaOOjJZ
n5AbRg58Is/PpB2HtcKEDOkrB8xBIJsGHezpIYQlXfE+6V5SPYWswaReJ8X6j9wY
XwKUJAT9Jg061ADMSeXo8MUaUcmcP4Rc++s40sUw94nssFonApqyHAepuQKBgQCW
6HsCQPOjIVkc7nRDfuYjaMeYUTH6xbo9zbtREk1Vz4E8wO6k6hn53b2rudNfadRq
V6DJQnhwfijhEQ65qRHBzLiS8nSpxZ7Cqnp8ghYnpnV6SS4kDF+FRT4fWWM6GIHW
pp88rkCs3AwDnlRmxy+YxTVr7OY0lPIeQXsRLn76kQKBgQCBEI/QwPhEp+sEgpOr
7/h9Em5eX47we8Op9kHm74PKI2pk4+9Pny6XUbjpwU1faqNNQZwxDtEhhAKft4Wg
x8QWN1IN+dWnBVsVoFgHKdnqdFFOX8p2WV6UJzgmEgRX0M/7f2+uhol/Ch+I+8Pt
3EFJh76JzsG8iSoLnPglGbAOiA==
-----END PRIVATE KEY-----""",
    "client_email": "conexionjupyter@bbrands1.iam.gserviceaccount.com",
    "client_id": "108650246531418048114",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/conexionjupyter%40bbrands1.iam.gserviceaccount.com"
}

credentials = service_account.Credentials.from_service_account_info(service_account_info)

bq_client = bigquery.Client(
    credentials=credentials,
    project=credentials.project_id,
    location="US"
)

# Configuramos la API de Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("No se encontró 'GOOGLE_API_KEY' en variables de entorno. Por favor, configúralo.")
    st.stop()
else:
    genai.configure(api_key=GOOGLE_API_KEY)

DEFAULT_MODEL_NAME = "gemini-1.5-flash"

#############################
# 2. DISEÑO ESTILIZADO
#############################
st.set_page_config(page_title="Analizador de Videos", layout="wide")

# Colores inspirados en Maihue
PRIMARY_COLOR = "#0A74DA"  # Azul principal
SECONDARY_COLOR = "#34A853"  # Verde sostenibilidad
BACKGROUND_COLOR = "#F9F9F9"  # Fondo claro
FONT_COLOR = "#333333"  # Texto principal

st.markdown(f"""
    <style>
        .main {{
            background-color: {BACKGROUND_COLOR};
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: {PRIMARY_COLOR};
            font-family: 'Arial', sans-serif;
        }}
        p, div, span {{
            color: {FONT_COLOR};
            font-family: 'Arial', sans-serif;
        }}
        .stButton>button {{
            background-color: {SECONDARY_COLOR};
            color: white;
            border-radius: 8px;
            border: none;
            font-size: 16px;
            padding: 10px 20px;
        }}
        .stButton>button:hover {{
            background-color: {PRIMARY_COLOR};
            color: white;
        }}
    </style>
""", unsafe_allow_html=True)

#############################
# 3. FUNCIONES PARA BQ
#############################
DATASET_ID = "fb_ads"  
TABLE_NAME = "vid_tst"   

def verify_dataset_and_table():
    """
    Verifica que la dataset y la tabla existan en BigQuery.
    Si no, lanza un error informativo o crea la dataset/tabla si lo prefieres.
    """
    dataset_ref = bq_client.dataset(DATASET_ID)
    table_ref = dataset_ref.table(TABLE_NAME)

    try:
        bq_client.get_dataset(dataset_ref)  # Si falla, no existe el dataset
    except NotFound:
        # Puedes crear la dataset automáticamente o solo alertar
        st.error(f"Dataset '{DATASET_ID}' no existe en la ubicación US. Por favor, créala o ajusta la location.")
        st.stop()
    
    try:
        bq_client.get_table(table_ref)  # Si falla, no existe la tabla
    except NotFound:
        # Puedes crear la tabla automáticamente o solo alertar
        st.error(f"La tabla '{TABLE_NAME}' no existe en '{DATASET_ID}'. Por favor, créala primero.")
        st.stop()

def get_average_ctr(ad_type=None):
    """
    Devuelve la media de la columna CTR para un tipo de anuncio específico.
    Lanza error si la dataset/tabla no existen.
    """
    verify_dataset_and_table()

    query = f"""
        SELECT AVG(CTR) as avg_ctr
        FROM {bq_client.project}.{DATASET_ID}.{TABLE_NAME}
        WHERE CTR IS NOT NULL
    """

    if ad_type:
        query += f" AND TIPO_DE_ANUNCIO = '{ad_type}'"

    query_job = bq_client.query(query)
    try:
        results = query_job.result()
    except NotFound as e:
        st.error(f"No se pudo ejecutar la consulta: {e}")
        st.stop()

    for row in results:
        return row["avg_ctr"]

def get_top_videos_data(ad_type=None):
    """
    Retorna los datos de los 5 videos con mejor CTR, ordenados por CTR descendente.
    """
    verify_dataset_and_table()  # Verifica que la tabla exista

    query = f"""
        SELECT
            Campaignname,
            AdSetname,
            Adname,
            CTR
        FROM {bq_client.project}.{DATASET_ID}.{TABLE_NAME}
    """

    if ad_type:
        query += f" WHERE TIPO_DE_ANUNCIO = '{ad_type}'"

    query += " ORDER BY CTR DESC LIMIT 5"

    query_job = bq_client.query(query)
    try:
        rows = list(query_job.result())
    except NotFound as e:
        st.error(f"No se pudo ejecutar la consulta: {e}")
        return []

    return rows


#############################
# 4. FUNCIÓN ANALIZAR VIDEO CON GEMINI
#############################
def analyze_video_with_gemini(local_file_path: str = None, prompt: str = None) -> dict:
    """
    Analiza un video subido utilizando la API de Gemini.
    """
    try:
        if local_file_path:  # Para el análisis inicial con un archivo
            uploaded_file = genai.upload_file(
                path=local_file_path,
                display_name="video_de_usuario"
            )
            file_info = uploaded_file
            while file_info.state.name == "PROCESSING":
                time.sleep(5)
                file_info = genai.get_file(file_info.name)

            if file_info.state.name == "FAILED":
                raise RuntimeError("Procesamiento del video falló en Gemini (estado FAILED)")

            # Llamar al modelo con el archivo subido y el prompt
            model = genai.GenerativeModel(model_name=DEFAULT_MODEL_NAME)
            response = model.generate_content(
                [file_info, prompt],
                request_options={"timeout": 120}
            )

        elif prompt:  # Para la interpretación en lenguaje natural
            model = genai.GenerativeModel(model_name=DEFAULT_MODEL_NAME)
            response = model.generate_content(
                [prompt],
                request_options={"timeout": 120}
            )
        else:
            raise ValueError("Debes proporcionar un archivo de video o un prompt para analizar.")

        try:
            cleaned_text = response.text.replace("json", "").replace("", "").strip()
            parsed_data = json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            logging.error(f"Error al parsear JSON: {e}, raw_response: {response.text}")
            parsed_data = {"raw_response": response.text, "warning": "La respuesta no se pudo parsear como JSON."}
        
        return parsed_data

    except Exception as e:
        logging.error(f"Error al interactuar con Gemini: {e}")
        st.error(f"Error al interactuar con Gemini: {e}")
        return {"error": str(e)}  # Devuelve un diccionario con el error


#############################
# 5. GENERAR SUGERENCIAS
#############################
def generate_suggestions(gemini_analysis: dict, videos_data: list, avg_ctr: float) -> (list, float):
    """
    Compara el análisis del video nuevo con la data histórica (videos_data) 
    y genera sugerencias. Retorna (lista_de_sugerencias, ctr_estimado).
    """
    suggestions = []

    # Extraemos "argumentos_comerciales" del nuevo video, si existen
    argumentos_nuevos = gemini_analysis.get("argumentos_comerciales", [])
    # Ejemplo: gemini_analysis["argumentos_comerciales"] = ["Salud y Bienestar", "Comodidad", ...]

    # 1. Ver argumentos comerciales más frecuentes en videos exitosos
    freq_args = {}
    for row in videos_data:
        # row["commercial_argument"] podría ser un texto con comas
        if row["commercial_argument"]:
            splitted = row["commercial_argument"].split(",")
            for a in splitted:
                a_clean = a.strip().lower()
                freq_args[a_clean] = freq_args.get(a_clean, 0) + 1

    # Ordenar desc
    sorted_args = sorted(freq_args.items(), key=lambda x: x[1], reverse=True)
    top_args = [arg for arg, _count in sorted_args[:3]]

    # Sugerir si el nuevo video no incluye los 3 top arguments
    for arg in top_args:
        if arg not in [x.lower() for x in argumentos_nuevos]:
            suggestions.append(f"Considera incluir el argumento comercial '{arg}' (muy frecuente en videos con CTR alto).")

    # 2. Revisar background_music
    bg_music = gemini_analysis.get("impacto_musical_global", {})
    tone = bg_music.get("emotional_tone", "").lower()
    if "alegre" not in tone and "positivo" not in tone:
        suggestions.append("Prueba un tono musical más alegre/positivo, ya que con frecuencia mejora el engagement.")

    # 3. Revisar personajes
    if "escenas" in gemini_analysis:
        chars_all_scenes = [char for scene in gemini_analysis.get("escenas", []) for char in scene.get("personajes",[])]
        if len(chars_all_scenes) == 0:
            suggestions.append("Considera incluir al menos un personaje con emoción clara para captar la atención.")
        else:
            # Ver si hay 'felicidad' o 'satisfacción' en el emotional_state
            if not any("felic" in c.get("emociones", []) or 
                       "satis" in c.get("emociones", [])
                       for c in chars_all_scenes):
                suggestions.append("Incorpora un momento de felicidad o satisfacción en los personajes.")
    else:
         suggestions.append("No se detectaron personajes. Considera añadir personajes para humanizar el anuncio.")


    # 4. Calcular CTR estimado (método heurístico)
    #    Ejemplo: tomamos la media de CTR de videos > media y le aplicamos un factor según coincidencias
    if videos_data:
        top_ctrs = [row["CTR"] for row in videos_data]
        if top_ctrs:
            avg_of_top = sum(top_ctrs) / len(top_ctrs)
        else:
            avg_of_top = avg_ctr
    else:
        avg_of_top = avg_ctr

    # Factor base
    similarity_factor = 0.5
    # Si la música es alegre, subimos el factor
    if "alegre" in tone:
        similarity_factor += 0.2
    # Si usamos 1+ argumentos top
    matched_args = sum(arg in [a.lower() for a in argumentos_nuevos] for arg in top_args)
    similarity_factor += matched_args * 0.1  # cada arg top sube 0.1

    estimated_ctr = avg_of_top * similarity_factor
    # No salirse de rangos razonables
    if estimated_ctr < avg_ctr * 0.5:
        estimated_ctr = avg_ctr * 0.5
    if estimated_ctr > avg_of_top * 1.5:
        estimated_ctr = avg_of_top * 1.5

    return suggestions, estimated_ctr

#############################
# 6. APLICACIÓN STREAMLIT
def display_video(video_path):
    """Muestra el video en la interfaz."""
    col1, col2, col3 = st.columns([2, 1, 2])  # Cambia el tamaño de las columnas
    with col2:  # Centrar el video en la columna del medio
       st.video(video_path, start_time=0, format="video/mp4")

def download_video(url, destination_path):
    """Descarga un video desde una URL."""
    try:
      with st.spinner("Descargando video..."):
        resp = httpx.get(url)
        resp.raise_for_status()
        with open(destination_path, "wb") as f:
           f.write(resp.content)
        st.success("Video descargado con éxito.")
        display_video(destination_path)
    except httpx.HTTPError as e:
       st.error(f"No se pudo descargar el video (HTTP Error): {e}")
       raise  # Re-lanza la excepción para que se maneje en el main
    except Exception as e:
        st.error(f"Error al descargar el video: {e}")
        raise # Re-lanza la excepción para que se maneje en el main

def main():
    st.title("Analizador de Videos")

    st.markdown("""
    **Objetivos**:
    1. Analizar información cualitativa de videos con la API de Gemini (música, emociones, etc.).
    2. Comparar con patrones de éxito (CTR) en BigQuery.
    3. Generar sugerencias de mejora.
    4. Estimar CTR potencial.
    """)

    # 6.1 Seleccionar tipo de anuncio
    ad_types = ["AUB", "Descuento", "PVP", "Experiencias", "Review", "Otros"]
    selected_ad_type = st.selectbox("Selecciona el tipo de anuncio a analizar", [None] + ad_types)

    # 6.2 Verificar CTR promedio
    if selected_ad_type is None:
        st.warning("Por favor, selecciona una categoría.")
        return  # Detiene la ejecución de la función hasta que se seleccione una categoría
    
    try:
        with st.spinner("Obteniendo CTR promedio..."):
            avg_ctr = get_average_ctr(selected_ad_type)
    except Exception as e:
        st.error(f"Ocurrió un error al obtener el CTR promedio: {e}")
        st.stop()

    if avg_ctr is None or avg_ctr == 0:
        st.warning("No se encuentran registros en esta categoría.")
        avg_ctr = 0.0  # fallback
    else:
        st.write(f"**CTR promedio de la categoría**: {avg_ctr:.2f}%")

    # 6.3 Obtener videos "exitosos"
    try:
        with st.spinner("Obteniendo datos de videos exitosos..."):
             successful_videos_info = get_top_videos_data(ad_type=selected_ad_type)
    except Exception as e:
        st.error(f"Ocurrió un error obteniendo datos de videos exitosos: {e}")
        successful_videos_info = []
    
    if successful_videos_info:
        st.write(f"Top 5 videos con mejor CTR:")
        for i, row in enumerate(successful_videos_info):
            st.write(f"{i+1}. ['{row['Campaignname']}'] | ['{row['AdSetname']}'] | ['{row['Adname']}'] | ['CTR': {row['CTR']:.2f}]")

    else:
        st.info("No se encontraron videos para esta categoría.")


    # 6.3a Subir video: local o drive
    st.subheader("Sube tu video para analizar")
    choice = st.radio("Selecciona la fuente del video:", ("Archivo local", "Enlace (Drive/otro)"))
    
    temp_video_path = "temp_uploaded_video.mp4"

    if choice == "Archivo local":
        video_file = st.file_uploader("Sube un archivo .mp4 o .mov", type=["mp4", "mov"])
        if video_file:
            try:
              with open(temp_video_path, "wb") as f:
                    f.write(video_file.getbuffer())
              display_video(temp_video_path)
            except Exception as e:
                st.error(f"Error al guardar el archivo: {e}")

    else:
        drive_url = st.text_input("Pega la URL del video (Google Drive u otro servidor)")
        if drive_url:
            if st.button("Descargar video"):
              try:
                download_video(drive_url, temp_video_path)
              except Exception:
                pass # Los errores se manejan en la función download_video


    # 6.4 Análisis con Gemini
    if os.path.exists(temp_video_path):
        # Botón 1: Generar JSON del análisis
        if st.button("Generar JSON del análisis"):
            prompt_analisis = """
            Rol: Actúa como un analista de publicidad de video experto en marketing digital y comunicación, con un enfoque en la extracción de datos estructurados para análisis comparativo.

Objetivo: Analiza el video proporcionado y extrae información objetiva y cuantificable sobre sus atributos visuales, auditivos y textuales.

Formato de Salida: Proporciona la información en formato JSON estructurado, optimizado para la inserción en una base de datos BigQuery.

Completitud: Asegúrate de que todos los campos del JSON estén presentes. Si algún dato no aplica, rellénalo con un valor nulo (null) en JSON.

Énfasis en la Extracción de Datos: Prioriza la extracción de datos concretos, sin interpretación subjetiva innecesaria.

Argumentos Comerciales: Identifica los argumentos comerciales presentes y ausentes, usando datos estructurados para su análisis.

Música: Analiza la música de forma precisa, extrayendo datos sobre su tono y instrumentación. Considera que siempre hay melodía, aunque sea sutil.

Optimización para API: El formato JSON debe ser lo más conciso y eficiente posible para la API. El JSON se estructurará de la siguiente forma:
{
  "analisis": {
    "escenas": [
      {
        "numero_escena": null,
        "descripcion": null,
        "timestamp_inicio": null,
        "timestamp_fin": null,
        "personajes": [
          {
            "nombre": null,
            "descripcion": null,
            "emociones": []
          }
        ],
        "emociones_principales": [],
         "impacto_musical": {
          "existente": "si",
          "tono_emocional": null,
          "instrumentacion": []
        },
        "texto_en_pantalla": [],
        "colores_usados": [],
        "proposito": null
      }
    ],
     "emociones_globales": [],
    "impacto_musical_global": {
      "existente": "si",
      "tono_emocional": null,
       "instrumentacion": []
     },
   "analisis_argumentos_comerciales": [
        {
          "timestamp": null,
          "argumento": null,
          "contexto": null
        }
      ],
    "argumentos_no_utilizados": []
  }
}

Formato de Datos:

timestamp_inicio y timestamp_fin: Formato "MM:SS".

emociones: Lista de strings (ej. ["alegría", "sorpresa"]). Si no hay emociones claras, usar una lista vacía [].

texto_en_pantalla: Lista de strings con el texto exacto (ej. ["Oferta", "2x1"]).

colores_usados: Lista de strings con los colores predominantes (ej. ["rojo", "blanco", "negro"]).

instrumentacion: Lista de strings con los instrumentos detectados y/o descripción de la melodía (ej ["melodía electrónica suave", "piano"]). Siempre habrá melodía o sonidos presente.

argumentos_comerciales: timestamp en formato "MM:SS", argumento en formato string (ej: "Diseño y Moderno") y contexto en formato string (ej: "Se ve un producto con diseño elegante").

argumentos_no_utilizados: Lista de strings con los argumentos que no aparecen en el video.

Si un campo no tiene valor, debe ser null en el JSON, no una string vacía.

Listas: Siempre usar listas para emociones, texto_en_pantalla, colores_usados e instrumentacion aunque haya un solo valor.

Consistencia: Mantén la misma estructura en todas las escenas.

Objetividad: Evita la interpretación subjetiva innecesaria. Céntrate en datos observables y cuantificables, pero no ignores los detalles emocionales que pueda transmitir el video.

Argumentos Comerciales:
* Salud y Bienestar
* Diseño y Moderno
* Comodidad y Prestigio
* Mantención sin costo adicional
* Sostenibilidad
* Evolución
* Agua purificada ilimitada
* Mejor Sabor

Énfasis en el Humor: Siempre que sea posible, detecta y describe los tintes cómicos que puedan estar presentes en el video, ya sea en las actuaciones, situaciones, o en el uso de la música o melodías.

Análisis Musical Detallado:
* Existencia: Siempre considera que hay un elemento musical o melodía presente, aunque sea sutil.
* Tono Emocional: Identifica el tono emocional de la música o melodía, utilizando una variedad de descriptores (ej. "alegre", "triste", "relajante", "tensa", "cómica", "irónica", etc.) y combinaciones (ej. "ligeramente tensa con toque cómico").
* Instrumentación/Melodía: Describe la instrumentación o los elementos que componen la melodía. No solo los instrumentos tradicionales, sino también los elementos electrónicos y cualquier otro sonido que contribuya a la música o melodía del video.
            """
            with st.spinner("Generando el JSON con Gemini..."):
                try:
                    gemini_result = analyze_video_with_gemini(temp_video_path, prompt_analisis)

                    if "error" in gemini_result:
                       st.error(f"Error generando el análisis con Gemini: {gemini_result['error']}")
                    else:
                        # Guardar en session_state
                        st.session_state["gemini_result"] = gemini_result
                        st.success("JSON generado con éxito.")
                        # Ocultar el json
                        with st.expander("Ver JSON generado"):
                            st.json(gemini_result)  # Muestra el JSON dentro de un expander
    
                except Exception as e:
                    st.error(f"Error generando el análisis con Gemini: {e}")

    
        # Botón 2: Interpretar JSON en lenguaje natural
        if st.button("Interpretar JSON en lenguaje natural"):
            if "gemini_result" not in st.session_state:
                st.warning("Primero debes generar el JSON del análisis.")
            else:
                gemini_result = st.session_state["gemini_result"]  # Recuperar el JSON
                prompt_interpretacion = f"""
                Dado el análisis JSON a continuación, interpreta cada sección en lenguaje natural. Destaca:
                - Secuencia de escenas (descripción, personajes, emociones, texto en pantalla, propósito).
                - Impacto de la música, si existe, y cómo contribuye a la narrativa.
                - Argumentos comerciales (identificando la timestamp, nombre y contexto).
                - Narrativa clave.
                - Recomendaciones generales para mejorar el anuncio.
                JSON a interpretar:
                {json.dumps(gemini_result, indent=2)}
                """
                with st.spinner("Generando interpretación en lenguaje natural..."):
                    try:
                        interpretation = analyze_video_with_gemini(prompt=prompt_interpretacion).get("raw_response","No se pudo obtener la interpretación.")
    
                        # Mostrar interpretación estilizada
                        st.markdown("""
                        <style>
                            .interpretation-title {
                                color: #0A74DA;
                                font-size: 22px;
                                font-weight: bold;
                            }
                            .section-title {
                                color: #34A853;
                                font-size: 18px;
                                font-weight: bold;
                            }
                            .content {
                                font-size: 16px;
                                color: #333333;
                                margin-bottom: 10px;
                            }
                        </style>
                        """, unsafe_allow_html=True)
    
                        st.markdown(f"""
                        <div class="interpretation-title">Interpretación del Video</div>
                        <div class="content">{interpretation}</div>
                        """, unsafe_allow_html=True)
                        
                        # Generar sugerencias
                        if successful_videos_info and isinstance(gemini_result, dict) and "analisis" in gemini_result:
                            suggestions, estimated_ctr = generate_suggestions(gemini_result["analisis"], successful_videos_info, avg_ctr)
                            
                            st.markdown(f"""
                                <div class="section-title">Sugerencias de Mejora:</div>
                                <ul class="suggestion-list">
                                    {"".join(f"<li>{s}</li>" for s in suggestions)}
                                </ul>
                                """, unsafe_allow_html=True)

                            st.markdown(f"""
                                <div class="content">
                                 **CTR Estimado:** {estimated_ctr:.2f}%
                                </div>
                            """, unsafe_allow_html=True)
                                                        
                        else:
                             st.info("No se pueden generar sugerencias. No se encontraron videos similares con CTR superior a la media o el analisis JSON no se pudo generar.")
        
                    except Exception as e:
                        st.error(f"Error generando la interpretación: {e}")
                        st.json(gemini_result)  # Mostrar el JSON si falla la interpretación
    
    else:
        st.info("Aún no has seleccionado ni subido/descargado un video.")

if __name__ == "__main__":
    main()