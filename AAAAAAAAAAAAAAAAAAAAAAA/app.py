import os
import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
from io import BytesIO
import base64
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.tools.sql_database.tool import SQLDatabase
from sqlalchemy import create_engine

st.set_page_config(page_title="SQL AI Assistant", layout="wide")

# Inicializar valores predeterminados en session_state si no existen
if "config" not in st.session_state:
    st.session_state.config = {
        "openai_api_key": "",
        "mysql_user": "root",
        "mysql_password": "rootpassword",
        "mysql_host": "10.23.63.61",
        "mysql_port": "3306",
        "mysql_db": "dev",
        "llm_model": "o3-mini",
        "chart_model": "gpt-3.5-turbo",
        "db_initialized": False
    }

# Menú lateral para configuración
with st.sidebar:
    st.title("Configuración")
    
    # Sección de API Key
    st.subheader("OpenAI API")
    api_key = st.text_input("OpenAI API Key", value=st.session_state.config["openai_api_key"], type="password")
    
    # Sección de configuración de base de datos
    st.subheader("Configuración de Base de Datos")
    mysql_host = st.text_input("Host", value=st.session_state.config["mysql_host"])
    mysql_port = st.text_input("Puerto", value=st.session_state.config["mysql_port"])
    mysql_user = st.text_input("Usuario", value=st.session_state.config["mysql_user"])
    mysql_password = st.text_input("Contraseña", value=st.session_state.config["mysql_password"], type="password")
    mysql_db = st.text_input("Base de Datos", value=st.session_state.config["mysql_db"])
    
    # Sección de configuración de modelos
    st.subheader("Modelos LLM")
    llm_model = st.selectbox(
        "Modelo para consultas SQL",
        options=["o3-mini", "gpt-3.5-turbo", "gpt-4"],
        index=0 if st.session_state.config["llm_model"] == "o3-mini" else 
              1 if st.session_state.config["llm_model"] == "gpt-3.5-turbo" else 2
    )
    
    chart_model = st.selectbox(
        "Modelo para gráficos y análisis",
        options=["gpt-3.5-turbo", "gpt-4"],
        index=0 if st.session_state.config["chart_model"] == "gpt-3.5-turbo" else 1
    )
    
    # Botón para actualizar configuración
    if st.button("Actualizar Configuración"):
        st.session_state.config.update({
            "openai_api_key": api_key,
            "mysql_user": mysql_user,
            "mysql_password": mysql_password,
            "mysql_host": mysql_host,
            "mysql_port": mysql_port,
            "mysql_db": mysql_db,
            "llm_model": llm_model,
            "chart_model": chart_model,
            "db_initialized": False  # Forzar reinicialización
        })
        st.success("Configuración actualizada correctamente!")

    st.subheader("Sobre la aplicación")
    st.markdown("""
    Este chatbot utiliza LangChain y OpenAI para:
    - Procesar consultas en lenguaje natural sobre bases de datos SQL
    - Traducir entre español e inglés
    - Generar visualizaciones automáticamente
    """)

# Inicialización de la conexión y agentes
@st.cache_resource(show_spinner=False)
def initialize_agents(_config):
    # Configurar API key
    os.environ["OPENAI_API_KEY"] = _config["openai_api_key"]
    
    # Configurar conexión a la base de datos
    DATABASE_URL = f"mysql+pymysql://{_config['mysql_user']}:{_config['mysql_password']}@{_config['mysql_host']}:{_config['mysql_port']}/{_config['mysql_db']}"
    
    try:
        engine = create_engine(DATABASE_URL)
        db = SQLDatabase(engine)
        
        # Inicializar modelos
        llm = ChatOpenAI(model=_config["llm_model"])
        chart_llm = ChatOpenAI(model=_config["chart_model"])
        
        # Crear agente SQL
        sql_agent = create_sql_agent(llm=llm, db=db, verbose=True)
        
        return {
            "db": db,
            "llm": llm,
            "chart_llm": chart_llm,
            "sql_agent": sql_agent,
            "status": "success"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

# Función para traducir texto (español a inglés y viceversa)
def translate_text(text, source_lang, target_lang, translation_llm):
    prompt = f"Traduce el siguiente texto de {source_lang} a {target_lang}. Solo responde con la traducción, sin explicaciones adicionales: {text}"
    response = translation_llm.invoke(prompt)
    return response.content

# Función para extraer datos JSON con LLM
def extract_json_with_llm(response_text, llm):
    prompt = f"""
    Dado el siguiente texto en español:

    "{response_text}"

    Extrae los datos numéricos explícitos mencionados en el texto con su correspondiente etiqueta asociada, y devuélvelos en el siguiente formato JSON:

    {{
        "etiqueta1": valor1,
        "etiqueta2": valor2
    }}

    Instrucciones estrictas:
    - Usa únicamente el contenido textual provisto.
    - NO agregues símbolos como "%", solo el número en formato decimal.
    - Proporciona exclusivamente el JSON solicitado, sin explicaciones adicionales.
    """

    response = llm.invoke(prompt)
    try:
        # Intentar extraer el JSON de la respuesta
        json_str = re.search(r'({.*})', response.content, re.DOTALL)
        if json_str:
            data = json.loads(json_str.group(1))
        else:
            data = json.loads(response.content.strip())
    except json.JSONDecodeError:
        # Si falla, intentar otro enfoque
        fallback_prompt = f"""
        Simplifica el proceso. Extrae SOLO los números con sus etiquetas del texto:
        "{response_text}"
        
        Y responde ÚNICAMENTE con un JSON simple, por ejemplo: {{"total": 500, "promedio": 42.5}}
        """
        fallback_response = llm.invoke(fallback_prompt)
        try:
            json_str = re.search(r'({.*})', fallback_response.content, re.DOTALL)
            if json_str:
                data = json.loads(json_str.group(1))
            else:
                data = json.loads(fallback_response.content.strip())
        except json.JSONDecodeError:
            data = {}
    
    return data

# Función para determinar si la consulta solicita un gráfico
def is_chart_request(query):
    chart_keywords = [
        "gráfico", "grafico", "gráfica", "grafica", "visualiza", 
        "visualización", "visualizacion", "muestra", "representa", "dibuja",
        "chart", "plot", "graph", "visualize", "show", "display", "draw"
    ]
    
    for keyword in chart_keywords:
        if keyword in query.lower():
            return True
    return False

# Función para determinar el tipo de gráfico solicitado
def get_chart_type(query):
    if any(kw in query.lower() for kw in ["torta", "pastel", "pie", "circular"]):
        return "pie"
    # Por defecto, usar gráfico de barras
    return "bar"

# Función para generar y guardar el gráfico
def generate_chart(data, chart_type="bar", title="Distribución de datos"):
    plt.figure(figsize=(10, 6))
    
    if chart_type == "bar":
        plt.bar(list(data.keys()), list(data.values()), color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'][:len(data)])
        plt.ylabel('Valores')
        plt.xlabel('Categorías')
    elif chart_type == "pie":
        plt.pie(list(data.values()), labels=list(data.keys()), autopct='%1.1f%%', 
                colors=['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'][:len(data)])
        plt.axis('equal')
    
    plt.title(title)
    
    # Guardar la imagen en un buffer en memoria
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight')
    img_buffer.seek(0)
    
    # Codificar la imagen en base64 para mostrarla
    img_str = base64.b64encode(img_buffer.read()).decode('utf-8')
    
    plt.close()
    return img_str

# Función principal de la aplicación
def main():
    st.title("🤖 SQL Assistant con IA")
    st.subheader("Consulta tu base de datos en español")
    
    # Inicializar agentes si es necesario
    if not st.session_state.config.get("db_initialized", False):
        with st.spinner("Conectando a la base de datos y configurando agentes..."):
            agents = initialize_agents(st.session_state.config)
            
            if agents["status"] == "error":
                st.error(f"Error al inicializar la conexión: {agents['error']}")
                st.warning("Por favor, verifica la configuración en el menú lateral.")
                return
            
            st.session_state.agents = agents
            st.session_state.config["db_initialized"] = True
    
    # Inicializar historial de chat si no existe
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Mostrar historial de chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("chart"):
                st.image(f"data:image/png;base64,{message['chart']}")
            if message.get("sql_query"):
                with st.expander("Ver consulta SQL generada"):
                    st.code(message["sql_query"], language="sql")
    
    # Input para la consulta
    if prompt := st.chat_input("Escribe tu consulta en español..."):
        # Agregar mensaje del usuario al historial
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Mostrar mensaje del usuario
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Respuesta del asistente
        with st.chat_message("assistant"):
            # Verificar si se solicita un gráfico
            chart_requested = is_chart_request(prompt)
            chart_type = get_chart_type(prompt) if chart_requested else None
            
            # Mostrar indicador de carga mientras se procesa
            with st.spinner("Pensando..."):
                try:
                    # Traducir la pregunta al inglés
                    progress_placeholder = st.empty()
                    progress_placeholder.info("🔄 Traduciendo consulta...")
                    
                    pregunta_en = translate_text(
                        prompt, 
                        "español", 
                        "inglés", 
                        st.session_state.agents["chart_llm"]
                    )
                    
                    # Procesar la consulta en inglés
                    progress_placeholder.info("🔍 Analizando la base de datos...")
                    respuesta_en = st.session_state.agents["sql_agent"].run(pregunta_en)
                    
                    # Intentar capturar la consulta SQL
                    try:
                        sql_query = st.session_state.agents["sql_agent"]._last_query
                    except AttributeError:
                        # Intentar extraer la consulta SQL del texto de respuesta
                        sql_match = re.search(r'```sql\n(.*?)\n```', respuesta_en, re.DOTALL)
                        sql_query = sql_match.group(1) if sql_match else "No se pudo obtener la consulta SQL"
                    
                    # Traducir la respuesta al español
                    progress_placeholder.info("🔄 Traduciendo respuesta...")
                    respuesta_es = translate_text(
                        respuesta_en, 
                        "inglés", 
                        "español", 
                        st.session_state.agents["chart_llm"]
                    )
                    
                    # Limpiar el mensaje de progreso
                    progress_placeholder.empty()
                    
                    # Mostrar la respuesta
                    st.markdown(respuesta_es)
                    
                    # Si se solicita un gráfico, generarlo
                    img_str = None
                    if chart_requested:
                        progress_placeholder.info("📊 Generando visualización...")
                        data = extract_json_with_llm(respuesta_es, st.session_state.agents["chart_llm"])
                        
                        if data:
                            # Generar título para el gráfico
                            chart_title_prompt = f"""
                            Genera un título corto y descriptivo para un gráfico basado en esta consulta:
                            "{prompt}"
                            Solo responde con el título, sin texto adicional.
                            """
                            chart_title_response = st.session_state.agents["chart_llm"].invoke(chart_title_prompt)
                            chart_title = chart_title_response.content.strip()
                            
                            # Generar el gráfico
                            img_str = generate_chart(data, chart_type, chart_title)
                            st.image(f"data:image/png;base64,{img_str}")
                        else:
                            st.warning("No se pudieron extraer datos adecuados para generar un gráfico")
                        
                        progress_placeholder.empty()
                    
                    # Guardar respuesta en el historial
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": respuesta_es,
                        "chart": img_str,
                        "sql_query": sql_query
                    })
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"Lo siento, ocurrió un error: {str(e)}",
                        "chart": None,
                        "sql_query": None
                    })

# Ejecución de la aplicación
if __name__ == "__main__":
    main()