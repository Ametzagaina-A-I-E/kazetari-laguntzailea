# ARGIA

ARGIA es un proyecto de IA multimodal enfocado en el procesamiento del euskera. Incluye herramientas para la transcripción de audio, análisis de noticias y sistemas RAG (Generación Aumentada por Recuperación).

## Características Principales

### 🧠 Modelos de IA
Este proyecto utiliza modelos de IA específicos para el euskera:
- **Latxa (LLM)**: Un Modelo de Lenguaje Grande (LLM) basado en Llama, ajustado específicamente para el euskera. Se utiliza para la generación de texto, resúmenes y el sistema RAG.
- **Whisper (Speech-to-Text)**: Utilizamos el modelo Whisper de OpenAI, ajustado para el euskera (fine-tuned), para realizar transcripciones de audio precisas.
- **Milvus (Base de Datos Vectorial)**: Utilizamos Milvus como base de datos vectorial para el sistema RAG. Las noticias se dividen en fragmentos (chunks), se vectorizan y se almacenan en Milvus. Esto permite realizar búsquedas semánticas rápidas y precisas para recuperar la información más relevante.

### 💻 Interfaz de Usuario
- **Streamlit**: Todo el proyecto cuenta con una interfaz gráfica amigable construida con Streamlit, lo que facilita la interacción con los modelos sin necesidad de código.

## Modos de Ejecución

El proyecto está diseñado para ser flexible y permite diferentes formas de ejecución:

### 1. Ejecución Local (Scripts)
Puedes ejecutar los scripts directamente en tu máquina local si dispones de los recursos necesarios como el GPU para los modelos.
- **RAG**: `RAG/RAG.py` (Ejecución local del sistema de preguntas y respuestas).
- **Whisper**: `audio_text/whisper_eu.py` (Script de transcripción).
- **Generador de Texto**: `albiste_analisia/text_generator.py`.

### 2. Ejecución Servidor / API (Streamlit)
Los archivos terminados en `_server.py` están optimizados para funcionar conectándose a APIs externas (como la API de Hugging Face o OpenAI) o para ser desplegados en servidores. En estos casos, se requiere un apikey para acceder a las APIs externas.
- **RAG Server**: `RAG/RAG_server.py`
- **Whisper Server**: `audio_text/whisper_eu_server.py`
- **Text Gen Server**: `albiste_analisia/text_generator_server.py`

## Instalación

1.  **Clonar el repositorio**:
    ```bash
    git clone https://github.com/tu_usuario/ARGIA.git
    cd ARGIA
    ```

2.  **Crear un entorno virtual** (recomendado):
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```

3.  **Instalar dependencias**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configurar variables de entorno**:
    - Copia el archivo de ejemplo:
      ```bash
      cp .env.example .env
      ```
    - Abre el archivo `.env` y añade tus claves (Hugging Face Token, API Keys) y la configuración de tu base de datos MySQL si es necesario.

5.  **Iniciar Milvus (Base de Datos Vectorial)**:
    Para que el sistema RAG funcione, necesitas tener [Milvus](https://milvus.io/) corriendo. Utilizamos Docker para esto.
    Asegúrate de tener Docker instalado y ejecuta:
    ```bash
    docker-compose up -d
    ```
    *Nota: En el Json de `chunk_guztiak` están todos los chunks contextualializados, para utilizarlos, hay que subirlos a milvus vectorizandolos, para ello está el script `MILVUS/milvus_db.py`*

## Uso

La forma principal de ejecutar la aplicación es a través de `app.py`, que proporciona una interfaz unificada para acceder a todas las herramientas.

### Ejecución Unificada
Por defecto, la aplicación se ejecuta en modo **local** (intentará cargar modelos en tu máquina):
```bash
streamlit run app.py
```

### Modos de Ejecución (Local vs Servidor)
Puedes controlar si la aplicación se ejecuta en modo local o si se conecta a servidores/APIs externas mediante el argumento `--local`.

- **Modo Local (Por defecto)**:
  ```bash
  streamlit run app.py -- --local true
  ```
  *Nota: Requiere GPU y recursos suficientes en tu máquina.*

- **Modo Servidor / API**:
  Si prefieres que la aplicación utilice APIs externas (como Hugging Face) o servidores remotos, ejecuta:
  ```bash
  streamlit run app.py -- --local false
  ```
  *Nota: Asegúrate de tener configuradas las API Keys en tu archivo `.env`.*


## Estructura del Proyecto

- `audio_text/`: Implementación de Whisper y servidor de transcripción.
- `albiste_analisia/`: Lógica de generación de texto y análisis de noticias.
- `RAG/`: Implementación del sistema RAG (Retrieval-Augmented Generation).
- `MILVUS/`: Configuraciones para la base de datos vectorial Milvus.

