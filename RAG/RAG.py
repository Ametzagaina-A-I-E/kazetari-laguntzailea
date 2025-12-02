import streamlit as st
import sqlite3
import uuid
import streamlit.components.v1 as components
import torch
from MILVUS.milvus_utils import get_milvus_client, get_search_results
from load_models import generate_text,get_reranker,get_llm,emb_text_bge
from RAG.config import config
import time
from datetime import datetime, timedelta
from transformers import pipeline
import requests
import json
from calendar import monthrange

# ---------------------------------------------------------
# Function: hemeroteka
# Description:
#   Main UI logic for the "Hemeroteka" section.
#   Handles:
#       - Loading models (LLM and reranker)
#       - Managing chat sessions
#       - Retrieving documents from Milvus
#       - Reranking and displaying top results
#       - Generating answers with LLM
# ---------------------------------------------------------
def hemeroteka():

    # -----------------------------
    # Load reranker model
    # -----------------------------

    start_time = time.time()
    tokenizer_rerank, model_rerank = get_reranker(config.reraking.model_name)


    # -----------------------------
    # Functions
    # -----------------------------   

    def normalize_duckling_output(user_query, duckling_resp, reference_time=None):
        """
        Normalize Duckling date/time parsing output into constraints for Milvus filtering.
        """
        constraints = []
        query_lower = user_query.lower()

        for entity in duckling_resp:
            if entity.get("dim") != "time":
                continue

            value = entity["value"]

            # Interval case
            if value.get("type") == "interval":
                from_date, to_date = None, None
                if value.get("from") and isinstance(value["from"].get("value"), str):
                    from_date = value["from"]["value"][:10]
                if value.get("to") and isinstance(value["to"].get("value"), str):
                    to_date = value["to"]["value"][:10]
                constraints.append({"from": from_date, "to": to_date})

            # Simple date case
            elif "value" in value and isinstance(value["value"], str):
                date_value = value["value"][:10]
                grain = value.get("grain")
                
                if grain == "year":
                    from_date = f"{date_value[:4]}-01-01"
                    to_date   = f"{date_value[:4]}-12-31"
                    constraints.append({"from": from_date, "to": to_date})   
                elif grain == "month":
                    year, month = int(date_value[:4]), int(date_value[5:7])
                    last_day = monthrange(year, month)[1]
                    from_date = f"{year}-{month:02d}-01"
                    to_date   = f"{year}-{month:02d}-{last_day:02d}"
                    constraints.append({"from": from_date, "to": to_date})
                    
                elif grain == "week":
                    # Convertir la fecha a objeto datetime
                    dt = datetime.strptime(date_value, "%Y-%m-%d")
                    
                    # Calcular el lunes (weekday 0) y domingo (weekday 6)
                    start_of_week = dt - timedelta(days=dt.weekday())
                    end_of_week = start_of_week + timedelta(days=6)
                    
                    from_date = start_of_week.strftime("%Y-%m-%d")
                    to_date = end_of_week.strftime("%Y-%m-%d")
                    
                    constraints.append({"from": from_date, "to": to_date})

                else:  # day
                    constraints.append({"from": date_value, "to": date_value})

        # Handle special case: "until present"
        if "actualidad" in query_lower and reference_time:
            constraints.append({"to": reference_time.strftime("%Y-%m-%d")})

        return constraints


    def build_milvus_filter(constraints, column="fetxa"):
        """
        Build a Milvus filter query from date constraints.
        """
        filters = []

        for c in constraints:
            if "from" in c or "to" in c:
                if c.get("from"):
                    filters.append(f'{column} >= "{c["from"]}"')
                if c.get("to"):
                    filters.append(f'{column} <= "{c["to"]}"')
            elif "operator" in c and "date" in c:
                filters.append(f'{column} {c["operator"]} "{c["date"]}"')

        return " and ".join(filters) if filters else None

    def rerank_with_cross_encoder(query, candidates):
        """
        Rerank candidate documents based on relevance to the query using a Cross-Encoder.

        Args:
            query (str): The user's question.
            candidates (list): List of dicts containing {'text', 'score', ...}.

        Returns:
            list: Reranked candidates sorted by relevance score.
        """
        # Create query-document pairs
        pairs = [(query, c['text']) for c in candidates]

        # Tokenize and predict scores
        inputs = tokenizer_rerank(
            pairs, padding=True, truncation=True, return_tensors="pt", max_length=512
        )
        with torch.no_grad():
            scores = model_rerank(**inputs).logits.squeeze(-1).tolist()

        # Attach scores and sort
        for idx, score in enumerate(scores):
            candidates[idx]['rerank_score'] = score

        return sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)

    def guardar_valoracion(pregunta, respuesta, valoracion, expansibles,respuesta_id,duration):
        urls = [fuente["url"] for fuente in expansibles]

        # in case there are not 6 urls
        urls_padded = urls[:6] + [None] * (6 - len(urls))

        # save in SQLite
        conn = sqlite3.connect("Balorazioak_RAG.db")
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                galdera TEXT,
                erantzuna TEXT,
                balorazioa INT,
                url1 TEXT,
                url2 TEXT,
                url3 TEXT,
                url4 TEXT,
                url5 TEXT,
                url6 TEXT,
                denbora REAL
            )
        """)
        cursor.execute("""
            INSERT INTO feedback (galdera, erantzuna, balorazioa, url1, url2, url3, url4, url5, url6, denbora)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (pregunta, respuesta, valoracion, *urls_padded, duration))
        conn.commit()
        conn.close()

        # vote save the 
        st.session_state.votaciones[respuesta_id] = valoracion


    # -----------------------------
    # Page configuration
    # -----------------------------
    st.set_page_config(
        page_title="ARGIA",
        page_icon="✨",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.image("img/argia.png", width=300)
    st.markdown("📰 **Hemeroteka**")
    st.markdown("---")

    # -----------------------------
    # Connect to Milvus
    # -----------------------------
    milvus_client = get_milvus_client(
        uri=config.milvus.MILVUS_ENDPOINT, token=config.milvus.MILVUS_TOKEN
    )

    # -----------------------------
    # Load or retrieve LLM
    # -----------------------------
    if "model" not in st.session_state or "tokenizer" not in st.session_state:
        try:
            tokenizer, model = get_llm(config.model.name,True)

            st.session_state.tokenizer = tokenizer
            st.session_state.model = model
        except Exception as e:
            st.error(f"Error loading LLM: {e}")
            st.stop()
    else:
        tokenizer = st.session_state.tokenizer
        model = st.session_state.model

    translator = pipeline("translation", model="HiTZ/mt-hitz-eu-es")

    # -----------------------------
    # Initialize chat session state
    # -----------------------------
    if "conversaciones" not in st.session_state:
        st.session_state.conversaciones = {}
    if "chat_actual" not in st.session_state:
        st.session_state.chat_actual = None
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None
    if "votaciones" not in st.session_state:
        st.session_state.votaciones = {}
    if "contador_respuestas" not in st.session_state:
        st.session_state.contador_respuestas = 0

    # Create a new chat if none exists
    if st.session_state.chat_actual is None:
        nuevo_id = str(uuid.uuid4())[:8]
        st.session_state.conversaciones[nuevo_id] = []
        st.session_state.chat_actual = nuevo_id
        st.rerun()
        st.stop()

    chat_id = st.session_state.chat_actual
    historial = st.session_state.conversaciones[chat_id]

    # -----------------------------
    # Display chat history
    # -----------------------------
    with st.container():
        for mensaje in historial:
            with st.chat_message(mensaje["role"], avatar=mensaje.get("avatar", "")):
                st.markdown(mensaje["content"])
                
                expansibles = mensaje.get("expansibles", [])

                if expansibles:
                    st.write('Hona hemen erlazionatutako albiste batzuk: ')
                    # group by 3
                    for i in range(0, len(expansibles), 3):
                        
                        grupo = expansibles[i:i+3]
                        cols = st.columns(len(grupo))  # only neccesary columns

                        for j, fuente in enumerate(grupo):
                            with cols[j]:
                                st.markdown(
                                    f"""
                                    <div style="
                                        background-color: #f8f9fa;
                                        padding: 15px;
                                        border-radius: 12px;
                                        margin: 10px;
                                        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                                        height: 200px;
                                        display: flex;
                                        flex-direction: column;
                                        justify-content: space-between;
                                    ">
                                        <h4 style="margin-bottom:10px; color:#333; font-size:16px;">📌 {fuente['izenburua']}</h4>
                                        <p style="color:#666; margin-bottom: 8px;">
                                            <b>📅 Argitalpen data:</b><br>{fuente['data']}
                                        </p>
                                        <a href="{fuente['url']}" target="_blank" style="
                                            display: inline-block;
                                            background-color: #f0f2f6;
                                            color: black;
                                            padding: 6px 12px;
                                            text-decoration: none;
                                            border-radius: 6px;
                                            text-align: center;
                                            font-size: 14px;
                                            font-weight: bold;
                                        ">🔗 Ikusi albistea</a>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                
                                )
                if mensaje['role'] == 'assistant':
                    respuesta_id = mensaje.get("id") 
                    if respuesta_id not in st.session_state.votaciones:
                        col1, col2, col3,col4,col5,col6 = st.columns([5, 1,1,1,1,10])
                        with col2:
                            st.button("1 ", key=f"1_{respuesta_id}",
                                    on_click=lambda r_id=respuesta_id, msg=mensaje: guardar_valoracion(
                                        msg['pregunta'], msg['content'], 1, msg['expansibles'], r_id, msg['duration']
                                    ))
                        with col3:
                            st.button("2 ", key=f"2_{respuesta_id}",
                                    on_click=lambda r_id=respuesta_id, msg=mensaje: guardar_valoracion(
                                        msg['pregunta'], msg['content'], 2, msg['expansibles'], r_id, msg['duration']
                                    ))
                        with col4:
                            st.button("3 ", key=f"3_{respuesta_id}",
                                    on_click=lambda r_id=respuesta_id, msg=mensaje: guardar_valoracion(
                                        msg['pregunta'], msg['content'], 3, msg['expansibles'], r_id, msg['duration']
                                    ))
                        with col5:
                            st.button("4 ", key=f"4_{respuesta_id}",
                                    on_click=lambda r_id=respuesta_id, msg=mensaje: guardar_valoracion(
                                        msg['pregunta'], msg['content'], 4, msg['expansibles'], r_id, msg['duration']
                                    ))
                        with col1:
                            st.info('Puntuatu emandako erantzuna 1-etik 4-ra')
                    else:
                        st.info("✅ Eskerrik asko erantzunagatik!") 
                                

    # Auto-scroll to bottom
    components.html("<script>window.scrollTo(0, document.body.scrollHeight);</script>", height=0)

    # -----------------------------
    # User input field
    # -----------------------------
    query = st.chat_input("Bota zure galdera...")

    if query:
        
        st.session_state.conversaciones[chat_id].append(
            {"role": "user", "avatar": "👤", "content": query}
        )
        st.session_state.pending_query = query
        st.rerun()

    # -----------------------------
    # Process pending query
    # -----------------------------
    if st.session_state.pending_query:
        with st.spinner('Erantzuna sortzen...'):
            query = st.session_state.pending_query
            st.session_state.pending_query = None

            ############## DATE ##############################

            query_es = translator(query, max_length=100)[0]['translation_text']
            print('traducion_es', query_es)

            resp = requests.post(
                "http://0.0.0.0:8001/parse",
                data={"locale": "es_ES", "text": query_es, "referenceTime": "2025-09-23T00:00:00Z"}
            )
            
            fetxa = resp.json()
            print(fetxa)
            constraints = normalize_duckling_output(query_es, fetxa)
            milvus_filter = build_milvus_filter(constraints)
            print('milvus filter', milvus_filter)
            # 1. Embed the query
            query_vector = emb_text_bge(query, is_query=True)

            # 2. Retrieve similar documents from Milvus
            search_res = get_search_results(
                milvus_client,
                config.milvus.COLLECTION_NAME,
                query_vector,
                ["text", "id_albiste", "url", "fetxa", "entradilla", "izenburua"],
                limit=config.milvus.num_context_chunks,
                filter = milvus_filter
            )
            
            # 3. Group by id_albiste and select best chunk per document
            doc_dict = {}
            for hit in search_res[0]:
                doc_id = hit.entity.get("id_albiste")
                print(hit.entity.get('izenburua'),hit.score)
                if hit.score < 0.5:
                    continue
                if doc_id not in doc_dict or hit.score > doc_dict[doc_id]["score"]:
                    doc_dict[doc_id] = {
                        "id_albiste": doc_id,
                        "entradilla":hit.entity.get('entradilla'),
                        "text": hit.entity.get("text"),
                        "izenburua": hit.entity.get("izenburua"),
                        "fetxa": hit.entity.get("fetxa"),
                        "url": hit.entity.get("url"),
                        "score": hit.score,
                    }
                    

            # 4. Rerank top documents
            
            if not doc_dict:
                context = 'ez daukazu informaziorik'
                final_results = {}
            else:
                candidates = list(doc_dict.values())
                reranked_results = rerank_with_cross_encoder(query, candidates)
                final_results = reranked_results[:config.milvus.num_show_chunks]

                # 5. Build context for LLM
                #context = "\n".join([res["text"] for res in final_results])
                context = "\n".join([
                    res["entradilla"] if res.get("entradilla", "").strip() else res["text"]
                    for res in final_results
                ])


            print('contextue' , context, 'bukau da kontextue')
            messages = [
                {"role": "system", "content": config.prompts.system_prompt},
                {"role": "user", "content": config.prompts.prompt_user.format(context=context, question=query)},
            ]

            # 6. Generate LLM response
            respuesta = generate_text(
                model=model,
                tokenizer=tokenizer,
                messages=messages,
                use_chat_template=True,
                max_tokens=1024,
                temperature=0.01,
                do_sample=True,
            )
            # Incrementar y asignar ID incremental
            st.session_state.contador_respuestas += 1
            respuesta_id = st.session_state.contador_respuestas
            end_time = time.time()

            duration = end_time - start_time
            
            # 7. Save assistant message with expandable sources
            if not final_results:
                 mensaje = {
                'id': respuesta_id,
                'pregunta': query,
                "role": "assistant",
                "avatar": "🤔",
                "content": respuesta,
                'duration': duration,
                "expansibles": 
                    {
                        "texto": "",
                        "izenburua": "",
                        "data": "",
                        "url": "",
                    }
                   
                ,}
            mensaje = {
                'id': respuesta_id,
                'pregunta': query,
                "role": "assistant",
                "avatar": "🤔",
                "content": respuesta,
                'duration': duration,
                "expansibles": [
                    {
                        "texto": res["text"],
                        "izenburua": res["izenburua"],
                        "data": res["fetxa"],
                        "url": res["url"],
                    }
                    for res in final_results
                ],
            }
            st.session_state.conversaciones[chat_id].append(mensaje)

        
        
        # 8. Display assistant message and top 3 articles
        with st.chat_message("assistant", avatar="🤔"):
            st.markdown(respuesta)
            if final_results:
                st.write('Hona hemen erlazionatutako albiste batzuk: ')

            # Agrupar en bloques de 3
            for i in range(0, len(final_results), 3):
                grupo = final_results[i:i+3]
                cols = st.columns(len(grupo))  # Solo crea las columnas necesarias
                
                for j, fuente in enumerate(grupo):
                    
                    with cols[j]:
                        st.markdown(
                            f"""
                            <div style="
                                background-color: #f8f9fa;
                                padding: 15px;
                                border-radius: 12px;
                                margin: 10px;
                                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                                height: 200px;
                                display: flex;
                                flex-direction: column;
                                justify-content: space-between;
                            ">
                                <h4 style="margin-bottom:10px; color:#333; font-size:16px;">📌 {fuente['izenburua']}</h4>
                                <p style="color:#666; margin-bottom: 8px;">
                                    <b>📅 Argitalpen data:</b><br>{fuente['fetxa']}
                                </p>
                                <a href="{fuente['url']}" target="_blank" style="
                                    display: inline-block;
                                    background-color: #f0f2f6;
                                    color: black;
                                    padding: 6px 12px;
                                    text-decoration: none;
                                    border-radius: 6px;
                                    text-align: center;
                                    font-size: 14px;
                                    font-weight: bold;
                                ">🔗 Ikusi albistea</a>
                            </div>
                            """,
                            unsafe_allow_html=True
                    )           

            
            if respuesta_id not in st.session_state.votaciones:
                col1, col2, col3,col4,col5,col6 = st.columns([5, 1,1,1,1,10])
                with col2:
                    st.button("1 ", key=f"1_{respuesta_id}",
                            on_click=lambda r_id=respuesta_id, msg=mensaje: guardar_valoracion(
                                msg['pregunta'], msg['content'], 1, msg['expansibles'], r_id, msg['duration']
                            ))
                with col3:
                    st.button("2 ", key=f"2_{respuesta_id}",
                            on_click=lambda r_id=respuesta_id, msg=mensaje: guardar_valoracion(
                                msg['pregunta'], msg['content'], 2, msg['expansibles'], r_id, msg['duration']
                            ))
                with col4:
                    st.button("3 ", key=f"3_{respuesta_id}",
                            on_click=lambda r_id=respuesta_id, msg=mensaje: guardar_valoracion(
                                msg['pregunta'], msg['content'], 3, msg['expansibles'], r_id, msg['duration']
                            ))
                with col5:
                    st.button("4 ", key=f"4_{respuesta_id}",
                            on_click=lambda r_id=respuesta_id, msg=mensaje: guardar_valoracion(
                                msg['pregunta'], msg['content'], 4, msg['expansibles'], r_id, msg['duration']
                            ))
                with col1:
                    st.info('Puntuatu emandako erantzuna 1-etik 4-ra')
            else:
                st.info("✅ Eskerrik asko erantzunagatik!")  # reemplaza botones por mensaje local


    # -----------------------------
    # Sidebar: Chat History
    # -----------------------------
    with st.sidebar:
        st.markdown("---")
        st.header("🗂️ Aurreko elkarrizketak")

        for cid, mensajes in st.session_state.conversaciones.items():
            texto_preview = (
                mensajes[0]["content"][:20] + "..." if mensajes else "Berria"
            )
            if st.button(texto_preview, key=f"chat_{cid}"):
                st.session_state.chat_actual = cid
                st.rerun()

        st.markdown("---")
        if st.button("🆕 Elkarrizketa berria"):
            nuevo_id = str(uuid.uuid4())[:8]
            st.session_state.conversaciones[nuevo_id] = []
            st.session_state.chat_actual = nuevo_id
            st.rerun()
