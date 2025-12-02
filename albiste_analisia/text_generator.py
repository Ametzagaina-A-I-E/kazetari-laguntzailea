import streamlit as st
import sqlite3
import re
import time
import hunspell
from RAG.config import config
from load_models import generate_text, get_llm
from albiste_analisia.config import config as conf_text    


def textu_generatzailea():
    if "model" not in st.session_state or "tokenizer" not in st.session_state:
        st.session_state.tokenizer, st.session_state.model = get_llm(conf_text.llm.model_name, True)

    tokenizer = st.session_state.tokenizer
    model = st.session_state.model

    st.set_page_config(layout="wide")
    st.image("img/argia.png", width=300)

    if "albiste_gordeak" not in st.session_state:
        st.session_state.albiste_gordeak = []

    col1, col2 = st.columns([1, 2])
    with col1:
        albistea = st.text_area("✍️ Albistea idatzi:", height=700, key="albistea_text")
        st.caption(f"Karaktereak: {len(albistea)}")

    with col2:

        # 🔹 Función genérica para generar texto con el LLM
        def generate_field(sys_prompt, albistea, model, tokenizer, max_tokens=1000, temperature=0.8, do_sample=True):
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": albistea}
            ]
           
            value = generate_text(
                model=model,
                tokenizer=tokenizer,
                messages=messages,
                use_chat_template=True,
                max_tokens=max_tokens,
                temperature=temperature,
                do_sample=do_sample
            )
            return value.strip()

        # 🔹 Guardar valoración en la base de datos
        def guardar_valoracion(campo, valor_texto, valoracion, row_id, duration, texto_original):
            conn = sqlite3.connect("Balorazioak_TEXT.db")
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    texto TEXT UNIQUE,
                    izenburua TEXT, izenburua_balorazioa INTEGER,
                    sarrera TEXT, sarrera_balorazioa INTEGER,
                    etiketak TEXT, etiketak_balorazioa INTEGER,
                    hobekuntza TEXT, hobekuntza_balorazioa INTEGER,
                    denbora REAL
                )
            """)
            cursor.execute("SELECT id FROM feedback WHERE texto = ?", (texto_original,))
            result = cursor.fetchone()
            if result:
                row_id = result[0]
                cursor.execute(f"""
                    UPDATE feedback
                    SET {campo} = ?, {campo}_balorazioa = ?, denbora = ?
                    WHERE id = ?
                """, (valor_texto, valoracion, duration, row_id))
            else:
                cursor.execute(f"""
                    INSERT INTO feedback (texto, {campo}, {campo}_balorazioa, denbora)
                    VALUES (?, ?, ?, ?)
                """, (texto_original, valor_texto, valoracion, duration))
                row_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return row_id

        # 🔹 Extraer correcciones de ortografía
        def extract_corrections(text):
            pattern = r'"original"\s*:\s*"([^"]+)"\s*,\s*"corrected"\s*:\s*"([^"]+)"'
            matches = re.findall(pattern, text, re.DOTALL)
            return [{"original": o, "corrected": c} for o, c in matches]

        if albistea:
            with st.spinner('Erantzuna sortzen...'):
                if "last_input" not in st.session_state or st.session_state.last_input != albistea:
                    st.session_state.last_input = albistea
                    st.session_state.votaciones_text = set()
                    st.session_state.row_id = None

                    start_time = time.time()

                    # 🔹 Generar campos principales
                    izenburua = generate_field(
                        conf_text.prompt.prompt_sys_generate_izenburua,
                        albistea, model, tokenizer,
                        max_tokens=10000, temperature=0.8, do_sample=True
                    )

                    sarrera = generate_field(
                        conf_text.prompt.prompt_sys_generate_sarrera,
                        albistea, model, tokenizer,
                        max_tokens=10000, temperature=0.8, do_sample=True
                    )

                    etiketak = generate_field(
                        conf_text.prompt.prompt_sys_generate_etiketak,
                        albistea, model, tokenizer,
                        max_tokens=10000, temperature=0.7, do_sample=False
                    )

                    end_time = time.time()
                    duration = end_time - start_time

                    st.session_state.generated_data = {
                        "izenburua": izenburua,
                        "sarrera": sarrera,
                        "etiketak": etiketak,
                        "duration": duration
                    }

                if "votaciones_text" not in st.session_state:
                    st.session_state.votaciones_text = set()
                if "row_id" not in st.session_state:
                    st.session_state.row_id = None

                # 🔹 Obtener datos generados
                data = st.session_state.generated_data
                izenburua = data["izenburua"]
                sarrera = data["sarrera"]
                etiketak = data["etiketak"]
                duration = data["duration"]

            # 🔹 Mostrar campos principales con valoración
            for campo, valor, icono, sys_prompt, max_tokens in [
                ("izenburua", izenburua, "📌 Izenburua", conf_text.prompt.prompt_sys_generate_izenburua, 10000),
                ("sarrera", sarrera, "🧾 Sarrera", conf_text.prompt.prompt_sys_generate_sarrera, 10000),
                ("etiketak", etiketak, "🏷️ Etiketak", conf_text.prompt.prompt_sys_generate_etiketak, 10000),
            ]:
                col1, col2, col3, col4, col5, col6, col7 = st.columns([4, 3, 1, 1, 1, 4, 3])
                with col1:
                    st.subheader(icono)
                with col2:
                    st.write("Baloratu emandako erantzuna:")
                if campo not in st.session_state.votaciones_text:
                    with col3:
                        if st.button("1", key=f"{campo}_1"):
                            st.session_state.row_id = guardar_valoracion(
                                campo, valor, 1, st.session_state.row_id, duration, albistea
                            )
                            st.session_state.votaciones_text.add(campo)
                    with col4:
                        if st.button("2", key=f"{campo}_2"):
                            st.session_state.row_id = guardar_valoracion(
                                campo, valor, 2, st.session_state.row_id, duration, albistea
                            )
                            st.session_state.votaciones_text.add(campo)
                    with col5:
                        if st.button("3", key=f"{campo}_3"):
                            st.session_state.row_id = guardar_valoracion(
                                campo, valor, 3, st.session_state.row_id, duration, albistea
                            )
                            st.session_state.votaciones_text.add(campo)
                    with col6:
                        if st.button("4", key=f"{campo}_4"):
                            st.session_state.row_id = guardar_valoracion(
                                campo, valor, 4, st.session_state.row_id, duration, albistea
                            )
                            st.session_state.votaciones_text.add(campo)
                    
                else:
                    with col3:
                        st.empty()
                    with col4:
                        st.empty()
                    with col5:
                        st.empty()
                    
                    with col6:
                        st.success("✅ Baloratuta")
                with col7:
                    if st.button("🔄 Berriz sortu", key=f"{campo}_regen"):
                        
                        new_value = generate_field(
                            sys_prompt, albistea, model, tokenizer, max_tokens
                        )
                        st.session_state.generated_data[campo] = new_value
                        valor = new_value
                        
                st.markdown(valor)
                st.markdown("---")
            
        