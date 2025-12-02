import os
import torchaudio
from audio_text.config import config
import streamlit as st
import tempfile
import time
from load_models import load_asr_model, load_diarization_model, get_llm, generate_text, run_model_safe
import sqlite3
import torch, gc


def format_time(seconds: float) -> str:
    """Convierte segundos a mm:ss"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

def clean_repetitions(text: str, max_repeat: int = 2) -> str:
    """
    Elimina repeticiones excesivas de palabras consecutivas.
    max_repeat = máximo número de repeticiones permitidas
    """
    words = text.split()
    cleaned = []
    count = 0

    for i, word in enumerate(words):
        if i > 0 and word.lower() == words[i-1].lower():
            count += 1
            if count < max_repeat:
                cleaned.append(word)
        else:
            count = 0
            cleaned.append(word)

    return " ".join(cleaned)
def free_gpu_memory(*models):
    for m in models:
        try:
            del m
        except:
            pass
    gc.collect()
    torch.cuda.empty_cache()
def guardar_valoracion(trankripzioa, laburpena, puntuak, balorazioa, duration, audio_path):
        if isinstance(trankripzioa, list):
            trankripzioa = "\n".join(
                [f"{item['speaker']} ({item['start_time']:.2f}-{item['end_time']:.2f}) → {item['text']}" for item in trankripzioa]
            )
        if isinstance(laburpena, list):
            laburpena = " ".join(laburpena)
        if isinstance(puntuak, list):
            puntuak = "\n".join(puntuak)
        conn = sqlite3.connect("Balorazioak_AUDIO.db")
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trankripzioa TEXT,
                laburpena TEXT,
                puntuak TEXT, 
                balorazioa INTEGER,
                audio_path TEXT,
                denbora REAL
            )
        """)
        cursor.execute("""
            INSERT INTO feedback (trankripzioa, laburpena, puntuak, balorazioa, audio_path, denbora)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (trankripzioa, laburpena, puntuak, balorazioa,audio_path, duration))
        conn.commit()
        conn.close()
        st.session_state.votaciones[id] = balorazioa
import tempfile




def audio_text():
    st.set_page_config(page_title="🎤 Transkripzioa eta Elkarrizketa", layout="wide")
    st.title("🎤 Hizlarien bereizketa duen transkripzioa")

    # Cargar LLM si no está en session_state
    if "model" not in st.session_state or "tokenizer" not in st.session_state:
        st.session_state.tokenizer, st.session_state.model = get_llm(config.llm.model_name, True)
        #st.session_state.tokenizer, st.session_state.model = (None,None)
    tokenizer = st.session_state.tokenizer
    model = st.session_state.model

    #####################################
    # Inicializar estados
    #####################################
    if "final_lines" not in st.session_state:
        st.session_state.final_lines = None
    if "resum" not in st.session_state:
        st.session_state.resum = None
    if "parts" not in st.session_state:
        st.session_state.parts = None
    if "votaciones" not in st.session_state:
        st.session_state.votaciones = {}
        
    st.session_state.duration = 0

    #####################################
    # Subir archivo
    #####################################
    uploaded_file = st.file_uploader("📂 Igo audio edo bideo fitxategia", type=["mp3", "wav", "mp4", "m4a", "ogg"])
    if uploaded_file is not None:
        if ("last_file_name" not in st.session_state) or (st.session_state.last_file_name != uploaded_file.name):
            # resetear estado si es un archivo distinto
            st.session_state.final_lines = None
            st.session_state.resum = None
            st.session_state.parts = None
            st.session_state.last_file_name = uploaded_file.name
            st.session_state.votaciones = {}
    if uploaded_file is not None and st.session_state.final_lines is None:
        start_time = time.time()
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmpfile:
            tmpfile.write(uploaded_file.read())
            st.session_state.audio_path = tmpfile.name

        if st.session_state.audio_path.endswith((".mp4", ".m4a")):
            st.video(st.session_state.audio_path)
        else:
            st.audio(st.session_state.audio_path)

        with st.spinner("⏳ Prozesatzen (hizlarien bereizketa + transkripzioa)..."):
   
            def run_asr_diarization(audio_path):
                # 1️⃣ Cargar y ejecutar ASR
                asr_pipeline = load_asr_model(config.whisper.WHISPER_ID)
                asr_result = asr_pipeline(audio_path)
                
                # 2️⃣ Cargar y ejecutar diarization
                diarization_pipeline = load_diarization_model(config.whisper.HF_TOKEN)

                # 3️⃣ Liberar memoria GPU
                
                free_gpu_memory( asr_pipeline, diarization_pipeline)
                
                # 4️⃣ Devolver resultados juntos
                return asr_result, diarization_result
            
            asr_result, diarization = run_model_safe(run_asr_diarization, st.session_state.audio_path)

            chunks = asr_result.get("chunks", [])

           
            info = torchaudio.info(st.session_state.audio_path)
            audio_duration = info.num_frames / info.sample_rate

            # 3️⃣ Corregir timestamps
            for i, c in enumerate(chunks):
                start, end = c["timestamp"]
                if start is None:
                    start = 0.0 if i == 0 else (chunks[i-1]["timestamp"][1] + 0.001)
                if end is None:
                    end = audio_duration if i == len(chunks)-1 else (chunks[i+1]["timestamp"][0] - 0.001)
                c["timestamp"] = (float(start), float(end))

            # 4️⃣ Merge por speaker
            segments_info = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments_info.append({"start": turn.start, "end": turn.end, "speaker": speaker})
            segments_info.sort(key=lambda x: x["start"])
            chunks.sort(key=lambda x: x["timestamp"][0])

            final_lines = []
            current_speaker, current_text, block_start, block_end = None, [], None, None
            for c in chunks:
                c_start, c_end = c["timestamp"]
                if not c["text"].strip():
                    continue
                # match speaker
                best_speaker, max_overlap = None, 0
                for seg in segments_info:
                    overlap_start = max(c_start, seg["start"] - 0.5)
                    overlap_end = min(c_end, seg["end"] + 0.5)
                    overlap = max(0, overlap_end - overlap_start)
                    if overlap > max_overlap:
                        max_overlap, best_speaker = overlap, seg["speaker"]
                if best_speaker != current_speaker:
                    if current_speaker and current_text:
                        b = []
                        
                        for a in current_text:
                            a = clean_repetitions(text=a, max_repeat=3 )
                            b.append(a)
                        final_lines.append({
                            "speaker": current_speaker,
                            "text": " ".join(b),
                            "start_time": block_start,
                            "end_time": block_end
                        })
                    current_speaker, current_text, block_start = best_speaker, [], c_start
                block_end = c_end
                current_text.append(c["text"].strip())
            if current_speaker and current_text:
                
                b = []
                       
                for a in current_text:
                    a = clean_repetitions(text=a, max_repeat=3 )
                    b.append(a)
                final_lines.append({
                    "speaker": current_speaker,
                    "text": " ".join(b),
                    "start_time": block_start,
                    "end_time": block_end
                })

            st.session_state.final_lines = final_lines

            #####################################
            # 5️⃣ Generar resumen y partes automáticamente
            #####################################
            transcription_text = " ".join([line["text"] for line in final_lines])

            def generate_field(sys_prompt, text):
                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": text}
                ]
                return generate_text(
                    model=model,
                    tokenizer=tokenizer,
                    messages=messages,
                    use_chat_template=True,
                    max_tokens=1000,
                    temperature=0.5,
                    do_sample=True
                ).strip()

            st.session_state.resum = generate_field(config.prompt.prompt_resum, transcription_text)
            st.session_state.parts = generate_field(config.prompt.pormpt_parts, transcription_text)
        end_time = time.time()
        st.session_state.duration = end_time - start_time
    #####################################
    # Mostrar resultados
    #####################################

    if st.session_state.final_lines:
        #####################################
        # Exportación flexible
        #####################################
        col_1,col3,col_2 = st.columns([3,0.5,1.5])
        with col_1:
            st.subheader("📥 Deskargatu TXT aukeratuta edukiekin")
            col1, col2, col3, col4 = st.columns([2,2,2,5])
            with col1:
                include_transcription = st.checkbox("Transkripzio osoa", value=True)
            with col2:
                include_summary = st.checkbox("Laburpena", value=True)
            with col3:
                include_parts = st.checkbox("Puntu garrantzitsuak", value=True)

            txt_sections = []
            if include_transcription:
                export_lines = [
                    f"{line['speaker']} ---- ({format_time(line['start_time'])}s, {format_time(line['end_time'])}s) --→ {line['text']}"
                    for line in st.session_state.final_lines
                ]
                txt_sections.append("### TRANSKRIPZIOA ###\n\n" + "\n".join(export_lines))
            if include_summary:
                txt_sections.append("### LABURPENA ###\n\n" + st.session_state.resum)
            if include_parts:
                txt_sections.append("### PUNTU GARRANTZITSUAK ###\n\n" + st.session_state.parts)

            txt_str = "\n\n".join(txt_sections)
            with col4:
                st.download_button("📥 Deskargatu TXT", data=txt_str, file_name="transkripzioa.txt", mime="text/plain")
        with col_2:
            st.markdown('Baloratu jasotako erantzuna:')
            if not st.session_state.votaciones:
                col1, col2, col3,col4 = st.columns([1,1,1,1])
                with col1:
                    st.button("1 ", key=1,
                         on_click=lambda: guardar_valoracion(st.session_state.final_lines ,st.session_state.resum, st.session_state.parts,1, st.session_state.duration,uploaded_file.name))
                with col2:
                    st.button("2 ", key=2,
                         on_click=lambda: guardar_valoracion(st.session_state.final_lines ,st.session_state.resum, st.session_state.parts,2, st.session_state.duration,uploaded_file.name))
                with col3:
                    st.button("3 ", key=3,
                         on_click=lambda: guardar_valoracion(st.session_state.final_lines ,st.session_state.resum, st.session_state.parts,3, st.session_state.duration,uploaded_file.name))
                with col4:
                    st.button("4 ", key=4,
                         on_click=lambda: guardar_valoracion(st.session_state.final_lines ,st.session_state.resum, st.session_state.parts,4, st.session_state.duration,uploaded_file.name))
            else:
                st.info("✅ Eskerrik asko erantzunagatik!") 

        
        col_trans,col2, col_res = st.columns([3,0.5,1.5])    
        with col_trans:
            st.subheader("💬 Elkarrizketa transkripzioa")
            for line in st.session_state.final_lines:
                speaker = line["speaker"]
                start = format_time(line["start_time"])
                end = format_time(line["end_time"])
                text = line["text"]
                st.markdown(f"**{speaker}** · *{start}s → {end}s* → {text}")

        
        with col2:
            st.markdown(
                "<div style='border-left: 1px solid #ccc; height: 100%;'></div>",
                unsafe_allow_html=True
            )
        with col_res:
            st.subheader("📝 Laburpena")
            st.write(st.session_state.resum)

            st.subheader("📌 Puntu garrantzitsuak")
            st.write(st.session_state.parts)

        
   