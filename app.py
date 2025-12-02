import streamlit as st
from RAG.RAG import hemeroteka
from RAG.RAG_server import hemeroteka_server
from albiste_analisia.text_generator import textu_generatzailea
from albiste_analisia.text_generator_server import textu_generatzailea_servers
from audio_text.whisper_eu import audio_text
import torch
from load_models import free_all_cached
import sqlite3


import argparse

# Configurar argumentos de línea de comandos
parser = argparse.ArgumentParser(description="Ejecutar la aplicación ARGIA.")
parser.add_argument('--local', type=str, default='true', help="Ejecutar en modo local (true/false). Por defecto es true.")
args = parser.parse_args()

# Convertir el argumento a booleano
LOCAL = args.local.lower() == 'true'
# Clear GPU cache to avoid memory issues
torch.cuda.empty_cache()


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="ARGIA",       # Title displayed in the browser tab
    page_icon="✨",           # Icon displayed in the browser tab
    layout="wide",            # Use wide layout for better visualization
    initial_sidebar_state="expanded"  # Sidebar starts expanded
)

# -----------------------------
# Session State Initialization
# -----------------------------
# Store the active section in session state to maintain UI state across interactions
if 'seccion_activa' not in st.session_state:
    st.session_state.seccion_activa = "🏠 Hasiera"  # Default section: Home

# -----------------------------
# Sidebar Navigation
# -----------------------------
# Sidebar buttons to switch between different sections
with st.sidebar:
    if st.button("🏠 Hasiera"):
        st.session_state.seccion_activa = "🏠 Hasiera"
    if st.button("📰 Hemeroteka"):
        st.session_state.seccion_activa = "📰 Hemeroteka"
    if st.button("✍️ Textu generatzailea"):
        st.session_state.seccion_activa = "✍️ Textu generatzailea"
    if st.button("🎙️ Transkripzioak"):
        st.session_state.seccion_activa = "🎙️ Transkripzioak"


# -----------------------------
# Home Section Function
# -----------------------------
def mostrar_hasiera():
    """Display the home section with an introduction and instructions."""
    
    st.image("img/argia.png", width=300)
    
    st.markdown("## 🌟 Ongi etorri Argia Bilatzaile Digitalera")

    st.write(
        """
        📚 Web aplikazio hau **Argiako aldizkariaren artxibo digitala esploratzeko tresna** da.  
        Hainbat hamarkadatan jasotako albiste, artikulu eta testuak modu arin, bisual eta erabilgarrian eskuratzeko aukera eskaintzen dizugu.  
        Ezkerreko menuan hainbat atal aurkituko dituzu — bakoitza esperimentatzeko eta probatzeko modukoa:
        """
    )

    
    # ---- Tarjetas en columnas ----
    col1, col2, col3 = st.columns(3)

    card_style = """
        background-color:#f9f9f9;
        padding:20px;
        border-radius:15px;
        box-shadow:2px 2px 8px rgba(0,0,0,0.1);
        text-align:left;
        min-height:400px;
        display:flex;
        flex-direction:column;
        justify-content:flex-start;
    """

    with col1:
        st.markdown(
            f"""
            <div style="{card_style}">
                <h3>📰 Hemeroteka</h3>
                <ul>
                    <li>Zure galderarekin erlazionatutako albisteak bilatzen ditu, eta albiste horien esplikazio txiki bat ematen dizu.</li>
                    <li>Ez da konbertsio bat, galderari erantzuten dio baina ez du konbentziorik ematen.</li>
                    <li>Albisterik ez badu bilatzen ez dizu albisterik bidaliko.</li>
                    <li>Galdera bakoitzaren erantzuna 1etik 4ra baloratzea eskertuko genuke.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div style="{card_style}">
                <h3>✍️ Testu generatzailea</h3>
                <ul>
                    <li>Zati honetan zuek artikulu bat txertatu eta honek izenburu posible bat, sarrera posible bat eta etiketa posibleak ematen dizkizu.</li>
                    <li>Ez bazaizu gustatzen zatiren bat, berriz beste bat sortzeko "Berriz sortu" botoiari eman diezaiokezu.</li>
                    <li>Zati bakoitza 1etik 4ra baloratzea eskertuko genuke.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div style="{card_style}">
                <h3>🎙️ Transkripzioak</h3>
                <ul>
                    <li>Euskarazko audioak transkribitzen ditu.</li>
                    <li>Denbora gehiago behar da audio luzeagoentzat (adibidez, 1 min audio gutxienez 1 min tardatzen du).</li>
                    <li>Transkripzioak hitzegiten duten pertsona desberdinak bereizten saiatzen da.</li>
                    <li>Transkripzioa ez da guztiz zuzena.</li>
                    <li>Ondoren transkripzio horren laburpena eta puntu garrantzitsuak ateratzen ditu.</li>
                    <li>Horiek deskargatu ditzakezu, nahi duzun informazioa gehituz.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True
        )

    # ---- Feedback expandible ----
    st.markdown('---')
    with st.expander("💡 Arazo edo proposamen berriren bat baduzu, idatzi hemen"):
        feedback = st.text_area("Zure iruzkina idatzi:", placeholder="Adibidez: arazo teknikoa, hobekuntza bat, ideia berria...")
        if st.button("📩 Bidali feedback-a"):
            if feedback.strip():
                conn = sqlite3.connect("Proposamenak.db")
                c = conn.cursor()

                # Taula sortu (bakarrik behin sortzen da)
                c.execute("""
                CREATE TABLE IF NOT EXISTS Proposamenak (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    comment TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """)
                
                c.execute("INSERT INTO Proposamenak (comment) VALUES (?)", (feedback,))
                conn.commit()
                conn.close()
                st.success("Eskerrik asko zure iruzkinagatik! 🙌")
                # Hemen gorde feedback-a fitxategi batean edo bidali backend-era
            else:
                st.warning("Mesedez, idatzi zerbait bidali aurretik.")



# -----------------------------
# Main Content Loader
# -----------------------------
# Show a spinner while loading the selected section
if LOCAL:
    if st.session_state.seccion_activa == "🏠 Hasiera":
        mostrar_hasiera()
    elif st.session_state.seccion_activa == "📰 Hemeroteka":
        hemeroteka()
    elif st.session_state.seccion_activa == "✍️ Textu generatzailea":
        textu_generatzailea()
    elif st.session_state.seccion_activa == "🎙️ Transkripzioak":
        #free_all_cached()
        audio_text()

else:
    if st.session_state.seccion_activa == "🏠 Hasiera":
        mostrar_hasiera()
    elif st.session_state.seccion_activa == "📰 Hemeroteka":
        hemeroteka_server()
    elif st.session_state.seccion_activa == "✍️ Textu generatzailea":
        textu_generatzailea_servers()
    elif st.session_state.seccion_activa == "🎙️ Transkripzioak":
        st.markdown('tresna hau ez dago erabilgarri lokalki ez bada')
