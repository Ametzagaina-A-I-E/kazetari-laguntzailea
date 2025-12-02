import mysql.connector
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_models import get_llm, generate_text
from MILVUS.config import config
import time
import datetime
import sqlite3
from langchain.schema import Document 
# -----------------------------
# Get text from MySQL
# -----------------------------
def get_text_mysql(config, query) -> list:
    """
    Executes a SQL query and returns a list of documents with metadata.
    """
    text_dict = []
    conn = mysql.connector.connect(**config)
    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    
    for row in rows:
        albiste_id, izenburua, url_izena, entradilla, alta_data, albistea = row

        # Normalize URL
        final_url = ""
        if url_izena and url_izena.strip() != "":
            url = url_izena.strip().strip("'")
            final_url = url if url.startswith("https:") else f"https://www.argia.eus/albistea/{url}"

        # Normalize date
        if isinstance(alta_data, datetime.datetime):
            alta_data = alta_data.strftime('%Y-%m-%d')

        text_dict.append({
            "document": Document(page_content=albistea, metadata={"id": albiste_id}),
            "url": final_url,
            "izenburua": izenburua,
            "entradilla": entradilla,
            "fecha": alta_data
        })

    return text_dict
 # Asegúrate de tener esta importación si usas LangChain

def get_text_sqlite(db_path, query) -> list:
    """
    Ejecuta una consulta SQL en una base de datos SQLite y devuelve una lista de documentos con metadatos.
    """
    text_dict = []
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    for row in rows:
        albiste_id, izenburua, url_izena, entradilla, alta_data, albistea = row

        # Normalizar URL
        final_url = ""
        if url_izena and url_izena.strip() != "":
            url = url_izena.strip().strip("'")
            final_url = url if url.startswith("https:") else f"https://www.argia.eus/albistea/{url}"
        print(type(alta_data))
        print(alta_data)
        # Normalizar fecha
        if alta_data is None:
            alta_data= None
        elif isinstance(alta_data, datetime.datetime):
            alta_data = alta_data.strftime('%Y-%m-%d')
        else: 
            alta_data = datetime.datetime.strptime(alta_data, "%Y-%m-%d %H:%M:%S").date()
        text_dict.append({
            "document": Document(page_content=albistea, metadata={"id": albiste_id}),
            "url": final_url,
            "izenburua": izenburua,
            "entradilla": entradilla,
            "fecha": alta_data
        })

    return text_dict
# -----------------------------
# Split text into chunks
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.chunks.CHUNK_SIZE,
    chunk_overlap=config.chunks.CHUNK_OVERLAP,
)

def create_chunks(document: Document) -> list:
    """Splits a Document into smaller chunks using LangChain splitter."""
    return text_splitter.split_documents([document])

# -----------------------------
# Generate contextualized chunk
# -----------------------------
def generate_context(tokenizer, model, document: str, chunk: str) -> str:
    """
    Generates a context summary for a chunk based on the full document.
    """
    prompt_text = config.prompts.insert_prompt.format_prompt(
        document=document, chunk=chunk
    ).to_string()

    messages = [
        {"role": "system", "content": "Generate a brief context for the following chunk:"},
        {"role": "user", "content": prompt_text}
    ]

    return generate_text(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        use_chat_template=True,
        max_tokens=config.chunks.max_new_tokens_chunk_context,
        temperature=0,
        do_sample=False
    )

# -----------------------------
# Generate resumen
# -----------------------------
def generar_resumen(tokenizer, model, document: Document) -> str:
    """
    Generates a summary for the given document using the LLM.
    """
    prompt_text = config.prompts.resumen_prompt.format_prompt(
        albistea=document.page_content
    ).to_string()

    messages = [
        {"role": "system", "content": "Generate a summary for the news article."},
        {"role": "user", "content": prompt_text}
    ]

    return generate_text(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        use_chat_template=True,
        max_tokens=config.chunks.max_new_tokens_summary,
        temperature=0,
        do_sample=False
    )

# -----------------------------
# Save docs to JSONL
# -----------------------------
def save_docs_to_json(docs, filename):
    with open(filename, 'a', encoding='utf-8') as jsonl_file:
        for doc in docs:
            jsonl_file.write(json.dumps(doc, ensure_ascii=False, default=str) + '\n')

# -----------------------------
# Main logic with modes
# -----------------------------
def process_documents(mode="chunks"):
    """
    mode options:
        - "chunks": Split documents into chunks (with or without context)
        - "resumen": Generate summaries for full documents
    """
    # Load data from MySQL
    # text_sql = get_text_mysql(config.mysql.database_config, config.mysql.query) # get text from MySQL
    text_sql = get_text_sqlite(config.mysql.mysql_lite,config.mysql.query)

    # Load LLM
    tokenizer, model = get_llm(config.llm.name,False)

    for albiste in text_sql:
        document = albiste['document']
        url = albiste['url']
        izenburua = albiste['izenburua']
        entradilla = albiste['entradilla']
        fecha = albiste['fecha']

        start_time = time.perf_counter()
        start_dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"▶️ Noticia {document.metadata['id']} - inicio {start_dt}")
        try:
            if mode == "chunks":
                # Split into chunks
                chunk_list = create_chunks(document)

                contextual_chunks = []
                if config.chunks.CONTEXTUALIZE_CHUNKS:
                    for chunk in chunk_list:
                        context = generate_context(tokenizer, model, document.page_content, chunk.page_content)
                        chunk_with_context = f"{context}\n\n{chunk.page_content}"
                        contextual_chunks.append(Document(page_content=chunk_with_context, metadata=chunk.metadata))
                else:
                    contextual_chunks = chunk_list

                # Save chunks
                chunks_to_save = []
                for chunk in contextual_chunks:
                    chunks_to_save.append({
                        "metadata": {
                            "id": chunk.metadata["id"],
                            "url": url,
                            "izenburua": izenburua,
                            "entradilla": entradilla,
                            "fecha": fecha
                        },
                        "text": chunk.page_content
                    })

                save_docs_to_json(chunks_to_save, config.milvus.file_json)

            elif mode == "resumen":
                # Generate resumen for full document
                resumen = generar_resumen(tokenizer, model, document)
                resumen_con_meta = {
                    "metadata": {
                        "id": document.metadata["id"],
                        "url": url,
                        "izenburua": izenburua,
                        "entradilla": entradilla,
                        "fecha": fecha
                    },
                    "resumen": resumen
                }
                save_docs_to_json([resumen_con_meta], config.milvus.file_json)
            end_time = time.perf_counter()
            print(f"✅ Noticia {document.metadata['id']} procesada en {end_time - start_time:.2f} segundos")
        except Exception as e:
            print(f"❌ Error processing document {document.metadata['id']}: {e}")
            continue

# Example of execution:
if __name__ == "__main__":
    # Choose mode: "chunks" or "resumen"
    process_documents(mode="chunks")
    # process_documents(mode="resumen")
