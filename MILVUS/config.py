from langchain.prompts import PromptTemplate
import os

class config:
    class milvus:
        COLLECTION_NAME =('argia_albisteak')
        MILVUS_ENDPOINT = ('http://localhost:19530')
        MILVUS_TOKEN = ""
        num_context_chunks = 50  # Bilatzerakoan hautatu nahi ditugun chunk kopurua
        num_show_chunks = 6 # Interfazean ikusi nahi ditugun chunk kopurua
        file_json = "chunk-2025_45666.json"

    class embedding:
        embedding_name = "BAAI/bge-m3"
    class llm:
        name = "HiTZ/Latxa-Llama-3.1-8B-Instruct"

    class chunks:
        CHUNK_SIZE = 350
        CHUNK_OVERLAP = 40
        CONTEXTUALIZE_CHUNKS = True # Datuak bektorizatzerakoan llm erabili edo ez
        max_new_tokens_summary = 500
        max_new_tokens_chunk_context = 100
    class mysql:
        query =  "SELECT albiste_id, izenburua, url_izena, entradilla, alta_data, albistea FROM albistea"
        database_config = {
            'host': os.getenv("DB_HOST", "localhost"),
            'user': os.getenv("DB_USER", "root"),
            'password': os.getenv("DB_PASSWORD", ""),
            'database': os.getenv("DB_NAME", "")
        }
        mysql_lite = 'argia_albistea_local.db'


    class prompts: 
        insert_prompt = PromptTemplate.from_template(
            '''
            Dokumentu honetatik ateratako zati bat azpian duzu. Sortu 512 token baino gutxiagoko azalpen ulerterraza, zati horrek dokumentu osoaren barruan duen zentzua argitzeko. Ez errepikatu zatia, baizik eta azaldu zer esaten duen eta zer funtzio betetzen duen dokumentuaren testuinguruan.

            Dokumentua: {document}

            Zatia: {chunk}

            Azalpena:
            '''
        )

        resumen_prompt = PromptTemplate.from_template(
            ''' I will give you a news article. From this article, you must write a concise summary (2 to 4 sentences) that captures the essential content
            Here is the article:
            {albistea} 
            Output query:
            ''' 
        )