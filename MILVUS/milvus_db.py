import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_models import emb_text_bge
from MILVUS.milvus_utils import get_milvus_client, create_collection
from sentence_transformers import SentenceTransformer
import json
from MILVUS.config import config
import time

# ---------------------------------------------------------
# Initialize Milvus client and collection
# ---------------------------------------------------------
start_time = time.time()

model = SentenceTransformer(config.embedding.embedding_name,device = 'cuda')
milvus_client = get_milvus_client(uri=config.milvus.MILVUS_ENDPOINT, token=config.milvus.MILVUS_TOKEN)
dim = model.get_sentence_embedding_dimension()
#create_collection(milvus_client=milvus_client, collection_name=config.milvus.COLLECTION_NAME, dim=dim, drop_old=False)

# ---------------------------------------------------------
# Insert documents into Milvus
# ---------------------------------------------------------
count = 0

with open("/home/ubuntu/ARGIA/MILVUS/todos_chunks2.json", "r", encoding="utf-8") as f:
    
    for line in f: 
        data = []
        try:
            doc = json.loads(line)
            doc_id = doc["metadata"]["id"]
            url = doc["metadata"]["url"]
            izenburua = doc["metadata"]["izenburua"]
            entradilla = doc["metadata"]["entradilla"]
            fetxa = doc['metadata']['fecha']
            text = doc["text"]
            
        
            vector = emb_text_bge( text,is_query=False)

            # Prepare data for Milvus
            data = [{
                "vector": vector,
                "text": text,
                "id_albiste": doc_id,
                "url": url,
                "izenburua": izenburua,
                "entradilla": entradilla,
                "fetxa": fetxa
            }]

            count += 1

        except Exception as e:
            print(f"Skipping doc ID {doc.get('id', 'unknown')} ({doc_id}) por error en embedding:\n{e}")
            continue

        try:
            milvus_client.insert(collection_name=config.milvus.COLLECTION_NAME, data=data)
        except Exception as milvus_error:
            print(f"Errorea -> {doc_id}: {milvus_error}")
        else:
            print("Gehituta:", doc_id)

    end_time = time.time()

    duration = end_time - start_time
    print('Denbora guztira: ', duration)
