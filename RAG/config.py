from langchain.prompts import PromptTemplate

class config:
    class openai:
        base_url = 'https://router.huggingface.co/v1'
        api_key = os.getenv("HF_API_KEY") #Huggingface API key 
    
    class model:
        name = "HiTZ/Latxa-Llama-3.1-8B-Instruct"
        temperature = 0.7
        max_new_tokens_query = 50
        max_new_tokens_answer = 200
        max_new_tokens_chunk_context = 100
        max_new_tokens_summary = 500


    class embedding:
        embedding_name = "BAAI/bge-m3"

    class milvus:
        COLLECTION_NAME =('argia_albisteak')
        MILVUS_ENDPOINT = ('http://localhost:19530')
        MILVUS_TOKEN = ""
        num_context_chunks = 50  
        num_show_chunks = 6 
        
    class reraking:
        model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"


    class prompts: 

        query_prompt = PromptTemplate.from_template(
            '''
            You are a query rewriter for a semantic search system. 
            Your task is to rewrite the user's query into a natural, expanded version in Basque (Euskara), suitable for semantic similarity search in a news database.

            Guidelines:
            - Keep the meaning of the original query, but make it more descriptive.
            - Include synonyms or related phrases to improve recall.
            - The output must be a single, natural sentence in Basque.
            - Do NOT include any explanation or extra text, only the rewritten query.

            User query: "{original_query}"

                '''
            )
     
        system_prompt = '''

            - You are a RAG assistant.
            - You must always answer using the context provided.
            - If the context is exactly 'ez daukazu informaziorik', say: "Ez daukazu informaziorik galderari erantzuteko."
            - Otherwise, always create an creative answer based on the context.
            - Never say you don't have information if the context contains content.
            - Respond in Basque, in a clear, fluid, and easy-to-understand way.
            - Always write at least one well-structured paragraph, combining the most relevant information from the context.
            - Make the answer engaging, like a short informative article, not just a single sentence.

            ''' 

        prompt_user = '''

        Hona hemen dokumentuetatik ateratako informazioa:

        <context>
        {context}
        </context>

        Erabili testuinguru hau soilik erantzuteko. 
        Ez asmatu informazio berririk eta ez baztertu emandakoa.

        Orain erantzun galderari:

        <question>
        {question}
        </question>
        Erabiltzaileak fetxarenbat esaten badizu egin kasu berezia.
        Erantzuna:
        
        '''

       

       
