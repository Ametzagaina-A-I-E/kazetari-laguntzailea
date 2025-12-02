import os

class config:
    class api:
        api_url = "https://router.huggingface.co/v1"
        api_key = os.getenv("HF_API_KEY") #Huggingface API key 
    class llm: 
        model_name = "HiTZ/Latxa-Llama-3.1-8B-Instruct"

    class embedding:
        embedding_name = "BAAI/bge-small-en-v1.5"

    class prompt:
        
        prompt_sys_generate_izenburua = """
            You will receive a Basque news article.
            Your task: Write a captivating and memorable title that grabs attention while summarizing the article's main point.
            Only return the title. Avoid extra words or commentary.

        """

        prompt_sys_generate_sarrera = """
            You will receive a Basque news article.  
            Your task: Write a concise summary in Basque (2 to 4 sentences) that captures the essential content.  
            Do NOT include any explanation, only return the summary text.  

        """

        prompt_sys_generate_etiketak = """
            You will receive a Basque news article.  
            Your task: Choose betweem 3-5 relevant labels (topics) that best describe the article in basque.  
            Return them strictly in JSON array format, like this: ["label1", "label2", "label3"].  
            Do NOT add any explanation.  

        """
       
        prompt_sys_hobekuntza = f"""
        You are a spelling corrector specialized in Basque.

        Instructions:
        - Detect ONLY real spelling mistakes (orthographic errors: misspelled words, wrong characters, extra/missing letters).
        - DO NOT correct grammar, style, wording, morphology, or phrasing.
        - Do NOT output words that are already correct.
        - Include ONLY words where the corrected form is different from the original.
        - DO NOT add words or change sentence structure.
        - Keep unchanged words and sentences intact if they are already correct.

        Output ONLY the corrections, in strict JSON format:
        [
            {{"original": "wrong_word", "corrected": "right_word"}},
            ...
        ]

        If the text has no spelling errors, return exactly: []

        Example:
        Text: "This is an exmple with a speling error."
        Output:
        [
            {{"original": "exmple", "corrected": "example"}},
            {{"original": "speling", "corrected": "spelling"}}
        ]
        """

