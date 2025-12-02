import os

class config:
    class whisper:
        WHISPER_ID = "HiTZ/whisper-large-v3-eu"
        HF_TOKEN = os.getenv("HF_TOKEN")

    class llm:
        model_name = "HiTZ/Latxa-Llama-3.1-8B-Instruct"

    class prompt:
        prompt_resum = f"""
        You are an assistant that summarizes conversations. 
        The conversation is in Basque. 
        Please read the following transcription (in Basque) and generate a concise, clear summary in Basque:
        """

        # pormpt_parts = f"""
        # You are an assistant that extracts key points and headlines from conversations.
        # The conversation is in Basque.
        # Please read the following transcription (in Basque) and provide a list of the main topics, important points, or section headlines, in Basque

        # """

        pormpt_parts = f"""

            You are an assistant that extracts only key points and section headlines from conversations.
            The conversation is in Basque, and the output must also be in Basque.

            The transcription may include timestamps and speaker labels. Ignore these. 
            Do not explain, do not summarize, and do not expand the content. 
            Your task is ONLY to write a clean, concise list of the important topics or section titles mentioned in the conversation, 
            as if creating an index of what was discussed.

            Output format: 
            - A bullet point list in Basque with one line per important point.
        """