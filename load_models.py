import torch
from transformers import BitsAndBytesConfig, AutoTokenizer,AutoModelForCausalLM,AutoModelForSequenceClassification
import streamlit as st
from RAG.config import config
import gc
from transformers import pipeline as hf_pipeline
from pyannote.audio import Pipeline as PyannotePipeline
import threading
import time


from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# Function: create_llm
# Description:
#   Initializes a language model (LLM) and its tokenizer.
#   Supports optional quantization for memory efficiency.
# ---------------------------------------------------------
def create_llm(name: str, quantize: bool):
    """
    Load a language model and tokenizer with optional 16-bit quantization.

    Args:
        name (str): The model name or path (Hugging Face format).
        quantize (bool): Whether to load the model with 16-bit quantization.

    Returns:
        tuple: (tokenizer, model)
    """
    # # Configure 16-bit quantization settings
    # bnb_config = BitsAndBytesConfig(
    #     load_in_16bit=True,
    #     bnb_16bit_compute_dtype=torch.bfloat16,
    # )
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    # bnb_config = BitsAndBytesConfig(
    #     load_in_8bit=True,
    #     llm_int8_threshold=6.0,
    #     llm_int8_skip_modules=None,
    #     llm_int8_enable_fp32_cpu_offload=False,
    #     llm_int8_has_fp16_weight=False
    # )



    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        name,
        trust_remote_code=True
    )

    # Load model with or without quantization
    if quantize:
        model = AutoModelForCausalLM.from_pretrained(
            name,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            name,
            device_map="auto",
            torch_dtype=torch.float16
        )

    return tokenizer, model

# ---------------------------------------------------------
# Function: generate_text
# Description:
#   Generic text-generation helper used by both summarization
#   and Q&A flows. Accepts either plain prompt text or a list
#   of chat-style messages (with roles), and returns the model
#   continuation only (without the input prompt).
# ---------------------------------------------------------
def generate_text(model, tokenizer, prompt_text: str = None, messages: list = None, max_tokens: int = 200, temperature: float = 0.7,
                   do_sample: bool = True, use_chat_template: bool = False):
    """
    Generic text generation function for both summary and Q&A tasks.

    Args:
        model: Loaded language model.
        tokenizer: Tokenizer associated with the model.
        prompt_text (str): The text prompt for standard generation.
        messages (list): Chat-style messages (if using chat template).
        max_tokens (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature.
        do_sample (bool): Whether to use sampling or greedy decoding.
        use_chat_template (bool): Whether to apply chat template (for chat-based prompts).

    Returns:
        str: Generated text from the model.
    """
    # Prepare inputs
    if use_chat_template and messages:
        
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt"
        ).to(model.device)
    elif prompt_text:
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    else:
        raise ValueError("You must provide either 'prompt_text' or 'messages'.")

    # Generate output
    outputs = model.generate(
        inputs,
        max_new_tokens=max_tokens,
        pad_token_id=tokenizer.eos_token_id,
        temperature=temperature,
        no_repeat_ngram_size=3,
        do_sample=do_sample
    )

    # Decode and return only generated tokens
    generated_text = tokenizer.decode(
        outputs[0][inputs.shape[-1]:],
        skip_special_tokens=True
    )

    return generated_text

@st.cache_resource
def get_reranker(model_name):
    
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tok, mdl

@st.cache_resource(show_spinner="🚀 LATXA LLM-a kargatzen...")
def get_llm(model_name,quantize:bool):
    tokenizer, model = create_llm(model_name, quantize)
    model.eval()
    return tokenizer, model

def free_all_cached():
    for k in ("model", "tokenizer"):
        if k in st.session_state:
            st.session_state.pop(k)
    st.cache_resource.clear()
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


@st.cache_resource(show_spinner="Dializatzeko modeloa kargatzen... ")
def load_diarization_model(HF_TOKEN,perfer_gpu = True):
    if perfer_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    diarization_pipeline = PyannotePipeline.from_pretrained(
        "pyannote/speaker-diarization-3.0",
        use_auth_token=HF_TOKEN
    ).to(device)
    print('hau erabakidu',device)
    return diarization_pipeline


@st.cache_resource(show_spinner= "Transkribitzeko modelo kargatzen...")
def load_asr_model(WHISPER_ID):
    asr_pipeline = hf_pipeline(
        "automatic-speech-recognition",
        model=WHISPER_ID,
        return_timestamps=True,
        device_map="auto"
    )
    return asr_pipeline

@st.cache_resource
def get_embedding_cache():
    """
    Returns a dictionary to cache embeddings and avoid redundant computation.
    Cached using Streamlit's cache_resource for performance optimization.
    """
    return {}
model_bge = SentenceTransformer(config.embedding.embedding_name, device = 'cpu')
embedding_cache = get_embedding_cache()

def emb_text_bge(text: str, is_query: bool = False):
    """
    Generate an embedding for a given text using the BAAI/BGE-M3 model.

    Args:
        text (str): Input text to be embedded.
        is_query (bool): If True, treat text as a query; otherwise as a passage.

    Returns:
        numpy.ndarray: The normalized embedding vector.
    """

    # Check if the text is already in the cache
    if text in embedding_cache:
        return embedding_cache[text]

    # Add prefix for better performance in retrieval tasks
    prefix = "query: " if is_query else "passage: "
    full_text = prefix + text

    # Generate and normalize embedding
    embedding = model_bge.encode(full_text, normalize_embeddings=True)

    # Store in cache for future use
    embedding_cache[text] = embedding

    return embedding


gpu_lock = threading.Lock()
gpu_condition = threading.Condition(gpu_lock)

MEM_NEEDED = 8.4 * 1024**3  # memoria mínima requerida (8 GB)
COOLDOWN_AFTER_FREE = 10  # segundos de espera tras liberar memoria

_last_gpu_use_time = 0  # para controlar el enfriamiento entre ejecuciones


def run_model_safe(model_func, *args, **kwargs):
    global _last_gpu_use_time

    placeholder = st.empty()  # mensaje temporal en Streamlit

    with gpu_condition:
        while True:
            # 🧠 Esperar un poco si otro acaba de liberar GPU (para evitar carreras)
            since_last = time.time() - _last_gpu_use_time
            if since_last < COOLDOWN_AFTER_FREE:
                wait_time = COOLDOWN_AFTER_FREE - since_last
                print(f"[GPU COOLDOWN] Esperando {wait_time:.1f}s para evitar carrera...")
                placeholder.info(f"🕐 Zain {wait_time:.0f} segundu GPU libre egon arte...")
                gpu_condition.wait(timeout=wait_time)
                continue

            total_mem = torch.cuda.get_device_properties(0).total_memory
            used_mem = torch.cuda.memory_allocated(0)
            free_mem = total_mem - used_mem

            print(f"[GPU CHECK] Libre: {free_mem/1e9:.2f} GB / {total_mem/1e9:.2f} GB -- > threshold {MEM_NEEDED/1e9:.2f}")

            if free_mem >= MEM_NEEDED:
                print("[GPU ACCESS GRANTED] Ejecutando modelo...")
                _last_gpu_use_time = time.time() 
                break
            else:
                placeholder.info("⚠️ Beste prozesuak bukatzeko zain zurearekin hasteko...")
                gpu_condition.wait(timeout=2)

    placeholder.empty()

    try:
        result = model_func(*args, **kwargs)
    finally:
        torch.cuda.empty_cache()
        gc.collect()
        with gpu_condition:
            _last_gpu_use_time = time.time()  # marca cuándo se liberó GPU
            gpu_condition.notify_all()        # avisa a los que esperan
        print("[GPU RELEASE] Memoria liberada y notificados los demás.")

    return result
