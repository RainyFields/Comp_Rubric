"""Offline tokenizer loading for the tests.

``FOLD_TOKENIZER_PATH`` (or the legacy ``QWEN3_TOKENIZER_PATH``) selects the tokenizer; otherwise the first available
candidate is used. Run the whole suite once per model family::

    for T in Qwen/Qwen3-8B ~/xiaoxuan/tokenizers/Qwen3.5-9B; do FOLD_TOKENIZER_PATH=$T python -m unittest discover -s tests -t . ; done
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

CANDIDATES = [
    os.environ.get("FOLD_TOKENIZER_PATH"),
    os.environ.get("QWEN3_TOKENIZER_PATH"),
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-1.7B",
    os.path.expanduser("~/xiaoxuan/tokenizers/Qwen3.5-9B"),
    "/mnt/hdfs/mlsys/users/xiaoxuan/fold-job-assets/tokenizers/Qwen3.5-9B",
    "/mnt/hdfs/mlsys/xiaoxuan/fold_replication/ckpt/foldgrpo/global_step_90/actor/huggingface",
]


def load_tokenizer():
    from transformers import AutoTokenizer
    for cand in CANDIDATES:
        if not cand:
            continue
        try:
            tok = AutoTokenizer.from_pretrained(cand)
        except Exception:
            continue
        if tok.chat_template is None and os.path.isdir(cand):
            jinja = os.path.join(cand, "chat_template.jinja")
            if os.path.exists(jinja):
                tok.chat_template = open(jinja).read()
        if tok.chat_template and "tool_response" in tok.chat_template and "last_query_index" in tok.chat_template:
            return tok
    return None


def profile_of(tok):
    from agents.model_profile import detect_profile
    return detect_profile(tok)
