import os

# --- Ollama 配置 ---
OLLAMA_BASE_URL = "http://localhost:11434"

# 统一推理和总结模型，使用能力更强的模型
# 例如，如果 'deepseek-r1:14b' 是你的主推理模型
OLLAMA_MODEL = "deepseek-r1:14b"  # 主要用于生成和扮演
OLLAMA_SUMMARY_MODEL = OLLAMA_MODEL # 总结模型也使用与推理相同的模型

# 默认用于文本嵌入的模型
OLLAMA_EMBEDDING_MODEL_NAME = "shaw/dmeta-embedding-zh:latest"


# --- 数据存储配置 ---
# 主数据目录
DATA_DIR = "data_worlds"
WORLDS_METADATA_FILE = os.path.join(DATA_DIR, "worlds_metadata.json")

# --- 世界观数据库配置 ---
WORLD_CHARACTERS_DB_FILENAME = "characters.json"
WORLD_WORLDVIEW_TEXTS_FILENAME = "worldview_texts.json"
WORLD_WORLDVIEW_FAISS_INDEX_FILENAME = "worldview_index.faiss"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
SIMILARITY_TOP_K = 3

# --- 世界观压缩配置 ---
COMPRESSION_THRESHOLD = 100
COMPRESSION_TARGET_CLUSTERS = 20

# --- 文本长度限制 ---
MAX_CHAR_FULL_DESC_LEN = 10000
# LLM 生成概要描述时，输入给 LLM 的最大文本长度 (字符)
# 这个值需要根据你的模型和硬件进行调整，过长可能导致OOM或性能问题
MAX_SUMMARY_INPUT_LEN_LLM = 4000 # 适当增加，因为现在用大模型处理

# 确保主数据目录存在
os.makedirs(DATA_DIR, exist_ok=True)