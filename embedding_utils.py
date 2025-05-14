# embedding_utils.py
import ollama
import numpy as np
from config import OLLAMA_BASE_URL, OLLAMA_EMBEDDING_MODEL_NAME

_ollama_client_for_embed = None
_embedding_dimension = None

def get_ollama_client_for_embeddings():
    global _ollama_client_for_embed # 使用不同的客户端变量名以区分LLM客户端
    if _ollama_client_for_embed is None:
        try:
            print(f"正在初始化 Ollama 嵌入客户端以连接到: {OLLAMA_BASE_URL}")
            _ollama_client_for_embed = ollama.Client(host=OLLAMA_BASE_URL)
            _ollama_client_for_embed.list() 
            print(f"Ollama 嵌入客户端已连接。将使用模型 '{OLLAMA_EMBEDDING_MODEL_NAME}'。")
        except Exception as e:
            print(f"连接到 Ollama (用于嵌入) 或初始化客户端时发生错误: {e}")
            _ollama_client_for_embed = None
            raise
    return _ollama_client_for_embed

def get_model_embedding_dimension() -> int:
    global _embedding_dimension
    if _embedding_dimension is None:
        client = get_ollama_client_for_embeddings()
        if not client:
            raise ValueError("Ollama 嵌入客户端未初始化，无法获取嵌入维度。")
        try:
            print(f"首次调用，正在为嵌入模型 '{OLLAMA_EMBEDDING_MODEL_NAME}' 获取嵌入维度...")
            test_response = client.embeddings(
                model=OLLAMA_EMBEDDING_MODEL_NAME,
                prompt="dim_test"
            )
            _embedding_dimension = len(test_response['embedding'])
            print(f"获取到嵌入模型 '{OLLAMA_EMBEDDING_MODEL_NAME}' 的维度: {_embedding_dimension}")
        except Exception as e:
            print(f"通过 Ollama API 获取嵌入维度时发生错误: {e}")
            raise ValueError(f"无法确定模型 '{OLLAMA_EMBEDDING_MODEL_NAME}' 的嵌入维度。") from e
    return _embedding_dimension

def get_embedding(text: str) -> np.ndarray:
    client = get_ollama_client_for_embeddings()
    if not client:
        raise ConnectionError("Ollama 嵌入客户端不可用。")

    dimension = get_model_embedding_dimension()

    if not text or not isinstance(text, str) or not text.strip():
        print(f"警告：接收到用于嵌入的空或无效文本: '{text}'。返回零向量。")
        return np.zeros(dimension, dtype=np.float32)
    
    try:
        response = client.embeddings(
            model=OLLAMA_EMBEDDING_MODEL_NAME,
            prompt=text
        )
        embedding = np.array(response['embedding'], dtype=np.float32)
        return embedding
    except Exception as e:
        print(f"使用 Ollama 生成嵌入时发生错误 (文本片段: '{text[:50]}...'): {e}")
        return np.zeros(dimension, dtype=np.float32)


def get_embeddings(texts: list[str]) -> np.ndarray:
    client = get_ollama_client_for_embeddings()
    if not client:
        raise ConnectionError("Ollama 嵌入客户端不可用。")

    dimension = get_model_embedding_dimension()

    if not texts:
        return np.empty((0, dimension), dtype=np.float32)

    all_embeddings = np.zeros((len(texts), dimension), dtype=np.float32)
    
    for i, text_content in enumerate(texts):
        if isinstance(text_content, str) and text_content.strip():
            try:
                response = client.embeddings(
                    model=OLLAMA_EMBEDDING_MODEL_NAME,
                    prompt=text_content
                )
                all_embeddings[i] = np.array(response['embedding'], dtype=np.float32)
            except Exception as e:
                print(f"为文本 (索引 {i}) '{text_content[:50]}...' 生成嵌入时发生错误: {e}。此文本嵌入将为零。")
        else:
            print(f"警告：批量嵌入中跳过无效或空文本，索引 {i}: '{text_content}'。此文本嵌入将为零。")
            
    return all_embeddings