# llm_utils.py
import ollama
from config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_SUMMARY_MODEL

_ollama_client = None

def get_ollama_client():
    global _ollama_client
    if _ollama_client is None:
        print(f"正在初始化 Ollama LLM 客户端以连接到: {OLLAMA_BASE_URL}")
        try:
            _ollama_client = ollama.Client(host=OLLAMA_BASE_URL)
            response_data = _ollama_client.list()
            print(f"成功连接到 Ollama，地址为: {OLLAMA_BASE_URL}")

            available_models_info = response_data.get('models', [])
            available_model_names = []

            if available_models_info:
                first_item = available_models_info[0]
                if hasattr(first_item, 'name'):
                    for m_obj in available_models_info:
                        if hasattr(m_obj, 'name'):
                            available_model_names.append(m_obj.name)
                elif hasattr(first_item, 'model'):
                    for m_obj in available_models_info:
                        if hasattr(m_obj, 'model'):
                            available_model_names.append(m_obj.model)
                else:
                    print("警告：无法从 Ollama 模型列表中提取模型名称。")
            
            # 检查默认生成模型和总结模型是否可用
            models_to_check = {
                "生成模型": OLLAMA_MODEL,
                "总结模型": OLLAMA_SUMMARY_MODEL
            }
            all_specified_models_available = True
            for desc, model_name in models_to_check.items():
                if model_name not in available_model_names:
                    print(f"警告：配置的 {desc} '{model_name}' 在 Ollama 中未找到。可用模型: {available_model_names}")
                    print(f"  请确保已拉取 '{model_name}' 或更新 config.py。")
                    all_specified_models_available = False
                else:
                    print(f"配置的 {desc} '{model_name}' 在 Ollama 中可用。")
            
            if not all_specified_models_available:
                # 可以选择在这里抛出异常，如果核心模型缺失
                pass # print("警告：一个或多个配置的LLM模型在Ollama中不可用，功能可能受限。")

        except Exception as e:
            print(f"连接到 Ollama 或列出模型时发生错误: {e}")
            _ollama_client = None
            raise
    return _ollama_client

def generate_text(prompt: str, system_message: str = None, model_name: str = None) -> str:
    """
    使用指定的 Ollama 模型生成文本。

    Args:
        prompt (str): 用户提示。
        system_message (str, optional): 系统消息。
        model_name (str, optional): 要使用的Ollama模型名称。如果为None，则使用config.py中的OLLAMA_MODEL。

    Returns:
        str: 生成的文本，或错误消息。
    """
    client = get_ollama_client()
    if not client:
        return "错误：Ollama 客户端不可用。"

    target_model = model_name if model_name else OLLAMA_MODEL
        
    messages = []
    if system_message:
        messages.append({'role': 'system', 'content': system_message})
    messages.append({'role': 'user', 'content': prompt})

    try:
        print(f"正在使用模型 '{target_model}' 生成文本...")
        response = client.chat(
            model=target_model,
            messages=messages
        )
        generated_content = response['message']['content']
        print(f"模型 '{target_model}' 响应长度: {len(generated_content)}")
        return generated_content
    except Exception as e:
        error_message = f"Error: 调用 Ollama API 时发生错误 (模型: {target_model}): {e}"
        if "model not found" in str(e).lower() or \
           (hasattr(e, 'response') and hasattr(e.response, 'status_code') and e.response.status_code == 404):
             error_message = f"错误：LLM 模型 '{target_model}' 在 Ollama 服务器上未找到。请拉取该模型或检查其名称。"
        print(error_message)
        return error_message

def generate_summary(text_to_summarize: str, max_input_len: int = None) -> str:
    """
    使用配置的总结模型为给定文本生成概要。
    """
    from config import MAX_SUMMARY_INPUT_LEN_LLM # 延迟导入以避免循环

    if max_input_len is None:
        max_input_len = MAX_SUMMARY_INPUT_LEN_LLM

    if not text_to_summarize.strip():
        return "" # 空文本直接返回空概要

    truncated_text = text_to_summarize
    if len(text_to_summarize) > max_input_len:
        truncated_text = text_to_summarize[:max_input_len] + "\n... [内容过长已截断]"
        print(f"用于总结的文本已从 {len(text_to_summarize)} 字符截断到 {max_input_len} 字符。")

    prompt = f"""请为以下文本生成一个简洁但尽量保留关键角色性格特质的的描述。概要应该捕捉文本的核心思想和关键细节，并且自身是一段连贯的文字。不要添加任何额外的解释或评论，直接给出概要。

原始文本：
---
{truncated_text}
---

概要："""
    system_message = "你是一个专业的文本摘要助手，擅长提炼核心信息并生成高质量的摘要。"
    
    # 使用配置中指定的总结模型
    summary = generate_text(prompt, system_message=system_message, model_name=OLLAMA_SUMMARY_MODEL)
    
    if summary.startswith("错误:") or summary.startswith("Error:"):
        print(f"使用 LLM 生成概要失败。将返回原始文本的前N个字符作为备用。错误: {summary}")
        # 备用方案：如果LLM失败，返回原始文本的一部分
        return text_to_summarize[:500] + "..." if len(text_to_summarize) > 500 else text_to_summarize
    
    return summary.strip()