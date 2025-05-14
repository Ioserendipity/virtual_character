import os
import json
import faiss
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from langchain.text_splitter import RecursiveCharacterTextSplitter
import shutil
import re # 用于 add_world 的ID校验

from config import (
    DATA_DIR as MAIN_DATA_DIR, WORLDS_METADATA_FILE,
    WORLD_CHARACTERS_DB_FILENAME, WORLD_WORLDVIEW_TEXTS_FILENAME,
    WORLD_WORLDVIEW_FAISS_INDEX_FILENAME,
    CHUNK_SIZE, CHUNK_OVERLAP, SIMILARITY_TOP_K, MAX_CHAR_FULL_DESC_LEN,
    MAX_SUMMARY_INPUT_LEN_LLM # 新增导入
)
from embedding_utils import get_embedding, get_embeddings, get_model_embedding_dimension
# 修改导入：现在直接使用 generate_text 来生成角色概要
from llm_utils import generate_text, get_ollama_client

# --- Worlds Metadata Management ---  显式标注键是字符串，值是字典
_worlds_metadata: Dict[str, Dict] = {}

def load_worlds_metadata():
    global _worlds_metadata
    if os.path.exists(WORLDS_METADATA_FILE):
        if os.path.getsize(WORLDS_METADATA_FILE) > 0:
            try:
                with open(WORLDS_METADATA_FILE, 'r', encoding='utf-8') as f:
                    _worlds_metadata = json.load(f)
            except json.JSONDecodeError:
                print(f"警告: {WORLDS_METADATA_FILE} 包含无效JSON。将初始化为空。")
                _worlds_metadata = {}
        else: # 文件为空
            _worlds_metadata = {}
    else: # 文件不存在
        _worlds_metadata = {}
        save_worlds_metadata()

def save_worlds_metadata():
    with open(WORLDS_METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(_worlds_metadata, f, ensure_ascii=False, indent=4)

def add_world(world_id: str, world_name: str) -> str:
    if not world_id or not re.match(r"^[a-zA-Z0-9_]+$", world_id) or " " in world_id:
        return "错误: 世界ID只能包含字母、数字和下划线，且不能包含空格或为空。"
    if world_id in _worlds_metadata:
        return f"错误: 世界ID '{world_id}' 已存在。"

    world_data_dir = os.path.join(MAIN_DATA_DIR, world_id)
    try:
        os.makedirs(world_data_dir, exist_ok=True)
    except OSError as e:
        return f"错误：创建世界目录 '{world_data_dir}' 失败: {e}"

    _worlds_metadata[world_id] = {"name": world_name, "data_dir": world_data_dir}
    save_worlds_metadata()

    initialize_character_db_for_world(world_id)
    initialize_worldview_db_for_world(world_id)

    return f"新世界 '{world_name}' (ID: {world_id}) 已添加。其数据将存储在 {world_data_dir}"

def delete_world(world_id: str) -> str:
    global _worlds_metadata, _active_world_id, _active_characters_db
    global _active_worldview_texts, _active_faiss_index, _active_next_worldview_doc_id

    if world_id not in _worlds_metadata:
        return f"错误: 世界ID '{world_id}' 不存在，无法删除。"

    world_data_to_delete = _worlds_metadata.pop(world_id)
    save_worlds_metadata()

    world_name_to_delete = world_data_to_delete.get("name", world_id)
    world_dir_to_delete = world_data_to_delete.get("data_dir")

    if _active_world_id == world_id:
        _active_world_id = None
        _active_characters_db = []
        _active_worldview_texts = {}
        _active_faiss_index = None
        _active_next_worldview_doc_id = 0
        print(f"活动世界 '{world_name_to_delete}' 已被删除，当前无活动世界。")

    if world_dir_to_delete and os.path.exists(world_dir_to_delete):
        try:
            shutil.rmtree(world_dir_to_delete)
            print(f"已成功删除世界 '{world_name_to_delete}' 的数据目录: {world_dir_to_delete}")
            return f"世界 '{world_name_to_delete}' (ID: {world_id}) 及其所有数据已成功删除。"
        except OSError as e:
            print(f"错误：删除世界 '{world_name_to_delete}' 的数据目录 {world_dir_to_delete} 失败: {e}")
            return f"世界 '{world_name_to_delete}' 的元数据已删除，但物理数据目录删除失败: {e}。请手动检查并删除 {world_dir_to_delete}。"
    else:
        return f"世界 '{world_name_to_delete}' (ID: {world_id}) 的元数据已删除（数据目录未找到或路径无效）。"


def get_world_data_dir(world_id: str) -> Optional[str]:
    return _worlds_metadata.get(world_id, {}).get("data_dir")

def get_world_display_name(world_id: str) -> Optional[str]:
    if not world_id: return None
    return _worlds_metadata.get(world_id, {}).get("name")

def get_available_worlds() -> Dict[str, str]:
    return {world_id: data["name"] for world_id, data in _worlds_metadata.items()}

_active_world_id: Optional[str] = None
_active_characters_db: List[Dict] = []
_active_worldview_texts: Dict[int, Dict] = {}
_active_faiss_index: Optional[faiss.Index] = None
_active_next_worldview_doc_id: int = 0


def switch_active_world(world_id: Optional[str]) -> bool:
    global _active_world_id, _active_characters_db, _active_worldview_texts
    global _active_faiss_index, _active_next_worldview_doc_id

    if not world_id:
        if _active_world_id is not None:
            pass
        _active_world_id = None; _active_characters_db = []; _active_worldview_texts = {}
        _active_faiss_index = None; _active_next_worldview_doc_id = 0
        return True

    if world_id not in _worlds_metadata:
        print(f"错误: 尝试切换到不存在的世界 '{world_id}'。将取消当前活动世界。")
        switch_active_world(None)
        return False

    if _active_world_id == world_id and _active_faiss_index is not None: # 已是活动状态，且数据已加载
        return True

    _active_world_id = world_id

    load_characters_db_for_world(world_id)
    load_worldview_db_for_world(world_id)

    return True

def get_character_db_path(world_id: str) -> Optional[str]:
    data_dir = get_world_data_dir(world_id)
    return os.path.join(data_dir, WORLD_CHARACTERS_DB_FILENAME) if data_dir else None

def initialize_character_db_for_world(world_id: str):
    global _active_characters_db
    char_db_path = get_character_db_path(world_id)
    if char_db_path:
        with open(char_db_path, 'w', encoding='utf-8') as f:
            json.dump([], f)
        if _active_world_id == world_id:
            _active_characters_db = []

def load_characters_db_for_world(world_id: str):
    global _active_characters_db
    char_db_path = get_character_db_path(world_id)
    current_world_chars = []
    if char_db_path and os.path.exists(char_db_path):
        if os.path.getsize(char_db_path) > 0:
            try:
                with open(char_db_path, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
                if isinstance(loaded_data, list):
                    current_world_chars = loaded_data
                    for char_data in current_world_chars:
                        if 'embedding' in char_data and isinstance(char_data.get('embedding'), list):
                            char_data['embedding'] = np.array(char_data['embedding'], dtype=np.float32)
                else:
                    print(f"警告: 世界 '{world_id}' 的角色文件 {char_db_path} 内容非列表。")
            except Exception as e:
                print(f"加载世界 '{world_id}' 的角色数据库 {char_db_path} 时出错: {e}")
    _active_characters_db = current_world_chars

def save_characters_db_for_active_world():
    if not _active_world_id:
        print("错误：没有活动的存储世界来保存角色数据。")
        return
    char_db_path = get_character_db_path(_active_world_id)
    if not char_db_path:
        print(f"错误：无法获取世界 '{_active_world_id}' 的角色数据库路径。")
        return

    serializable_db = []
    for char_data in _active_characters_db:
        char_copy = char_data.copy()
        if 'embedding' in char_copy and isinstance(char_copy['embedding'], np.ndarray):
            char_copy['embedding'] = char_copy['embedding'].tolist()
        serializable_db.append(char_copy)

    try:
        with open(char_db_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_db, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"保存世界 '{_active_world_id}' 的角色数据到 {char_db_path} 时出错: {e}")

def add_character(name: str, full_description: str) -> str:
    if not _active_world_id: return "错误：请先选择一个活动世界。"
    if len(full_description) > MAX_CHAR_FULL_DESC_LEN:
        return f"错误：角色描述过长 (超过 {MAX_CHAR_FULL_DESC_LEN} 字符)。请缩减后再试。"

    # --- 修改开始：使用主推理模型生成角色概要 ---
    # 截断输入文本以适应模型限制
    truncated_full_desc = full_description
    if len(full_description) > MAX_SUMMARY_INPUT_LEN_LLM: # 使用 config 中的限制
        truncated_full_desc = full_description[:MAX_SUMMARY_INPUT_LEN_LLM] + "\n...[内容已截断]"
        print(f"为角色 '{name}' 生成概要时，输入描述已从 {len(full_description)} 截断到 {len(truncated_full_desc)} 字符。")

    # 新的 Prompt，引导模型生成更具角色内心独白或核心设定的概要
    summary_prompt = f"""
请深入分析以下提供的角色完整描述。你的任务不是简单地总结，而是要提炼出这个角色的【核心驱动力、性格特质、标志性口头禅或内心独白】。
想象你就是这个角色，用【第一人称】或非常贴近角色视角的第三人称，写一段文字,适当凝练，但是一定要足够精准地传达角色内心，作为这个角色的“灵魂”或“核心设定”。
这段文字应该能描绘出一个活生生地人。
角色完整描述：
---
{truncated_full_desc}
---
他/她的“灵魂”/“核心设定”(不要给出依据，只输出结论)：以'(角色名)是'开头
"""
    system_message_for_summary = "你是一个富有洞察力的角色分析师和剧作家，擅长捕捉角色的核心本质并用生动的语言表达出来。"

    # 使用 llm_utils.generate_text 和主推理模型（在 config.py 中 OLLAMA_MODEL）
    # 注意：generate_text 的 model_name 参数默认为 OLLAMA_MODEL，所以这里可以不传
    summary_description = generate_text(prompt=summary_prompt, system_message=system_message_for_summary)

    if not summary_description or summary_description.startswith("错误:") or summary_description.startswith("Error:"):
        error_msg = summary_description if summary_description else "LLM未返回内容"
        print(f"为角色 '{name}' 生成概要描述失败。将使用完整描述的前250字符作为备用。LLM错误: {error_msg}")
        # 备用方案：如果LLM失败，返回完整描述的一部分
        summary_description = full_description[:250] + ("..." if len(full_description) > 250 else "")
    else:
        summary_description = summary_description.strip() # 清理可能的空白

    # --- 修改结束 ---

    embedding = get_embedding(summary_description)
    if embedding.size == 0:
        return f"为角色 '{name}' 的概要描述生成嵌入失败。"

    new_char_data = {
        'name': name,
        'full_description': full_description,
        'summary_description': summary_description, # 这是新生成的概要
        'embedding': embedding
    }
    for i, char_data in enumerate(_active_characters_db):
        if char_data['name'] == name:
            _active_characters_db[i] = new_char_data
            save_characters_db_for_active_world()
            return f"角色 '{name}' (世界: {get_world_display_name(_active_world_id)}) 已更新。"

    _active_characters_db.append(new_char_data)
    save_characters_db_for_active_world()
    return f"角色 '{name}' (世界: {get_world_display_name(_active_world_id)}) 已添加。概要：'{summary_description[:50]}...'"


def get_character(name: str) -> Optional[Dict]:
    if not _active_world_id: return None
    for char_data in _active_characters_db:
        if char_data['name'] == name:
            return char_data
    return None

def get_all_characters() -> List[Dict]:
    return _active_characters_db if _active_world_id else []

def get_character_names() -> List[str]:
    return [char['name'] for char in _active_characters_db] if _active_world_id else []

def delete_character(name: str) -> str:
    if not _active_world_id:
        return "错误：请先选择一个活动世界。"
    if not name:
        return "错误：未指定要删除的角色名称。"

    char_index_to_delete = -1
    for i, char_data in enumerate(_active_characters_db):
        if char_data['name'] == name:
            char_index_to_delete = i
            break

    if char_index_to_delete != -1:
        del _active_characters_db[char_index_to_delete]
        save_characters_db_for_active_world()
        return f"角色 '{name}' (世界: {get_world_display_name(_active_world_id)}) 已成功删除。"
    else:
        return f"错误：在当前世界 '{get_world_display_name(_active_world_id)}' 中未找到角色 '{name}'。"

_text_splitter = None
try:
    _text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        length_function=len, is_separator_regex=False,
    )
except NameError:
    print("警告: CHUNK_SIZE 或 CHUNK_OVERLAP 未在config.py中定义。文本分割器将使用默认值。")
    _text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)


def get_worldview_texts_path(world_id: str) -> Optional[str]:
    data_dir = get_world_data_dir(world_id)
    return os.path.join(data_dir, WORLD_WORLDVIEW_TEXTS_FILENAME) if data_dir else None

def get_worldview_faiss_path(world_id: str) -> Optional[str]:
    data_dir = get_world_data_dir(world_id)
    return os.path.join(data_dir, WORLD_WORLDVIEW_FAISS_INDEX_FILENAME) if data_dir else None

def initialize_faiss_index_for_world(world_id: str):
    global _active_faiss_index
    faiss_path = get_worldview_faiss_path(world_id)
    if not faiss_path:
        print(f"错误：无法获取世界 '{world_id}' 的FAISS路径。")
        if _active_world_id == world_id: _active_faiss_index = None
        return

    try:
        dimension = get_model_embedding_dimension()
        base_index = faiss.IndexFlatL2(dimension)
        temp_faiss_index = faiss.IndexIDMap(base_index)
        faiss.write_index(temp_faiss_index, faiss_path)
        if _active_world_id == world_id:
            _active_faiss_index = temp_faiss_index
    except Exception as e:
        print(f"为世界 '{world_id}' 初始化 FAISS 索引时出错: {e}")
        if _active_world_id == world_id:
            _active_faiss_index = None

def initialize_worldview_db_for_world(world_id: str):
    global _active_worldview_texts, _active_next_worldview_doc_id

    texts_path = get_worldview_texts_path(world_id)
    if texts_path:
        with open(texts_path, 'w', encoding='utf-8') as f:
            json.dump({}, f)

    initialize_faiss_index_for_world(world_id)

    if _active_world_id == world_id:
        _active_worldview_texts = {}
        _active_next_worldview_doc_id = 0

def load_worldview_db_for_world(world_id: str):
    global _active_worldview_texts, _active_faiss_index, _active_next_worldview_doc_id

    texts_path = get_worldview_texts_path(world_id)
    faiss_path = get_worldview_faiss_path(world_id)

    current_world_texts = {}
    current_next_id = 0

    if texts_path and os.path.exists(texts_path):
        if os.path.getsize(texts_path) > 0:
            try:
                with open(texts_path, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
                if isinstance(loaded_data, dict):
                    # Ensure keys are integers after loading from JSON
                    current_world_texts = {int(k): v for k, v in loaded_data.items()}
                if current_world_texts:
                    current_next_id = max([k for k in current_world_texts.keys()] + [-1]) + 1 # k is already int
            except Exception as e:
                print(f"加载世界 '{world_id}' 的世界观文本 {texts_path} 时出错: {e}")

    _active_worldview_texts = current_world_texts
    _active_next_worldview_doc_id = current_next_id

    current_faiss_index = None
    if faiss_path and os.path.exists(faiss_path):
        try:
            current_model_dim = get_model_embedding_dimension()
            current_faiss_index = faiss.read_index(faiss_path)
            if current_faiss_index.d != current_model_dim:
                print(f"警告: 世界 '{world_id}' 的FAISS索引维度 ({current_faiss_index.d}) 与当前模型维度 ({current_model_dim}) 不符。将重新初始化索引。")
                initialize_faiss_index_for_world(world_id)
                current_faiss_index = _active_faiss_index
        except Exception as e:
            print(f"加载世界 '{world_id}' 的FAISS索引 {faiss_path} 时出错: {e}。将重新初始化。")
            initialize_faiss_index_for_world(world_id)
            current_faiss_index = _active_faiss_index
    else:
        initialize_faiss_index_for_world(world_id)
        current_faiss_index = _active_faiss_index

    _active_faiss_index = current_faiss_index

def save_worldview_db_for_active_world():
    if not _active_world_id:
        print("错误：没有活动的存储世界来保存世界观数据。")
        return

    texts_path = get_worldview_texts_path(_active_world_id)
    faiss_path = get_worldview_faiss_path(_active_world_id)

    if texts_path:
        try:
            # Ensure keys are strings when saving to JSON
            with open(texts_path, 'w', encoding='utf-8') as f:
                json.dump({str(k): v for k, v in _active_worldview_texts.items()}, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"保存世界 '{_active_world_id}' 的世界观文本到 {texts_path} 时出错: {e}")

    if _active_faiss_index is not None and faiss_path:
        try:
            faiss.write_index(_active_faiss_index, faiss_path)
        except Exception as e:
            print(f"保存世界 '{_active_world_id}' 的FAISS索引到 {faiss_path} 时出错: {e}")
    elif faiss_path: # Only print warning if path exists but index is None for some reason
        print(f"警告: 世界 '{_active_world_id}' 的FAISS索引为None，但路径有效。无法保存到 {faiss_path}。")


def add_worldview_text(full_text: str) -> str:
    global _active_next_worldview_doc_id
    if not _active_world_id: return "错误：请先选择一个活动世界。"
    if not _active_faiss_index:
        load_worldview_db_for_world(_active_world_id) # Try to load/initialize
        if not _active_faiss_index:
            return "错误：FAISS索引初始化失败，无法添加文本。"

    if not _text_splitter: return "错误: 文本分割器未初始化。"

    chunks = _text_splitter.split_text(full_text)
    if not chunks: return "未生成文本块。文本可能过短或为空。"

    chunk_embeddings = get_embeddings(chunks)
    if chunk_embeddings.size == 0 or (chunk_embeddings.ndim == 2 and chunk_embeddings.shape[0] == 0):
        return "无法为提供的文本生成嵌入向量。"
    if chunk_embeddings.ndim == 1: # Should not happen if get_embeddings returns correctly shaped array
        expected_dim = get_model_embedding_dimension()
        if chunk_embeddings.shape[0] == expected_dim:
            chunk_embeddings = chunk_embeddings.reshape(1, -1)
        else: return "生成的嵌入维度不正确。"

    ids_to_add = []; new_text_entries = {}; temp_next_id = _active_next_worldview_doc_id
    for chunk_content in chunks:
        current_id_to_assign = temp_next_id
        # Ensure unique ID, though _active_next_worldview_doc_id should already be the next available
        while current_id_to_assign in _active_worldview_texts:
            temp_next_id += 1; current_id_to_assign = temp_next_id
        ids_to_add.append(current_id_to_assign)
        new_text_entries[current_id_to_assign] = {"full_text": chunk_content, "summary_text": None}
        temp_next_id +=1 # Increment for the next potential ID

    ids_np = np.array(ids_to_add, dtype=np.int64)
    try:
        _active_faiss_index.add_with_ids(chunk_embeddings, ids_np)
        _active_worldview_texts.update(new_text_entries)
        _active_next_worldview_doc_id = temp_next_id # Update the global next ID
        save_worldview_db_for_active_world()
        return f"已添加 {len(chunks)} 个文本块到世界 '{get_world_display_name(_active_world_id)}'。总条目数: {_active_faiss_index.ntotal}。"
    except Exception as e:
        print(f"向FAISS索引添加嵌入时发生错误: {e}")
        # Rollback: remove added text entries if FAISS add failed
        for failed_id in ids_to_add: _active_worldview_texts.pop(failed_id, None)
        # Note: Rolling back FAISS index changes is harder, might require re-init or selective removal if supported easily.
        # For now, we assume if add_with_ids fails, the index might be in an inconsistent state for those IDs.
        return f"添加嵌入到FAISS时失败: {e}"


def search_worldview(query_text: str, k: int = -1) -> List[str]:
    if k < 0: k = SIMILARITY_TOP_K
    if not _active_world_id or not _active_faiss_index or _active_faiss_index.ntotal == 0: return []
    query_embedding = get_embedding(query_text)
    if query_embedding.size == 0: print("警告：搜索查询的嵌入为空。"); return []
    query_embedding_np = np.array([query_embedding], dtype=np.float32) # FAISS expects 2D array
    actual_k = min(k, _active_faiss_index.ntotal) # Cannot retrieve more than what's in index
    if actual_k == 0: return []
    try:
        distances, doc_ids_found = _active_faiss_index.search(query_embedding_np, k=actual_k)
    except Exception as e: print(f"FAISS 搜索时发生错误: {e}"); return []
    retrieved_texts = []
    if doc_ids_found.size > 0:
        for doc_id in doc_ids_found[0]: # doc_ids_found is 2D array [[id1, id2, ...]]
            if doc_id != -1 and doc_id in _active_worldview_texts: # doc_id can be -1 if less than k results found
                retrieved_texts.append(_active_worldview_texts[doc_id]["full_text"])
    return retrieved_texts

def get_worldview_size() -> int:
    return _active_faiss_index.ntotal if _active_world_id and _active_faiss_index else 0

def get_all_worldview_data_for_active_world() -> Tuple[Optional[faiss.Index], Dict[int, Dict]]:
    if not _active_world_id: return None, {}
    # Return copies to prevent direct modification of cached data outside of db_utils functions
    return _active_faiss_index, _active_worldview_texts.copy()

def rebuild_worldview_from_data(new_texts_with_ids: Dict[int, Dict], new_embeddings: np.ndarray):
    global _active_worldview_texts, _active_faiss_index, _active_next_worldview_doc_id
    if not _active_world_id: print("错误：没有活动世界，无法重建世界观。"); return

    initialize_faiss_index_for_world(_active_world_id) # Creates a new empty index for the active world
    _active_worldview_texts.clear() # Clear in-memory text map

    if not _active_faiss_index: print("错误：重建世界观时 FAISS 索引未能初始化。"); return

    if not new_texts_with_ids or not isinstance(new_embeddings, np.ndarray) or \
       new_embeddings.ndim != 2 or new_embeddings.shape[0] == 0:
        _active_next_worldview_doc_id = 0
        save_worldview_db_for_active_world() # Save empty state
        return

    if len(new_texts_with_ids) != new_embeddings.shape[0]:
        print(f"错误：重建数据时，文本数量 ({len(new_texts_with_ids)}) 与嵌入数量 ({new_embeddings.shape[0]}) 不匹配。世界观将为空。")
        initialize_worldview_db_for_world(_active_world_id) # Re-initialize to empty state
        return

    ids_np = np.array(list(new_texts_with_ids.keys()), dtype=np.int64)
    try:
        _active_faiss_index.add_with_ids(new_embeddings, ids_np)
        _active_worldview_texts = new_texts_with_ids.copy() # Use a copy of the new data
        if _active_worldview_texts:
            _active_next_worldview_doc_id = max(list(_active_worldview_texts.keys()) + [-1]) + 1
        else:
            _active_next_worldview_doc_id = 0
        save_worldview_db_for_active_world()
    except Exception as e:
        print(f"在重建过程中向FAISS添加数据时出错: {e}。世界观可能未正确重建，将重置为空。")
        initialize_worldview_db_for_world(_active_world_id) # Reset to empty on error

load_worlds_metadata()

if __name__ == '__main__':
    print("直接运行 database_utils.py 进行多世界数据库管理基础测试...")

    try:
        get_model_embedding_dimension()
        get_ollama_client()
        print("嵌入和LLM客户端初始化成功。")
    except Exception as e:
        print(f"初始化依赖项时出错: {e}")
        exit()

    print("\n--- 世界管理功能基础测试 ---")
    print(f"模块启动时，已加载/初始化的可用世界: {get_available_worlds()}")

    print("\nDB Util 基础功能测试点完成。")
    print("现在您可以通过 main_app.py 启动UI来创建和管理世界。")