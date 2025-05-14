# compression_utils.py
import numpy as np
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple

from config import COMPRESSION_TARGET_CLUSTERS, CHUNK_SIZE # CHUNK_SIZE 用于LLM失败时的备用文本截断
from database_utils import (
    get_all_worldview_data_for_active_world, # 获取当前活动世界的数据
    rebuild_worldview_from_data,             # 重建当前活动世界的数据
    get_worldview_size,                      # 获取当前活动世界的大小
    _active_world_id, get_world_display_name # 用于打印信息
)
from embedding_utils import get_embedding, get_embeddings # For new summary embeddings and re-embedding
from llm_utils import generate_text # For summarizing clusters

def compress_worldview_db_for_active_world(force_compress: bool = False) -> str:
    """
    使用K-Means聚类和LLM总结来压缩当前活动世界的世界观数据库。
    """
    if not _active_world_id:
        return "错误：没有活动的存储世界来进行压缩。"

    world_name = get_world_display_name(_active_world_id)
    current_size = get_worldview_size() # 这现在会获取活动世界的大小
    
    if current_size <= COMPRESSION_TARGET_CLUSTERS * 1.5 and not force_compress:
        return f"世界 '{world_name}' 的世界观大小 ({current_size}) 未达到压缩标准或已接近目标 ({COMPRESSION_TARGET_CLUSTERS})。除非强制，否则不执行压缩。"

    faiss_index, worldview_texts_map = get_all_worldview_data_for_active_world()

    if not faiss_index or faiss_index.ntotal == 0:
        return f"世界 '{world_name}' 的世界观为空。无需压缩。"
    if faiss_index.ntotal <= 1:
        return f"世界 '{world_name}' 的世界观只有1条记录。无法压缩。"

    print(f"开始压缩世界 '{world_name}' 的世界观。当前大小: {faiss_index.ntotal}")

    all_original_ids_np_list = []
    all_texts_for_reembed_list = []

    # 从 worldview_texts_map 中提取ID和 full_text 用于重新嵌入
    # 这是更安全的方式，因为我们总是基于原始内容进行聚类
    for doc_id, text_data in worldview_texts_map.items():
        all_original_ids_np_list.append(doc_id)
        all_texts_for_reembed_list.append(text_data["full_text"])
    
    if not all_texts_for_reembed_list:
        return f"世界 '{world_name}' 的 worldview_texts_map 中没有找到文本。压缩中止。"

    print(f"正在为世界 '{world_name}' 的 {len(all_texts_for_reembed_list)} 条文本重新生成嵌入以进行压缩...")
    all_vectors_np = get_embeddings(all_texts_for_reembed_list) # 使用批量嵌入
    all_original_ids_np = np.array(all_original_ids_np_list, dtype=np.int64)

    if all_vectors_np.shape[0] == 0:
        return f"世界 '{world_name}' 中没有找到用于压缩的向量。"

    num_samples = all_vectors_np.shape[0]
    n_clusters = min(num_samples, COMPRESSION_TARGET_CLUSTERS)
    if num_samples > 1 and n_clusters <= 0: n_clusters = 1
    if num_samples <= n_clusters: n_clusters = num_samples
    if n_clusters == 0: return f"世界 '{world_name}' 无法确定有效的聚类数量。"
    
    print(f"正在将世界 '{world_name}' 的 {num_samples} 个向量聚类为 {n_clusters} 个簇。")
    
    try: # Scikit-learn KMeans n_init 兼容性处理
        from sklearn import __version__ as sklearn_version
        if sklearn_version < '1.2': kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif sklearn_version < '1.4': kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        else: kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    except ImportError: kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        
    cluster_labels = kmeans.fit_predict(all_vectors_np)

    new_worldview_texts_with_ids: Dict[int, Dict] = {} # 新ID -> {"full_text": summary, "summary_text": None}
    new_worldview_embeddings_list: List[np.ndarray] = []
    
    print(f"正在为世界 '{world_name}' 的 {n_clusters} 个簇生成LLM摘要...")
    next_new_id = 0 # 为压缩后的条目使用新的连续ID
    for i in range(n_clusters):
        cluster_indices = np.where(cluster_labels == i)[0]
        if len(cluster_indices) == 0: continue

        texts_in_cluster = [
            worldview_texts_map[all_original_ids_np[idx]]["full_text"] 
            for idx in cluster_indices 
            if all_original_ids_np[idx] in worldview_texts_map
        ]
        
        if not texts_in_cluster:
            print(f"警告 (世界 '{world_name}'): 簇 {i} 没有关联的文本。跳过。")
            continue

        summary_text_content: str
        if len(texts_in_cluster) == 1:
            summary_text_content = texts_in_cluster[0] # 单条文本直接使用
        else:
            # 为了避免LLM输入过长，可以对每个文本先做初步总结，或只取一部分
            # 这里简单拼接，但实际应用中可能需要更复杂的处理
            combined_text_for_summary = "\n\n---\n\n".join(texts_in_cluster)
            
            # generate_summary 函数内部会处理截断
            summary_text_content = generate_summary(combined_text_for_summary)

            if summary_text_content.startswith("错误:") or summary_text_content.startswith("Error:"):
                print(f"LLM对簇 {i} (世界 '{world_name}') 的总结失败。使用合并文本的截断作为备用。")
                # 使用一个简单的备用：连接所有文本并截断
                fallback_summary = (" ".join(texts_in_cluster))[:CHUNK_SIZE * 2] # CHUNK_SIZE 来自 config
                summary_text_content = fallback_summary
        
        summary_embedding = get_embedding(summary_text_content)
        if summary_embedding.size == 0:
            print(f"警告 (世界 '{world_name}'): 簇 {i} 的摘要嵌入生成失败。跳过此簇。")
            continue
            
        new_worldview_texts_with_ids[next_new_id] = {
            "full_text": summary_text_content,
            "summary_text": None # 压缩后的条目，其 full_text 就是摘要，所以 summary_text 设为 None
        }
        new_worldview_embeddings_list.append(summary_embedding)
        print(f"世界 '{world_name}' 的簇 {i} 已总结。原始文本数: {len(texts_in_cluster)}, 摘要长度: {len(summary_text_content)}")
        next_new_id += 1

    if not new_worldview_texts_with_ids or not new_worldview_embeddings_list:
        return f"世界 '{world_name}' 的压缩未能产生新的世界观条目。原始数据可能过于稀疏或LLM总结失败。"

    new_worldview_embeddings_np = np.array(new_worldview_embeddings_list, dtype=np.float32)

    print(f"正在为世界 '{world_name}' 重建FAISS索引和文本映射...")
    rebuild_worldview_from_data(new_worldview_texts_with_ids, new_worldview_embeddings_np) # 这会作用于当前活动世界
    
    final_size = get_worldview_size()
    return f"世界 '{world_name}' 的世界观压缩完成。条目数从 {current_size} 减少到 {final_size}。"


if __name__ == '__main__':
    # 此测试依赖于 database_utils.py 中的 __main__ 先创建和填充一些世界的数据
    # 并确保 Ollama 服务运行，相关模型可用
    print("运行压缩工具测试 (需要先通过 database_utils.py 创建并激活一个世界)...")
    
    # 假设 database_utils.py 的 __main__ 已经运行并激活了一个世界
    # 或者我们在这里手动激活一个
    from database_utils import switch_active_world, get_available_worlds, add_world, add_worldview_text

    # 确保必要的客户端已初始化
    try:
        from embedding_utils import get_model_embedding_dimension
        from llm_utils import get_ollama_client
        get_model_embedding_dimension()
        get_ollama_client()
    except Exception as e:
        print(f"测试压缩前初始化模型失败: {e}")
        exit()

    available_worlds = get_available_worlds()
    test_world_id = "compression_test_world"
    test_world_name = "压缩测试世界"

    if test_world_id not in available_worlds:
        print(add_world(test_world_id, test_world_name))
    
    if switch_active_world(test_world_id):
        print(f"已激活世界 '{get_world_display_name(_active_world_id)}' 进行压缩测试。")
        initial_size = get_worldview_size()
        print(f"压缩前大小: {initial_size}")

        if initial_size < 5: # 添加一些数据以确保可以压缩
            print("为压缩测试添加更多数据...")
            add_worldview_text("关于宇宙的奥秘，星球的诞生与毁灭。")
            add_worldview_text("黑洞是宇宙中最神秘的天体之一，具有巨大的引力。")
            add_worldview_text("人工智能技术正在飞速发展，深刻影响社会。")
            add_worldview_text("机器学习是人工智能的一个重要分支。")
            add_worldview_text("气候变化是全球面临的严峻挑战。")
            add_worldview_text("可再生能源是应对气候变化的关键。")
            initial_size = get_worldview_size()
            print(f"添加数据后的大小: {initial_size}")
        
        if initial_size > 1:
            # 确保 COMPRESSION_TARGET_CLUSTERS 合理，例如小于 initial_size
            # config.COMPRESSION_TARGET_CLUSTERS = max(2, initial_size // 2) # 仅为测试动态调整
            
            print("\n开始压缩测试...")
            message = compress_worldview_db_for_active_world(force_compress=True)
            print(message)
            print(f"压缩后大小: {get_worldview_size()}")
        else:
            print("数据不足，无法进行有意义的压缩测试。")
    else:
        print(f"无法激活世界 '{test_world_id}' 进行压缩测试。")