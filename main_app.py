import gradio as gr
import time
import database_utils

from config import (
    COMPRESSION_THRESHOLD, OLLAMA_MODEL, OLLAMA_EMBEDDING_MODEL_NAME
)
from database_utils import (
    add_world, get_available_worlds, switch_active_world, get_world_display_name, delete_world,
    add_character, get_character, get_character_names, get_all_characters, delete_character,
    add_worldview_text, search_worldview, get_worldview_size, SIMILARITY_TOP_K
)
from llm_utils import generate_text, get_ollama_client
from embedding_utils import get_model_embedding_dimension
from compression_utils import compress_worldview_db_for_active_world

_initial_world_activated_on_startup = False
_initial_active_world_id_on_startup = None

try:
    print("正在执行 main_app.py 的初始化...")
    get_model_embedding_dimension()
    get_ollama_client()
    available_worlds_init = get_available_worlds()
    if available_worlds_init:
        first_world_id = list(available_worlds_init.keys())[0]
        if switch_active_world(first_world_id):
            _initial_world_activated_on_startup = True
            _initial_active_world_id_on_startup = first_world_id
            print(f"已在启动时自动激活默认世界: '{get_world_display_name(first_world_id)}'")
        else:
            print(f"警告：启动时未能自动激活默认世界 '{first_world_id}'。")
    else:
        print("提示：当前没有已创建的世界。请在'世界管理'标签页中添加新世界。")
    print("main_app.py 初始化完成。")
except Exception as e:
    print(f"致命错误：main_app.py 初始设置期间出错: {e}")

def refresh_world_dropdown_choices_for_gradio():
    available_worlds = get_available_worlds()
    return [(name, id) for id, name in available_worlds.items()]

def refresh_character_dropdown_choices():
    return get_character_names()

def refresh_worldview_status_display_text():
    if not database_utils._active_world_id:
        return "无活动世界。世界观信息不可用。"
    size = get_worldview_size()
    world_name = get_world_display_name(database_utils._active_world_id)
    status_text = f"世界 '{world_name}' | 世界观条目数: {size}. "
    ct = COMPRESSION_THRESHOLD
    if isinstance(ct, (int, float)) and ct > 0:
        if size > ct: status_text += f"建议压缩 (阈值: {ct})."
        else: status_text += f"压缩阈值: {ct}."
    else: status_text += "压缩阈值未有效配置。"
    return status_text

def get_active_world_markdown_text_for_global_display():
    if database_utils._active_world_id:
        return f"当前活动世界: **'{get_world_display_name(database_utils._active_world_id)}'** (ID: `{database_utils._active_world_id}`)"
    else:
        return "<p style='color:orange;'>当前无活动世界。请从上方选择或在“世界管理”中创建一个新世界。</p>"

def clear_textboxes_and_checkboxes(*args):
    updates = []
    for arg in args:
        if isinstance(arg, gr.Checkbox):
            updates.append(gr.update(value=False))
        else:
            updates.append(gr.update(value=""))
    return tuple(updates) if len(updates) > 1 else updates[0] if updates else gr.update()

def update_ui_after_world_change(feedback_message: str = "", feedback_component_elem_id=None):
    world_choices_for_dd = refresh_world_dropdown_choices_for_gradio()
    current_active_id = database_utils._active_world_id

    char_choices = refresh_character_dropdown_choices()
    wv_status_text = refresh_worldview_status_display_text()
    global_md_text = get_active_world_markdown_text_for_global_display()
    is_world_active = current_active_id is not None

    updates = {
        "world_select_dropdown": gr.update(choices=world_choices_for_dd, value=current_active_id),
        "global_active_world_display": gr.update(value=global_md_text),
        "char_select_dropdown_pred_tab": gr.update(choices=char_choices, value=None, interactive=is_world_active),
        "character_select_for_delete_dropdown": gr.update(choices=char_choices, value=None, interactive=is_world_active),
        "worldview_status_display": gr.update(value=wv_status_text),
        "char_name_input": gr.update(interactive=is_world_active, value=""),
        "char_full_desc_input": gr.update(interactive=is_world_active, value=""),
        "add_char_button": gr.update(interactive=is_world_active),
        "delete_char_button": gr.update(interactive=is_world_active),
        "view_chars_button": gr.update(interactive=is_world_active),
        "worldview_text_input": gr.update(interactive=is_world_active, value=""),
        "add_worldview_button": gr.update(interactive=is_world_active),
        "compress_worldview_button": gr.update(interactive=is_world_active),
        "situation_query_input": gr.update(interactive=is_world_active, value=""),
        "predict_button": gr.update(interactive=is_world_active),
        "delete_world_button": gr.update(interactive=is_world_active),
        "confirm_delete_world_checkbox": gr.update(interactive=is_world_active, value=False),
    }

    feedback_fields_to_clear = [
        "world_switch_feedback", "add_world_feedback_output", "delete_world_feedback_output",
        "char_op_feedback_output", "worldview_feedback_output",
        "compression_status_output", "prediction_output", "view_characters_output"
    ]
    for field_id in feedback_fields_to_clear:
        if feedback_component_elem_id and feedback_component_elem_id == field_id:
            updates[field_id] = gr.update(value=feedback_message)
        else:
            updates[field_id] = gr.update(value="") # Clear others if not the target feedback

    # Ensure the primary feedback is set if it was accidentally cleared above
    if feedback_component_elem_id and feedback_component_elem_id in updates:
         updates[feedback_component_elem_id] = gr.update(value=feedback_message)

    return updates

def handle_add_world(world_id_input: str, world_name_input: str):
    feedback_msg = ""
    if not world_id_input or not world_name_input:
        feedback_msg = "错误：世界ID和世界名称不能为空。"
    else:
        message = add_world(world_id_input, world_name_input)
        feedback_msg = message
        if "已添加" in message:
            if switch_active_world(world_id_input):
                world_name = get_world_display_name(world_id_input)
                feedback_msg += f" 并已激活 '{world_name}'。"
            else:
                feedback_msg += " 但激活失败，请手动选择。"
    all_updates = update_ui_after_world_change(feedback_msg, "add_world_feedback_output")
    return map_updates_to_ordered_list(all_updates, ordered_output_components_for_world_change)


def handle_delete_world(confirm_delete_checkbox: bool):
    feedback_msg = ""
    world_id_to_delete = database_utils._active_world_id
    if not world_id_to_delete:
        feedback_msg = "错误：没有活动的或选中的世界可供删除。"
    elif not confirm_delete_checkbox:
        feedback_msg = "错误：请勾选确认框以删除当前活动世界。"
    else:
        world_name_to_delete = get_world_display_name(world_id_to_delete)
        message = delete_world(world_id_to_delete)
        feedback_msg = message
        if "已成功删除" in message:
            feedback_msg = f"世界 '{world_name_to_delete}' 已被成功删除。"
    all_updates = update_ui_after_world_change(feedback_msg, "delete_world_feedback_output")
    return map_updates_to_ordered_list(all_updates, ordered_output_components_for_world_change)

def handle_switch_world(world_id_selected: str):
    feedback_msg = ""
    if not world_id_selected:
        if database_utils._active_world_id is not None:
            switch_active_world(None)
        feedback_msg = "已取消活动世界。"
    elif switch_active_world(world_id_selected):
        world_name = get_world_display_name(world_id_selected)
        feedback_msg = f"已激活世界: '{world_name}'"
    else:
        feedback_msg = f"切换到世界 '{get_world_display_name(world_id_selected) if world_id_selected else '选择项'}' 失败。"
    all_updates = update_ui_after_world_change(feedback_msg, "world_switch_feedback")
    return map_updates_to_ordered_list(all_updates, ordered_output_components_for_world_change)

def handle_add_character(name: str, full_description: str):
    feedback_msg = "请先选择并激活一个世界。"
    char_choices_update1 = gr.update()
    char_choices_update2 = gr.update()
    if database_utils._active_world_id:
        if not name or not full_description:
            feedback_msg = "角色名称和完整描述不能为空。"
        else:
            message = add_character(name, full_description)
            feedback_msg = message
            new_char_choices = refresh_character_dropdown_choices()
            char_choices_update1 = gr.update(choices=new_char_choices, value=None)
            char_choices_update2 = gr.update(choices=new_char_choices, value=None)
    return feedback_msg, char_choices_update1, char_choices_update2

def handle_delete_character(character_name_to_delete: str):
    feedback_msg = "错误：请先选择活动世界。"
    char_choices_update1 = gr.update()
    char_choices_update2 = gr.update()
    if database_utils._active_world_id:
        if not character_name_to_delete:
            feedback_msg = "错误：请从下拉列表中选择要删除的角色。"
        else:
            message = delete_character(character_name_to_delete)
            feedback_msg = message
            new_char_choices = refresh_character_dropdown_choices()
            char_choices_update1 = gr.update(choices=new_char_choices, value=None)
            char_choices_update2 = gr.update(choices=new_char_choices, value=None)
    return feedback_msg, char_choices_update1, char_choices_update2

def handle_view_characters():
    if not database_utils._active_world_id: return "请先选择并激活一个世界。"
    chars = get_all_characters()
    if not chars: return f"当前活动世界 '{get_world_display_name(database_utils._active_world_id)}' 中没有角色。"
    output = f"当前活动世界 '{get_world_display_name(database_utils._active_world_id)}' 的角色列表:\n" + "="*20 + "\n"
    for char_data in chars:
        desc_to_show = char_data.get('summary_description', char_data.get('full_description', '（无描述）'))
        output += f"  名称: {char_data['name']}\n  概要: {desc_to_show[:150].replace(chr(10), ' ')}...\n---\n"
    return output

def handle_add_worldview(text_input: str):
    feedback_msg = "请先选择并激活一个世界。"
    wv_status_update = gr.update()
    if database_utils._active_world_id:
        if not text_input:
            feedback_msg = "世界观文本不能为空。"
        else:
            message = add_worldview_text(text_input)
            feedback_msg = message
            auto_compress_message = ""
            current_size = get_worldview_size()
            ct = COMPRESSION_THRESHOLD
            if isinstance(ct, (int, float)) and ct > 0 and current_size > ct:
                if current_size > ct * 1.5:
                    compress_result = compress_worldview_db_for_active_world(force_compress=False)
                    auto_compress_message = f"\n自动压缩结果: {compress_result}"
                else:
                    auto_compress_message = f"\n提示: 世界观条目数 ({current_size}) 接近压缩阈值 ({ct})。可考虑手动压缩。"
            feedback_msg += auto_compress_message
        wv_status_update = gr.update(value=refresh_worldview_status_display_text())
    return feedback_msg, wv_status_update

def handle_compress_worldview_button():
    feedback_msg = "请先选择并激活一个世界才能进行压缩。"
    if not database_utils._active_world_id:
        yield feedback_msg
        return
    world_name = get_world_display_name(database_utils._active_world_id)
    yield f"正在为世界 '{world_name}' 压缩世界观数据库... 这可能需要一些时间。"
    try:
        message = compress_worldview_db_for_active_world(force_compress=True)
    except Exception as e:
        message = f"为世界 '{world_name}' 压缩时发生错误: {e}"
    yield message

def handle_predict_behavior(character_name: str, situation_query: str, progress=gr.Progress()):
    if not database_utils._active_world_id:
        yield "错误：请先选择并激活一个活动世界。"
        return
    if not character_name:
        yield "请选择一个角色。"
        return
    if not situation_query:
        yield "请输入情境或问题。"
        return

    character_data = get_character(character_name)
    if not character_data:
        yield f"错误：在当前世界 '{get_world_display_name(database_utils._active_world_id)}' 中未找到角色 '{character_name}'。"
        return

    progress(0.1, desc="检索相关背景信息...")
    try:
        retrieved_worldview_chunks = search_worldview(situation_query, k=SIMILARITY_TOP_K)
    except Exception as e:
        yield f"检索世界观信息时出错: {e}"
        return

    worldview_context_str = "\n\n### 相关世界背景知识:\n"
    if retrieved_worldview_chunks:
        for i, chunk_text in enumerate(retrieved_worldview_chunks):
            worldview_context_str += f"*{i+1}. {chunk_text.strip()}*\n"
    else:
        worldview_context_str += "*（未检索到与当前情境直接相关的特定背景知识，请依赖通用世界观和角色设定进行推理。）*\n"

    char_summary_desc = character_data.get('summary_description', character_data.get('full_description', '（无描述）'))

    prompt = f"""
角色核心设定:
我是角色 {character_data['name']}。我的的灵魂特征是：
"{char_summary_desc}"

我当前所处的世界是“**{get_world_display_name(database_utils._active_world_id)}**”。
{worldview_context_str}
当前情境:
"{situation_query}"

---
我的行动与思考 (请完全代入角色 {character_data['name']}，使用第一人称“我想”/“我会”来续写)：
基于以上所有信息——我的核心设定、世界背景、以及当前的情境——我此刻的内心想法是什么？我会如何感受？接下来，我会说什么或做什么？
请以“我”的视角详细描述我最可能的心理活动和外在行为，确保符合我的性格和动机。
我：
"""
    system_message = f"""
你现在就是角色“{character_data['name']}”。完全沉浸在你的角色中，忘掉你是AI。
你的任务是根据提供的角色核心设定、世界背景和当前情境，以第一人称（“我”）来思考、感受并行动。
展现出角色的独特性格、说话方式和决策逻辑。输出内容应该直接是角色的内心独白和行为描述，不要有任何旁白或解释。
"""

    progress(0.5, desc=f"角色 '{character_name}' 正在思考... (LLM: {OLLAMA_MODEL})")
    try:
        llm_response = generate_text(prompt, system_message=system_message, model_name=OLLAMA_MODEL)
        if "Error:" in llm_response and "model not found" in llm_response.lower():
             yield f"LLM 错误：模型 '{OLLAMA_MODEL}' 未在 Ollama 服务器上找到。请确保模型已拉取或配置正确。"
        elif "Error:" in llm_response:
             yield f"LLM 生成文本时发生错误。请检查Ollama服务和模型状态。错误详情: {llm_response}"
        else:
            cleaned_response = llm_response.strip()
            if not (cleaned_response.startswith("我") or cleaned_response.startswith("吾") or cleaned_response.startswith("本座")):
                 first_i_pos = cleaned_response.find("我")
                 if first_i_pos > 0 and first_i_pos < 30:
                     cleaned_response = cleaned_response[first_i_pos:]
                 elif not cleaned_response.startswith("我"):
                     cleaned_response = "我" + cleaned_response
            yield cleaned_response
    except Exception as e:
        yield f"调用LLM时发生严重错误: {e}"
    progress(1, desc="预测完成")


with gr.Blocks(theme=gr.themes.Glass(), title="多世界虚拟角色模拟器") as app:
    gr.Markdown(f"""
    # 🌌 多世界虚拟角色模拟器 🎭
    <p align="center">LLM: 🧠 <b>{OLLAMA_MODEL}</b> (扮演与总结) | 🔗 嵌入: <b>{OLLAMA_EMBEDDING_MODEL_NAME}</b></p>
    """)

    with gr.Row(variant="panel"):
        with gr.Column(scale=3):
            world_select_dropdown = gr.Dropdown(label="选择或切换活动世界", elem_id="world_select_dropdown", show_label=True)
        with gr.Column(scale=2):
            world_switch_feedback = gr.Textbox(label="世界操作反馈", interactive=False, elem_id="world_switch_feedback", show_label=True, max_lines=1)
    global_active_world_display = gr.Markdown(value=get_active_world_markdown_text_for_global_display(), elem_id="global_active_world_display")

    with gr.Tabs() as tabs_main:
        with gr.TabItem("🌍 世界管理", id="tab_world_management"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### ✨ 创建新世界")
                    new_world_id_input = gr.Textbox(label="世界ID (字母数字下划线)", placeholder="e.g., cyberpunk_2077", elem_id="new_world_id_input")
                    new_world_name_input = gr.Textbox(label="世界显示名称", placeholder="e.g., 赛博朋克 2077", elem_id="new_world_name_input")
                    add_world_button = gr.Button("创建并激活新世界", variant="primary", elem_id="add_world_button")
                    add_world_feedback_output = gr.Textbox(label="创建状态", interactive=False, elem_id="add_world_feedback_output", max_lines=2)
                with gr.Column(scale=1, min_width=300):
                    gr.Markdown("### 🗑️ 删除当前活动世界")
                    gr.HTML(
                        "<div style='padding: 10px; border: 1px solid red; border-radius: 5px; background-color: #fee; color: #c00;'>"
                        "<b>警告:</b> 此操作将永久删除当前选中的活动世界及其所有数据 (角色、世界观等)，无法恢复！"
                        "</div>"
                    )
                    confirm_delete_world_checkbox = gr.Checkbox(
                        label="我已了解风险，确认删除当前活动世界。", value=False,
                        elem_id="confirm_delete_world_checkbox", interactive=False
                    )
                    delete_world_button = gr.Button(
                        "永久删除此世界", variant="stop", elem_id="delete_world_button", interactive=False
                    )
                    delete_world_feedback_output = gr.Textbox(label="删除状态", interactive=False, elem_id="delete_world_feedback_output", max_lines=2)

        with gr.TabItem("👥 角色管理", id="tab_character_management"):
            gr.Markdown("管理当前活动世界的角色。如果无活动世界，请先选择或创建一个。")
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("#### 添加/更新角色")
                    char_name_input = gr.Textbox(label="角色名称", elem_id="char_name_input", interactive=False)
                    char_full_desc_input = gr.Textbox(label="角色完整描述 (将用于生成角色核心设定)", lines=5, elem_id="char_full_desc_input", interactive=False)
                    add_char_button = gr.Button("保存角色 (添加/更新)", variant="secondary", elem_id="add_char_button", interactive=False)
                with gr.Column(scale=1):
                    gr.Markdown("#### 删除角色")
                    character_select_for_delete_dropdown = gr.Dropdown(label="选择要删除的角色", elem_id="character_select_for_delete_dropdown", interactive=False)
                    delete_char_button = gr.Button("删除选中角色", variant="stop", elem_id="delete_char_button", interactive=False)
            char_op_feedback_output = gr.Textbox(label="角色操作状态", interactive=False, elem_id="char_op_feedback_output", lines=2, max_lines=3)
            gr.Markdown("---")
            gr.Markdown("#### 查看所有角色")
            view_chars_button = gr.Button("刷新查看当前世界角色列表", elem_id="view_chars_button", interactive=False)
            view_characters_output = gr.Textbox(label="角色列表", lines=8, interactive=False, elem_id="view_characters_output", show_copy_button=True)

        with gr.TabItem("📚 世界观管理", id="tab_worldview_management"):
            gr.Markdown("为当前活动世界添加和管理世界观知识。")
            worldview_text_input = gr.Textbox(label="添加世界观文本块 (会自动分块和嵌入)", lines=6, elem_id="worldview_text_input", interactive=False)
            add_worldview_button = gr.Button("添加文本到世界观", variant="secondary", elem_id="add_worldview_button", interactive=False)
            worldview_feedback_output = gr.Textbox(label="添加状态", interactive=False, elem_id="worldview_feedback_output", max_lines=2)
            worldview_status_display = gr.Textbox(label="世界观数据库状态", interactive=False, elem_id="worldview_status_display")
            gr.Markdown("---")
            compress_worldview_button = gr.Button("手动压缩当前世界观 (耗时操作)", elem_id="compress_worldview_button", interactive=False)
            compression_status_output = gr.Textbox(label="压缩结果", interactive=False, elem_id="compression_status_output", max_lines=2)

        with gr.TabItem("💬 交互与预测", id="tab_prediction"):
            gr.Markdown("选择角色，输入情境，观察LLM如何根据角色设定和世界观进行预测。")
            char_select_dropdown_pred_tab = gr.Dropdown(label="选择角色进行交互", elem_id="char_select_dropdown_pred_tab", interactive=False)
            situation_query_input = gr.Textbox(label="输入情境 / 对角色提问", lines=4, elem_id="situation_query_input", interactive=False)
            predict_button = gr.Button("🚀 预测角色行为/回应", variant="primary", elem_id="predict_button", interactive=False)
            prediction_output = gr.Textbox(label="LLM预测结果 (角色内心独白与行动)", lines=12, interactive=False, show_copy_button=True, elem_id="prediction_output")

    # --- Moved component list definition here, after all components are defined ---
    ordered_output_components_for_world_change = [
        world_select_dropdown, global_active_world_display, world_switch_feedback,
        new_world_id_input, new_world_name_input, add_world_button, add_world_feedback_output,
        confirm_delete_world_checkbox, delete_world_button, delete_world_feedback_output,
        char_name_input, char_full_desc_input, add_char_button,
        character_select_for_delete_dropdown, delete_char_button, char_op_feedback_output,
        view_chars_button, view_characters_output,
        worldview_text_input, add_worldview_button, worldview_feedback_output,
        worldview_status_display, compress_worldview_button, compression_status_output,
        char_select_dropdown_pred_tab, situation_query_input, predict_button, prediction_output
    ]

    # Helper function to map dictionary updates to an ordered list for Gradio outputs
    def map_updates_to_ordered_list(updates_dict, ordered_components_list):
        # Ensure all components in ordered_components_list have an elem_id or are directly usable as keys
        # For safety, we'll try to get elem_id, otherwise assume the component object itself is the key (less robust)
        return tuple(
            updates_dict.get(getattr(comp, 'elem_id', comp), gr.update()) for comp in ordered_components_list
        )

    # --- Event Binding ---
    world_select_dropdown.change(
        fn=handle_switch_world,
        inputs=[world_select_dropdown],
        # The outputs list here directly uses the ordered_output_components_for_world_change
        # handle_switch_world must return a tuple of updates in this exact order
        outputs=ordered_output_components_for_world_change,
        show_progress="full"
    )

    add_world_button.click(
        fn=handle_add_world,
        inputs=[new_world_id_input, new_world_name_input],
        outputs=ordered_output_components_for_world_change,
        show_progress="full"
    ).then(
        fn=lambda: clear_textboxes_and_checkboxes(new_world_id_input, new_world_name_input),
        outputs=[new_world_id_input, new_world_name_input]
    )

    delete_world_button.click(
        fn=handle_delete_world,
        inputs=[confirm_delete_world_checkbox],
        outputs=ordered_output_components_for_world_change,
        show_progress="full"
    )

    add_char_button.click(
        fn=handle_add_character,
        inputs=[char_name_input, char_full_desc_input],
        outputs=[char_op_feedback_output, char_select_dropdown_pred_tab, character_select_for_delete_dropdown],
        show_progress="full"
    ).then(
        fn=lambda: clear_textboxes_and_checkboxes(char_name_input, char_full_desc_input),
        outputs=[char_name_input, char_full_desc_input]
    )

    delete_char_button.click(
        fn=handle_delete_character,
        inputs=[character_select_for_delete_dropdown],
        outputs=[char_op_feedback_output, char_select_dropdown_pred_tab, character_select_for_delete_dropdown],
        show_progress="full"
    )

    view_chars_button.click(handle_view_characters, outputs=view_characters_output, show_progress="minimal")

    add_worldview_button.click(
        fn=handle_add_worldview,
        inputs=[worldview_text_input],
        outputs=[worldview_feedback_output, worldview_status_display],
        show_progress="full"
    ).then(
        fn=lambda: clear_textboxes_and_checkboxes(worldview_text_input), outputs=[worldview_text_input]
    )

    compress_worldview_button.click(
        fn=handle_compress_worldview_button,
        outputs=[compression_status_output],
        show_progress="full"
    ).then(
        fn=lambda: gr.update(value=refresh_worldview_status_display_text()),
        outputs=[worldview_status_display]
    )

    predict_button.click(
        fn=handle_predict_behavior,
        inputs=[char_select_dropdown_pred_tab, situation_query_input],
        outputs=[prediction_output],
        show_progress="full"
    )

    def initial_load_ui_elements():
        global _initial_active_world_id_on_startup
        current_true_active_id = _initial_active_world_id_on_startup
        if current_true_active_id and database_utils._active_world_id != current_true_active_id:
            if not switch_active_world(current_true_active_id):
                 print(f"App.load: 同步激活世界 '{current_true_active_id}' 失败。")
        all_updates = update_ui_after_world_change(feedback_message="应用已加载。", feedback_component_elem_id="world_switch_feedback")
        return map_updates_to_ordered_list(all_updates, ordered_output_components_for_world_change)

    app.load(
        fn=initial_load_ui_elements,
        outputs=ordered_output_components_for_world_change,
        show_progress="full"
    )

if __name__ == "__main__":
    print("正在启动 Gradio 多世界角色模拟器应用...")
    app.launch(inbrowser=True, server_name="0.0.0.0")