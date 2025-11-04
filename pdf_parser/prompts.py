import json
from typing import List, Dict, Any

def get_vlm_image_analysis_prompt():

    prompt_text = """
[任务1: 分类]
请根据以下分类对第一张图片进行分类，选择最符合的类别：
- Device Package
- Functional Block Diagram
- Application Circuit
- Timing Diagram
- Test Waveform
- Characteristic Graphs
- PCB Layout Guidelines
- Mechanical Characteristics
- Company_Logo
- Else

如果图片内容无明显的意义：例如只有文字信息，或是不符合上述任何类别，种类直接输出“Else”。

[任务2: 描述与标题]
结合图像，从PDF页面图像（第二张图片）中找到与第一张图片内容完全匹配的图表，描述第一张图片，并直接输出该图表的标题。
注意页面图像中可能包含多个图表、标题和总标题，优先返回具体图表的标题，例如：
总标题：Typical Performance Characteristics at 25 °C
图表标题：Figure 1. Resistance vs Forward Current @ 1 GHz
则应返回：Figure 1. Resistance vs Forward Current @ 1 GHz
如果没有具体的标题，则返回PDF页面图像中的总标题。
如果没有找到任何标题，请返回 'none'。

[任务3: 器件识别]
请你通过图像和第二页的页面图像，一并判断该图表适用于哪些电子器件。器件的名称不仅包括完整的器件型号（如 ALQ00013、ALQ00014），也可以是系列号（如 xxxx Series）、封装形式（如 SOT-23、TO-220、 -204）、产品代号、或图中出现的标注型号。你可以从图表中的图例、标题、标签、坐标轴名称或图中标注的文字中提取这些信息。
如果你能识别出多个相关型号或封装，请尽可能完整地列出，并输出为数组。
如果该图像适用于所有型号（例如整份手册通用、适用于全系列器件或所有封装），请将 "applicable_models" 设置为 ["all"]。
如果无法判断，请将 "applicable_models" 字段设置为空数组 []。

请严格按照以下 JSON 格式输出，不要增加多余的解释：
{{
  "classification": "选择的分类",
  "title": "图表标题",
  "description": "图片内容的描述",
  "applicable_models": ["型号1", "型号2"]
}}
"""
    return prompt_text


def get_model_extraction_prompt():
    """
    获取 LLM 型号抽取的提示词。
    """

    Model_PROMPT = """
请你从上方的电子器件数据手册段落中提取所有出现的具体型号，并判断该段是否为公共信息。
要求：
1. "models_found" 是此文本段落内出现的所有具体电子器件型号，必须输出所有在文本中真实存在的器件型号，不允许构造不存在的器件型号，如果没有任何型号出现，则返回一个空数组 []。
   - 请只提取段落中出现的“主电子器件型号”，即主要介绍的器件型号，不要提取用于外围或配套的被动元件（如磁珠MPZ、电阻、电容GRM等，如果该段落就是这类器件的数据手册则这一要求可以省略）；
   - 不要提取出现在订购信息中的器件型号，例如出现 "Ordering Information" 等字段；
   - 避免提取以下类型的标识符，除非它们本身是该数据手册片段的核心介绍对象：
     - 用于替换的型号：例如，在描述中明确提到“可替换 XXX 型号”。
     - 应用电路中使用的辅助元器件：例如，推荐电路中使用的其他晶体管、电阻、电感等。
     - 封装或外壳型号。
     - PCB布局建议编号或名称。
     - PCB设计中要求出现的器件名称。
     - 环境、测试标准或认证编号。
     - 测试设备型号或厂商名称。
     - 数据手册自身的文档编号或版本号。
   - 数据手册内容是从 PDF 自动转换为 Markdown，可能存在由于格式问题造成的拼写连写或格式错误，特别是在表格中，多个型号可能会被无空格、无标点连写在一起，实际上是两个型号，例如 "74VHCT245AFT#74VHCV245FT#" 实际上是74VHCT245AFT#和74VHCV245FT#，需要分开处理；
   - 请根据型号命名规则的合理性、编号结构的明显差异，尝试识别并拆分这类连写内容，尽可能准确识别多个型号；
   - 遇到这类情况时，只要合理拆分出的部分看起来像真实型号，也请一并加入到 `models_found` 中（但仍需符合上述“主器件”的判断标准）；
2. 电子器件型号通常具有如下特征：
   - 器件名称包含大写字母 + 数字；
   - 相邻多个型号通常结构相似，可能只相差中间几位字符；
3. 如果该段落似乎是公共信息（即对所有器件通用或并未特指某一型号），请把 "possible_common" 设为 true。可以参考以下判断标准：
   - 若文本明确使用「Series / All Devices / Entire Family / Whole Series / All xxxx」等描述，表示其适用于该系列全部型号，或普适于整份数据手册里的所有器件，则可认为是公共信息。
   - 如果文本一并提及多个子型号，但使用统一描述或强调它们拥有相同特征，也可以视为公共信息。
   - 该段落中仅描述通用特征、共性参数、普适标准，而未提及任何具体型号；
   - 标题或内容明确指明这是全系列共用特性或适用范围（如“绝对最大额定值”、“电气特性”等）；
   - 文本中出现表格，且表格标题或内容表明适用于多个型号（例如“Table 1. xxxx Series Absolute Maximum Ratings”）；
   - 文本中未提及具体型号，但描述了与器件相关的通用信息（如封装类型、工作温度范围等），且未限定到某个单一型号，也可能是公共信息。
   - 其他能表明这是普遍适用于本手册所有器件的描述。
4. 如果以上公共信息的条件不满足，请将 "possible_common" 设为 false。
   -如果文本只提到某个具体型号且没有使用任何“全系列/所有型号”的语言，则应视为非公共信息 (possible_common=false)。
请严格按照以下格式输出JSON结构，禁止输出任何多余解释文字：

{
  "models_found": [
    "型号1",
    "型号2",
    ...
    ]
  "possible_common": true/false
}
"""
    return Model_PROMPT


# --- 阶段5：器件融合提示词 ---

def get_model_merging_tool_def():
    """
    返回 LLM 器件融合 Tool 的定义。
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "restructure_model_group",
                "description": (
                    "Based on the provided group of models, define merge and delete operations. "
                    "A 'series' model's information should typically be merged into its specific variants, "
                    "and the series model might then be deleted. Specific variants usually remain distinct entries."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operations": {
                            "type": "array",
                            "description": "List of merge or delete operations for the models in the input group.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "action": {"type": "string", "enum": ["merge", "delete"]},
                                    "target": {"type": "string",
                                               "description": "The target model name for the action. Must be one of the models in the input group."},
                                    "sources": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "For 'merge' action, list of source model names whose chunks are merged into target. Must be from input group."
                                    }
                                },
                                "required": ["action", "target"]
                            }
                        }
                    },
                    "required": ["operations"]
                }
            }
        }
    ]


def get_model_merging_system_prompt():
    """
    返回 LLM 器件融合的 System Prompt。
    """
    return (
        "You are an expert in electronic component model names and data sheet interpretation. "
        "You will first be given a consolidated block of context texts, where each context is sourced from a specific document chunk ID. "
        "Then, you will receive a list of potentially related model names. These models are grouped because their cleaned versions are identical or one is a subsequence of another. For each model name, a 'primary_context_source_chunk_id' will indicate which chunk ID (from the consolidated block) its main context was derived from. If this ID is null, that model had no specific context chunk associated. "
        "Your task is to analyze all this information and decide how to consolidate the listed models by defining 'merge' and 'delete' operations. "
        "Key principles: "
        "1. 'Series' or 'family' models (e.g., 'APD0505') often contain general information. This information should be merged into EACH of its specific variant models (e.g., 'APD0505-203', 'APD0505-210'). The original series model might then be deleted if all its unique information is distributed. "
        "2. Specific variant models (e.g., 'APD0505-203') should typically remain as distinct entities, possibly augmented with information from a series model. "
        "3. Handling Package or Minor Variants: Models within a group (e.g., 'RN2102', 'RN2102MFV') might differ by suffixes or other additions (like 'MFV', 'FT', 'FU') that frequently indicate specific package types, grades, or slight functional variations. Crucially, if the provided context, especially tables listing part numbers (like ordering information or package options), shows these models as separate line items, in different package columns, or otherwise clearly distinguished as individual orderable options, they MUST be treated as distinct entities. In such cases, you should NOT propose merge operations between them simply because their names are similar or one is a subsequence of the other. These models should remain separate in your output (achieved by proposing no operations for them or ensuring they are not sources in a merge). A merge operation involving such variants is only appropriate if the context explicitly states one directly replaces the other, one is a fully interchangeable alias, or one is clearly a non-orderable base series name whose entire definition is only fleshed out by its specific, orderable suffixed variants (and this is not just a package difference). If, in a rare and explicitly justified case based on the text, a merge of such variants is performed, carefully select the target. "
        "4. Models like '-210' (which are just suffixes or parts of other models) should have their information merged into the relevant complete model(s) within the group. "
        "5. Distinguishing Datasheet Subject vs. Orderable Part Number: If one model name (e.g., 'TMAG6180-Q1') appears to be the primary subject of the datasheet (often found in general descriptions, features, and document titles, and might be shorter or contain hyphens), and another model name within the group (e.g., 'TMAG6180EDGKRQ1') appears to be a specific 'Orderable Device' code or Manufacturer Part Number (MPN) for it (often found in package/ordering tables, may be longer, and might lack hyphens), the 'datasheet subject' name (e.g., 'TMAG6180-Q1') should generally be the `target_model_to_keep`. The 'Orderable Device' name's information should be merged into it, and the 'Orderable Device' name itself should be listed in `sources` (to be merged and then conceptually discarded). "
        "All model names used in 'target' or 'sources' MUST be from the initially provided list of model names for this group. "
        "Output a list of operations using the 'restructure_model_group' function. If you determine that no models in the provided group need to be merged or deleted with each other, call the function with an empty 'operations' list."
    )


def get_model_merging_user_prompt(
        consolidated_contexts_string: str,
        model_references_for_llm: list,
        model_names_in_group: list
) -> str:
    """
    构建 LLM 器件融合的 User Prompt。
    """
    prompt_model_list_summary = []
    for m in model_references_for_llm:
        prompt_model_list_summary.append({
            "model_name": m["name"],
            "primary_context_source_chunk_id": m["selected_chunk_id_for_context"]
        })

    return (
            "Here is the consolidated context from relevant document chunks. Each context is demarcated by '--- Context from Chunk ID X ---' or indicates if context was not found for a chunk ID:\n\n"
            f"{consolidated_contexts_string}\n\n"
            "Now, please analyze the following group of component models. For each model, 'primary_context_source_chunk_id' indicates which of the above contexts is most relevant to it (or null if none was specifically selected):\n\n"
            + json.dumps(prompt_model_list_summary, ensure_ascii=False, indent=2) + "\n\n"
                                                                                    f"The full list of model names in this group is: {model_names_in_group}\n"
                                                                                    "Remember, all 'target' and 'sources' in your operations must be from this list. "
                                                                                    "Call the 'restructure_model_group' function."
    )


# --- 阶段6：参数提取提示词 ---

def build_single_device_prompt(device_name: str) -> str:
    return f"""
你是一位电子器件资料分析专家。请从以上文本中提取与器件型号 `{device_name}` 相关的所有参数和通用的器件信息。该文本仅包含 `{device_name}` 一个型号。

提取要求如下：
- 若文中存在符号难以理解的表达，请转换为更清晰易懂的形式；
- 如果参数仅在特定条件下成立（例如“Ta=25℃”），请在参数名称中注明该条件，如 `"Forward Voltage @Ta=25℃"`，不要将特定条件作为参数值处理；如果参数没有标注条件则无需在参数名中注明。
- 若参数是最小值（Min）、典型值（Typ）或最大值（Max），请在参数名称中标注，例如 `"Capacitance (Typ)"`、`"Leakage Current (Max)"`,如果文中没有明确说明则无需标注；
- 出现描述性信息，也放置在参数名称中标注，键值只需要对应的参数值。
- 如果某字段在文本中未出现，请省略该字段。
- 只输出合法 JSON，结构层次清晰，键值对准确表达语义；
- 如果参数值大于0，可以省略正号 "+"；如果参数值小于0，必须填写负号 "-"。
- 如果参数值中含有单位，请确保数值与单位中有一个空格，例如："Junction Temperature": "-130°C to 130°C" → "Junction Temperature": "-130 °C to 130 °C"；
- 对于 "Parameters" 字段下，可以抽取所有与器件相关的参数信息，但只需要文本中明确提及的参数项及其对应数值；
- 不用处理 "封装尺寸" 的相关信息。
- 如果参数以表格形式给出（例如 S 参数随频率变化的表格），请尽可能提取**全部频率点**，所有频点的格式必须保持一致。将每一个频点下的所有参数按以下格式展开为扁平化键值对：
  示例：
  "S11 (Mag) @ Vds=3V, Ids=15mA, Freq=1.0 GHz": "0.79",
  "S21 (dB) @ Vds=3V, Ids=15mA, Freq=1.0 GHz": "20.68",
  ...

请严格按照以下结构输出：

{{
  "Manufacturer": {{
    "Name": "生产厂商"
  }},
  "Package type": {{
    "Package": "封装类型1"
  }},
  "Basic information": {{
    "Description": "器件的简要描述",
    "Function": "功能描述1,功能描述2,...",
    "Applications": "适用的应用1,适用的应用2,..."
  }},
  "Pin Configuration": {{
    "Pin 1": "引脚名称1(引脚功能1)",
    "Pin 2": "引脚名称2(引脚功能2)"
  }},
  "Parameters": {{
    "参数名称1": "参数值1",
    "参数名称2": "参数值2"
  }},
  "Order Information": {{
    "购买信息1": "数据1",
    "购买信息2": "数据2"
  }}
}}

请确保最终输出为合法 JSON，不包含任何解释性文本或注释。
"""


def build_device_prompt(device_name: str) -> str:
    return f"""
请从上述文本中抽取与器件型号 `{device_name}` 相关的结构化参数，并以 JSON 格式输出结果。

提取要求如下：
- 仅提取与 `{device_name}` 明确相关或可合理适用的内容，不要抽取与 `{device_name}` 无关的参数信息；
- 若文中存在符号难以理解的表达，请转换为更清晰易懂的形式；
- 如果参数仅在特定条件下成立（例如“Ta=25℃”），请在参数名称中注明该条件，如 `"Forward Voltage @Ta=25℃"`，不要将特定条件作为参数值处理；如果参数没有标注条件则无需在参数名中注明。
- 若参数是最小值（Min）、典型值（Typ）或最大值（Max），请在参数名称中标注，例如 `"Capacitance (Typ)"`、`"Leakage Current (Max)"` ,如果文中没有明确说明则无需标注；
- 出现描述性信息，也放置在参数名称中标注，键值只需要对应的参数值。
- 如果某字段在文本中未出现，请省略该字段。
- 只输出合法 JSON，结构层次清晰，键值对准确表达语义；
- 如果参数值大于0，可以省略正号 "+"；如果参数值小于0，必须填写负号 "-"。
- 如果参数值中含有单位，请确保数值与单位中有一个空格，例如："Junction Temperature": "-130°C to 130°C" → "Junction Temperature": "-130 °C to 130 °C"；
- 对于 "Parameters" 字段下，可以抽取所有与器件相关的参数信息，但只需要文本中明确提及的参数项及其对应数值；
- 不用处理 "封装尺寸" 的相关信息。
- 如果参数以表格形式给出（例如 S 参数随频率变化的表格），请尽可能提取**全部频率点**，所有频点的格式必须保持一致。将每一个频点下的所有参数按以下格式展开为扁平化键值对：
  示例：
  "S11 (Mag) @ Vds=3V, Ids=15mA, Freq=1.0 GHz": "0.79",
  "S21 (dB) @ Vds=3V, Ids=15mA, Freq=1.0 GHz": "20.68",
  ...
请严格按照以下结构输出，示例格式（仅供参考）：

{{
  "Manufacturer": {{
    "Name": "生产厂商"
  }},
  "Package type": {{
    "Package": "封装类型1"
  }},
  "Basic information": {{
    "Description": "器件的简要描述",
    "Function": "功能描述1,功能描述2,...",
    "Applications": "适用的应用1,适用的应用2,..."
  }},
  "Pin Configuration": {{
    "Pin 1": "引脚名称1(引脚功能1)",
    "Pin 2": "引脚名称2(引脚功能2)"
  }},
  "Parameters": {{
    "参数名称1": "参数值1",
    "参数名称2": "参数值2"
  }},
  "Order Information": {{
    "购买信息1": "数据1",
    "购买信息2": "数据2"
  }}
}}

请确保最终输出为合法 JSON，不包含任何解释性文本或注释。
"""


# --- 阶段7 (步骤2)：参数融合细化 (Refinement) ---

def get_param_refinement_tool_def():
    """
    返回 LLM 参数融合细化 Tool 的定义。
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "record_resolved_parameter",
                "description": "记录分析后解析/合并的参数信息。必须调用此函数以输出对每个参数的分析结果。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "parameter_path": {"type": "string", "description": "被分析参数的完整路径。"},
                        "resolution_strategy_applied": {
                            "type": "string",
                            "enum": ["MERGED_SEMANTICALLY_EQUIVALENT", "DECIDED_FROM_CONFLICT", "CONSOLIDATED_LIST",
                                     "NORMALIZED_KEY", "KEPT_AS_IS_NO_CONFLICT", "NO_RESOLUTION_FLAG_FOR_REVIEW"],
                            "description": "用于解析参数值的策略。"
                        },
                        "final_entries": {
                            "type": "array",
                            "description": "参数的最终解析条目列表。",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "value": {"type": ["string", "number", "boolean", "null"],
                                              "description": "解析后的值。"},
                                    "source_chunks": {"type": "array", "items": {"type": "string"},
                                                      "description": "来源 chunk ID 列表。"},
                                    "reasoning": {"type": "string", "description": "决策说明。"}
                                },
                                "required": ["value", "source_chunks", "reasoning"]
                            }
                        },
                        "normalized_key": {"type": "string", "description": "标准化的键名（如果适用）。"}
                    },
                    "required": ["parameter_path", "resolution_strategy_applied", "final_entries"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_chunk_contexts",
                "description": "当需要解决冲突且当前值不足以判断时，调用此函数请求特定 chunk ID 的文本上下文。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "parameter_path": {"type": "string", "description": "正在处理的参数路径。"},
                        "chunk_ids_for_context": {"type": "array", "items": {"type": "string"},
                                                  "description": "需要上下文的 chunk ID 列表 (LLM的建议，脚本可能会基于此扩展范围)。"},
                        "reason_for_request": {"type": "string", "description": "请求上下文的原因。"}
                    },
                    "required": ["parameter_path", "chunk_ids_for_context", "reason_for_request"]
                }
            }
        }
    ]


def get_param_refinement_system_prompt(device_model_name: str, is_follow_up_call: bool = False) -> str:
    """
    构建 LLM 参数融合细化的 System Prompt。
    """

    specific_instructions = ""
    if is_follow_up_call:
        specific_instructions = (
            "You previously requested text contexts for some chunks. "
            "The (potentially de-duplicated and combined) context is now provided. "
            "Using all this information, you MUST now make a final decision and call the `record_resolved_parameter` function. "
            "Do NOT call `get_chunk_contexts` again for this parameter."
        )

    template = f"""You are an expert data processing assistant specializing in fusing and refining structured information extracted from electronic component datasheets.
You are currently processing data for the device model: '{device_model_name}'.
Your goal is to ensure data accuracy, consistency, and completeness for this specific device.
For each parameter presented to you, analyze the candidate values.
Apply the specified 'parameter type hint' to guide your resolution strategy.
Be concise and accurate in your reasoning.
For package types like "DFN 5 x 6" and "DFN 5 x6mm (MS)", consider them equivalent and normalize to a consistent format like "DFN 5x6mm" unless context strongly suggests otherwise.
**Preference for Full Names:** When an abbreviation and its full name (e.g., "D" and "Drain", "G" and "Gate", "Celeritek" and "CELERITEK") are deemed equivalent for a parameter, you **MUST** prefer the full name or the most canonical/formal representation (e.g., proper capitalization for names) as the final resolved value and consolidate all their source_chunks.
If a value is explicitly not found or not applicable from the context, represent its 'value' as JSON null.
IMPORTANT: When you provide the 'value' in the 'final_entries' of the 'record_resolved_parameter' function, ensure it is ALWAYS a string. For example, if a quantity is 5000, return "5000". If it's a boolean true, return "true". If it's null, use JSON null (which will be stringified to "null" by the script later).
{specific_instructions}"""

    return template


def get_param_refinement_user_prompt(
        parameter_path: str,
        values_with_source_ids_only: List[Dict],
        parameter_type_hint: str,
        is_follow_up_call: bool = False,
        combined_requested_context_str: str = ""
) -> str:
    """
    构建 LLM 参数融合细化的 User Prompt (首次调用或后续调用)。
    """

    # --- 指导方针 ---
    guidelines = []
    if is_follow_up_call:
        guidelines.append(
            "You have now received context. Use this context to make your final decision for the specified device model. Your primary goal is to resolve the initial list of candidate values into a *single, canonical entry* for this parameter if appropriate (unless hint is 'list_enhancement'), or a consolidated list if the hint is 'list_enhancement'."
        )
        if parameter_type_hint == "list_enhancement":
            guidelines.append(
                "- For 'list_enhancement': Based on the context and original values, identify all unique items. Normalize similar items. Combine all into a *single, comma-separated string*. Call `record_resolved_parameter` with `resolution_strategy_applied` as 'CONSOLIDATED_LIST' and a `final_entries` array containing *exactly one object* where 'value' is this string."
            )
        else:
            guidelines.append(
                "1.  **Find a Single Best/Correct Entry:** Does the context help you identify a single, most accurate/appropriate value for the specified device model? This could be by: \n"
                "    a.  Confirming one value is correct and others are typos/errors.\n"
                "    b.  Confirming an abbreviation and its full name are equivalent (in which case, **you MUST choose the full name** and consolidate source_chunks).\n"
                "    c.  Identifying a super-set or most complete description if multiple text snippets are provided for a single descriptive field.\n"
                "    If YES to any of these: Call `record_resolved_parameter` with a single entry in `final_entries`. Set `resolution_strategy_applied` to 'MERGED_SEMANTICALLY_EQUIVALENT' or 'DECIDED_FROM_CONFLICT'."
            )
            guidelines.append(
                "2.  **If No Single Entry Can Be Determined (Conflict Persists or Multiple Distinct Valid Values):** If, even with context, the values are genuinely conflicting and represent distinct pieces of information that *cannot be merged into one single representative value for this parameter path for the specified device model* (e.g., a device has two truly different, valid package types that are not variants of each other but distinct options), OR if the conflict remains unresolvable: \n"
                "    Call `record_resolved_parameter` with `resolution_strategy_applied` as 'NO_RESOLUTION_FLAG_FOR_REVIEW'. The `final_entries` in this case should be the original list of conflicting values you were given (each value object as a separate entry in the final_entries list), each with reasoning explaining the persistent conflict or why they are distinct valid options that cannot be merged."
            )
    else:  # 首次调用
        guidelines.extend([
            "- 'single_value_equivalence': Your goal is to find a single, canonical representation for the parameter of the specified device model. \n"
            "  a. **Direct Merge for Obvious Equivalents:** If candidate values are *clearly and unambiguously* minor variations of the same information (e.g., only capitalization differences like 'Celeritek' vs 'CELERITEK'; minor spacing/punctuation like 'Isolink, Inc.' vs 'Isolink Inc'; or very common, standard abbreviations like 'V' for 'Volts' for a voltage parameter, 'D' for 'Drain' for a pin name), AND you are **highly confident** no context is needed to confirm this, then call `record_resolved_parameter` directly. Choose the most complete/formal/standard representation (e.g., 'Celeritek', 'Isolink, Inc.', 'Drain', 'Volts') and consolidate all their `source_chunks`.\n"
            "  b. **Request Context for Non-Obvious Differences:** If values are *different in a more substantial way* (e.g., 'TO252' vs 'TO255-2L'), or if an abbreviation is not universally standard for this parameter type, or if you have *any doubt* about their equivalence without context, your default action is to call `get_chunk_contexts`.",
            "- 'single_value_conflict_resolution': For genuinely different values for the parameter of the specified device model, your **primary action is to call `get_chunk_contexts`** to examine context. Only if *absolutely certain initially* that context is useless, may you call `record_resolved_parameter` to report conflict.",
            "- 'key_normalization': If normalizing this key, what would it be? Store in 'normalized_key'.",
            "- 'KEPT_AS_IS_NO_CONFLICT': Single value, no changes needed (this hint is determined by the script, LLM won't be called if this is the case).",
            "- 'NO_RESOLUTION_FLAG_FOR_REVIEW': If confident resolution is not possible even after context (more relevant for follow-up)."
        ])
        if parameter_type_hint == "list_enhancement":
            guidelines.append(
                "- 'list_enhancement' (initial): Your **primary goal is to directly synthesize** all provided textual descriptions into a single, comprehensive, de-duplicated, comma-separated string. Call `record_resolved_parameter` with `resolution_strategy_applied` as 'CONSOLIDATED_LIST' and this single string as the value in a single entry. Only if the provided text values contain *direct and irreconcilable contradictions* or are so ambiguous that a coherent summary *cannot* be formed from the values alone, should you then call `get_chunk_contexts`."
            )
        else:
            guidelines.append(
                "- 'list_enhancement' (general description if not current hint): Consolidate items into a de-duplicated list, usually as a single comma-separated string."
            )

    guidelines_str = "\n".join(guidelines)

    # --- 构建最终 Prompt ---
    if is_follow_up_call:
        return f"""For parameter path: '{parameter_path}' (hint: '{parameter_type_hint}') for the device model specified in the system instructions, you requested context.
Original candidate values (each with its representative source chunk IDs):
{json.dumps(values_with_source_ids_only, indent=2, ensure_ascii=False)}
Requested text context (de-duplicated & combined from ALL representative chunks of ALL conflicting values):
--- BEGIN CONTEXT ---
{combined_requested_context_str}
--- END CONTEXT ---
Based on the original values AND the provided context, follow these guidelines to determine the parameter value for the specified device model:
{guidelines_str}
Now, make a final decision and call `record_resolved_parameter`.
REMINDER: 'value' in 'final_entries' must be a string (or JSON null which script will stringify).
"""
    else:  # 首次调用
        return f"""Please analyze and resolve the parameter at path: '{parameter_path}' for the device model specified in the system instructions.
The parameter type hint for your processing is: '{parameter_type_hint}'.

Candidate values (each with its representative source chunk IDs, which might have been pre-filtered by the script if many original values were present):
{json.dumps(values_with_source_ids_only, indent=2, ensure_ascii=False)}

Follow these specific guidelines based on the hint:
{guidelines_str}

**Decision Steps for this Initial Call:**
1.  **Assess Values:** Look at the candidate values provided.
2.  **If hint is 'list_enhancement':**
    * Your **primary action** is to directly synthesize all provided textual descriptions into a single, comprehensive, de-duplicated, comma-separated string. Call `record_resolved_parameter` with this single string.
    * Only if the provided text values contain *direct and irreconcilable contradictions* or are so ambiguous that a coherent summary *cannot* be formed from the values alone, should you then call `get_chunk_contexts`.
3.  **If hint is NOT 'list_enhancement':**
    a.  **Clear Semantic Equivalence (including standard abbreviations like capitalization or minor punctuation, or common pin/unit abbreviations)?** Are the values clearly minor, purely stylistic variations of the exact same information (e.g., "Celeritek" vs "CELERITEK", "50V" vs "50 V") 
        OR a common, standard, and unambiguous abbreviation and its full name specifically for *this type of parameter* (e.g., for a pin name, "D" and "Drain"; for a unit, "V" and "Volts")?
        If YES, AND you are **highly confident** that no further context is needed to confirm this equivalence for the specified device model: 
        Call `record_resolved_parameter`. **Ensure you select the full name or most canonical form as the `value` and consolidate all `source_chunks`** from the original, pre-filtered list of source chunks associated with these equivalent values.
    b.  **Context Needed (Default for other differences/conflicts/ambiguities):** If values are meaningfully different (e.g., "TO252" vs "TO255-2L", "4000" vs "5000"), 
        OR if their relationship is ambiguous, 
        OR if it's an abbreviation case but you are *not highly confident* it's a standard, unambiguous one for this parameter without seeing context for the specified device model: 
        Your **default and primary action** is to call `get_chunk_contexts`. Provide all representative chunk IDs from ALL conflicting/ambiguous values.
    c.  **(Very Rare Exception) Reporting Unresolvable Conflict *Without* Context:** Only if you are *absolutely certain* that: (i) The values are genuinely and irreconcilably conflicting for the specified device model, AND (ii) No amount of chunk context could *possibly* help clarify, explain the difference, or confirm if multiple distinct options are valid.
        Then, and only then, may you call `record_resolved_parameter` to report the conflict. **Prefer `get_chunk_contexts` if any doubt.**

You MUST call either `record_resolved_parameter` or `get_chunk_contexts`.
REMINDER: 'value' in 'final_entries' must be a string (or JSON null which script will stringify).
"""