# rag_retriever/retriever.py

import logging
import json
import time
import re
import numpy as np
from openai import OpenAI
from collections import deque, defaultdict, Counter
from sklearn.preprocessing import minmax_scale
from django.conf import settings
from django.db.models import Q
from pgvector.django import L2Distance
import pandas as pd
from typing import List, Dict, Any, Tuple, Set
from django.conf import settings  # <--- 新增
import os  # <--- 新增

# 导入 pdf_parser 的模型
from pdf_parser.models import GraphNode, GraphEdge, GraphChunk, NodeSourceLink, PDFParsingTask
# 导入辅助函数和提示词
from pdf_parser.prompts import get_rag_entity_extraction_prompt, get_rag_final_answer_prompt
from pdf_parser.graph_construction import get_embeddings_batch  # 复用批量嵌入函数

logger = logging.getLogger(__name__)


class PgVectorRetriever:

    def __init__(self):
        logger.info("Initializing PgVector-based RAG Retriever...")

        # 1. 加载配置
        llm_conf = {
            "API_KEY": settings.DEFAULT_LLM_API_KEY,
            "API_URL": settings.DEFAULT_LLM_API_URL,
            "MODEL_NAME": settings.DEFAULT_LLM_MODEL_NAME
        }
        emb_conf = {
            "API_KEY": settings.DEFAULT_EMBEDDING_API_KEY,
            "API_URL": settings.DEFAULT_EMBEDDING_API_URL,
            "MODEL_NAME": settings.DEFAULT_EMBEDDING_MODEL_NAME,
            "DIMENSIONS": settings.DEFAULT_EMBEDDING_DIMENSIONS,
            "BATCH_SIZE": settings.DEFAULT_EMBEDDING_BATCH_SIZE
        }
        rag_conf = settings.PDF_PARSER_RAG

        # 2. 初始化 API 客户端
        self.embedding_client = OpenAI(api_key=emb_conf["API_KEY"], base_url=emb_conf["API_URL"])
        self.llm_client = OpenAI(api_key=llm_conf["API_KEY"], base_url=llm_conf["API_URL"])
        self.embedding_model = emb_conf["MODEL_NAME"]
        self.embedding_dim = emb_conf["DIMENSIONS"]
        self.embedding_batch_size = emb_conf["BATCH_SIZE"]
        self.llm_model = llm_conf["MODEL_NAME"]

        # 3. 加载 RAG 算法参数
        self.TOP_P_PER_ENTITY = rag_conf['TOP_P_PER_ENTITY']
        self.BFS_DEPTH = rag_conf['BFS_DEPTH']
        # self.TOP_K_ORPHANS_TO_BRIDGE = rag_conf['TOP_K_ORPHANS_TO_BRIDGE'] # <--- MODIFIED: 移除

        # 4. 加载评分权重
        self.CHUNK_SCORE_ALPHA = rag_conf['CHUNK_SCORE_ALPHA']
        self.SEED_DENSITY_BONUS = rag_conf['SEED_DENSITY_BONUS']
        self.TOP_REC_K_FOR_SIMILARITY = rag_conf['TOP_REC_K_FOR_SIMILARITY']
        self.STRONG_CHUNK_RECOMMENDATION_BONUS = rag_conf['STRONG_CHUNK_RECOMMENDATION_BONUS']
        self.WEAK_CHUNK_RECOMMENDATION_BONUS = rag_conf['WEAK_CHUNK_RECOMMENDATION_BONUS']
        self.TEXT_CONFIRMATION_BONUS = rag_conf['TEXT_CONFIRMATION_BONUS']
        # self.ENDORSEMENT_BASE_BONUS = rag_conf['ENDORSEMENT_BASE_BONUS'] # <--- MODIFIED: 移除
        # self.ENDORSEMENT_DECAY_FACTOR = rag_conf['ENDORSEMENT_DECAY_FACTOR'] # <--- MODIFIED: 移除

        # 5. 构建内存图（为BFS做准备）
        self.adj_list, self.edge_map = self._build_adjacency_list()

        logger.info("Retriever initialized successfully.")

    def _get_formatted_node_name(self, node: GraphNode) -> str:
        """
        根据节点类型格式化节点的可读名称。
        """
        if not node:
            return "Unknown"

        # 对于 'Device', 'Manufacturer', 'Category'，我们使用 "Community: Value"
        if node.community in ["Device", "Manufacturer", "Category"]:
            return f"{node.community}: {node.value}"

        # 对于 'Parameters', 'Image' 等，我们使用 "Name: Value"
        # (node.name 对应 'Forward Voltage', node.value 对应 '1.3 V')
        else:
            return f"{node.name}: {node.value}"

    def _get_embedding(self, text: str) -> np.ndarray:
        """获取单个嵌入向量并返回 np.ndarray"""
        emb_map = get_embeddings_batch(
            [text], self.embedding_client, self.embedding_model,
            self.embedding_dim, self.embedding_batch_size
        )
        emb = emb_map.get(text)
        return np.array(emb).astype('float32') if emb else None

    def _build_adjacency_list(self):
        """
        从 GraphEdge 表构建内存邻接表和边映射，用于快速BFS。
        """
        logger.info("Building in-memory graph adjacency list...")
        adj_list = defaultdict(set)
        edge_map = {}

        edges_query = GraphEdge.objects.all().values_list(
            'source_node_id',
            'target_node_id',
            'type',
            'id'  # 使用 edge.id 作为 relation_id
        )

        for source_id, target_id, rel_type, rel_id in edges_query:
            edge_key = tuple(sorted((source_id, target_id)))

            edge_map[edge_key] = {
                "relation_id": str(rel_id),
                "relation_type": rel_type,
                "description": rel_type
            }
            adj_list[source_id].add(target_id)
            adj_list[target_id].add(source_id)

        logger.info(f"Adjacency list built with {len(adj_list)} nodes and {len(edge_map)} edges.")
        return adj_list, edge_map

    def _extract_entities_from_query(self, query: str):
        """使用LLM从查询中提取种子实体"""
        prompt = get_rag_entity_extraction_prompt(query)
        usage = None
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
            if match:
                content = match.group(1)

            entities = json.loads(content)

            # --- (*** 关键修复 ***) ---
            # 将 CompletionUsage 对象转换为可序列化的字典
            usage_dict = None
            if response.usage:
                usage_dict = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            return entities, usage_dict
            # --- (修复结束) ---

        except Exception as e:
            response_content = response.choices[0].message.content if 'response' in locals() else 'N/A'
            logger.error(f"Error extracting entities: {e}. Response: {response_content}")
            try:
                entities = json.loads(response_content)
                return entities, None  # usage 为 None 是可序列化的
            except json.JSONDecodeError:
                logger.error("Failed to decode LLM response as JSON.")
                return [], None

    def _find_seed_entities(self, extracted_entities: list[str]) -> dict:
        """使用pgvector查找与文本实体最匹配的图节点"""
        if not extracted_entities:
            return {}

        seed_entities = {}
        for entity_name in extracted_entities:
            emb = self._get_embedding(entity_name)
            if emb is None:
                logger.warning(f"Could not get embedding for entity: {entity_name}")
                continue

            # --- (*** 关键修复 ***) ---
            # 将 np.ndarray 转换为 list
            emb_list = emb.tolist()
            # --- (修复结束) ---

            candidates = GraphNode.objects.annotate(
                distance=L2Distance('embedding', emb_list)  # <--- 使用转换后的 list
            ).order_by('distance')[:self.TOP_P_PER_ENTITY]

            for node in candidates:
                score = max(0, 1.0 - (node.distance / 2.0))

                if node.node_id not in seed_entities or score > seed_entities[node.node_id]['score']:
                    seed_entities[node.node_id] = {'id': node.node_id, 'score': score, 'origin': 'initial_entity'}

        return seed_entities

    def _graph_pathfinding(self, seed_entity_ids: set) -> list:
        """图谱路径发现BFS (来自您的代码)"""
        if not seed_entity_ids:
            return []
        completed_paths = []
        queue = deque([(seed_id, [seed_id]) for seed_id in seed_entity_ids])

        while queue:
            current_id, current_path = queue.popleft()
            neighbors = self.adj_list.get(current_id, set())
            valid_neighbors = [n for n in neighbors if n not in current_path]

            if len(current_path) - 1 >= self.BFS_DEPTH:
                if len(current_path) > 1:
                    completed_paths.append(current_path)
                continue
            if not valid_neighbors:
                if len(current_path) > 1:
                    completed_paths.append(current_path)
                continue
            for neighbor_id in valid_neighbors:
                new_path = current_path + [neighbor_id]
                queue.append((neighbor_id, new_path))

        return [list(p) for p in set(tuple(path) for path in completed_paths)]

    def _batch_get_similarity(self, node_ids: list, query_embedding_np: np.ndarray) -> dict:
        """
        批量获取节点与查询的相似度
        """
        if not node_ids or query_embedding_np is None:
            return {}

        nodes = GraphNode.objects.filter(node_id__in=node_ids).values('node_id', 'embedding')

        # --- (*** 关键修复 ***) ---
        # 错误行: id_map = {n['node_id']: n['embedding'] for n in nodes if n['embedding']}
        # 修复: 显式检查 None
        id_map = {n['node_id']: n['embedding'] for n in nodes if n['embedding'] is not None}
        # --- (修复结束) ---

        valid_embeddings, valid_ids = [], []
        for node_id in node_ids:
            if node_id in id_map:
                valid_embeddings.append(id_map[node_id])
                valid_ids.append(node_id)

        if not valid_ids:
            return {}

        embeddings_np = np.array(valid_embeddings).astype('float32')
        distances = np.linalg.norm(embeddings_np - query_embedding_np.reshape(1, -1), axis=1)
        scores = np.maximum(0, 1.0 - (distances / 2.0))

        return {id: float(score) for id, score in zip(valid_ids, scores)}

    def _score_paths_component_based(self, paths: list, query_embedding_np: np.ndarray, seed_entity_ids: set):
        """对路径进行评分 (移除了 degree)"""
        if not paths:
            return []

        unique_entity_ids = {eid for path in paths for eid in path}
        entity_sim_map = self._batch_get_similarity(list(unique_entity_ids), query_embedding_np)

        final_scored_paths = []
        for path in paths:
            total_component_score = 0

            for eid in path:
                sim = entity_sim_map.get(eid, 0)
                total_component_score += sim

            num_components = len(path) + max(0, len(path) - 1)
            avg_quality_score = total_component_score / num_components if num_components > 0 else 0

            num_seeds = len(set(path) & seed_entity_ids)
            density_bonus_factor = 1.0
            if num_seeds > 1 and len(path) > 1:
                path_length = len(path) - 1
                density = num_seeds / path_length
                density_bonus_factor = 1 + self.SEED_DENSITY_BONUS * density

            base_score = avg_quality_score * density_bonus_factor
            final_scored_paths.append({
                'path': path,
                'score': base_score,
                'reason': f'AvgQuality({avg_quality_score:.2f}) * DensityBonus({density_bonus_factor:.2f})'
            })

        return final_scored_paths

    # def _find_shortest_bridge_path(self, start_node: str, target_nodes: set) -> tuple[list | None, str | None]:
    #     """查找最短桥接路径 (来自您的代码)"""
    #     # <--- MODIFIED: 移除
    #     pass

    def _get_text_chunks(self, query_embedding: list, top_k: int) -> Set[str]:
        """使用 pgvector 检索 Top-K 文本块"""
        candidates = GraphChunk.objects.annotate(
            distance=L2Distance('embedding', query_embedding)
        ).order_by('distance')[:top_k]
        return {chunk.unique_id for chunk in candidates}

    def _get_chunk_similarities(self, chunk_ids: List[str], query_embedding_np: np.ndarray) -> Dict[str, float]:
        """批量获取 Chunks 与查询的相似度"""
        if not chunk_ids or query_embedding_np is None:
            return {}
        chunks = GraphChunk.objects.filter(unique_id__in=chunk_ids).values('unique_id', 'embedding')

        # --- (*** 关键修复 ***) ---
        # 错误行: id_map = {c['unique_id']: c['embedding'] for c in chunks if c['embedding']}
        # 修复: 显式检查 None
        id_map = {c['unique_id']: c['embedding'] for c in chunks if c['embedding'] is not None}
        # --- (修复结束) ---

        valid_embeddings, valid_ids = [], []
        for chunk_id in chunk_ids:
            if chunk_id in id_map:
                valid_embeddings.append(id_map[chunk_id])
                valid_ids.append(chunk_id)

        if not valid_ids:
            return {}

        embeddings_np = np.array(valid_embeddings).astype('float32')
        distances = np.linalg.norm(embeddings_np - query_embedding_np.reshape(1, -1), axis=1)
        scores = np.maximum(0, 1.0 - (distances / 2.0))

        return {id: float(score) for id, score in zip(valid_ids, scores)}

    def _score_chunks(self, initial_chunk_ids, chunk_recommendations_from_graph, query_embedding_np):
        """对 Chunks 评分 (来自您的代码)"""
        graph_only_recs = {
            cid: count for cid, count in chunk_recommendations_from_graph.items()
            if cid not in initial_chunk_ids
        }
        top_k_recs_to_score = sorted(graph_only_recs.items(), key=lambda item: item[1], reverse=True)[
                              :self.TOP_REC_K_FOR_SIMILARITY]
        top_k_rec_ids = {cid for cid, count in top_k_recs_to_score}

        all_candidate_ids_to_score_sim = list(initial_chunk_ids | top_k_rec_ids)
        all_sim_scores = self._get_chunk_similarities(all_candidate_ids_to_score_sim, query_embedding_np)

        candidate_scores = {}
        for cid in all_candidate_ids_to_score_sim:
            rec_count = chunk_recommendations_from_graph.get(cid, 0)
            rec_bonus = self.STRONG_CHUNK_RECOMMENDATION_BONUS if cid in initial_chunk_ids else self.WEAK_CHUNK_RECOMMENDATION_BONUS
            candidate_scores[cid] = {
                'sim_score': all_sim_scores.get(cid, 0.0),
                'rec_score': rec_count * rec_bonus
            }

        if not candidate_scores:
            return [], []

        scoring_df = pd.DataFrame.from_dict(candidate_scores, orient='index')

        if scoring_df['sim_score'].nunique() > 1:
            scoring_df['norm_sim'] = minmax_scale(scoring_df['sim_score'])
        else:
            scoring_df['norm_sim'] = scoring_df['sim_score'].apply(lambda x: 1.0 if x > 0 else 0.0)

        if scoring_df['rec_score'].nunique() > 1:
            scoring_df['norm_rec'] = minmax_scale(scoring_df['rec_score'])
        else:
            scoring_df['norm_rec'] = scoring_df['rec_score'].apply(lambda x: 1.0 if x > 0 else 0.0)

        alpha = self.CHUNK_SCORE_ALPHA
        scoring_df['final_score'] = (alpha * scoring_df['norm_sim']) + ((1 - alpha) * scoring_df['norm_rec'])

        final_chunk_scores_list = scoring_df.reset_index().rename(columns={'index': 'id'}).to_dict('records')
        return list(candidate_scores.keys()), final_chunk_scores_list

    def _get_canonical_path_key(self, path: list) -> frozenset:
        """为路径生成一个与方向无关的范式键 (来自您的代码)"""
        if len(path) < 2:
            return frozenset(path)
        edges = set()
        for i in range(len(path) - 1):
            edge = frozenset([path[i], path[i + 1]])
            edges.add(edge)
        return frozenset(edges)

    def search(self, query: str, top_k_chunks: int = 5, top_k_paths: int = 10):
        """主检索函数 (移植自您的代码)"""
        diagnostics = {}
        total_start_time = time.time()
        logger.info(f"\n{'=' * 20} Starting New Search {'=' * 20}")
        logger.info(f"Query: {query}")

        # --- STAGE 1: 初始检索 ---
        stage_start_time = time.time()
        query_embedding_np = self._get_embedding(query)
        if query_embedding_np is None:
            logger.error("Failed to get query embedding. Aborting search.")
            return {"top_paths": [], "top_chunks": []}, {"error": "Failed to get query embedding."}

        query_embedding_list = query_embedding_np.tolist()

        extracted_entities, usage = self._extract_entities_from_query(query)
        diagnostics['llm_extraction'] = {'entities': extracted_entities, 'usage': usage}

        seed_entities_dict = self._find_seed_entities(extracted_entities)
        seed_entity_ids = set(seed_entities_dict.keys())

        initial_paths = self._graph_pathfinding(seed_entity_ids)
        if not initial_paths and seed_entity_ids:
            logger.info("  - ⚠️ No multi-hop paths found. Falling back to single-entity results.")
            initial_paths = [[seed_id] for seed_id in seed_entity_ids]
        logger.info(
            f"  - Graph Channel: Found {len(initial_paths)} initial paths from {len(seed_entities_dict)} seed(s).")

        initial_chunk_ids = self._get_text_chunks(query_embedding_list, top_k_chunks * 2)
        logger.info(f"  - Text Channel: Found {len(initial_chunk_ids)} initial candidate chunks.")
        diagnostics['time_stage1_retrieval'] = f"{time.time() - stage_start_time:.2f}s"

        # --- STAGE 2: 协调与评分 ---
        stage_start_time = time.time()

        scored_paths = self._score_paths_component_based(initial_paths, query_embedding_np, seed_entity_ids)
        logger.info(f"  - Initial path scoring complete.")

        entities_from_paths = {eid for p_info in scored_paths for eid in p_info['path']}

        entities_from_chunks_query = NodeSourceLink.objects.filter(
            source_chunks__unique_id__in=initial_chunk_ids
        ).values_list('node_id', flat=True).distinct()
        entities_from_chunks = set(entities_from_chunks_query)

        for p_info in scored_paths:
            overlap = len(set(p_info['path']).intersection(entities_from_chunks))
            if overlap > 0:
                p_info['score'] += self.TEXT_CONFIRMATION_BONUS * overlap
                p_info['reason'] += f" + TextConfirm({overlap})"
        logger.info(f"  - Applied text confirmation bonus to paths.")

        # --- (*** MODIFIED: 移除孤儿实体桥接逻辑 ***) ---
        # orphan_entities = entities_from_chunks - entities_from_paths
        # endorsing_bridges_map = defaultdict(list) # 移除
        # bridged_path_objects = [] # 移除

        # if orphan_entities and entities_from_paths:
        # ... (移除整个 if 块)
        # --- (*** 修改结束 ***) ---

        chunk_recs_query = NodeSourceLink.objects.filter(
            node_id__in=entities_from_paths
        ).values_list('source_chunks__unique_id', flat=True)
        chunk_recommendations_from_graph = Counter(c for c in chunk_recs_query if c is not None)

        all_candidate_ids, final_chunk_scores_list = self._score_chunks(
            initial_chunk_ids,
            chunk_recommendations_from_graph,
            query_embedding_np
        )
        logger.info(f"  - Scored a total of {len(all_candidate_ids)} candidate chunks.")

        merged_paths = {}
        paths_for_merging = scored_paths  # <--- MODIFIED: 移除 + bridged_path_objects
        for p_info in paths_for_merging:
            canonical_key = self._get_canonical_path_key(p_info['path'])
            if canonical_key not in merged_paths:
                merged_paths[canonical_key] = p_info
            else:
                if p_info['score'] > merged_paths[canonical_key]['score']:
                    merged_paths[canonical_key] = p_info

        all_scored_paths = list(merged_paths.values())
        logger.info(f"  - Merged paths down to {len(all_scored_paths)} unique paths.")
        diagnostics['time_stage2_fusion'] = f"{time.time() - stage_start_time:.2f}s"

        # --- STAGE 3: 排序与格式化 ---
        stage_start_time = time.time()

        final_ranked_paths = sorted(all_scored_paths, key=lambda x: x['score'], reverse=True)[:top_k_paths]

        # --- (*** MODIFIED: 移除桥接信息添加 ***) ---
        # for p_info in final_ranked_paths:
        #     canonical_key = self._get_canonical_path_key(p_info['path'])
        #     p_info['endorsing_bridges'] = endorsing_bridges_map.get(canonical_key, [])
        # --- (*** 修改结束 ***) ---

        final_ranked_chunks = sorted(final_chunk_scores_list, key=lambda x: x['final_score'], reverse=True)[
                              :top_k_chunks]

        logger.info(
            f"  - Ranked and selected top {len(final_ranked_paths)} paths and {len(final_ranked_chunks)} chunks.")
        diagnostics['time_stage3_ranking'] = f"{time.time() - stage_start_time:.2f}s"

        # 4. 格式化输出
        results = {
            "top_paths": [self.get_path_details(p) for p in final_ranked_paths],
            "top_chunks": [self.get_chunk_details(c) for c in final_ranked_chunks]
        }
        diagnostics['time_total_retrieval'] = f"{time.time() - total_start_time:.2f}s"
        logger.info(f"✅ Search complete. Total time: {diagnostics['time_total_retrieval']}.")

        return results, diagnostics

    def get_chunk_details(self, item):
        """从DB获取Chunk详细信息"""
        chunk_id = item['id']  # 'id' is unique_id (e.g., "39_1")
        output_dir = None  # <--- 新增
        doc_name = "Unknown"  # <--- 新增

        try:
            chunk = GraphChunk.objects.select_related('task').get(unique_id=chunk_id)
            doc_name = chunk.task.get_pdf_filename()
            output_dir = chunk.task.output_directory  # <--- (*** 关键修改：获取 output_directory ***)
        except Exception:
            return {"id": chunk_id, "error": "Chunk not found", "content": "", "output_directory": None}

        details = item.copy()
        if 'final_score' in details: details['score'] = details['final_score']

        details.update({
            'type': 'chunk',
            'name': f"Chunk from {doc_name}",
            'source_document': doc_name,
            'content': chunk.text,
            'output_directory': output_dir  # <--- (*** 关键修改：将 output_directory 发送给前端 ***)
        })
        return details

    def get_path_details(self, path_info):
        """从DB获取Path详细信息 (移植自您的代码)"""
        path_ids = path_info['path']
        # 批量获取路径中所有节点的详细信息
        nodes_in_path = GraphNode.objects.filter(node_id__in=path_ids).in_bulk()

        path_segments = []

        # --- (为可视化准备节点和边) ---
        vis_nodes = []
        vis_edges = []
        added_nodes = set()
        # --- (准备结束) ---

        start_node = nodes_in_path.get(path_ids[0])
        path_readable_parts = [self._get_formatted_node_name(start_node)]

        # 添加起始节点到可视化列表
        if start_node and start_node.node_id not in added_nodes:
            vis_nodes.append({
                "id": start_node.node_id,
                "label": self._get_formatted_node_name(start_node),
                "group": start_node.community,
                "value": start_node.value,
                "name": start_node.name,
            })
            added_nodes.add(start_node.node_id)

        for i in range(len(path_ids) - 1):
            source_id, target_id = path_ids[i], path_ids[i + 1]
            edge_key = tuple(sorted((source_id, target_id)))
            edge_info = self.edge_map.get(edge_key, {})

            source_node = nodes_in_path.get(source_id)
            target_node = nodes_in_path.get(target_id)

            source_name_formatted = self._get_formatted_node_name(source_node)
            target_name_formatted = self._get_formatted_node_name(target_node)
            rel_type = edge_info.get('relation_type', 'related to')

            path_readable_parts.extend([f" --[{rel_type}]--> ", target_name_formatted])

            # --- (*** 关键修改：提取图片路径并转换为相对URL ***) ---

            def get_node_description(node):
                """
                辅助函数：提取描述。
                如果是 Image 节点，则将绝对路径转换为 Web 相对路径。
                """
                if not node:
                    return ''
                props = node.properties or {}
                description = props.get('description', '')

                # 检查是否是 'Image' 社区
                if node.community == 'Image':
                    abs_path = props.get('value', '')  # e.g., D:\..._web\media\md_results\52\image\...

                    # 转换为相对于 MEDIA_ROOT 的路径
                    try:
                        media_root_path = str(settings.MEDIA_ROOT)  # e.g., D:\..._web\media
                        norm_abs_path = os.path.normpath(abs_path)
                        norm_media_root = os.path.normpath(media_root_path)

                        relative_path = os.path.relpath(norm_abs_path, norm_media_root)

                        # 转换为 Web 路径 (使用 /)
                        web_path = relative_path.replace('\\', '/')

                        return f"描述: {description}, 路径: {web_path}"  # e.g., md_results/52/image/...

                    except (ValueError, TypeError) as e:
                        # 路径转换失败（例如跨驱动器）或 abs_path 为 None
                        logger.warning(f"无法为图像创建相对路径: {abs_path}. 错误: {e}")
                        return f"描述: {description}"
                else:
                    # 对于非Image节点，返回其 'description' (如果存在)
                    return description

            source_desc = get_node_description(source_node)
            target_desc = get_node_description(target_node)
            # --- (*** 修改结束 ***) ---

            path_segments.append({
                "source": source_name_formatted,
                "target": target_name_formatted,
                "keywords": rel_type,
                "description": edge_info.get('description', ''),
                "source_desc": source_desc,  # <--- 使用新变量
                "target_desc": target_desc  # <--- 使用新变量
            })

            # --- (添加节点和边到可视化列表) ---
            if source_node and source_node.node_id not in added_nodes:
                vis_nodes.append({
                    "id": source_node.node_id,
                    "label": source_name_formatted,
                    "group": source_node.community,
                    "value": source_node.value,
                    "name": source_node.name,
                })
                added_nodes.add(source_node.node_id)

            if target_node and target_node.node_id not in added_nodes:
                vis_nodes.append({
                    "id": target_node.node_id,
                    "label": target_name_formatted,
                    "group": target_node.community,
                    "value": target_node.value,
                    "name": target_node.name,
                })
                added_nodes.add(target_node.node_id)

            vis_edges.append({
                "from": source_id,
                "to": target_id,
                "label": rel_type,
                "source": source_id,
                "target": target_id,
                "value": rel_type
            })
            # --- (添加结束) ---

        details = {
            "path_readable": "".join(path_readable_parts),
            "segments": path_segments,
            "score": path_info['score'],
            "reason": path_info['reason'],
            "entity_ids": path_ids,
            "graph_data": {
                "nodes": vis_nodes,
                "edges": vis_edges
            }
        }

        # --- (*** MODIFIED: 移除桥接信息添加 ***) ---
        # if path_info.get('endorsing_bridges'):
        # ... (移除整个 if 块)
        # --- (*** 修改结束 ***) ---

        return details

    def generate_answer(self, query: str, top_chunks: list, top_paths: list, mode: str):
        """生成最终答案 (移植自您的代码)"""
        logger.info(f"\n[STAGE 5] Generating final answer with mode: {mode}...")
        paths_context, chunks_context = "", ""

        if mode in ["full_context", "paths_only"]:
            paths_context = "无相关知识图谱路径。\n"
            if top_paths:
                context_parts = []
                for i, p in enumerate(top_paths):
                    context_parts.append(f"核心路径 {i + 1}: {p['path_readable']}")
                    described_entities_in_path = set()
                    for segment in p['segments']:
                        source_name, target_name = segment['source'], segment['target']
                        if source_name not in described_entities_in_path:
                            desc = segment['source_desc']
                            context_parts.append(f"  - 实体: {source_name} (描述: {desc})")
                            described_entities_in_path.add(source_name)

                        context_parts.append(
                            f"  - 关系: 从 '{source_name}' 到 '{target_name}' (类型: {segment['keywords']})")

                        if target_name not in described_entities_in_path:
                            desc = segment['target_desc']
                            context_parts.append(f"  - 实体: {target_name} (描述: {desc})")
                            described_entities_in_path.add(target_name)

                    # --- (*** MODIFIED: 移除桥接信息添加 ***) ---
                    # if p.get('endorsing_bridges'):
                    # ... (移除整个 if 块)
                    # --- (*** 修改结束 ***) ---
                paths_context = "\n".join(context_parts)

        if mode in ["full_context", "chunks_only"]:
            chunks_context = "无相关文本证据。\n"
            if top_chunks:
                context_parts = [f"证据 {i + 1} (来源文档: {chunk['source_document']}):\n'''\n{chunk['content']}\n'''"
                                 for i, chunk in enumerate(top_chunks)]
                chunks_context = "\n\n".join(context_parts)

        prompt = get_rag_final_answer_prompt(query=query, paths_context=paths_context,
                                             chunks_context=chunks_context)

        try:
            return self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                stream=True
            )
        except Exception as e:
            logger.error(f"Error during final answer generation: {e}")

            # 返回一个可迭代的错误
            def error_generator():
                yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

            return error_generator()