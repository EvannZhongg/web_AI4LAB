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
from django.conf import settings
import os

# Import models
from pdf_parser.models import GraphNode, GraphEdge, GraphChunk, NodeSourceLink, PDFParsingTask
# Import prompts
from pdf_parser.prompts import get_rag_entity_extraction_prompt, get_rag_final_answer_prompt
from pdf_parser.graph_construction import get_embeddings_batch

logger = logging.getLogger(__name__)


class PgVectorRetriever:

    def __init__(self):
        logger.info("Initializing PgVector-based RAG Retriever (Path-centric SBEA Aligned)...")

        # 1. Load Configurations
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

        # 2. Initialize Clients
        self.embedding_client = OpenAI(api_key=emb_conf["API_KEY"], base_url=emb_conf["API_URL"])
        self.llm_client = OpenAI(api_key=llm_conf["API_KEY"], base_url=llm_conf["API_URL"])
        self.embedding_model = emb_conf["MODEL_NAME"]
        self.embedding_dim = emb_conf["DIMENSIONS"]
        self.embedding_batch_size = emb_conf["BATCH_SIZE"]
        self.llm_model = llm_conf["MODEL_NAME"]

        # 3. Load RAG Algorithm Parameters
        self.TOP_P_PER_ENTITY = rag_conf.get('TOP_P_PER_ENTITY', 3)
        self.BFS_DEPTH = rag_conf.get('BFS_DEPTH', 3)
        self.TOP_K_ORPHANS_TO_BRIDGE = rag_conf.get('TOP_K_ORPHANS_TO_BRIDGE', 3)  # <--- NEW: For bridging

        # 4. Load Scoring Weights
        self.CHUNK_SCORE_ALPHA = rag_conf.get('CHUNK_SCORE_ALPHA', 0.6)
        self.SEED_DENSITY_BONUS = rag_conf.get('SEED_DENSITY_BONUS', 0.5)
        self.TOP_REC_K_FOR_SIMILARITY = rag_conf.get('TOP_REC_K_FOR_SIMILARITY', 5)
        self.STRONG_CHUNK_RECOMMENDATION_BONUS = rag_conf.get('STRONG_CHUNK_RECOMMENDATION_BONUS', 0.25)
        self.WEAK_CHUNK_RECOMMENDATION_BONUS = rag_conf.get('WEAK_CHUNK_RECOMMENDATION_BONUS', 0.15)
        self.TEXT_CONFIRMATION_BONUS = rag_conf.get('TEXT_CONFIRMATION_BONUS', 0.5)
        
        # New Scoring Parameters from PathSBERetriever
        self.ENDORSEMENT_BASE_BONUS = rag_conf.get('ENDORSEMENT_BASE_BONUS', 0.1)
        self.ENDORSEMENT_DECAY_FACTOR = rag_conf.get('ENDORSEMENT_DECAY_FACTOR', 0.85)
        self.ENTITY_DEGREE_WEIGHT = 0.01 # Default weight for node degree
        self.RELATION_DEGREE_WEIGHT = 0.01 

        # 5. Build Memory Graph (for BFS)
        self.adj_list, self.edge_map, self.node_degrees = self._build_adjacency_list()

        logger.info("Retriever initialized successfully.")

    def _get_formatted_node_name(self, node: GraphNode) -> str:
        if not node:
            return "Unknown"
        if node.community in ["Device", "Manufacturer", "Category"]:
            return f"{node.community}: {node.value}"
        else:
            return f"{node.name}: {node.value}"

    def _get_embedding(self, text: str) -> np.ndarray:
        emb_map = get_embeddings_batch(
            [text], self.embedding_client, self.embedding_model,
            self.embedding_dim, self.embedding_batch_size
        )
        emb = emb_map.get(text)
        return np.array(emb).astype('float32') if emb else None

    def _build_adjacency_list(self):
        """
        Builds in-memory adjacency list, edge map, and node degrees.
        """
        logger.info("Building in-memory graph adjacency list...")
        adj_list = defaultdict(set)
        edge_map = {}
        node_degrees = defaultdict(int)

        edges_query = GraphEdge.objects.all().values_list(
            'source_node_id',
            'target_node_id',
            'type',
            'id'
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
            
            # Count degrees
            node_degrees[source_id] += 1
            node_degrees[target_id] += 1

        logger.info(f"Adjacency list built with {len(adj_list)} nodes and {len(edge_map)} edges.")
        return adj_list, edge_map, node_degrees

    def _extract_entities_from_query(self, query: str):
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
            usage_dict = None
            if response.usage:
                usage_dict = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            return entities, usage_dict

        except Exception as e:
            logger.error(f"Error extracting entities: {e}.")
            return [], None

    def _find_seed_entities(self, extracted_entities: list[str]) -> dict:
        if not extracted_entities:
            return {}

        seed_entities = {}
        for entity_name in extracted_entities:
            emb = self._get_embedding(entity_name)
            if emb is None:
                continue

            emb_list = emb.tolist()
            candidates = GraphNode.objects.annotate(
                distance=L2Distance('embedding', emb_list)
            ).order_by('distance')[:self.TOP_P_PER_ENTITY]

            for node in candidates:
                score = max(0, 1.0 - (node.distance / 2.0))
                if node.node_id not in seed_entities or score > seed_entities[node.node_id]['score']:
                    seed_entities[node.node_id] = {'id': node.node_id, 'score': score, 'origin': 'initial_entity'}

        return seed_entities

    def _graph_pathfinding(self, seed_entity_ids: set) -> list:
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

    # --- NEW: Bridging Logic (Replaces SQL implementation with in-memory BFS) ---
    def _find_shortest_bridge_path(self, start_node: str, target_nodes: set) -> list:
        """
        Find shortest path from start_node (orphan) to any node in target_nodes using BFS.
        """
        if start_node in target_nodes:
            return [start_node]
        
        queue = deque([(start_node, [start_node])])
        visited = {start_node}
        
        # Limit bridge depth to keep performance high and relevance tight
        max_bridge_depth = 3 

        while queue:
            current_id, path = queue.popleft()
            
            if len(path) - 1 >= max_bridge_depth:
                continue
                
            neighbors = self.adj_list.get(current_id, set())
            for neighbor in neighbors:
                if neighbor in target_nodes:
                    return path + [neighbor]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None

    # --- NEW: Redundant Path Filtering ---
    def _filter_redundant_paths(self, paths: list) -> list:
        """
        Filter out paths that are subsets of longer, higher-scoring paths.
        """
        if not paths: return []

        keep_indices = set(range(len(paths)))
        path_sets = [set(p['path']) for p in paths]

        for i in range(len(paths)):
            if i not in keep_indices: continue

            for j in range(len(paths)):
                if i == j: continue

                # If path i is a subset of path j
                if path_sets[i].issubset(path_sets[j]):
                    # Case 1: Strict subset (j is longer) -> Remove i
                    if len(path_sets[i]) < len(path_sets[j]):
                        keep_indices.discard(i)
                        break

                    # Case 2: Identical sets -> Keep the one with higher score
                    elif len(path_sets[i]) == len(path_sets[j]):
                        if paths[i]['score'] < paths[j]['score']:
                            keep_indices.discard(i)
                            break
                        elif paths[i]['score'] == paths[j]['score'] and i > j:
                             # Tie-break: keep first encountered
                            keep_indices.discard(i)
                            break

        filtered_paths = [paths[i] for i in sorted(list(keep_indices))]
        if len(paths) > len(filtered_paths):
            logger.info(f"  - Filtered {len(paths) - len(filtered_paths)} redundant sub-paths.")

        return filtered_paths

    def _batch_get_similarity(self, node_ids: list, query_embedding_np: np.ndarray) -> dict:
        if not node_ids or query_embedding_np is None:
            return {}

        nodes = GraphNode.objects.filter(node_id__in=node_ids).values('node_id', 'embedding')
        id_map = {n['node_id']: n['embedding'] for n in nodes if n['embedding'] is not None}

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
        """
        Score paths based on entity similarity, seed density, AND Node Degree.
        """
        if not paths:
            return []

        unique_entity_ids = {eid for path in paths for eid in path}
        entity_sim_map = self._batch_get_similarity(list(unique_entity_ids), query_embedding_np)

        final_scored_paths = []
        for path in paths:
            total_component_score = 0

            for eid in path:
                sim = entity_sim_map.get(eid, 0)
                # Apply Degree Weighting
                degree = self.node_degrees.get(eid, 0)
                total_component_score += sim * (1 + self.ENTITY_DEGREE_WEIGHT * degree)

            # Note: Edge degree weighting skipped as edges usually don't have degrees in this schema
            # We focus on Node degree which is the primary structural importance indicator.

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
                'reason': f'AvgQual({avg_quality_score:.2f})*Density({density_bonus_factor:.2f})',
                'endorsing_bridges': [] # Initialize empty list for bridges
            })

        return final_scored_paths

    def _get_text_chunks(self, query_embedding: list, top_k: int) -> Set[str]:
        candidates = GraphChunk.objects.annotate(
            distance=L2Distance('embedding', query_embedding)
        ).order_by('distance')[:top_k]
        return {chunk.unique_id for chunk in candidates}

    def _get_chunk_similarities(self, chunk_ids: List[str], query_embedding_np: np.ndarray) -> Dict[str, float]:
        if not chunk_ids or query_embedding_np is None:
            return {}
        chunks = GraphChunk.objects.filter(unique_id__in=chunk_ids).values('unique_id', 'embedding')
        id_map = {c['unique_id']: c['embedding'] for c in chunks if c['embedding'] is not None}

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
        if len(path) < 2:
            return frozenset(path)
        edges = set()
        for i in range(len(path) - 1):
            edge = frozenset([path[i], path[i + 1]])
            edges.add(edge)
        return frozenset(edges)

    def search(self, query: str, top_k_chunks: int = 5, top_k_paths: int = 10):
        """
        Main Search Logic (Aligned with PathSBERetriever).
        """
        diagnostics = {}
        total_start_time = time.time()
        logger.info(f"\n{'=' * 20} Starting New Search {'=' * 20}")
        logger.info(f"Query: {query}")

        # --- STAGE 1: Initial Retrieval ---
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

        # --- STAGE 2: Fusion, Bridging & Scoring ---
        stage_start_time = time.time()

        # Score initial paths
        scored_paths = self._score_paths_component_based(initial_paths, query_embedding_np, seed_entity_ids)
        
        # Text Confirmation Bonus
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
        
        # --- NEW: Orphan Bridging Logic ---
        orphan_entities = entities_from_chunks - entities_from_paths
        bridged_path_objects = []
        endorsing_bridges_map = defaultdict(list)

        if orphan_entities and entities_from_paths:
            # Sort orphans by degree (heuristic: important orphans have higher degree)
            orphans_with_degree = [{'id': eid, 'degree': self.node_degrees.get(eid, 0)} for eid in orphan_entities]
            sorted_orphans = sorted(orphans_with_degree, key=lambda x: x['degree'], reverse=True)
            orphans_to_process = sorted_orphans[:self.TOP_K_ORPHANS_TO_BRIDGE]

            node_to_initial_path_map = defaultdict(list)
            for p_info in scored_paths:
                for node in p_info['path']:
                    node_to_initial_path_map[node].append(p_info)

            all_found_bridge_paths = []

            for rank, orphan in enumerate(orphans_to_process, 1):
                # Find shortest path from orphan to ANY node in the main paths
                bridge_path = self._find_shortest_bridge_path(orphan['id'], entities_from_paths)

                if bridge_path and len(bridge_path) > 1:
                    target_node = bridge_path[-1] # The connection point
                    all_found_bridge_paths.append(bridge_path)

                    bridge_len = len(bridge_path) - 1
                    # Calculate endorsement bonus
                    bonus_score = self.ENDORSEMENT_BASE_BONUS * (1.0 / rank) * (
                                self.ENDORSEMENT_DECAY_FACTOR ** bridge_len)

                    logger.info(f"    - Bridged orphan via {bridge_len}-hop path. Bonus: {bonus_score:.3f}")

                    # Apply bonus to all main paths that contain the target node
                    if target_node in node_to_initial_path_map:
                        for target_path_info in node_to_initial_path_map[target_node]:
                            target_path_info['score'] *= (1 + bonus_score)
                            target_path_info['reason'] += f" + Endorsed"
                            # Store the bridge path for context
                            canonical_key = tuple(sorted(target_path_info['path']))
                            endorsing_bridges_map[canonical_key].append(bridge_path)
            
            # Score and add the bridge paths themselves to the results
            if all_found_bridge_paths:
                scored_bridged_paths = self._score_paths_component_based(
                    all_found_bridge_paths, query_embedding_np, seed_entity_ids
                )
                for p_info in scored_bridged_paths:
                    p_info['reason'] = 'Bridged Path'
                bridged_path_objects = scored_bridged_paths

        # --- Chunk Recommendations from Graph ---
        # Update entities list to include bridged paths
        final_path_entities = entities_from_paths.union({eid for p_info in bridged_path_objects for eid in p_info['path']})
        
        chunk_recs_query = NodeSourceLink.objects.filter(
            node_id__in=final_path_entities
        ).values_list('source_chunks__unique_id', flat=True)
        chunk_recommendations_from_graph = Counter(c for c in chunk_recs_query if c is not None)

        all_candidate_ids, final_chunk_scores_list = self._score_chunks(
            initial_chunk_ids,
            chunk_recommendations_from_graph,
            query_embedding_np
        )
        logger.info(f"  - Scored a total of {len(all_candidate_ids)} candidate chunks.")

        # --- Merge Paths (Deduplicate by set) ---
        merged_paths = {}
        # Combine original paths and new bridge paths
        paths_for_merging = scored_paths + bridged_path_objects 
        for p_info in paths_for_merging:
            canonical_key = self._get_canonical_path_key(p_info['path'])
            if canonical_key not in merged_paths:
                merged_paths[canonical_key] = p_info
            else:
                if p_info['score'] > merged_paths[canonical_key]['score']:
                    merged_paths[canonical_key] = p_info

        all_scored_paths = list(merged_paths.values())
        
        # --- NEW: Redundant Path Filtering ---
        filtered_paths = self._filter_redundant_paths(all_scored_paths)

        logger.info(f"  - Merged & Filtered paths down to {len(filtered_paths)} unique paths.")
        diagnostics['time_stage2_fusion'] = f"{time.time() - stage_start_time:.2f}s"

        # --- STAGE 3: Ranking & Output ---
        stage_start_time = time.time()

        final_ranked_paths = sorted(filtered_paths, key=lambda x: x['score'], reverse=True)[:top_k_paths]

        # Attach endorsing bridges info
        for p_info in final_ranked_paths:
            canonical_key = tuple(sorted(p_info['path']))
            p_info['endorsing_bridges'] = endorsing_bridges_map.get(canonical_key, [])

        final_ranked_chunks = sorted(final_chunk_scores_list, key=lambda x: x['final_score'], reverse=True)[
                              :top_k_chunks]

        logger.info(
            f"  - Ranked and selected top {len(final_ranked_paths)} paths and {len(final_ranked_chunks)} chunks.")
        diagnostics['time_stage3_ranking'] = f"{time.time() - stage_start_time:.2f}s"

        # 4. Format Output
        results = {
            "top_paths": [self.get_path_details(p) for p in final_ranked_paths],
            "top_chunks": [self.get_chunk_details(c) for c in final_ranked_chunks]
        }
        diagnostics['time_total_retrieval'] = f"{time.time() - total_start_time:.2f}s"
        logger.info(f"✅ Search complete. Total time: {diagnostics['time_total_retrieval']}.")

        return results, diagnostics

    def get_chunk_details(self, item):
        chunk_id = item['id']
        output_dir = None
        doc_name = "Unknown"

        try:
            chunk = GraphChunk.objects.select_related('task').get(unique_id=chunk_id)
            doc_name = chunk.task.get_pdf_filename()
            output_dir = chunk.task.output_directory
        except Exception:
            return {"id": chunk_id, "error": "Chunk not found", "content": "", "output_directory": None}

        details = item.copy()
        if 'final_score' in details: details['score'] = details['final_score']

        details.update({
            'type': 'chunk',
            'name': f"Chunk from {doc_name}",
            'source_document': doc_name,
            'content': chunk.text,
            'output_directory': output_dir
        })
        return details

    def get_path_details(self, path_info):
        path_ids = path_info['path']
        nodes_in_path = GraphNode.objects.filter(node_id__in=path_ids).in_bulk()
        
        # Prepare for visualization
        path_segments = []
        vis_nodes = []
        vis_edges = []
        added_nodes = set()
        
        # Helper to safely get nodes
        def get_node(nid): return nodes_in_path.get(nid)

        start_node = get_node(path_ids[0])
        path_readable_parts = [self._get_formatted_node_name(start_node)]

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

            source_node = get_node(source_id)
            target_node = get_node(target_id)

            source_name_formatted = self._get_formatted_node_name(source_node)
            target_name_formatted = self._get_formatted_node_name(target_node)
            rel_type = edge_info.get('relation_type', 'related to')

            path_readable_parts.extend([f" --[{rel_type}]--> ", target_name_formatted])

            def get_node_description(node):
                if not node: return ''
                props = node.properties or {}
                description = props.get('description', '')
                if node.community == 'Image':
                    abs_path = props.get('value', '')
                    try:
                        media_root_path = str(settings.MEDIA_ROOT)
                        norm_abs_path = os.path.normpath(abs_path)
                        norm_media_root = os.path.normpath(media_root_path)
                        relative_path = os.path.relpath(norm_abs_path, norm_media_root)
                        web_path = relative_path.replace('\\', '/')
                        return f"描述: {description}, 路径: {web_path}"
                    except (ValueError, TypeError) as e:
                        return f"描述: {description}"
                else:
                    return description

            path_segments.append({
                "source": source_name_formatted,
                "target": target_name_formatted,
                "keywords": rel_type,
                "description": edge_info.get('description', ''),
                "source_desc": get_node_description(source_node),
                "target_desc": get_node_description(target_node)
            })

            # Add to vis
            for node in [source_node, target_node]:
                if node and node.node_id not in added_nodes:
                    vis_nodes.append({
                        "id": node.node_id,
                        "label": self._get_formatted_node_name(node),
                        "group": node.community,
                        "value": node.value,
                        "name": node.name,
                    })
                    added_nodes.add(node.node_id)

            vis_edges.append({
                "from": source_id,
                "to": target_id,
                "label": rel_type,
                "source": source_id,
                "target": target_id,
                "value": rel_type
            })

        # Process Endorsing Bridges for Display
        endorsing_bridges_readable = []
        if path_info.get('endorsing_bridges'):
            # Pre-fetch bridge nodes
            bridge_node_ids = {nid for bridge in path_info['endorsing_bridges'] for nid in bridge}
            bridge_nodes = GraphNode.objects.filter(node_id__in=bridge_node_ids).in_bulk()
            
            for bridge_path in path_info['endorsing_bridges']:
                names = []
                for nid in bridge_path:
                    node = bridge_nodes.get(nid)
                    names.append(self._get_formatted_node_name(node) if node else "Unknown")
                endorsing_bridges_readable.append(" -> ".join(names))

        details = {
            "path_readable": "".join(path_readable_parts),
            "segments": path_segments,
            "score": path_info['score'],
            "reason": path_info['reason'],
            "entity_ids": path_ids,
            "endorsing_bridges": endorsing_bridges_readable, # Add human readable bridges
            "graph_data": {
                "nodes": vis_nodes,
                "edges": vis_edges
            }
        }

        return details

    def generate_answer(self, query: str, top_chunks: list, top_paths: list, mode: str):
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

                    # --- Include Endorsing Bridges in Context ---
                    if p.get('endorsing_bridges'):
                        context_parts.append("  - 该路径被以下补全证据所支持 (Endorsing Bridges):")
                        for bridge_readable in p['endorsing_bridges']:
                            context_parts.append(f"    - 补全路径: {bridge_readable}")

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

            def error_generator():
                yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

            return error_generator()
