import os
import random
import numpy as np
import torch
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from collections import defaultdict
from transformers import AutoTokenizer
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from tqdm import tqdm
from typing import Dict, List


class GLiClassBEIREvaluator(EvaluateRetrieval):
    def __init__(self, model_id: str = 'knowledgator/gliclass_msmarco_merged', rerank_k: int = 10):
        super().__init__()
        self.rerank_k = rerank_k
        self.model = GLiClassModel.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, add_prefix_space=True)
        self.pipeline = ZeroShotClassificationPipeline(self.model, self.tokenizer, classification_type='multi-label',
                                                       device='cuda:0', max_length=2048, progress_bar=False)
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

    @staticmethod
    def load_dataset(dataset: str):
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
        out_dir = os.path.join(os.getcwd(), "gliclass_datasets")
        data_path = util.download_and_unzip(url, out_dir)
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
        return corpus, queries, qrels

    @staticmethod
    def prepare_data(corpus: Dict, queries: Dict, qrels: Dict):
        unique_doc_ids = set()
        for rels in qrels.values():
            unique_doc_ids.update(rels.keys())
        documents = [
            f"{corpus[doc_id].get('title', '')} {corpus[doc_id].get('text', '')}".strip()
            for doc_id in unique_doc_ids
            if doc_id in corpus
        ]
        qrels_list = []
        for query_id in qrels:
            example_dict = {
                'query': queries[query_id],
                'documents': documents
            }
            qrels_list.append(example_dict)

        return qrels_list

    @staticmethod
    def text_to_id(text_scores_dict: Dict, corpus, queries):
        docs_to_id_mapping = {
            f"{doc.get('title', '')} {doc.get('text', '')}".strip(): str(doc_id)
            for doc_id, doc in corpus.items()
        }
        queries_to_id_mapping = {
            text: query_id for query_id, text in queries.items()
        }
        result = {}
        for query_text, doc_scores in text_scores_dict.items():
            query_id = queries_to_id_mapping.get(query_text, None)
            if query_id is not None:
                result[query_id] = {}
                for doc_text_trimmed, score in doc_scores.items():
                    matched_doc_id = None
                    for full_doc_text, doc_id in docs_to_id_mapping.items():
                        if full_doc_text.startswith(doc_text_trimmed):
                            matched_doc_id = doc_id
                            break
                    if matched_doc_id:
                        result[query_id][matched_doc_id] = score
        return result

    @staticmethod
    def chunk(dataset, chunk_size=25):
        processed_dataset = []
        for example in tqdm(dataset):
            query = example['query']
            documents = example['documents']
            for i in range(0, len(documents), chunk_size):
                doc_chunk = documents[i:i + chunk_size]
                processed_dataset.append({
                    'query': query,
                    'documents': doc_chunk
                })

        return processed_dataset

    def seq_process(self, data):
        max_len = 2048
        processed_data = []
        for example in tqdm(data):
            query_tokens = self.tokenizer.encode(example['query'], add_special_tokens=False)
            query_len = len(query_tokens)
            doc_token_lists = [
                self.tokenizer.encode(doc, add_special_tokens=False)
                for doc in example['documents']
            ]
            doc_lengths = [len(tokens) for tokens in doc_token_lists]
            total_doc_len = sum(doc_lengths)
            available = max_len - query_len
            if available <= 0:
                continue
            processed_docs = []
            for tokens, length in zip(doc_token_lists, doc_lengths):
                proportion = length / total_doc_len if total_doc_len > 0 else 0
                keep_len = max(1, int(proportion * available))
                trimmed_tokens = tokens[:keep_len]
                trimmed_doc = self.tokenizer.decode(trimmed_tokens, skip_special_tokens=True)
                processed_docs.append(trimmed_doc)
            processed_example = {
                'query': example['query'],
                'documents': processed_docs
            }
            processed_data.append(processed_example)
        return processed_data

    def score_docs(self, dataset: List[Dict]):
        text_scores_dict = defaultdict(dict)
        for example in tqdm(dataset):
            query_text = example['query']
            doc_list = example['documents']
            results = self.pipeline(query_text, doc_list, threshold=0.0)[0]
            for result in results:
                pred_document = result['label']
                pred_doc_score = result['score']
                text_scores_dict[query_text][pred_document] = pred_doc_score
        return text_scores_dict

    def eval(self, dataset: str):
        corpus, queries, qrels = self.load_dataset(dataset)
        dataset = self.prepare_data(corpus, queries, qrels)
        dataset_chunk = self.chunk(dataset)
        data_chunk_and_trim = self.seq_process(dataset_chunk)
        text_scores = self.score_docs(data_chunk_and_trim)
        results = self.text_to_id(text_scores, corpus, queries)
        ndcg, _map, recall, precision = self.evaluate(qrels, results, [self.rerank_k])
        mrr = self.evaluate_custom(qrels, results, [self.rerank_k], metric="mrr")
        return ndcg, _map, mrr


evaluator = GLiClassBEIREvaluator('knowledgator/gliclass_msmarco_merged', 10)
evaluator.eval('scifact')
