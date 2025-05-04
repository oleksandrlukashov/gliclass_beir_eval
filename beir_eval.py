import os
import random
import numpy as np
import torch
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from transformers import AutoTokenizer
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from typing import Dict
from labels_truncation_pipeline import ZeroShotWithTruncationPipeline


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
        unique_doc_ids = list(set(corpus.keys()))
        documents = [
            f"{corpus[doc_id].get('title', '')} {corpus[doc_id].get('text', '')}".strip()
            for doc_id in unique_doc_ids
            if doc_id in corpus
        ]
        queries_list = []
        for query_id in qrels:
            queries_list.append(queries[query_id])
        return queries_list, documents

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

    def eval(self, dataset: str):
        corpus, queries, qrels = self.load_dataset(dataset)
        texts, documents = self.prepare_data(corpus, queries, qrels)
        pipeline = ZeroShotWithTruncationPipeline(model='knowledgator/gliclass_msmarco_merged',
                                                  tokenizer='knowledgator/gliclass_msmarco_merged',
                                                  max_classes=25, max_length=2048, classification_type='multi-label',
                                                  device='cuda:0')
        results = pipeline(texts, documents, threshold=0.5)
        print(results)
        return results


evaluator = GLiClassBEIREvaluator('knowledgator/gliclass_msmarco_merged', 10)
evaluator.eval('scifact')
