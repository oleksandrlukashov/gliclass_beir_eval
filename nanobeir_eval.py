from typing import Literal
from tqdm import tqdm
import numpy as np
from sentence_transformers.util import is_datasets_available
from sentence_transformers.cross_encoder.evaluation import CrossEncoderNanoBEIREvaluator, CrossEncoderRerankingEvaluator
from labels_truncation_pipeline import ZeroShotWithTruncationPipeline
import logging
import os
import csv

logger = logging.getLogger(__name__)

DatasetNameType = Literal[
    "climatefever",
    "dbpedia",
    "fever",
    "fiqa2018",
    "hotpotqa",
    "msmarco",
    "nfcorpus",
    "nq",
    "quoraretrieval",
    "scidocs",
    "arguana",
    "scifact",
    "touche2020",
]

dataset_name_to_id = {
    "climatefever": "sentence-transformers/NanoClimateFEVER-bm25",
    "dbpedia": "sentence-transformers/NanoDBPedia-bm25",
    "fever": "sentence-transformers/NanoFEVER-bm25",
    "fiqa2018": "sentence-transformers/NanoFiQA2018-bm25",
    "hotpotqa": "sentence-transformers/NanoHotpotQA-bm25",
    "msmarco": "sentence-transformers/NanoMSMARCO-bm25",
    "nfcorpus": "sentence-transformers/NanoNFCorpus-bm25",
    "nq": "sentence-transformers/NanoNQ-bm25",
    "quoraretrieval": "sentence-transformers/NanoQuoraRetrieval-bm25",
    "scidocs": "sentence-transformers/NanoSCIDOCS-bm25",
    "arguana": "sentence-transformers/NanoArguAna-bm25",
    "scifact": "sentence-transformers/NanoSciFact-bm25",
    "touche2020": "sentence-transformers/NanoTouche2020-bm25",
}

dataset_name_to_human_readable = {
    "climatefever": "ClimateFEVER",
    "dbpedia": "DBPedia",
    "fever": "FEVER",
    "fiqa2018": "FiQA2018",
    "hotpotqa": "HotpotQA",
    "msmarco": "MSMARCO",
    "nfcorpus": "NFCorpus",
    "nq": "NQ",
    "quoraretrieval": "QuoraRetrieval",
    "scidocs": "SCIDOCS",
    "arguana": "ArguAna",
    "scifact": "SciFact",
    "touche2020": "Touche2020",
}


class GLiClassRerankingEvaluator(CrossEncoderRerankingEvaluator):
    def __call__(
            self, model: ZeroShotWithTruncationPipeline, output_path: str = None, epoch: int = -1, steps: int = -1,
            labels_chunk_size: int = -1
    ) -> dict[str, float]:

        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps"
        else:
            out_txt = ""

        logger.info(f"GLiClassRerankingEvaluator: Evaluating the model on the {self.name} dataset{out_txt}:")

        base_mrr_scores = []
        base_ndcg_scores = []
        base_ap_scores = []
        all_mrr_scores = []
        all_ndcg_scores = []
        all_ap_scores = []
        num_queries = 0
        num_positives = []
        num_negatives = []
        for instance in tqdm(self.samples, desc="Evaluating samples", disable=not self.show_progress_bar, leave=False):
            if "query" not in instance:
                raise ValueError("GLiClassRerankingEvaluator requires a 'query' key in each sample.")
            if "positive" not in instance:
                raise ValueError("GLiClassRerankingEvaluator requires a 'positive' key in each sample.")
            if ("negative" in instance and "documents" in instance) or (
                    "negative" not in instance and "documents" not in instance
            ):
                raise ValueError(
                    "GLiClassRerankingEvaluator requires exactly one of 'negative' and 'documents' in each sample."
                )

            query = instance["query"]
            positive = instance["positive"]
            if isinstance(positive, str):
                positive = [positive]

            negative = instance.get("negative", None)
            documents = instance.get("documents", None)

            if documents:
                base_is_relevant = [int(sample in positive) for sample in documents]
                if sum(base_is_relevant) == 0:
                    base_mrr, base_ndcg, base_ap = 0, 0, 0
                else:
                    # If not all positives are in documents, we need to add them at the end
                    base_is_relevant += [1] * (len(positive) - sum(base_is_relevant))
                    base_pred_scores = np.array(range(len(base_is_relevant), 0, -1))
                    base_mrr, base_ndcg, base_ap = self.compute_metrics(base_is_relevant, base_pred_scores)
                base_mrr_scores.append(base_mrr)
                base_ndcg_scores.append(base_ndcg)
                base_ap_scores.append(base_ap)

                if self.always_rerank_positives:
                    docs = positive + [doc for doc in documents if doc not in positive]
                    is_relevant = [1] * len(positive) + [0] * (len(docs) - len(positive))
                else:
                    docs = documents
                    is_relevant = [int(sample in positive) for sample in documents]
            else:
                docs = positive + negative
                is_relevant = [1] * len(positive) + [0] * len(negative)

            num_queries += 1

            num_positives.append(len(positive))
            num_negatives.append(len(is_relevant) - sum(is_relevant))

            if sum(is_relevant) == 0:
                all_mrr_scores.append(0)
                all_ndcg_scores.append(0)
                all_ap_scores.append(0)
                continue

            model_input = [[query, doc] for doc in docs]
            gliclass_outputs = model([query], docs, threshold=0.9, batch_size=4)

            pred_scores = np.array([item['score'] for item in gliclass_outputs[0]])
            # Add the ignored positives at the end
            if num_ignored_positives := len(is_relevant) - len(pred_scores):
                pred_scores = np.concatenate([pred_scores, np.zeros(num_ignored_positives)])

            mrr, ndcg, ap = self.compute_metrics(is_relevant, pred_scores)

            all_mrr_scores.append(mrr)
            all_ndcg_scores.append(ndcg)
            all_ap_scores.append(ap)

        mean_mrr = np.mean(all_mrr_scores)
        mean_ndcg = np.mean(all_ndcg_scores)
        mean_ap = np.mean(all_ap_scores)
        metrics = {
            "map": mean_ap,
            f"mrr@{self.at_k}": mean_mrr,
            f"ndcg@{self.at_k}": mean_ndcg,
        }

        logger.info(
            f"Queries: {num_queries}\t"
            f"Positives: Min {np.min(num_positives):.1f}, Mean {np.mean(num_positives):.1f}, Max {np.max(num_positives):.1f}\t"
            f"Negatives: Min {np.min(num_negatives):.1f}, Mean {np.mean(num_negatives):.1f}, Max {np.max(num_negatives):.1f}"
        )
        if documents:
            mean_base_mrr = np.mean(base_mrr_scores)
            mean_base_ndcg = np.mean(base_ndcg_scores)
            mean_base_ap = np.mean(base_ap_scores)
            base_metrics = {
                "base_map": mean_base_ap,
                f"base_mrr@{self.at_k}": mean_base_mrr,
                f"base_ndcg@{self.at_k}": mean_base_ndcg,
            }
            logger.info(f"{' ' * len(str(self.at_k))}       Base  -> Reranked")
            logger.info(f"MAP:{' ' * len(str(self.at_k))}   {mean_base_ap * 100:.2f} -> {mean_ap * 100:.2f}")
            logger.info(f"MRR@{self.at_k}:  {mean_base_mrr * 100:.2f} -> {mean_mrr * 100:.2f}")
            logger.info(f"NDCG@{self.at_k}: {mean_base_ndcg * 100:.2f} -> {mean_ndcg * 100:.2f}")

            model_card_metrics = {
                "map": f"{mean_ap:.4f} ({mean_ap - mean_base_ap:+.4f})",
                f"mrr@{self.at_k}": f"{mean_mrr:.4f} ({mean_mrr - mean_base_mrr:+.4f})",
                f"ndcg@{self.at_k}": f"{mean_ndcg:.4f} ({mean_ndcg - mean_base_ndcg:+.4f})",
            }
            model_card_metrics = {
                key: float(value.split()[0]) if isinstance(value, str) else value
                for key, value in model_card_metrics.items()
            }
            model_card_metrics = self.prefix_name_to_metrics(model_card_metrics, self.name)

            metrics.update(base_metrics)
            metrics = self.prefix_name_to_metrics(metrics, self.name)
        else:
            logger.info(f"MAP:{' ' * len(str(self.at_k))}   {mean_ap * 100:.2f}")
            logger.info(f"MRR@{self.at_k}:  {mean_mrr * 100:.2f}")
            logger.info(f"NDCG@{self.at_k}: {mean_ndcg * 100:.2f}")

            metrics = self.prefix_name_to_metrics(metrics, self.name)
            self.store_metrics_in_model_card_data(model, metrics, epoch, steps)

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a" if output_file_exists else "w", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, mean_ap, mean_mrr, mean_ndcg])

        return metrics


class GLiClassNanoBEIREvaluator(CrossEncoderNanoBEIREvaluator):
    def _load_dataset(self, dataset_name, **ir_evaluator_kwargs) -> CrossEncoderRerankingEvaluator:
        if not is_datasets_available():
            raise ValueError(
                "datasets is not available. Please install it to use the CrossEncoderNanoBEIREvaluator via `pip install datasets`."
            )
        from datasets import load_dataset

        dataset_path = dataset_name_to_id[dataset_name.lower()]
        corpus = load_dataset(dataset_path, "corpus", split="train")
        corpus_mapping = dict(zip(corpus["_id"], corpus["text"]))
        queries = load_dataset(dataset_path, "queries", split="train")
        query_mapping = dict(zip(queries["_id"], queries["text"]))
        relevance = load_dataset(dataset_path, "relevance", split="train")

        def mapper(sample, corpus_mapping: dict[str, str], query_mapping: dict[str, str], rerank_k: int):
            query = query_mapping[sample["query-id"]]
            positives = [corpus_mapping[positive_id] for positive_id in sample["positive-corpus-ids"]]
            documents = [corpus_mapping[document_id] for document_id in sample["bm25-ranked-ids"][:rerank_k]]
            return {
                "query": query,
                "positive": positives,
                "documents": documents,
            }

        relevance = relevance.map(
            mapper,
            fn_kwargs={"corpus_mapping": corpus_mapping, "query_mapping": query_mapping, "rerank_k": self.rerank_k},
        )

        human_readable_name = self._get_human_readable_name(dataset_name)
        return GLiClassRerankingEvaluator(
            samples=list(relevance),
            name=human_readable_name,
            **ir_evaluator_kwargs,
        )

    def __call__(
            self, model: ZeroShotWithTruncationPipeline, output_path: str = None, epoch: int = -1, steps: int = -1,
            *args, **kwargs
    ) -> dict[str, float]:
        per_metric_results = {}
        per_dataset_results = {}
        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps"
        else:
            out_txt = ""
        logger.info(f"NanoBEIR Evaluation of the model on {self.dataset_names} dataset{out_txt}:")

        for evaluator in tqdm(self.evaluators, desc="Evaluating datasets", disable=not self.show_progress_bar):
            logger.info(f"Evaluating {evaluator.name}")
            evaluation = evaluator(model, output_path, epoch, steps)
            for k in evaluation:
                dataset, _rerank_k, metric = k.split("_", maxsplit=2)
                if metric not in per_metric_results:
                    per_metric_results[metric] = []
                per_dataset_results[f"{dataset}_R{self.rerank_k}_{metric}"] = evaluation[k]
                per_metric_results[metric].append(evaluation[k])
            logger.info("")

        agg_results = {}
        for metric in per_metric_results:
            agg_results[metric] = self.aggregate_fn(per_metric_results[metric])

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                fOut = open(csv_path, mode="w", encoding="utf-8")
                fOut.write(",".join(self.csv_headers))
                fOut.write("\n")

            else:
                fOut = open(csv_path, mode="a", encoding="utf-8")

            output_data = [
                epoch,
                steps,
                agg_results["map"],
                agg_results[f"mrr@{self.at_k}"],
                agg_results[f"ndcg@{self.at_k}"],
            ]

            fOut.write(",".join(map(str, output_data)))
            fOut.write("\n")
            fOut.close()

        logger.info("CrossEncoderNanoBEIREvaluator: Aggregated Results:")
        logger.info(f"{' ' * len(str(self.at_k))}       Base  -> Reranked")
        logger.info(
            f"MAP:{' ' * len(str(self.at_k))}   {agg_results['base_map'] * 100:.2f} -> {agg_results['map'] * 100:.2f}"
        )
        logger.info(
            f"MRR@{self.at_k}:  {agg_results[f'base_mrr@{self.at_k}'] * 100:.2f} -> {agg_results[f'mrr@{self.at_k}'] * 100:.2f}"
        )
        logger.info(
            f"NDCG@{self.at_k}: {agg_results[f'base_ndcg@{self.at_k}'] * 100:.2f} -> {agg_results[f'ndcg@{self.at_k}'] * 100:.2f}"
        )

        model_card_metrics = {
            "map": f"{agg_results['map']:.4f} ({agg_results['map'] - agg_results['base_map']:+.4f})",
            f"mrr@{self.at_k}": f"{agg_results[f'mrr@{self.at_k}']:.4f} ({agg_results[f'mrr@{self.at_k}'] - agg_results[f'base_mrr@{self.at_k}']:+.4f})",
            f"ndcg@{self.at_k}": f"{agg_results[f'ndcg@{self.at_k}']:.4f} ({agg_results[f'ndcg@{self.at_k}'] - agg_results[f'base_ndcg@{self.at_k}']:+.4f})",
        }

        agg_results = self.prefix_name_to_metrics(agg_results, self.name)
        per_dataset_results.update(agg_results)

        return per_dataset_results


if __name__ == '__main__':
    from gliclass import GLiClassModel
    from transformers import AutoTokenizer

    chunk_pipeline = True

    model_path = "knowledgator/gliclass_msmarco_merged"

    model = GLiClassModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)
    pipeline = ZeroShotWithTruncationPipeline(model=model,
                                              tokenizer=tokenizer,
                                              max_classes=25, max_length=2048, classification_type='multi-label',
                                              device='cuda:0')
    dataset_names = [
        "climatefever",
        "dbpedia",
        "fever",
        "fiqa2018",
        "hotpotqa",
        "msmarco",
        "nfcorpus",
        "nq",
        "quoraretrieval",
        "scidocs",
        "arguana",
        "scifact",
        "touche2020",
    ]
    evaluator = GLiClassNanoBEIREvaluator(dataset_names)
    results = evaluator(pipeline)
    print(results)