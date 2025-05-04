import torch
from gliclass.pipeline import BaseZeroShotClassificationPipeline
from gliclass import GLiClassModel
from tqdm import tqdm


class ZeroShotWithTruncationPipeline(BaseZeroShotClassificationPipeline):
    def __init__(self, model, tokenizer, max_classes=25, max_length=2048, classification_type='multi-label',
                 device='cuda:0'):
        super().__init__(model, tokenizer, max_classes, max_length, classification_type, device)
        if isinstance(model, str):
            self.model = GLiClassModel.from_pretrained(model).to(device)
        else:
            self.model = model

        if self.model.device != self.device:
            self.model.to(self.device)

    def prepare_input(self, text, labels):
        text_tokens = self.tokenizer.encode(text, add_special_tokens=False)
        label_marker_len = len(self.tokenizer.encode('<<LABEL>>', add_special_tokens=False))
        sep_marker_len = len(self.tokenizer.encode('<<SEP>>', add_special_tokens=False))
        label_tokens_list = []
        for label in labels:
            label_tokens = self.tokenizer.encode(label, add_special_tokens=False)
            label_tokens_list.append(label_tokens)
        labels_lengths = [len(tokens) for tokens in label_tokens_list]
        total_labels_len = sum(labels_lengths)
        available = self.max_length - len(text_tokens) - len(labels) * label_marker_len - sep_marker_len
        truncated_input = []
        for tokens, length in zip(label_tokens_list, labels_lengths):
            proportion = length / total_labels_len
            keep_len = max(1, int(proportion * available))
            trunc_label = tokens[:keep_len]
            decoded = self.tokenizer.decode(trunc_label, skip_special_tokens=True)
            decoded_tag = f"<<LABEL>>{decoded.lower()}"
            truncated_input.append(decoded_tag)
        truncated_input.append('<<SEP>>')
        input_text = ''.join(truncated_input) + text
        return input_text

    def prepare_inputs(self, texts, labels, same_labels=False):
        inputs = []
        for text in texts:
            inputs.append(self.prepare_input(text, labels))
        tokenized_inputs = self.tokenizer(inputs, truncation=True,
                                          max_length=self.max_length,
                                          padding="longest", return_tensors="pt").to(self.device)
        return tokenized_inputs

    @torch.no_grad()
    def __call__(self, texts, labels, threshold=0.5, batch_size=8, labels_chunk_size=25):
        labels = list(set(labels))
        num_batches = (len(texts) + batch_size - 1) // batch_size
        num_label_chunks = (len(labels) + labels_chunk_size - 1) // labels_chunk_size
        total_steps = num_batches * num_label_chunks
        progress = tqdm(total=total_steps, desc='processing') if self.progress_bar else None

        results = []
        iterable = range(0, len(texts), batch_size)

        for idx in iterable:
            batch_texts = texts[idx:idx + batch_size]
            batch_results = [[] for _ in batch_texts]

            for labels_batch in range(0, len(labels), labels_chunk_size):
                curr_labels = labels[labels_batch:labels_batch + labels_chunk_size]
                tokenized_inputs = self.prepare_inputs(batch_texts, curr_labels)
                model_output = self.model(**tokenized_inputs)
                logits = model_output.logits

                if self.classification_type == 'multi-label':
                    sigmoid = torch.nn.Sigmoid()
                    probs = sigmoid(logits)
                    for i in range(len(batch_texts)):
                        for j, prob in enumerate(probs[i][:len(curr_labels)]):
                            score = prob.item()
                            if score >= threshold:
                                batch_results[i].append({'label': curr_labels[j], 'score': score})
                else:
                    raise ValueError("Unsupported classification type: choose 'single-label' or 'multi-label'")

                if progress:
                    progress.update(1)
            results.extend(batch_results)
        if progress:
            progress.close()
        return results
