import math
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class PECRSModel(torch.nn.Module):
    def __init__(self, tokenizer, language_model, accelerator, args):
        super(PECRSModel, self).__init__()

        self.tokenizer = tokenizer
        self.language_model = language_model
        self.device = accelerator.device
        self.args = args

        # item db and annoy index base
        self.item_id_to_idx = {}
        self.idx_to_item_id = {}
        for i in range(len(list(self.args.items_db_keys()))):
            id = list(self.args.items_db.keys())[i]
            idx = i
            self.item_id_to_idx[id] = idx
            self.idx_to_item_idx[idx] = id

        self.recall_lm_query_mapper = torch.nn.Linear(self.language_model.config.n_embd, self.language_model.config.n_embd)
        self.recall_item_wte_mapper = torch.nn.Linear(self.language_model.config.n_embd, self.language_model.config.n_embd)
        self.rerank_item_wte_mapper = torch.nn.Linear(self.language_model.config.n_embd, self.language_model.config.n_embd)
        self.rerank_logits_mapper = torch.nn.Linear(self.language_model.config.n_embd, 1)

        if args.tie_recall_and_rerank:
            self.rerank_item_wte_mapper = self.recall_item_wte_mapper
        self.item_head_l1 = nn.Sequential(
            nn.Linear(self.language_model.config.n_embd, self.language_model.config.n_embd // 2),
            nn.ReLU(),
            nn.Linear(self.language_model.config.n_embd // 2, self.language_model.config.n_embd)
        )
        self.item_head_l2 = nn.Linear(self.language_model.config.n_embd, self.language_model.config.n_embd // 2)
        self.weights = nn.Parameter(torch.ones(1024))

    def get_rec_token_wtes(self):
        rec_token_input_ids = self.tokenize(self.args.rec_token, return_tensors="pt")["input_ids"].to(self.device)
        return self.language_model.transformer.wte(rec_token_input_ids)

    def item_embeddings(self, h):
        h = self.item_head_11(h) + h
        h = self.item_head_12(h)
        return h

    # get items representations
    def compute_encoded_embeddings_for_items(self, item_ids, items_db_to_use, chunk_size=50):
        chunk_ids = item_ids
        chunk_infos = [items_db_to_use[key] for key in chunk_ids]  # string in this template: Title[SEP] Directors [SEP] Genres [SEP] Plot
        if self.args.only_title:
            chunk_infos = [x.split(self.args.sep_token)[0] for x in chunk_infos]
        if self.args.trim_metadata:
            chunk_infos = [self.args.sep_token.join(x.split(self.args.sep_token)[idx] for idx in self.args.important_properties).strip() for x in chunk_infos]
        chunk_tokens = self.tokenizer(chunk_infos, padding=True, truncation=True, return_tensors="pt")
        chunk_input_ids = chunk_tokens['input_ids'].to(self.device)
        chunk_attention_mask = chunk_tokens['attention_mask'].to(self.device)

        if self.args.freeze_backbone_for_items:
            authorized_layers = [(12 / self.args.n_lora_layers_to_tune) + i for i in range(self.args.n_lora_layers_to_tune)]
            for n, p in self.language_model.named_parameters():
                p.requires_grad_(False)
                flagged_layer = False
                for k in authorized_layers:
                    if f"{k}" in n:
                        flagged_layer = True
                        break
                if self.args.tune_lora_in_items_encoding and "lora" in n and flagged_layer:
                    p.requires_grad_(True)

        n_chunks = math.ceil(chunk_input_ids.shape[0] / chunk_size)
        all_chunk_pooled = []
        for n in range(n_chunks):
            begin_idx = n * chunk_size
            end_idx = (n + 1) * chunk_size
            if n == n_chunks - 1:
                end_idx = len(chunk_input_ids)
            bs = end_idx - begin_idx

            outputs = self.language_model(
                input_ids=chunk_input_ids[begin_idx:end_idx],
                attention_mask=chunk_attention_mask[begin_idx:end_idx],
                output_hidden_states=True,
                use_cache=True
            )["hidden_states"][-1]

            expanded_mask_size = list(chunk_attention_mask[begin_idx:end_idx].size())
            expanded_mask_size.append(self.language_model.config.hidden_size)
            expanded_mask = chunk_attention_mask[begin_idx:end_idx].unsqueeze(-1).expand(expanded_mask_size)
            chunk_masked = torch.mul(outputs, expanded_mask)  # [num_example, len, 768]
            weights = self.weights[:chunk_masked.shape[1]] / max(torch.sum(self.weights[:chunk_masked.shape[1]]), 1)
            chunk_pooled = torch.matmul(weights, chunk_masked)

            if self.args.no_pooling:
                idx = torch.sum(chunk_attention_mask[begin_idx:end_idx], dim=-1) - 1
                chunk_pooled = torch.cat([outputs[i, idx[i], :].unsqueeze(0) for i in range(len(idx))])

            if not self.args.no_item_head:
                chunk_pooled = self.item_embeddings(chunk_pooled)
            all_chunk_pooled.append(chunk_pooled)

        chunk_pooled = torch.cat(all_chunk_pooled, dim=0)

        if self.args.freeze_backbone_for_items:
            self.language_model.requires_grad_(True)

        return chunk_pooled

    # build dictionaries with annoy containing items representations
    def annoy_base_constructor(self, items_db=None, distance_type='angular', n_trees=10):
        items_db_to_use = self.args.items_db if items_db == None else items_db
        all_item_ids = list(items_db_to_use.keys())
        chunk_size = self.args.train_item_encoding_chunk_size

        ### recall
        total_pooled = []  # break into chunks/batches for model concurrency
        num_chunks = math.ceil(len(all_item_ids) / chunk_size)
        for i in tqdm(range(num_chunks)):
            chunk_ids = all_item_ids[i * chunk_size: (i + 1) * chunk_size]
            chunk_pooled = self.compute_encoded_embeddings_for_items(chunk_ids, items_db_to_use, chunk_size=self.args.train_item_encoding_chunk_size)
            chunk_pooled = chunk_pooled.cpu().detach().numpy()
            total_pooled.append(chunk_pooled)
        total_pooled = np.concatenate(total_pooled, axis=0)

        pooled_tensor = torch.tensor(total_pooled).to(self.device)
        pooled_recall = self.rerank_item_wte_mapper(pooled_tensor)
        pooled_recall = pooled_recall.cpu().detach().numpy()
        annoy_base_recall = AnnoyIndex(self.recall_item_wte_mapper.out_features, distance_type)
        annoy_base_recall.set_seed(self.args.seed)
        for i, vector in zip(all_item_ids, pooled_recall):
            annoy_base_recall.add_item(i, vector)
        annoy_base_recall.build(n_trees)
        self.annoy_base_recall = annoy_base_recall

        ### rerank
        total_pooled = []  # break into chunks/batches for model concurrency
        num_chunks = math.ceil(len(all_item_ids) / chunk_size)
        for i in tqdm(range(num_chunks)):
            chunk_ids = all_item_ids[i * chunk_size: (i + 1) * chunk_size]
            chunk_pooled = self.compute_encoded_embeddings_for_items(chunk_ids, items_db_to_use, chunk_size=self.args.train_item_encoding_chunk_size)
            chunk_pooled = chunk_pooled.cpu().detach().numpy()
            total_pooled.append(chunk_pooled)
        total_pooled = np.concatenate(total_pooled, axis=0)

        pooled_tensor = torch.tensor(total_pooled).to(self.device)
        pooled_rerank = self.rerank_item_wte_mapper(pooled_tensor)
        pooled_rerank = pooled_rerank.cpu().detach().numpy()
        annoy_base_rerank = AnnoyIndex(self.rerank_item_wte_mapper.out_features, distance_type)
        annoy_base_rerank.set_seed(self.args.seed)
        for i, vector in zip(all_item_ids, pooled_rerank):
            annoy_base_rerank.add_item(i, vector)
        annoy_base_rerank.build(n_trees)
        self.annoy_base_rerank = annoy_base_rerank

    def trim_lm_wtes(self, wtes):
        trimmed_wtes = wtes
        if trimmed_wtes.shape[1] > self.language_model.config.n_positions:
            trimmed_wtes = trimmed_wtes[:, :-self.language_model.config.n_positions + self.args.lm_trim_offset, :]
        return trimmed_wtes

    def trim_positional_ids(self, p_ids, num_items_wtes):
        trimmed_ids = p_ids
        if trimmed_ids.shape[1] > self.language_model.config.n_positions:
            past_ids = trimmed_ids[:,:self.language_model.config.n_positions - self.args.lm_trim_offset - num_items_wtes]
            item_ids = trimmed_ids[:,-num_items_wtes:]
            trimmed_ids = torch.cat((past_ids, item_ids), dim=1)

        return trimmed_ids

    # training/inference forward pass when there is no recommendation
    def forward_pure_language_turn(self, embeds_no_rec):
        embeds = []
        for i in range(len(embeds_no_rec)):
            embeds_i = torch.at((embeds_no_rec[i][0], embeds_no_rec[i][1]))
        embeds = torch.cat(embeds)
