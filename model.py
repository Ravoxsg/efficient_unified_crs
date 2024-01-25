import math
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from annoy import AnnoyIndex

from utils import sample_ids_from_db


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
        for i in range(len(list(self.args.items_db.keys()))):
            id = list(self.args.items_db.keys())[i]
            idx = i
            self.item_id_to_idx[id] = idx
            self.idx_to_item_id[idx] = id

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
        self.item_head_l2 = nn.Linear(self.language_model.config.n_embd, self.language_model.config.n_embd)
        self.weights = nn.Parameter(torch.ones(1024))

    def get_rec_token_wtes(self):
        rec_token_input_ids = self.tokenizer(self.args.rec_token, return_tensors="pt")["input_ids"].to(self.device)
        return self.language_model.transformer.wte(rec_token_input_ids)

    def get_rec_end_token_wtes(self):
        rec_end_token_input_ids = self.tokenizer(self.args.rec_end_token, return_tensors="pt")["input_ids"].to(self.device)
        return self.language_model.transformer.wte(rec_end_token_input_ids)

    def item_embeddings(self, h):
        h = self.item_head_l1(h) + h
        h = self.item_head_l2(h)
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
            embeds_i = torch.cat((embeds_no_rec[i][0], embeds_no_rec[i][1]))
            embeds.append(embeds_i.unsqueeze(0))
        embeds = torch.cat(embeds)

        lm_outputs = self.language_model(inputs_embeds=embeds)
        train_logits = lm_outputs.logits[:, :-1, :].contiguous()  # skip the last one

        return train_logits

    # training/inference forward pass when there is one recommendation, doing both response prediction +
    def forward_recall(self,
                       indices,
                       context_with_utterances_tokens,
                       embeds_has_rec,
                       context_lengths,
                       targets,
                       num_samples,
                       previous_recommended_ids=None,
                       ):
        # recall step 1. construct LM sequence output

        REC_wtes = self.get_rec_token_wtes()
        REC_END_wtes = self.get_rec_end_token_wtes()
        REC_targets = self.tokenizer(self.args.rec_token, return_tensors="pt")['input_ids'].to(self.device)
        bs = len(indices)

        gt_items_wte = self.compute_encoded_embeddings_for_items(targets, self.args.items_db, chunk_size=self.args.train_item_encoding_chunk_size)
        gt_items_wte = self.rerank_item_wte_mapper(gt_items_wte)

        embeds = []
        for i in range(len(embeds_has_rec)):
            gt_items_wte_i = gt_items_wte[i:(i + 1)]
            extra_tokens = torch.cat((REC_wtes[0], gt_items_wte_i, REC_END_wtes[0]))
            embeds_i = torch.cat((embeds_has_rec[i][0], extra_tokens, embeds_has_rec[i][1]))
            embeds.append(embeds_i.unsqueeze(0))
        embeds = torch.cat(embeds)
        lm_wte_inputs = self.trim_lm_wtes(embeds)  # trim for len > self.language_model.config.n_positions

        # recall step 2. get gpt output logits and hidden states
        lm_outputs = self.language_model(inputs_embeds=lm_wte_inputs, output_hidden_states=True)

        # recall step 3. pull logits (recall, rec_token and language logits of current turn) and compute

        # recall logit(s)
        hidden_states = lm_outputs.hidden_states[-1]
        rec_token_hidden = hidden_states[torch.arange(hidden_states.shape[0]), context_lengths]
        rec_query_vector = self.recall_lm_query_mapper(rec_token_hidden)

        all_sampled_item_ids, all_gt_item_id_indices = [], []
        for i in range(len(indices)):
            # sample num_samples item ids to train recall with "recommendation as classification"
            previous_ids = None
            if self.args.previous_recommended_ids_negative:
                previous_ids = previous_recommended_ids[i]
            sampled_item_ids = sample_ids_from_db(targets[i], num_samples, self.args, previous_recommended_ids=previous_ids)
            all_sampled_item_ids += sampled_item_ids
            gt_item_id_index = sampled_item_ids.index(targets[i])
            all_gt_item_id_indices.append(gt_item_id_index)

        if self.args.share_batch_negatives:
            if self.args.previous_recommended_ids_negative:
                pos_ids = []
                for i in range(len(indices)):
                    pos_ids += [previous_recommended_ids[i]]
                    pos_ids.append(all_sampled_item_ids[num_samples-1 + i*num_samples])
                neg_ids = [x for x in all_sampled_item_ids if not (x in pos_ids)]
                p = np.random.permutation(len(neg_ids))
                neg_ids = [neg_ids[x] for x in p]
                min_n_previous = min([len(x) for x in previous_recommended_ids])
                neg_ids = neg_ids[:(num_samples-1-min_n_previous)]
                if len(neg_ids) < num_samples-1-min_n_previous:
                    n_extra = (num_samples-1-min_n_previous) - len(neg_ids)
                    neg_ids = neg_ids + neg_ids[:n_extra]
                all_sampled_item_ids = neg_ids + pos_ids
                short_encoded_items_embeddings = self.compute_encoded_embeddings_for_items(all_sampled_item_ids, self.args.items_db, chunk_size=self.args.train_item_encoding_chunk_size)
                encoded_items_embeddings = torch.zeros((bs * num_samples, short_encoded_items_embeddings.shape[-1]), dtype=torch.float, device=self.device)
                begin_pos = num_samples-1-min_n_previous
                for i in range(bs):
                    begin = i*num_samples
                    end = (i+1)*num_samples
                    n_previous = len(previous_recommended_ids[i])
                    end_pos = begin_pos + n_previous + 1
                    encoded_items_embeddings[begin:(end-1-n_previous), :] = short_encoded_items_embeddings[:(num_samples-1-n_previous), :]
                    encoded_items_embeddings[(end-1-n_previous):end, :] = short_encoded_items_embeddings[begin_pos:end_pos, :]
                    begin_pos = end_pos
            else:
                pos_ids = [all_sampled_item_ids[(num_samples-1) + i * num_samples] for i in range(len(indices))]
                neg_ids = [x for x in all_sampled_item_ids if not (x in pos_ids)]
                p = np.random.permutation(len(neg_ids))
                neg_ids = [neg_ids[x] for x in p]
                neg_ids = neg_ids[:(num_samples-1)]
                all_sampled_item_ids = neg_ids + pos_ids
                short_encoded_items_embeddings = self.compute_encoded_embeddings_for_items(all_sampled_item_ids, self.args.items_db, chunk_size=self.args.train_item_encoding_chunk_size)
                encoded_items_embeddings = torch.zeros((bs*num_samples, short_encoded_items_embeddings.shape[-1]), dtype=torch.float, device=self.device)
                for i in range(bs):
                    begin = i*num_samples
                    end = (i+1)*num_samples
                    encoded_items_embeddings[begin:(end-1), :] = short_encoded_items_embeddings[:(num_samples-1), :]
                    encoded_items_embeddings[(end-1), :] = short_encoded_items_embeddings[num_samples-1+i, :]
        else:
            encoded_items_embeddings = self.compute_encoded_embeddings_for_items(all_sampled_item_ids, self.args.items_db, chunk_size=self.args.train_item_encoding_chunk_size)
        # to compute dot product with rec_query_vector
        items_key_vectors = self.recall_item_wte_mapper(encoded_items_embeddings)
        items_key_vectors = items_key_vectors.reshape((bs, num_samples, items_key_vectors.shape[-1]))
        expanded_rec_query_vector = rec_query_vector.unsqueeze(1).expand(rec_query_vector.shape[0], items_key_vectors.shape[1], rec_query_vector.shape[1])
        recall_logits = torch.sum(expanded_rec_query_vector * items_key_vectors, dim=2)

        # REC_TOKEN prediction and future sentence prediction
        # hidden rep of the token that's right before REC_TOKEN
        logits = lm_outputs.logits
        all_logits, all_targets = [], []
        for i in range(logits.shape[0]):
            logits_i = logits[i]
            logits_i = torch.cat((logits_i[:(context_lengths[i]), :], logits_i[(context_lengths[i]+3):-1, :]))
            all_logits.append(logits_i.unsqueeze(0))
            targets_i = context_with_utterances_tokens[i]
            targets_i = torch.cat((targets_i[1:(context_lengths[i])], REC_targets[0], targets_i[(context_lengths[i]+1):]))
            all_targets.append(targets_i.unsqueeze(0))
        language_logits = torch.cat(all_logits)
        language_targets = torch.cat(all_targets)

        return recall_logits, all_gt_item_id_indices, language_logits, language_targets, encoded_items_embeddings

    # training/inference forward pass when there is a recommendation, to re-rank items
    def forward_rerank(self,
                       indices,
                       context_tokens,
                       context_lengths,
                       targets,
                       num_samples,
                       encoded_items_embeddings=None,
                       previous_recommended_ids=None,
                       ):
        # REC wte
        REC_wtes = self.get_rec_token_wtes()

        if encoded_items_embeddings is None:
            all_sampled_item_ids, all_gt_item_id_indices = [], []
            for i in range(context_tokens.shape[0]):
                # sample num_samples item ids to train recall with "recommendation as classification"
                sampled_item_ids = sample_ids_from_db(targets[i], num_samples, self.args, previous_recommended_ids=previous_recommended_ids)
                all_sampled_item_ids += sampled_item_ids
                gt_item_id_index = sampled_item_ids.index(targets[i])
                all_gt_item_id_indices.append(gt_item_id_index)
            encoded_items_embeddings = self.compute_encoded_embeddings_for_items(all_sampled_item_ids, self.args.items_db, chunk_size=self.args.train_item_encoding_chunk_size)
        else:
            all_gt_item_id_indices = [num_samples-1] * len(indices)
        items_wtes = self.rerank_item_wte_mapper(encoded_items_embeddings)
        bs = len(indices)
        total_wtes = items_wtes.reshape((bs, num_samples, items_wtes.shape[-1]))
        total_wtes_len = total_wtes.shape[1]

        combined_position_ids, embeds = [], []
        expected_len = context_tokens.shape[1]+1+total_wtes_len
        for i in range(context_tokens.shape[0]):
            past_length = context_lengths[i]
            position_ids_i = torch.arange(0, past_length+1, dtype=torch.long, device=self.device)
            combined_position_ids_i = torch.cat((position_ids_i, torch.zeros((expected_len-1-past_length), dtype=torch.long, device=self.device)))
            combined_position_ids.append(combined_position_ids_i.unsqueeze(0))
            embeds_i = []
            for j in range(context_tokens.shape[1]):
                if context_tokens[i, j].item() < len(self.tokenizer):
                    embeds_i_j = self.language_model.transformer.wte(context_tokens[i, j])
                else:
                    item_id = self.args.pseudo_tokens_to_item_ids[context_tokens[i, j].item()]
                    embeds_i_j = self.compute_encoded_embeddings_for_items([item_id], self.args.items_db, chunk_size=self.args.train_item_encoding_chunk_size)[0]
                    embeds_i_j = self.rerank_item_wte_mapper(embeds_i_j)
                embeds_i.append(embeds_i_j.unsqueeze(0))
                if j == past_length - 1:
                    embeds_i.append(REC_wtes[0])
                    embeds_i.append(total_wtes[i])
            embeds_i = torch.cat(embeds_i)
            embeds.append(embeds_i.unsqueeze(0))
        combined_position_ids = torch.cat(combined_position_ids)
        embeds = torch.cat(embeds)

        # trim sequence to smaller length (len < self.language_model.config.n_positions - self.args.lm_trim_offset)
        combined_position_ids_trimmed = self.trim_positional_ids(combined_position_ids, total_wtes_len)
        embeds_trimmed = self.trim_lm_wtes(embeds)
        assert (combined_position_ids.shape[1] == embeds.shape[1])

        inductive_attention_mask = torch.zeros((context_tokens.shape[0], embeds.shape[1], embeds.shape[1]), dtype=torch.float, device=self.device)
        for i in range(context_tokens.shape[0]):
            start = context_lengths[i]+1
            end = context_lengths[i]+1+total_wtes_len
            inductive_attention_mask[i, start:end, start:end] = 1
        rerank_lm_outputs = self.language_model(
            inputs_embeds=embeds_trimmed,
            inductive_attention_mask=inductive_attention_mask,
            position_ids=combined_position_ids_trimmed,
            output_hidden_states=True
        )

        rerank_lm_hidden = rerank_lm_outputs.hidden_states[-1]
        rerank_lm_hidden = torch.cat([rerank_lm_hidden[i, (context_lengths[i]+1):(context_lengths[i]+1+total_wtes_len), :].unsqueeze(0) for i in range(context_tokens.shape[0])])
        rerank_logits = self.rerank_logits_mapper(rerank_lm_hidden).squeeze(-1)

        return rerank_logits, all_gt_item_id_indices

    # inference forward pass of the recall step
    def validation_perform_recall(self, context_tokens, context_lengths, topk):
        REC_wtes = self.get_rec_token_wtes()

        embeds = []
        for i in range(context_tokens.shape[0]):
            past_length = context_lengths[i]
            embeds_i = []
            for j in range(context_tokens.shape[1]):
                if context_tokens[i, j].item() < len(self.tokenizer):
                    embeds_i_j = self.language_model.transformer.wte(context_tokens[i, j])
                else:
                    item_id = self.args.pseudo_tokens_to_item_ids[context_tokens[i, j].item()]
                    embeds_i_j = self.compute_encoded_embeddings_for_items([item_id], self.args.items_db, chunk_size=self.args.train_item_encoding_chunk_size)[0]
                    embeds_i_j = self.rerank_item_wte_mapper(embeds_i_j)
                embeds_i.append(embeds_i_j.unsqueeze(0))
                if j == past_length-1:
                    embeds_i.append(REC_wtes[0])
            embeds_i = torch.cat(embeds_i)
            embeds.append(embeds_i.unsqueeze(0))
        embeds = torch.cat(embeds)

        lm_wte_inputs = self.trim_lm_wtes(embeds)
        lm_outputs = self.language_model(inputs_embeds=lm_wte_inputs, output_hidden_states=True)

        rec_token_hidden = lm_outputs.hidden_states[-1][torch.arange(context_tokens.shape[0]), context_lengths, :]
        rec_query_vectors = self.recall_lm_query_mapper(rec_token_hidden) # [bs, 768]
        rec_query_vectors = rec_query_vectors.cpu().detach().numpy()
        recall_results = []
        for i in range(rec_query_vectors.shape[0]):
            recall_results_i = self.annoy_base_recall.get_nns_by_vector(rec_query_vectors[i], topk)
            recall_results.append(recall_results_i)

        return recall_results

    # inference forward pass of the re-ranking step
    def validation_perform_rerank(self, context_tokens, context_lengths, recalled_ids):
        REC_wtes = self.get_rec_token_wtes()

        total_wtes = []
        for i in range(len(recalled_ids)):
            total_wtes_i = [self.annoy_base_rerank.get_item_vector(r_id) for r_id in recalled_ids[i]]
            total_wtes_i = [torch.tensor(wte).reshape(-1, self.language_model.config.n_embd).to(self.device) for wte in total_wtes_i]
            total_wtes_i = torch.cat(total_wtes_i, dim=0)
            total_wtes.append(total_wtes_i.unsqueeze(0))
        total_wtes = torch.cat(total_wtes)

        REC_wtes_len = REC_wtes.shape[1]
        total_wtes_len = total_wtes.shape[1]

        combined_position_ids, embeds = [], []
        expected_len = context_tokens.shape[1]+1+total_wtes_len
        combined_position_ids = torch.zeros((context_tokens.shape[0], expected_len), dtype=torch.long, device=self.device)
        for i in range(context_tokens.shape[0]):
            past_length = context_lengths[i]
            position_ids_i = torch.arange(0, past_length + REC_wtes_len, dtype=torch.long, device=self.device)
            combined_position_ids[i, :(past_length + REC_wtes_len)] = position_ids_i
            embeds_i = []
            for j in range(context_tokens.shape[1]):
                if context_tokens[i, j].item() < len(self.tokenizer):
                    embeds_i_j = self.language_model.transformer.wte(context_tokens[i, j])
                else:
                    item_id = self.args.pseudo_tokens_to_item_ids[context_tokens[i, j].item()]
                    embeds_i_j = self.compute_encoded_embeddings_for_items([item_id], self.args.items_db, chunk_size=self.args.train_item_encoding_chunk_size)[0]
                    embeds_i_j = self.rerank_item_wte_mapper(embeds_i_j)
                embeds_i.append(embeds_i_j.unsqueeze(0))
                if j == past_length-1:
                    embeds_i.append(REC_wtes[0])
                    embeds_i.append(total_wtes[i])
            embeds_i = torch.cat(embeds_i)
            embeds.append(embeds_i.unsqueeze(0))
        embeds = torch.cat(embeds)

        # trim sequence to smaller length (len < self.language_model.config.n_positions - self.args.lm_trim_offset)
        combined_position_ids_trimmed = self.trim_positional_ids(combined_position_ids, total_wtes_len)
        embeds_trimmed = self.trim_lm_wtes(embeds)
        assert (combined_position_ids.shape[1] == embeds.shape[1])

        inductive_attention_mask = torch.zeros((context_tokens.shape[0], embeds.shape[1], embeds.shape[1]), dtype=torch.float, device=self.device)
        for i in range(context_tokens.shape[0]):
            start = context_lengths[i]+1
            end = context_lengths[i]+1+total_wtes_len
            inductive_attention_mask[i, start:end, start:end] = 1

        rerank_lm_outputs = self.language_model(
            inputs_embeds = embeds_trimmed,
            inductive_attention_mask = inductive_attention_mask,
            position_ids = combined_position_ids_trimmed,
            output_hidden_states = True
        )

        rerank_lm_hidden = rerank_lm_outputs.hidden_states[-1]
        rerank_lm_hidden = torch.cat([rerank_lm_hidden[i, (context_lengths[i]+1):(context_lengths[i]+1+total_wtes_len), :].unsqueeze(0) for i in range(context_tokens.shape[0])])
        rerank_logits = self.rerank_logits_mapper(rerank_lm_hidden).squeeze(-1)

        return rerank_logits






