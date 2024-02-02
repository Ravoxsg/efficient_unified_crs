import numpy as np
import torch
import copy
from tqdm import tqdm
from torch.utils.data import Dataset


class MovieRecDataset(Dataset):
    def __init__(self, split, data, tokenizer, logger, args):
        self.split = split
        self.data = data
        logger.info(f"The {split} dataset has {len(data)} points")
        self.tokenizer = tokenizer
        self.logger = logger
        self.args = args

        if self.split != "train":
            p = np.random.permutation(len(self.data))
            p = p[:self.args.max_val_size]
            if logger is not None:
                logger.info(f"{split} permutation: {p[:10]}")
            #self.data = [self.data[x] for x in p]

        if self.args.debug:
            p = np.random.permutation(len(self.data))
            p = p[:args.debug_size]
            if logger is not None:
                logger.info(f"{split} DEBUG permutation: {p[:10]}")
            self.data = [self.data[x] for x in p]

        logger.info(f"The {split} dataset final version has {len(self.data)} points")

        self.turn_ending = torch.tensor([628, 198])  # end of turn, '\n\n\n't
        self.REC_token = self.tokenizer(args.rec_token, return_tensors="pt")['input_ids'][0]
        self.REC_END_token = self.tokenizer(args.rec_end_token, return_tensors="pt")['input_ids'][0]
        self.SEP_token = self.tokenizer(args.sep_token, return_tensors="pt")['input_ids'][0]

        metadata, data = self.prepare_data()
        (indexes, user_ids, turn_nums, repeated, previous_ids) = metadata
        self.indexes = indexes  # dialogue IDs
        self.user_ids = user_ids  # user IDs
        self.turn_nums = turn_nums  # utterance turn number
        self.repeated = repeated  # whether the target utterance is a repeat (happens when there are several recommended items)
        self.previous_ids = previous_ids  # item IDs previously recommended to the user
        logger.info(f"{len(indexes)} {len(user_ids)} {len(turn_nums)} {len(repeated)} {len(previous_ids)}")

        (raw_contexts, contexts, raw_utterances, utterances, targets) = data
        self.raw_contexts = raw_contexts  # contexts without special tokens
        self.contexts = contexts  # contexts
        self.raw_utterances = raw_utterances  # utterances without special tokens
        self.utterances = utterances  # utterances
        self.targets = targets  # target items
        logger.info(f"{len(raw_contexts)} {len(contexts)} {len(raw_utterances)} {len(utterances)} {len(targets)}")

    def prepare_data(self):
        self.logger.info("Preparing the data...")
        n_utt = 0
        seeker_count, recommender_count = 0, 0
        indexes, user_ids, turn_nums, repeated, previous_ids = [], [], [], [], []  # metadata
        raw_contexts, contexts, raw_utterances, utterances, targets = [], [], [], [], []  # data
        for i in tqdm(range(len(self.data))):
            user_id, dialogue = self.data[i]
            raw_past_tokens, past_tokens = None, None
            past_recommended_ids = []
            for turn_num, (utterance, gt_ind) in enumerate(dialogue):
                n_utt += 1
                raw_utterance = utterance.split()
                if gt_ind is not None:
                    if len(gt_ind) == raw_utterance.count(self.args.placeholder_token):
                        movie_idx = 0
                        for j in range(len(raw_utterance)):
                            if raw_utterance[j] == self.args.placeholder_token:
                                movie_id = gt_ind[movie_idx]
                                movie_name = self.args.items_db[movie_id].split(self.args.sep_token)[0]
                                raw_utterance[j] = movie_name
                                movie_idx += 1

                raw_utterance = " ".join(raw_utterance)
                raw_utt_tokens = self.tokenizer(raw_utterance, return_tensors="pt")['input_ids'][0]
                raw_utt_tokens = raw_utt_tokens[:self.args.utt_max_length]
                raw_utt_tokens = torch.cat((raw_utt_tokens, self.turn_ending), dim=0)
                utt_tokens = self.tokenizer(utterance, return_tensors="pt")['input_ids'][0]
                utt_tokens = utt_tokens[:self.args.utt_max_length]
                utt_tokens = torch.cat((utt_tokens, self.turn_ending), dim=0)

                if past_tokens == None:
                    raw_past_tokens = raw_utt_tokens
                    past_tokens = utt_tokens
                    continue

                # the seeker speaks -> we just append it to the context
                if utterance.startswith("B"):
                    seeker_count += 1
                    if gt_ind == None:
                        raw_past_tokens = torch.cat((raw_past_tokens, raw_utt_tokens))
                        past_tokens = torch.cat((past_tokens, utt_tokens))
                    else:
                        ids = torch.tensor([self.args.item_ids_to_pseudo_tokens[x] for x in gt_ind])
                        raw_past_tokens = torch.cat((raw_past_tokens, raw_utt_tokens))
                        combined_tokens = torch.cat((utt_tokens, self.SEP_token, ids, self.SEP_token), dim=0)
                        past_tokens = torch.cat((past_tokens, combined_tokens))

                # the recommender speaks -> we need to predict the utterance and make a new data point
                else:
                    recommender_count += 1
                    # data point with no recommended movie
                    if gt_ind == None:
                        indexes.append(i)
                        user_ids.append(user_id)
                        turn_nums.append(turn_num)
                        repeated.append(0)
                        raw_contexts.append(raw_past_tokens[-self.args.context_max_length:])
                        contexts.append(past_tokens[-self.args.context_max_length:])
                        raw_utterances.append(raw_utt_tokens)
                        utterances.append(utt_tokens)
                        targets.append(-1)
                        past_recommended_ids_j = copy.deepcopy(past_recommended_ids)
                        previous_ids.append(past_recommended_ids_j)
                        raw_past_tokens = torch.cat((raw_past_tokens, raw_utt_tokens))
                        past_tokens = torch.cat((past_tokens, utt_tokens))
                    # data point has at least 1 recommended movie -> we make 1 data point per movie
                    else:
                        for j in range(len(gt_ind)):
                            recommended_id = gt_ind[j]
                            indexes.append(i)
                            user_ids.append(user_id)
                            turn_nums.append(turn_num)
                            repeated.append(int(j > 0))
                            raw_contexts.append(raw_past_tokens[-self.args.context_max_length:])
                            contexts.append(past_tokens[-self.args.context_max_length:])
                            raw_utterances.append(raw_utt_tokens)
                            utterances.append(utt_tokens)
                            targets.append(recommended_id)
                            past_recommended_ids_j = copy.deepcopy(past_recommended_ids)
                            previous_ids.append(past_recommended_ids_j)
                        ids = torch.tensor([self.args.item_ids_to_pseudo_tokens[x] for x in gt_ind])
                        raw_past_tokens = torch.cat((raw_past_tokens, raw_utt_tokens))
                        combined_tokens = torch.cat((self.REC_token, ids, self.REC_END_token))
                        past_tokens = torch.cat((past_tokens, combined_tokens))
                        past_tokens = torch.cat((past_tokens, utt_tokens))
                        for x in gt_ind:
                            if x not in past_recommended_ids:
                                past_recommended_ids.append(x)

        self.logger.info(f"Seeker speaks {seeker_count} times, recommender {recommender_count} times")
        self.logger.info(f"Total # utterances: {n_utt}")
        self.logger.info(f"The dataset leads to {len(indexes)} data points")
        n_without_rec = len([x for x in targets if x == -1])
        n_with_rec = len([x for x in targets if x != -1])
        self.logger.info(f"Data points without recommendation: {n_without_rec}")
        self.logger.info(f"Data points with recommendation: {n_with_rec}")

        metadata = (indexes, user_ids, turn_nums, repeated, previous_ids)
        data = (raw_contexts, contexts, raw_utterances, utterances, targets)

        return metadata, data

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, ix):
        index = self.indexes[ix]
        user_id = self.user_ids[ix]
        turn_num = self.turn_nums[ix]
        repeated = self.repeated[ix]
        previous_recommended_ids = self.previous_ids[ix]

        raw_context = self.raw_contexts[ix]
        raw_context_length = len(raw_context)
        context = self.contexts[ix]
        context_length = len(context)
        raw_utterance = self.raw_utterances[ix]
        raw_utterance_length = len(raw_utterance)
        utterance = self.utterances[ix]
        utterance_length = len(utterance)
        target = self.targets[ix]

        data_point = {
            "index": [index],
            "user_id": [user_id],
            "turn_num": [turn_num],
            "repeated": [repeated],
            "previous_recommended_ids": previous_recommended_ids,

            "raw_context": raw_context,
            "raw_context_length": [raw_context_length],
            "context": context,
            "context_length": [context_length],
            "raw_utterance": raw_utterance,
            "raw_utterance_length": [raw_utterance_length],
            "utterance": utterance,
            "utterance_length": [utterance_length],
            "target": [target],
        }

        return data_point

class MovieRecDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        indices = np.array([x["index"][0] for x in batch])
        user_ids = np.array([x["user_id"][0] for x in batch])
        turn_nums = np.array([x["turn_num"][0] for x in batch])
        repeated = np.array([x["repeated"][0] for x in batch])
        previous_recommended_ids = [x["previous_recommended_ids"] for x in batch]

        raw_context_lengths = np.array([x["raw_context_length"][0] for x in batch])
        context_lengths = np.array([x["context_length"][0] for x in batch])
        raw_utterance_lengths = np.array([x["raw_utterance_length"][0] for x in batch])
        utterance_lengths = np.array([x["utterance_length"][0] for x in batch])
        targets = np.array([x["target"][0] for x in batch])

        # pad contexts and utterances
        max_raw_context_length = np.max(raw_context_lengths)
        max_context_length = np.max(context_lengths)
        max_raw_utterance_length = np.max(raw_utterance_lengths)
        max_utterance_length = np.max(utterance_lengths)
        context_with_utterances, raw_contexts, contexts, contexts_padded_left, raw_utterances, utterances = [], [], [], [], [], []
        for x in batch:
            # context + utterance packed together (for language modeling)
            context_with_utterance = torch.cat((x["context"], x["utterance"]))
            if len(context_with_utterance) < (max_context_length + max_utterance_length):
                extra_tokens = [self.tokenizer.pad_token_id]*((max_context_length+max_utterance_length)-len(context_with_utterance))
                extra_tokens = torch.tensor(extra_tokens)
                context_with_utterance = torch.cat([context_with_utterance, extra_tokens])
            context_with_utterances.append(context_with_utterance.unsqueeze(0))

            # raw context
            if len(x["raw_context"]) < max_raw_context_length:
                extra_tokens = [self.tokenizer.pad_token_id]*(max_raw_context_length-len(x["raw_context"]))
                extra_tokens = torch.tensor(extra_tokens)
                x["raw_context"] = torch.cat([x["raw_context"], extra_tokens])
            raw_contexts.append(x["raw_context"].unsqueeze(0))

            # context
            if len(x["context"]) < max_context_length:
                extra_tokens = [self.tokenizer.pad_token_id]*(max_context_length-len(x["context"]))
                extra_tokens = torch.tensor(extra_tokens)
                x["context"] = torch.cat([x["context"], extra_tokens])
            contexts.append(x["context"].unsqueeze(0))

            # context padded left (for generation)
            if len(x["context"]) < max_context_length:
                extra_tokens = [self.tokenizer.pad_token_id]*(max_context_length-len(x["context"]))
                extra_tokens = torch.tensor(extra_tokens)
                x["context"] = torch.cat([extra_tokens, x["context"]])
            contexts_padded_left.append(x["context"].unsqueeze(0))

            # raw utterance
            if len(x["raw_utterance"]) < max_raw_utterance_length:
                extra_tokens = [self.tokenizer.pad_token_id]*(max_raw_utterance_length-len(x["raw_utterance"]))
                extra_tokens = torch.tensor(extra_tokens)
                x["raw_utterance"] = torch.cat([x["raw_utterance"], extra_tokens])
            raw_utterances.append(x["raw_utterance"].unsqueeze(0))

            # utterance
            if len(x["utterance"]) < max_utterance_length:
                extra_tokens = [self.tokenizer.pad_token_id]*(max_utterance_length-len(x["utterance"]))
                extra_tokens = torch.tensor(extra_tokens)
                x["utterance"] = torch.cat([x["utterance"], extra_tokens])
            utterances.append(x["utterance"].unsqueeze(0))

        context_with_utterances = torch.cat(context_with_utterances)
        raw_contexts = torch.cat(raw_contexts)
        contexts = torch.cat(contexts)
        contexts_padded_left = torch.cat(contexts_padded_left)
        raw_utterances = torch.cat(raw_utterances)
        utterances = torch.cat(utterances)

        return {
            "indices": indices,
            "user_ids": user_ids,
            "turn_nums": turn_nums,
            "repeated": repeated,
            "previous_recommended_ids": previous_recommended_ids,

            "context_with_utterances": context_with_utterances,
            "raw_contexts": raw_contexts,
            "contexts": contexts,
            "context_lengths": context_lengths,
            "contexts_padded_left": contexts_padded_left,
            "raw_utterances": raw_utterances,
            "utterances": utterances,
            "utterance_lengths": utterance_lengths,
            "targets": targets,
        }



