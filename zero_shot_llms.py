# Source code for the PECRS model (EACL 2024)
# Parts of the code are taken from the MESE source code: https://github.com/by2299/mese

import argparse
import torch
import numpy as np
import tqdm
import pickle
import os
import logging
from rouge_score import rouge_scorer
from nltk.tokenize import word_tokenize, sent_tokenize
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

from utils import seed_everything


parser = argparse.ArgumentParser()

root = "/data/mathieu/efficient_unified_crs/" # todo: change to your home directory

# general
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--cuda", type=bool, default=True)
parser.add_argument("--debug", type=bool, default=False)
parser.add_argument("--debug_size", type=int, default=20)
parser.add_argument("--dataset_name", type=str, default="INSPIRED", choices=["REDIAL", "INSPIRED"])
parser.add_argument("--root", type=str, default=root)
parser.add_argument('--model', type=str, default = "llama-2-7b-chat", choices=["llama-2-7b-chat", "vicuna-1.5-7b"])
parser.add_argument('--torch_dtype', default = torch.bfloat16)
parser.add_argument('--max_output_length', type=int, default = 128)

args = parser.parse_args()

dataset_names = ["REDIAL", "INSPIRED"]
index = dataset_names.index(args.dataset_name)
train_paths = [
    root+f"data/{args.dataset_name}/train_data_processed", # size: 10006 -> 65670
    root+f"data/{args.dataset_name}/train_data_processed" # size: 801 -> 8152
]
test_paths = [
    root+f"data/{args.dataset_name}/test_data_processed", # size: 1342 -> 8329
    #root+f"data/{args.dataset_name}/dev_data_processed", # size: 99 -> 977
    root+f"data/{args.dataset_name}/test_data_processed" # size: 99 -> 993
]

args.train_path = train_paths[index]
args.test_path = test_paths[index]
args.items_db_path = root+f"data/{args.dataset_name}/movie_db"
args.test_split = args.test_path.split("/")[-1].split("_")[0]


models = ["llama-2-7b-chat", "vicuna-1.5-7b"]
index = models.index(args.model)
model_names = ["meta-llama/Llama-2-7b-chat-hf",  "lmsys/vicuna-7b-v1.5"]
clean_model_names = ["llama_2_7b", "vicuna_7b_16k"]

args.model_name = model_names[index]
args.clean_model_name = clean_model_names[index]


def main(args):
    print(args)

    # seed
    seed_everything(args.seed)

    # items
    items_db = torch.load(args.items_db_path)
    args.items_db = items_db

    test_data = torch.load(args.test_path)

    contexts, responses, labels = [], [], []
    for i in tqdm(range(len(test_data))):
        current_context = ""
        dialogue = test_data[i][1]
        for j in range(len(dialogue)):
            (raw_utt, movie_ids) = dialogue[j]
            speaker = raw_utt.split()[0]
            if "B" in speaker:
                speaker = "Seeker"
            else:
                speaker = "Recommender"
            words = raw_utt.split()
            words[0] = speaker + ":"
            movie_count = 0
            for k in range(len(words)):
                if words[k] in ["[MOVIE_ID]", "\"[MOVIE_ID]\"", "[MOVIE_ID],", "[MOVIE_ID].", "[MOVIE_ID]?", "[MOVIE_ID]!", "\"[MOVIE_ID]\",", "\"[MOVIE_ID]\".", "\"[MOVIE_ID]\"?", "\"[MOVIE_ID]\"!"]:
                    movie_id = movie_ids[movie_count]
                    movie_desc = items_db[movie_id]
                    title = movie_desc.split("[SEP]")[0].strip()
                    words[k] = title
                    movie_count += 1
            utt = " ".join(words)
            
            if speaker == "Recommender" and movie_ids != None:
                for movie_id in movie_ids:
                    movie_desc = items_db[movie_id]
                    title = movie_desc.split("[SEP]")[0].strip()
                    contexts.append(current_context)
                    responses.append(utt)
                    labels.append(title)
            else:
                current_context += "\n" + utt
    print(len(contexts), len(responses), len(labels))

    if args.debug:
        contexts = contexts[:args.debug_size]
        responses = responses[:args.debug_size]
        labels = labels[:args.debug_size]
        print(len(contexts), len(responses), len(labels))

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=f"../hf_models/{args.model}",
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=args.torch_dtype,
        cache_dir=f"../hf_models/{args.model}"
    )
    n_params = sum(p.numel() for p in model.parameters())
    print("\nThe model has {} parameters".format(n_params))
    params = list(model.parameters())[0]
    print(params.dtype)

    generated = []
    for i in tqdm(range(len(contexts))):
        prompt = "Imagine you are an expert at recommending movies. The following is a conversation between a seeker, and yourelf, a movie recommender expert."
        prompt += " Predict the recommender response, and nothing else. Your response should include a movie name."
        prompt += f"\n\nDialogue:\n{contexts[i]}"
        prompt += "\nRecommender:"

        tok = tokenizer(prompt, return_tensors="pt")
        length = tok["input_ids"].shape[1]

        outputs = model.generate(
            input_ids=tok["input_ids"].cuda(),
            attention_mask=tok["attention_mask"].cuda(),
            num_beams=1,
            do_sample=False,
            temperature=0,
            min_new_tokens=4,
            max_new_tokens=args.max_output_length,
            return_dict_in_generate=True,
        )
        tokens = outputs["sequences"]
        response = tokenizer.batch_decode(tokens[:, length:], skip_special_tokens=True)[0]
        response = "Recommender: " + response
        generated.append(response)

    rec_1s, rec_ids, r1s, r2s, rls = [], [], [], [], []
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
    for i in range(len(generated)):
        rec = 0
        movie_id = labels[i]
        if movie_id.lower() in generated[i].lower():
            rec = 1
            rec_ids.append(movie_id)
        rec_1s.append(rec)

        response = "\n".join(sent_tokenize(responses[i]))
        gen = "\n".join(sent_tokenize(generated[i]))
        rouge_scores = scorer.score(response, gen)
        r1 = 100 * rouge_scores["rouge1"].fmeasure
        r2 = 100 * rouge_scores["rouge2"].fmeasure
        rl = 100 * rouge_scores["rougeLsum"].fmeasure
        r1s.append(r1)
        r2s.append(r2)
        rls.append(rl)
    rec_1s, rec_ids, r1s, r2s, rls = 100 * np.array(rec_1s), np.array(rec_ids), np.array(r1s), np.array(r2s), np.array(rls)
    print(f"\nRec@1: {np.mean(rec_1s):.4f} # Unique: {len(rec_ids)} || R-1: {np.mean(r1s):.2f}, R-2: {np.mean(r2s):.2f}, R-L: {np.mean(rls):.2f}")
    

if __name__ == '__main__':
    main(args)
