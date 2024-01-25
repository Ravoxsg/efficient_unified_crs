import argparse
import torch
import numpy as np
import tqdm
import pickle
import os
import logging
from accelerate.logging import get_logger
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast
from transformers import AdamW, get_linear_schedule_with_warmup
from accelerate import Accelerator
from peft import get_peft_model, LoraConfig

from utils import seed_everything, SequenceCrossEntropyLoss, count_parameters
from dataset import MovieRecDataset, MovieRecDataCollator
from model_utils import GPT2InductiveAttentionHeadModel
from model import PECRSModel
from engine import training_loop, validate


parser = argparse.ArgumentParser()

root = "/data/mathieu/efficient_unified_crs/" # todo: change to your home directory

# general
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--cuda", type=bool, default=True)
parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
parser.add_argument("--debug", type=bool, default=True)
parser.add_argument("--debug_size", type=int, default=10)
parser.add_argument("--max_val_size", type=int, default=10000)
parser.add_argument("--root", type=str, default=root)

# data
parser.add_argument("--dataset_name", type=str, default="REDIAL", choices=["REDIAL", "INSPIRED"])

# model
parser.add_argument("--decoder", type=str, default="../hf_models/gpt2-small")
parser.add_argument("--rec_token", type=str, default="[REC]")
parser.add_argument("--rec_end_token", type=str, default="[REC_END]")
parser.add_argument("--sep_token", type=str, default="[SEP]")
parser.add_argument("--placeholder_token", type=str, default="[MOVIE_ID]")
parser.add_argument("--lm_trim_offset", type=int, default=100, help="offset to trim language model wte inputs length = (1024-lm_trim_offset)")
### response generation
parser.add_argument("--context_max_length", type=int, default=256) # 256
parser.add_argument("--utt_max_length", type=int, default=64) # 64
parser.add_argument("--check_learned_weights", type=bool, default=True)
parser.add_argument("--freeze_backbone_for_items", type=bool, default=True)
parser.add_argument("--train_item_encoding_chunk_size", type=int, default=50)
parser.add_argument("--tie_recall_and_rerank", type=bool, default=True)
### parameter efficiency (LORA)
parser.add_argument("--only_tune_new_tokens", type=bool, default=False)
parser.add_argument("--n_lora_layers_to_tune", type=int, default=3)
parser.add_argument("--tune_lora_in_items_encoding", type=bool, default=False)
parser.add_argument("--lora_r", type=int, default=16, help="lora attention dimension")
parser.add_argument("--lora_alpha", type=int, default=16, help="lora alpha")
parser.add_argument("--lora_dropout", type=float, default=0.1, help="lora dropout rate")
parser.add_argument("--task_type", type=str, default="CAUSAL_LM", help="task type [SEQ_2_SEQ_LM | CAUSAL_LM]")

# optimization
### standard optimization
parser.add_argument("--num_epochs", type=int, default=20)
parser.add_argument("--train_bs", type=int, default=8)
parser.add_argument("--eval_bs", type=int, default=8)
parser.add_argument("--num_gradients_accumulation", type=int, default=1)
parser.add_argument("--num_samples_recall_train", type=int, default=150) # 150
parser.add_argument("--num_samples_rerank_train", type=int, default=150) # 150
parser.add_argument("--validation_recall_size", type=int, default=700) # Not expended:
parser.add_argument("--expanded_reranking", type=bool, default=False)
parser.add_argument("--validation_rerank_block_size", type=int, default=500)
parser.add_argument("--temperature", type=float, default=1.2)
parser.add_argument("--language_loss_train_coeff", type=float, default=0.15) # 0.15
parser.add_argument("--language_loss_train_coeff_beginning_turn", type=float, default=1.0) # 1.0
parser.add_argument("--recall_loss_train_coeff", type=float, default=0.8) # 0.8
parser.add_argument("--rerank_loss_train_coeff", type=float, default=1.0) # 1.0
parser.add_argument("--lr", type=float, default=3e-5, help="learning rate")
parser.add_argument("--ls", type=float, default=0.02, help="label smoothing")
parser.add_argument("--scheduler", type=str, default="linear")
### negative items sampling
parser.add_argument("--previous_recommended_ids_negative", type=bool, default=False)
parser.add_argument("--share_batch_negatives", type=bool, default=True)
parser.add_argument("--tie_sampled_ids_recall_rerank", type=bool, default=True)
### acceleration
parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp8", "fp16", "bf16"])
### evaluation
parser.add_argument("--validate", type=bool, default=True)
parser.add_argument("--epoch_0", type=bool, default=True)
parser.add_argument("--print_every", type=int, default=100)
parser.add_argument("--eval_every", type=int, default=45000)
parser.add_argument("--show_weights_diff", type=bool, default=False)
parser.add_argument("--generate", type=bool, default=True)
parser.add_argument("--find_examples", type=bool, default=False)

# generation
parser.add_argument("--generation_method", type=str, default="top_k_sampling", choices=["beam_search", "diverse_beam_search", "top_k_sampling"])
parser.add_argument("--num_beams", type=int, default=2)
### top-k sampling
parser.add_argument("--top_k", type=int, default=50)
parser.add_argument("--sampling_temperature", type=float, default=1.15)
### diverse beam search
parser.add_argument("--num_beam_groups", type=int, default=2)
parser.add_argument("--diversity_penalty", type=float, default=1.0)

# ablations
parser.add_argument("--only_title", type=bool, default=False)
parser.add_argument("--trim_metadata", type=bool, default=False)
parser.add_argument("--item_properties", type=list, default=[0,1,2,3,4]) # 0: title 1: actors 2: directors 3: genres 4: plot
parser.add_argument("--no_pooling", type=bool, default=False)
parser.add_argument("--no_item_head", type=bool, default=False)

# export
parser.add_argument("--save", type=bool, default=True)
parser.add_argument("--exp_name", type=str, default="temp")
# checkpoint (for args.mode == "eval" only)
parser.add_argument("--load_model_path", type=str, default="Outputs/REDIAL/temp/CRS_Train_8.pt")

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
remove_unused_items = [False, True]
args.train_path = train_paths[index]
args.test_path = test_paths[index]
args.items_db_path = root+f"data/{args.dataset_name}/movie_db"
args.test_split = args.test_path.split("/")[-1].split("_")[0]
args.remove_unused_items = remove_unused_items[index]


def main(args):
    # seed
    seed_everything(args.seed)

    # output dir
    if not os.path.exists(root+f"Outputs/{args.dataset_name}/{args.exp_name}/"):
        os.makedirs(root+f"Outputs/{args.dataset_name}/{args.exp_name}/CRS_Train_")
    args.model_saved_path = root+f"Outputs/{args.dataset_name}/{args.exp_name}/CRS_Train_"

    # accelerator
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        deepspeed_plugin=None,
        gradient_accumulation_steps=args.num_gradients_accumulation,
        device_placement=True,
        cpu=not args.cuda
    )

    # logger
    log_path = root+f"Outputs/{args.dataset_name}/{args.exp_name}/CRS_{args.exp_name}_{args.mode}.txt"
    open(log_path, 'w').close()
    logger = get_logger(__name__, log_level="INFO")
    file_handler = logging.FileHandler(log_path)
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)
    logger.logger.addHandler(file_handler)
    logging.basicConfig(
        format=log_format,
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(f"Logging everything to {log_path}")
    logger.info("*"*50)
    logger.info(args)
    logger.info("\n")

    # tokenizer and base models
    tokenizer = GPT2TokenizerFast.from_pretrained(args.decoder)
    dec_model = GPT2InductiveAttentionHeadModel.from_pretrained(args.decoder)
    lora_config = LoraConfig(
        peft_type="LORA",
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        task_type=args.task_type
    )
    dec_model = get_peft_model(dec_model, peft_config=lora_config)

    # extra tokens:
    args.n_original_tokens = len(tokenizer)
    tokenizer.add_tokens([args.rec_token, args.rec_end_token, args.sep_token, args.placeholder_token])
    gpt2_special_tokens_dict = {
        'pad_token': '<pad>'
    }
    tokenizer.add_special_tokens(gpt2_special_tokens_dict)
    dec_model.config.pad_token_id = tokenizer.pad_token_id
    dec_model.resize_token_embeddings(len(tokenizer))
    args.original_token_emb_size = dec_model.get_input_embeddings().weight.shape[0]
    logger.info(f"Final vocab size: {len(tokenizer)}")

    # metadata
    # items
    items_db = torch.load(args.items_db_path)
    logger.info(f"DB has {len(items_db)} items")
    # prune the items db
    if args.remove_unused_items:
        used_items = {}
        if args.dataset_name == "REDIAL":
            train_data = torch.load(args.train_path)
            test_data = torch.load(args.test_path)
            datas = [train_data, test_data]
        elif args.dataset_name == "INSPIRED":
            train_data = torch.load(args.train_path)
            dev_data = torch.load(root+f"data/{args.dataset_name}/dev_data_processed")
            test_data = torch.load(root+f"data/{args.dataset_name}/test_data_processed")
            datas = [train_data, dev_data, test_data]
        for data in datas:
            for i in tqdm.tqdm(range(len(data))):
                _, dialogue = data[i]
                for _, (_, gt_ind) in enumerate(dialogue):
                    if gt_ind is not None:
                        for item in gt_ind:
                            if not(item in used_items.keys()):
                                used_items[item] = 0
        unused_items = [x for x in items_db.keys() if not(x) in used_items.keys()]
        logger.inf(f"There are {len(unused_items)} unused items")
        for x in unused_items:
            del items_db[x]
        logger.info(f"Items DB now has size {len(items_db.keys())}")
    args.items_db = items_db
    # genres
    genres_db = {}
    for k in items_db.keys():
        metadata = items_db[k]
        genres = metadata.split(args.sep_token)[3]
        genres = genres.lower().replace(",", " ")
        if len(genres.split()) > 0:
            top_genre = genres.split()[0]
            genres_db[k] = top_genre
    args.genres_db = genres_db
    genres_freqs = {}
    unique_genres = np.unique(np.array(list(genres_db.values())))
    for unique_genre in unique_genres:
        count = len([x for x in genres_db.keys() if genres_db[x] == unique_genre])
        frac = 100 * count / len(genres_db.keys())
        s = f"Genre {unique_genre}"
        logger.info(f"{s:20s} has {count} movies or {frac:.4f}% of total movies")
        genres_freqs[unique_genre] = frac / 100
    logger.info("\n")
    args.genres_freqs = genres_freqs

    # items <-> pseudo tokens mapping
    item_ids_to_pseudo_tokens, pseudo_tokens_to_item_ids = {}, {}
    item_ids = list(args.items_db.keys())
    for i in range(len(item_ids)):
        item_id = item_ids[i]
        pseudo_token = len(tokenizer) + i
        item_ids_to_pseudo_tokens[item_id] = pseudo_token
        pseudo_tokens_to_item_ids[pseudo_token] = item_id
    args.item_ids_to_pseudo_tokens = item_ids_to_pseudo_tokens
    args.pseudo_tokens_to_item_ids = pseudo_tokens_to_item_ids

    # datasets and data loaders
    data_collator = MovieRecDataCollator(tokenizer=tokenizer)
    train_dataset = MovieRecDataset("train", torch.load(args.train_path), tokenizer, logger, args)
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=args.train_bs, collate_fn=data_collator)
    test_dataset = MovieRecDataset("test", torch.load(args.test_path), tokenizer, logger, args)
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=args.eval_bs, collate_fn=data_collator)

    # model
    model = PECRSModel(tokenizer, dec_model, accelerator, args)
    param_optimizer = list(model.language_model.named_parameters()) + \
                      list(model.recall_lm_query_mapper.named_parameters()) + \
                      list(model.recall_item_wte_mapper.named_parameters()) + \
                      list(model.rerank_item_wte_mapper.named_parameters()) + \
                      list(model.rerank_logits_mapper.named_parameters()) + \
                      list(model.item_head_l1.named_parameters()) + \
                      list(model.item_head_l2.named_parameters())
    param_optimizer += [("weights", model.weights)]

    # parameters
    count_parameters(logger, tokenizer, dec_model, model, param_optimizer, args)

    # load checkpoint
    if args.mode != "train":
        loaded = torch.load(args.load_model_path)
        accelerator.unwrap_model(model).load_state_dict(loaded)
        if accelerator.is_main_process:
            logger.info("\n")
            logger.info(f"Loaded model weights! {args.load_model_path}")
        if args.check_learned_weights:
            block_size = 32
            n_blocks = int(1024 / block_size)
            for n in range(n_blocks):
                mean_weight = accelerator.unwrap_model(model).state_dict()["weights"][(n * block_size):((n + 1) * block_size)].mean()
                logger.info(f"Block of features {n}, mean weight: {mean_weight:.4f}")

    # loss functions
    criterion_language = SequenceCrossEntropyLoss()
    criterion_recall = torch.nn.CrossEntropyLoss()
    criterion_rerank_train = torch.nn.CrossEntropyLoss()
    criterions = (criterion_language, criterion_recall, criterion_rerank_train)
    args.rerank_encoder_chunk_size = int(args.num_samples_rerank_train / 15)

    if args.mode == "train":
        # steps
        num_train_optimization_steps = len(train_dataset) * args.num_epochs // args.train_bs // args.num_gradients_accumulation
        if accelerator.is_main_process:
            logger.info("\n")
            logger.info(f"# Optimization steps: {num_train_optimization_steps}")

        # optimizer and scheduler
        no_decay = ['bias', 'ln', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-06)
        scheduler = None
        if args.scheduler == "linear":
            num_warmup_steps = len(train_dataset) // args.train_bs // args.num_gradients_accumulation
            logger.info(f"# Linear warmup steps: {num_warmup_steps}")
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_optimization_steps)

        train_dataloader, test_dataloader, model, optimizer, scheduler = accelerator.prepare(train_dataloader, test_dataloader, model, optimizer, scheduler)
        model = model.to(accelerator.device)

        logger.info("Single-GPU training...")
        training_loop(train_dataloader, test_dataloader, tokenizer, model, optimizer, scheduler, criterions, logger, accelerator, args)

    else:
        model, test_dataloader = accelerator.prepare(model, test_dataloader)
        model = model.to(accelerator.device)

        logger.info("Single-GPU inference...")
        logger.info(accelerator.device)
        validate(1, test_dataloader, tokenizer, model, criterions, logger, accelerator, args)


if __name__ == '__main__':
    main(args)
