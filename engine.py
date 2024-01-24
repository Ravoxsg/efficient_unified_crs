import numpy as np
import torch
import tqdm
import time
import copy
import gc
import re
# import moxing as mox
from sklearn.metrics import recall_score, precision_score, f1_score
from rouge_score import rouge_scorer
from evaluation import distinct_metrics
from analysis import further_analysis

# overall training loop, on the entire dataset
def training_loop(train_dataloader, test_dataloader, tokenizer, model, optimizer, scheduler, criterions, logger, accelerator, args):
    if args.validate and args.epoch_0:
        validate(0, test_dataloader, tokenizer, model, criterions, logger, accelerator, args)
        model.train()
    else:
        model.eval()
        accelerator.unwrap_model(model).annoy_base_constructor()
        model.train()

    ppls, all_loss_ppl, all_loss_recall, all_loss_rerank = [], [], [], []
    times, times_embeds, times_language, times_recall, times_rerank = [], [], [], [], []
    lm_layer_name = "language_model.transformer.h.11.mlp.c_fc.weight"
    if args.use_lora:
        lm_layer_name = "language_model.base_model.model.transformer.h.11.mlp.c_fc.weight"
    lm_weights = copy.deepcopy(accelerator.unwrap_model(model).state_dict()[lm_layer_name])
    for ep in range(1, args.num_epochs + 1):
        if args.previous_recommended_ids_negative:
            args.previous_count = []
        if args.continue_training:
            if ep < args.continue_training_epoch:
                for batch in tqdm.tqdm(train_dataloader, disable=not accelerator.is_main_process):
                    with accelerator.accumulate(model):
                        scheduler.step()
                continue
        # training round of the epoch
        logger.info("\n")
        logger.info(f"Training epoch {ep}...")
        model.train()
        update_count, optim_count = 0, 0
        for batch in tqdm.tqdm(train_dataloader, disable=not accelerator.is_main_process):
            with accelerator.accumulate(model):
                # batch size of train_dataloader is 1
                t1 = time.time()
                avg_ppl, loss_ppl, loss_recall, loss_rerank, batch_times = train_one_iteration(batch, tokenizer, model, criterions, accelerator, args)
                (time_embeds, time_language, time_recall, time_rerank) = batch_times
                avg_ppl = np.nan_to_num(avg_ppl)
                loss_ppl = np.nan_to_num(loss_ppl)
                loss_recall = np.nan_to_num(loss_recall)
                loss_rerank = np.nan_to_num(loss_rerank)
                ppls.append(avg_ppl)
                all_loss_ppl.append(loss_ppl)
                all_loss_recall.append(loss_recall)
                all_loss_rerank.append(loss_rerank)
                update_count += 1
                if args.only_tune_new_tokens:
                    accelerator.unwrap_model(model).language_model.transformer.wte.weight.grad[args.n_original_tokens] = 0
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
                t2 = time.time()
                times.append(t2 - t1)
                times_embeds.append(time_embeds)
                times_language.append(time_language)
                times_recall.append(time_recall)
                times_rerank.append(time_rerank)

                if (update_count % args.num_gradients_accumulation == args.num_gradients_accumulation - 1) or (update_count == len(train_dataloader)):
                    # update for gradient accumulation
                    optim_count += 1
                    lr = optimizer.param_groups[0]['lr']
                    if args.show_weights_diff:
                        lm_diff = (accelerator.unwrap_model(model).state_dict()[lm_layer_name] -
                                   lm_weights) / lm_weights
                        lm_diff = lm_diff.mean()
                        logger.info(f"LM weights diff: {lm_diff:.8f}")
                        lm_weights = copy.deepcopy(accelerator.unwrap_model(model).state_dict()[lm_layer_name])

                if (update_count % args.print_every == 0):
                    median_ppl = np.percentile(np.array(ppls), 50)
                    mean_ppl = np.mean(np.array(ppls))
                    mean_loss_ppl = np.mean(np.array(all_loss_ppl))
                    mean_loss_recall = np.mean(np.array(all_loss_recall))
                    mean_loss_rerank = np.mean(np.array(all_loss_rerank))
                    lr = optimizer.param_groups[0]['lr']
                    logger.info(f"Epoch {ep}, Batch {update_count}, # optim steps: {optim_count}, LR: {lr:.10f}")
                    logger.info(f"median ppl: {median_ppl:.4f}, mean ppl: {mean_ppl:.4f}, loss ppl: {mean_loss_ppl: .4f}, loss recall: {mean_loss_recall: .4f}, loss_rerank: {mean_loss_rerank: .4f}")
                ppls, all_loss_ppl, all_loss_recall, all_loss_rerank = [], [], [], []
                mean_time = np.nan_to_num(np.mean(np.array(times)))
                mean_time_embeds = np.nan_to_num(np.mean(np.array(times_embeds)))
                mean_time_language = np.nan_to_num(np.mean(np.array(times_language)))
                mean_time_recall = np.nan_to_num(np.mean(np.array(times_recall)))
                mean_time_rerank = np.nan_to_num(np.mean(np.array(times_rerank)))
                logger.info(f"Time/batch: {mean_time:.4f}, time for embeds: {mean_time_embeds:.4f}, LM: {mean_time_language: .4f}, recall: {mean_time_recall: .4f}, rerank: {mean_time_rerank: .4f}")
                times, times_embeds, times_language, times_recall, times_rerank = [], [], [], [], []

                if (update_count % args.eval_every == 0):
                    validate(ep, test_dataloader, tokenizer, model, criterions, logger, accelerator, args)
                    model.train()
                    if args.save:
                        save_path = args.model_saved_path + str(ep) + f"_{update_count}.pt"
                        state_dict = accelerator.unwrap_model(model).state_dict()
                        accelerator.save(state_dict, save_path)
                        logger.info(f"saved model! at {save_path}")

        if args.previous_recommended_ids_negative:
            previous_count = np.mean(args.previous_count)
            logger.info(f"Added {previous_count:.4f} hard negatives on average through previously mentioned movies")
            args.previous_count = []
            # validation round of the epoch
            if args.validate:
                validate(ep, test_dataloader, tokenizer, model, criterions, logger, accelerator, args)
            model.train()
            if args.save:
                save_path = args.model_saved_path + str(ep) + ".pt"
            state_dict = accelerator.unwrap_model(model).state_dict()
            accelerator.save(state_dict, save_path)
            logger.info(f"saved model! at {save_path}")

# training on 1 batch
def train_one_iteration(batch, tokenizer, model, criterions, accelerator, args):
    (criterion_language, criterion_recall, criterion_rerank_train) = criterions
    ppl_history = []
    all_loss_ppl, all_loss_recall, all_loss_rerank = [], [], []

    no_rec_idx = [i for i in range(len(batch["targets"])) if batch["targets"][i] == -1]
    has_rec_idx = [i for i in range(len(batch["targets"])) if batch["targets"][i] != -1]

    times_embeds, times_language, times_recall, times_rerank = [], [], [], []

    t0 = time.time()
    embeds = []
    for i in range(batch["context_with_utterances"].shape[0]):
        embeds_i = []
        for j in range(batch["context_with_utterances"].shape[1]):
            if batch["context_with_utterances"][i,j].item() < len(tokenizer):
                embeds_i_j = accelerator.unwrap_model(model).language_model.transformer.wte(batch[
                    "context_with_utterances"][i,j])
            else:
                item_id = args.pseudo_tokens_to_item_ids[batch["context_with_utterances"][i,j].item()]
                embeds_i_j = accelerator.unwrap_model(model).compute_encoded_embeddings_for_items([
                    item_id])[0]
                embeds_i_j = accelerator.unwrap_model(model).rerank_item_wte_mapper(embeds_i_j)
            embeds_i.append(embeds_i_j)
        embeds.append(embeds_i)
    embeds_no_rec = [embeds[x] for x in no_rec_idx]
    embeds_has_rec = [embeds[x] for x in has_rec_idx]
    t1 = time.time()
    times_embeds.append(t1 - t0)

    # data points without recommendation (just response generation aka language modeling)
    if len(no_rec_idx) > 0:
        with accelerator.autocast():
            t0 = time.time()
            language_targets = batch["context_with_utterances"][no_rec_idx][:, 1:].contiguous()
            language_targets[language_targets >= len(tokenizer)] = 0
            language_logits = accelerator.unwrap_model(model).forward_pure_language_turn(embeds_no_rec)
            language_targets_mask = torch.zeros_like(language_targets).float()
            for i in range(batch["context_with_utterances"][no_rec_idx].shape[0]):
                context_length = batch["context_lengths"][no_rec_idx][i]
                utterance_length = batch["utterance_lengths"][no_rec_idx][i]
                language_targets_mask[i, context_length:(context_length + utterance_length - 1)] = 1
            loss_ppl = criterion_language(language_logits, language_targets, language_targets_mask, label_smoothing=args.label_smoothing, reduce='batch')
            perplexity = np.exp(min(300, torch.nan_to_num(loss_ppl).item()))
            ppl_history.append(perplexity)
            all_loss_ppl.append(loss_ppl.item())
            loss_ppl = args.language_loss_train_coeff * loss_ppl
            accelerator.backward(loss_ppl)

            del loss_ppl;
            del language_logits;
            del language_targets
            gc.collect()
            t1 = time.time()
            times_language.append(t1 - t0)

    # data points with recommended items
    if len(has_rec_idx) > 0:
        with accelerator.autocast():
            t0 = time.time()
            # recall
            previous_ids = None
            if args.previous_recommended_ids_negative:
                previous_ids = [batch["previous_recommended_ids"][x] for x in has_rec_idx]
            recall_logits, recall_true_index, language_logits, language_targets, encoded_items_embeddings = accelerator.unwrap_model(model).forward_recall(
                batch["indices"][has_rec_idx],
                batch["context_with_utterances"][has_rec_idx],
                embeds_has_rec,
                batch["context_lengths"][has_rec_idx],
                batch["targets"][has_rec_idx],
                args.num_samples_recall_train,
                previous_recommended_ids=previous_ids,
            )
            # recall items loss
            recall_targets = torch.LongTensor(recall_true_index).to(accelerator.device)
            loss_recall = criterion_recall(recall_logits, recall_targets)
            all_loss_recall.append(loss_recall.item())
            loss_recall = args.recall_loss_train_coeff * loss_recall
            # language loss in recall turn, REC_TOKEN, Language on conditional generation
            language_targets_mask = torch.zeros_like(language_targets).float()
            for i in range(batch["context_with_utterances"][has_rec_idx].shape[0]):
                context_length = batch["context_lengths"][has_rec_idx][i]
                utterance_length = batch["utterance_lengths"][has_rec_idx][i]
                language_targets_mask[i, (context_length - 1):(context_length - 1 + utterance_length)] = 1
            language_targets[language_targets >= len(tokenizer)] = 0
            loss_ppl = criterion_language(language_logits, language_targets, language_targets_mask, label_smoothing=args.ls, reduce="batch")
            perplexity = np.exp(min(300, torch.nan_to_num(loss_ppl).item()))
            ppl_history.append(perplexity)
            all_loss_ppl.append(loss_ppl.item())
            loss_ppl = args.language_loss_train_coeff * loss_ppl

            # combined loss
            recall_total_loss = loss_recall + loss_ppl
            if not (args.tied_sample_ids_recall_rerank):
                accelerator.backward(recall_total_loss)

            del loss_ppl
            del language_logits
            del language_targets
            del loss_recall
            del recall_logits
            del recall_targets
            gc.collect()
            t1 = time.time()
            times_recall.append(t1 - t0)

            # rerank
            t0 = time.time()
            encoded_items_transfer = None
            if args.tie_sampled_ids_recall_rerank:
                encoded_items_transfer = encoded_items_embeddings
            rerank_logits, rerank_true_index = accelerator.unwrap_model(model).forward_rerank(
                batch["indices"][has_rec_idx],
                batch["contexts"][has_rec_idx],
                batch["context_lengths"][has_rec_idx],
                batch["targets"][has_rec_idx],
                args.num_samples_rerank_train,
                encoded_items_embeddings=encoded_items_transfer,
                previous_recommended_ids=None,
            )
            rerank_logits /= args.temperature

            # rerank loss
            rerank_targets = torch.LongTensor(rerank_true_index).to(accelerator.device)
            loss_rerank = criterion_rerank_train(rerank_logits, rerank_targets)
            all_loss_rerank.append(loss_rerank.item())
            loss_rerank = args.rerank_loss_train_coeff * loss_rerank
            if args.tie_sampled_ids_recall_rerank:
                accelerator.backward(recall_total_loss + loss_rerank)
            else:
                accelerator.backward(loss_rerank)

            del loss_rerank
            del rerank_logits
            del rerank_targets
            gc.collect()
            t1 = time.time()
            times_rerank.append(t1 - t0)

    time_embeds = np.mean(np.array(times_embeds))
    time_language = np.mean(np.array(times_language))
    time_recall = np.mean(np.array(times_recall))
    time_rerank = np.mean(np.array(times_rerank))
    times = (time_embeds, time_language, time_recall, time_rerank)

    return np.mean(ppl_history), np.mean(all_loss_ppl), np.mean(all_loss_recall), np.mean(all_loss_rerank), times


# validation on the entire dataset
def validate(ep, dataloader, tokenizer, model, criterions, logger, accelerator, args):
    logger.info("\n")
    logger.info("Validating...")
    model.eval()
    accelerator.unwrap_model(model).annoy_base_constructor()

    # collect all predictions
    cold_start_tags, turn_nums, n_points, n_rec = [], [], 0, 0 # metadata
    ppl_losses, ppls = [], [] # response
    recall_losses, n_recall_success, recall_top100, recall_top300, recall_top500 = [], 0, [], [], [] # recall
    rerank_losses, total_rerank_top1, rerank_top1, rerank_top10, rerank_top50 = [], [], [], [], [] # re-ranking
    gt_ids, gt_ranks, total_predicted_ids, all_predicted_ids = [], [], [], [] # final recommendation
    for batch in tqdm.tqdm(dataloader, disable=not accelerator.is_main_process):
        metadata, response, recall, rerank, recommendation = validate_one_iteration(batch, tokenizer, model, criterions, accelerator, args)
        (cold_start_tags_batch, turn_nums_batch, n_points_batch, n_rec_batch) = metadata
        (ppl_losses_batch, ppls_batch) = response
        (recall_losses_batch, n_recall_success_batch, recall_top100_batch, recall_top300_batch, recall_top500_batch) = recall
        (rerank_losses_batch, total_rerank_top1_batch, rerank_top1_batch, rerank_top10_batch, rerank_top50_batch) = rerank
        (gt_ids_batch, gt_ranks_batch, total_predicted_ids_batch) = recommendation

        cold_start_tags += cold_start_tags_batch
        turn_nums += turn_nums_batch
        n_points += n_points_batch
        n_rec += n_rec_batch

        ppl_losses += ppl_losses_batch
        ppls += ppls_batch

        recall_losses += recall_losses_batch
        n_recall_success += n_recall_success_batch
        recall_top100 += recall_top100_batch
        recall_top300 += recall_top300_batch
        recall_top500 += recall_top500_batch

        rerank_losses += rerank_losses_batch
        total_rerank_top1 += total_rerank_top1_batch
        rerank_top1 += rerank_top1_batch
        rerank_top10 += rerank_top10_batch
        rerank_top50 += rerank_top50_batch

        gt_ids += gt_ids_batch
        gt_ranks += gt_ranks_batch
        total_predicted_ids += total_predicted_ids_batch
        all_predicted_ids.append(total_predicted_ids_batch)

    cold_start_tags, turn_nums = np.array(cold_start_tags), np.array(turn_nums)
    ppl_losses, ppls = np.array(ppl_losses), np.array(ppls)
    gt_ids, total_predicted_ids = np.array(gt_ids), np.array(total_predicted_ids)
    recall_losses, recall_top100, recall_top300, recall_top500 = np.array(recall_losses), np.array(recall_top100), np.array(recall_top300), np.array(recall_top500)
    rerank_losses, rerank_top1, rerank_top10, rerank_top50 = np.array(rerank_losses), np.array(rerank_top1), np.array(rerank_top10), np.array(rerank_top50)

    logger.info(f"# Data points: {n_points}, # with rec: {n_rec}, # recall successful: {n_recall_success}")
    logger.info(f"Epoch {ep}, ppl loss: {np.mean(ppl_losses):.4f}, recall loss: {np.mean(recall_losses):.4f}, rerank loss: {np.mean(rerank_losses):.4f}")
    logger.info(f"ppl: {np.mean(ppls):.4f}, min {np.min(ppls):.4f} 10%: {np.percentile(ppls, 10):.4f}, mean: {np.mean(ppls):.4f}, 90 %: {np.percentile(ppls, 90):.4f}, 99 %: {np.percentile(ppls, 99):.4f}, ppl max: {np.max(ppls):.4f}")

    if args.generate:
        batch_count, keep_ids, sources = 0, [], []
        gt_rec, raw_gt_sens, gt_sens, gt_n_tokens = [], [], [], []
        pred_rec, gen_sens, tok_gen_sens, gen_n_tokens = [], [], [], []
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        r1s, r2s, rLs = [], [], []
        for batch in tqdm.tqdm(dataloader, disable=not accelerator.is_main_process):
            keep_ids_batch = [0] * len(batch["repeated"])
            for j in range(len(batch["repeated"])):
                if batch["repeated"][j] == 0:
                    keep_ids_batch[j] = 1
            keep_ids += keep_ids_batch

            sources_batch, ground_truths, predicted = validate_language_metrics_batch_embeds(tokenizer, batch, model, accelerator, all_predicted_ids[batch_count], args)
            (gt_rec_batch, raw_gt_sens_batch, gt_sens_batch) = ground_truths
            (pred_rec_batch, gen_sens_batch, tok_gen_sens_batch) = predicted

            sources += sources_batch
            for j in range(len(gt_sens_batch)):
                gt_n_tokens.append(len(tokenizer(gt_sens_batch[j], return_tensors='pt')['input_ids'][0]))
                gen_n_tokens.append(len(tokenizer(gen_sens_batch[j], return_tensors='pt')['input_ids'][0]))

            rouge_scores = scorer.score(gt_sens_batch[j], gen_sens_batch[j])
            r1s.append(100 * rouge_scores['rouge1'].fmeasure)
            r2s.append(100 * rouge_scores['rouge2'].fmeasure)
            rLs.append(100 * rouge_scores['rougeL'].fmeasure)

            gt_rec += gt_rec_batch
            pred_rec += pred_rec_batch
            raw_gt_sens += raw_gt_sens_batch
            gt_sens += gt_sens_batch
            gen_sens += gen_sens_batch
            tok_gen_sens += tok_gen_sens_batch
            batch_count += 1

        logger.info(f">>>>>>>>>>> Generation:")
        logger.info(f"Generated {len(r1s)} sentences, including {sum(gt_rec)} with a required recommendation")
        logger.info(f">>>>>>>>>>> Generation metrics:")
        gt_n_tokens = np.array(gt_n_tokens)
        gen_n_tokens = np.array(gen_n_tokens)
        logger.info(f"# Tokens (GT): {np.mean(gt_n_tokens):.4f}, # Tokens (predicted): {np.mean(gen_n_tokens):.4f}")
        gt_rec = np.array(gt_rec)
        pred_rec = np.array(pred_rec)
        r = recall_score(gt_rec, pred_rec)
        p = precision_score(gt_rec, pred_rec)
        f1 = f1_score(gt_rec, pred_rec)
        logger.info(f"Prediction of recommendation: recall: {r:.4f}, precision: {p:.4f}, F-1: {f1:.4f} (GT count: {np.sum(gt_rec)} / Pred count: {np.sum(pred_rec)})")
        dist1, dist2, dist3, dist4 = distinct_metrics(tok_gen_sens)
        logger.info(f"Dist1: {dist1:.4f}, Dist2: {dist2:.4f}, Dist3: {dist3:.4f}, Dist4: {dist4:.4f}")
        r1, r2, rl = np.mean(r1s), np.mean(r2s), np.mean(rLs)
        logger.info(f"ROUGE-1: {r1:.4f}, ROUGE-2: {r2:.4f}, ROUGE-L: {rl:.4f}")

        if args.find_examples:
            print("\nTrying to find a good example...")
            print(len(total_rerank_top1), len(keep_ids), len(sources), len(pred_rec), len(raw_gt_sens), len(gt_sens), len(gen_sens))
            gen_idx = 0
            for j in range(len(keep_ids)):
                if keep_ids[j] == 1:
                    if (total_rerank_top1[j] == 1) and (pred_rec[gen_idx] == 1):
                        print("*" * 50)
                        print("Source:")
                        clean_source = re.sub(r'\n+', '\n', sources[gen_idx])
                        print(clean_source)
                        print("Predicted response:")
                        print(gen_sens[gen_idx])
                        print("Predicted movie:")
                        movie_name = args.items_db[total_predicted_ids[j]].split("[SEP]")[0]
                        print(f"{total_predicted_ids[j]} {movie_name}")
                        print("Ground truth response (with movie names):")
                        print(raw_gt_sens[gen_idx])
                        print("Ground truth response:")
                        print(gt_sens[gen_idx])
                    gen_idx += 1
    logger.info(f">>>>>>>>>>> Recommendation metrics:")
    gt_ids_unique = len(np.unique(gt_ids))
    predicted_ids_unique = len(np.unique(total_predicted_ids)) - 1
    logger.info(f"Unique (GT): {gt_ids_unique}, Unique (predicted): {predicted_ids_unique}")
    recall_ratio = 100 * n_recall_success / n_rec
    logger.info(f"Recall is successful (gt_id is in recommended ids): {recall_ratio:.4f}")
    rc100 = 100 * np.mean(recall_top100)
    rc300 = 100 * np.mean(recall_top300)
    rc500 = 100 * np.mean(recall_top500)
    mean_rc = (rc100 + rc300 + rc500) / 3
    logger.info(f"mean recall (%): {mean_rc:.4f}, recall top100 (%): {rc100:.4f}, top300 (%): {rc300:.4f}, top500( %): {rc500: .4f}")
    rr1 = 100 * np.mean(rerank_top1)
    rr10 = 100 * np.mean(rerank_top10)
    rr50 = 100 * np.mean(rerank_top50)
    mean_rr = (rr1 + rr10 + rr50) / 3
    logger.info(f"mean rerank (%): {mean_rr:.4f}, rerank top1 (%): {rr1:.4f}, top10 (%): {rr10:.4f}, top50( %): {rr50:.4f}")
    logger.info(f'\n')

    # Further analysis (optional)
    recalls = (recall_top100, recall_top300, recall_top500)
    reranks = (rerank_top1, rerank_top10, rerank_top50)
    further_analysis(cold_start_tags, turn_nums, gt_ids, total_predicted_ids, gt_ranks, recalls, reranks, logger, args)

    model.train()

# validate on just 1 batch -> perplexity + recommendation part
def validate_one_iteration(batch, tokenizer, model, criterions, accelerator, args):
    (criterion_language, criterion_recall, criterion_rerank) = criterions

    # split data points in no rec / rec
    no_rec_idx = [i for i in range(len(batch["targets"])) if batch["targets"][i] == -1]
    has_rec_idx = [i for i in range(len(batch["targets"])) if batch["targets"][i] != -1]

    # metrics to track
    cold_start_tags, turn_nums, n_points, n_rec = [], [], 0, 0 # metadata
    ppl_losses, ppls = [], [] # response
    recall_losses, n_recall_success, recall_top100, recall_top300, recall_top500 = [], 0, [], [], [] # recall
    rerank_losses, total_rerank_top1, rerank_top1, rerank_top10, rerank_top50 = [], [] * (len(no_rec_idx) + len(has_rec_idx)), [], [], []  # re-ranking
    gt_ids, gt_ranks, total_predicted_ids = [], [], [-1] * (len(no_rec_idx) + len(has_rec_idx))  # recommendation

    embeds = []
    for i in range(batch["context_with_utterances"].shape[0]):
        embeds_context_i, embeds_utterance_i = [], []
        for j in range(batch["context_with_utterances"].shape[1]):
            if batch["context_with_utterances"][i, j].item() < len(tokenizer):
                embeds_i_j = accelerator.unwrap_model(model).language_model.transformer.wte(batch["context_with_utterances"][i, j])
            else:
                item_id = args.pseudo_tokens_to_item_ids[batch["context_with_utterances"][i, j].item()]
                embeds_i_j = accelerator.unwrap_model(model).compute_encoded_embeddings_for_items([item_id])[0]
                embeds_i_j = accelerator.unwrap_model(model).rerank_item_wte_mapper(embeds_i_j)
            if j < batch["context_lengths"][i]:
                embeds_context_i.append(embeds_i_j.unsqueeze(0))
            else:
                embeds_utterance_i.append(embeds_i_j.unsqueeze(0))
        embeds_context_i = torch.cat(embeds_context_i)
        embeds_utterance_i = torch.cat(embeds_utterance_i)
        embeds.append((embeds_context_i, embeds_utterance_i))
    embeds_no_rec = [embeds[x] for x in no_rec_idx]
    embeds_has_rec = [embeds[x] for x in has_rec_idx]

