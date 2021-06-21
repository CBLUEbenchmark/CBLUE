import os
import json
import numpy as np
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader

from cblue.utils import seed_everything, ProgressBar, TokenRematch
from cblue.metrics import sts_metric, qic_metric, qqr_metric, qtr_metric, \
    ctc_metric, ee_metric, er_metric, re_metric, cdn_cls_metric, cdn_num_metric
from cblue.metrics import sts_commit_prediction, qic_commit_prediction, qtr_commit_prediction, \
    qqr_commit_prediction, ctc_commit_prediction, ee_commit_prediction, cdn_commit_prediction
from cblue.models import convert_examples_to_features, save_zen_model


class Trainer(object):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            model_class,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):

        self.args = args
        self.model = model
        self.data_processor = data_processor
        self.tokenizer = tokenizer

        if train_dataset is not None and isinstance(train_dataset, Dataset):
            self.train_dataset = train_dataset

        if eval_dataset is not None and isinstance(eval_dataset, Dataset):
            self.eval_dataset = eval_dataset

        self.logger = logger
        self.model_class = model_class
        self.ngram_dict = ngram_dict

    def train(self):
        args = self.args
        logger = self.logger
        model = self.model
        model.to(args.device)

        train_dataloader = self.get_train_dataloader()

        num_training_steps = len(train_dataloader) * args.epochs
        num_warmup_steps = num_training_steps * args.warmup_proportion
        num_examples = len(train_dataloader.dataset)

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_training_steps)
        
        if args.task_name in ['qic', 'qqr', 'qtr', 'sts']:
            seed_everything(args.seed)
            model.zero_grad()

        logger.info("***** Running training *****")
        logger.info("Num samples %d", num_examples)
        logger.info("Num epochs %d", args.epochs)
        logger.info("Num training steps %d", num_training_steps)
        logger.info("Num warmup steps %d", num_warmup_steps)

        global_step = 0
        best_step = None
        best_score = .0
        cnt_patience = 0
        for i in range(args.epochs):
            pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
            for step, item in enumerate(train_dataloader):
                loss = self.training_step(model, item)
                pbar(step, {'loss': loss.item()})

                if args.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                if args.task_name in ['qic', 'qqr', 'qtr', 'sts']:
                    model.zero_grad()
                else:
                    optimizer.zero_grad()

                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    print("")
                    score = self.evaluate(model)
                    if score > best_score:
                        best_score = score
                        best_step = global_step
                        cnt_patience = 0
                        self._save_checkpoint(model, global_step)
                    else:
                        cnt_patience += 1
                        self.logger.info("Earlystopper counter: %s out of %s", cnt_patience, args.earlystop_patience)
                        if cnt_patience >= self.args.earlystop_patience:
                            break
            if cnt_patience >= args.earlystop_patience:
                break

        logger.info("Training Stop! The best step %s: %s", best_step, best_score)
        if args.device == 'cuda':
            torch.cuda.empty_cache()

        self._save_best_checkpoint(best_step=best_step)

        return global_step, best_step

    def evaluate(self, model):
        raise NotImplementedError

    def _save_checkpoint(self, model, step):
        raise NotImplementedError

    def _save_best_checkpoint(self, best_step):
        raise NotImplementedError

    def training_step(self, model, item):
        raise NotImplementedError

    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True
        )

    def get_eval_dataloader(self):
        return DataLoader(
            self.eval_dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False
        )

    def get_test_dataloader(self, test_dataset, batch_size=None):
        if not batch_size:
            batch_size = self.args.eval_batch_size

        return DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )


class EETrainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            model_class,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(EETrainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
            model_class=model_class,
            ngram_dict=ngram_dict
        )

    def training_step(self, model, item):
        model.train()

        input_ids = item[0].to(self.args.device)
        token_type_ids = item[1].to(self.args.device)
        attention_mask = item[2].to(self.args.device)
        labels = item[3].to(self.args.device)

        if self.args.model_type == 'zen':
            input_ngram_ids = item[4].to(self.args.device)
            ngram_attention_mask = item[5].to(self.args.device)
            ngram_token_type_ids = item[6].to(self.args.device)
            ngram_position_matrix = item[7].to(self.args.device)

        if self.args.model_type == 'zen':
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            labels=labels, ngram_ids=input_ngram_ids, ngram_positions=ngram_position_matrix,
                            ngram_attention_mask=ngram_attention_mask, ngram_token_type_ids=ngram_token_type_ids)
        else:
            outputs = model(labels=labels, input_ids=input_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask)

        loss = outputs[0]
        loss.backward()

        return loss.detach()

    def evaluate(self, model):
        args = self.args
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)

        preds = None
        eval_labels = None

        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        for step, item in enumerate(eval_dataloader):
            model.eval()

            input_ids = item[0].to(self.args.device)
            token_type_ids = item[1].to(self.args.device)
            attention_mask = item[2].to(self.args.device)
            labels = item[3].to(self.args.device)

            if args.model_type == 'zen':
                input_ngram_ids = item[4].to(self.args.device)
                ngram_attention_mask = item[5].to(self.args.device)
                ngram_token_type_ids = item[6].to(self.args.device)
                ngram_position_matrix = item[7].to(self.args.device)

            with torch.no_grad():
                if self.args.model_type == 'zen':
                    outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                    labels=labels, ngram_ids=input_ngram_ids,
                                    ngram_positions=ngram_position_matrix,
                                    ngram_token_type_ids=ngram_token_type_ids,
                                    ngram_attention_mask=ngram_attention_mask)
                else:
                    outputs = model(labels=labels, input_ids=input_ids, token_type_ids=token_type_ids,
                                    attention_mask=attention_mask)

                # outputs = model(labels=labels, **inputs)
                loss, logits = outputs[:2]
                # active_index = inputs['attention_mask'].view(-1) == 1
                active_index = attention_mask.view(-1) == 1
                active_labels = labels.view(-1)[active_index]
                logits = logits.argmax(dim=-1)
                active_logits = logits.view(-1)[active_index]

            if preds is None:
                preds = active_logits.detach().cpu().numpy()
                eval_labels = active_labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, active_logits.detach().cpu().numpy(), axis=0)
                eval_labels = np.append(eval_labels, active_labels.detach().cpu().numpy(), axis=0)

        p, r, f1, _ = ee_metric(preds, eval_labels)
        logger.info("%s-%s precision: %s - recall: %s - f1 score: %s", args.task_name, args.model_name, p, r, f1)
        return f1

    def predict(self, model, test_dataset):
        args = self.args
        logger = self.logger
        test_dataloader = self.get_test_dataloader(test_dataset)
        num_examples = len(test_dataloader.dataset)
        model.to(args.device)

        predictions = []

        logger.info("***** Running prediction *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(test_dataloader), desc='Prediction')
        for step, item in enumerate(test_dataloader):
            model.eval()

            input_ids = item[0].to(self.args.device)
            token_type_ids = item[1].to(self.args.device)
            attention_mask = item[2].to(self.args.device)

            if args.model_type == 'zen':
                input_ngram_ids = item[3].to(self.args.device)
                ngram_attention_mask = item[4].to(self.args.device)
                ngram_token_type_ids = item[5].to(self.args.device)
                ngram_position_matrix = item[6].to(self.args.device)

            with torch.no_grad():
                if self.args.model_type == 'zen':
                    outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                    ngram_ids=input_ngram_ids,
                                    ngram_positions=ngram_position_matrix,
                                    ngram_token_type_ids=ngram_token_type_ids,
                                    ngram_attention_mask=ngram_attention_mask)
                else:
                    outputs = model(input_ids=input_ids, token_type_ids=token_type_ids,
                                    attention_mask=attention_mask)

                if args.model_type == 'zen':
                    logits = outputs.detach()
                else:
                    logits = outputs[0].detach()
                # active_index = (inputs['attention_mask'] == 1).cpu()
                active_index = attention_mask == 1
                preds = logits.argmax(dim=-1).cpu()

                for i in range(len(active_index)):
                    predictions.append(preds[i][active_index[i]].tolist())
            pbar(step=step, info="")

        # test_inputs = [list(text) for text in test_dataset.texts]
        test_inputs = test_dataset.texts
        predictions = [pred[1:-1] for pred in predictions]
        predicts = self.data_processor.extract_result(predictions, test_inputs)
        ee_commit_prediction(dataset=test_dataset, preds=predicts, output_dir=args.result_output_dir)

    def _save_checkpoint(self, model, step):
        output_dir = os.path.join(self.args.output_dir, 'checkpoint-{}'.format(step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if self.args.model_type == 'zen':
            save_zen_model(output_dir, model=model, tokenizer=self.tokenizer,
                           ngram_dict=self.ngram_dict, args=self.args)
        else:
            model.save_pretrained(output_dir)
            torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))
            self.tokenizer.save_vocabulary(save_directory=output_dir)
        self.logger.info('Saving models checkpoint to %s', output_dir)

    def _save_best_checkpoint(self, best_step):
        model = self.model_class.from_pretrained(os.path.join(self.args.output_dir, f'checkpoint-{best_step}'),
                                                 num_labels=self.data_processor.num_labels)
        if self.args.model_type == 'zen':
            save_zen_model(self.args.output_dir, model=model, tokenizer=self.tokenizer,
                           ngram_dict=self.ngram_dict, args=self.args)
        else:
            model.save_pretrained(self.args.output_dir)
            torch.save(self.args, os.path.join(self.args.output_dir, 'training_args.bin'))
            self.tokenizer.save_vocabulary(save_directory=self.args.output_dir)
        self.logger.info('Saving models checkpoint to %s', self.args.output_dir)


class STSTrainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            model_class,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(STSTrainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
            model_class=model_class,
            ngram_dict=ngram_dict
        )

    def training_step(self, model, item):
        model.train()

        text1 = item[0]
        text2 = item[1]
        labels = item[2].to(self.args.device)

        if self.args.model_type == 'zen':
            inputs = convert_examples_to_features(text1=text1, text2=text2, ngram_dict=self.ngram_dict,
                                                  tokenizer=self.tokenizer, max_seq_length=self.args.max_length,
                                                  return_tensors=True)
        else:
            inputs = self.tokenizer(text1, text2, return_tensors='pt', padding='max_length',
                                    truncation='longest_first', max_length=self.args.max_length)

        inputs['input_ids'] = inputs['input_ids'].to(self.args.device)
        inputs['attention_mask'] = inputs['attention_mask'].to(self.args.device)
        inputs['token_type_ids'] = inputs['token_type_ids'].to(self.args.device)

        if self.args.model_type == 'zen':
            inputs['input_ngram_ids'] = inputs['input_ngram_ids'].to(self.args.device)
            inputs['ngram_position_matrix'] = inputs['ngram_position_matrix'].to(self.args.device)
            inputs['ngram_attention_mask'] = inputs['ngram_attention_mask'].to(self.args.device)
            inputs['ngram_token_type_ids'] = inputs['ngram_token_type_ids'].to(self.args.device)

        # default using 'Transformers' library models.
        outputs = model(labels=labels, **inputs)
        loss = outputs[0]
        loss.backward()

        return loss.detach()

    def evaluate(self, model):
        args = self.args
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)

        preds = None
        eval_labels = None

        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        for step, item in enumerate(eval_dataloader):
            model.eval()

            text1 = item[0]
            text2 = item[1]
            labels = item[2].to(args.device)

            if self.args.model_type == 'zen':
                inputs = convert_examples_to_features(text1=text1, text2=text2, ngram_dict=self.ngram_dict,
                                                      tokenizer=self.tokenizer, max_seq_length=self.args.max_length,
                                                      return_tensors=True)
            else:
                inputs = self.tokenizer(text1, text2, return_tensors='pt', padding='max_length',
                                        truncation='longest_first', max_length=self.args.max_length)
            inputs['input_ids'] = inputs['input_ids'].to(self.args.device)
            inputs['attention_mask'] = inputs['attention_mask'].to(self.args.device)
            inputs['token_type_ids'] = inputs['token_type_ids'].to(self.args.device)

            if self.args.model_type == 'zen':
                inputs['input_ngram_ids'] = inputs['input_ngram_ids'].to(self.args.device)
                inputs['ngram_position_matrix'] = inputs['ngram_position_matrix'].to(self.args.device)
                inputs['ngram_attention_mask'] = inputs['ngram_attention_mask'].to(self.args.device)
                inputs['ngram_token_type_ids'] = inputs['ngram_token_type_ids'].to(self.args.device)

            with torch.no_grad():
                outputs = model(labels=labels, **inputs)
                loss, logits = outputs[:2]

            if preds is None:
                preds = logits.detach().cpu().numpy()
                eval_labels = labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                eval_labels = np.append(eval_labels, labels.detach().cpu().numpy(), axis=0)

        preds = np.argmax(preds, axis=1)
        p, r, f1, _ = sts_metric(preds, eval_labels)
        logger.info("%s-%s precision: %s - recall: %s - f1 score: %s", args.task_name, args.model_name, p, r, f1)
        return f1

    def _save_checkpoint(self, model, step):
        output_dir = os.path.join(self.args.output_dir, 'checkpoint-{}'.format(step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if self.args.model_type == 'zen':
            save_zen_model(output_dir, model=model, tokenizer=self.tokenizer,
                           ngram_dict=self.ngram_dict, args=self.args)
        else:
            model.save_pretrained(output_dir)
            torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))
            self.tokenizer.save_vocabulary(save_directory=output_dir)
        self.logger.info('Saving models checkpoint to %s', output_dir)

    def predict(self, test_dataset, model):
        args = self.args
        logger = self.logger
        test_dataloader = self.get_test_dataloader(test_dataset)
        num_examples = len(test_dataloader.dataset)
        model.to(args.device)

        preds = None

        logger.info("***** Running prediction *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(test_dataloader), desc='Prediction')
        for step, item in enumerate(test_dataloader):
            model.eval()

            text1 = item[0]
            text2 = item[1]

            if self.args.model_type == 'zen':
                inputs = convert_examples_to_features(text1=text1, text2=text2, ngram_dict=self.ngram_dict,
                                                      tokenizer=self.tokenizer, max_seq_length=self.args.max_length,
                                                      return_tensors=True)
            else:
                inputs = self.tokenizer(text1, text2, return_tensors='pt', padding='max_length',
                                        truncation='longest_first', max_length=self.args.max_length)

            if self.args.model_type == 'zen':
                inputs['input_ngram_ids'] = inputs['input_ngram_ids'].to(self.args.device)
                inputs['ngram_position_matrix'] = inputs['ngram_position_matrix'].to(self.args.device)
                inputs['ngram_attention_mask'] = inputs['ngram_attention_mask'].to(self.args.device)
                inputs['ngram_token_type_ids'] = inputs['ngram_token_type_ids'].to(self.args.device)

            inputs['input_ids'] = inputs['input_ids'].to(self.args.device)
            inputs['attention_mask'] = inputs['attention_mask'].to(self.args.device)
            inputs['token_type_ids'] = inputs['token_type_ids'].to(self.args.device)

            with torch.no_grad():
                outputs = model(**inputs)

                if args.model_type == 'zen':
                    logits = outputs
                else:
                    logits = outputs[0]

            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)

            pbar(step=step, info="")
        preds = np.argmax(preds, axis=1)
        sts_commit_prediction(dataset=test_dataset, preds=preds, output_dir=args.result_output_dir,
                              id2label=self.data_processor.id2label)

        return preds

    def _save_best_checkpoint(self, best_step):
        model = self.model_class.from_pretrained(os.path.join(self.args.output_dir, f'checkpoint-{best_step}'),
                                                 num_labels=self.data_processor.num_labels)

        if self.args.model_type == 'zen':
            save_zen_model(self.args.output_dir, model=model, tokenizer=self.tokenizer,
                           ngram_dict=self.ngram_dict, args=self.args)
        else:
            model.save_pretrained(self.args.output_dir)
            torch.save(self.args, os.path.join(self.args.output_dir, 'training_args.bin'))
            self.tokenizer.save_vocabulary(save_directory=self.args.output_dir)
        self.logger.info('Saving models checkpoint to %s', self.args.output_dir)


class QICTrainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            model_class,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None

    ):
        super(QICTrainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
            model_class=model_class,
            ngram_dict=ngram_dict
        )

    def training_step(self, model, item):
        model.train()

        text1 = item[0]
        labels = item[1].to(self.args.device)

        if self.args.model_type == 'zen':
            inputs = convert_examples_to_features(text1=text1, ngram_dict=self.ngram_dict,
                                                  tokenizer=self.tokenizer, max_seq_length=self.args.max_length,
                                                  return_tensors=True)
        else:
            inputs = self.tokenizer(text1, padding='max_length', max_length=self.args.max_length,
                                    truncation=True, return_tensors='pt')

        if self.args.model_type == 'zen':
            inputs['input_ngram_ids'] = inputs['input_ngram_ids'].to(self.args.device)
            inputs['ngram_position_matrix'] = inputs['ngram_position_matrix'].to(self.args.device)
            inputs['ngram_attention_mask'] = inputs['ngram_attention_mask'].to(self.args.device)
            inputs['ngram_token_type_ids'] = inputs['ngram_token_type_ids'].to(self.args.device)

        inputs['input_ids'] = inputs['input_ids'].to(self.args.device)
        inputs['attention_mask'] = inputs['attention_mask'].to(self.args.device)
        inputs['token_type_ids'] = inputs['token_type_ids'].to(self.args.device)

        # default using 'Transformers' library models.
        outputs = model(labels=labels, **inputs)
        loss = outputs[0]
        loss.backward()

        return loss.detach()

    def evaluate(self, model):
        args = self.args
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)

        preds = None
        eval_labels = None

        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        for step, item in enumerate(eval_dataloader):
            model.eval()

            text1 = item[0]
            labels = item[1].to(args.device)

            if self.args.model_type == 'zen':
                inputs = convert_examples_to_features(text1=text1, ngram_dict=self.ngram_dict,
                                                      tokenizer=self.tokenizer, max_seq_length=self.args.max_length,
                                                      return_tensors=True)
            else:
                inputs = self.tokenizer(text1, return_tensors='pt', padding='max_length',
                                        truncation='longest_first', max_length=self.args.max_length)
            inputs['input_ids'] = inputs['input_ids'].to(self.args.device)
            inputs['attention_mask'] = inputs['attention_mask'].to(self.args.device)
            inputs['token_type_ids'] = inputs['token_type_ids'].to(self.args.device)

            if self.args.model_type == 'zen':
                inputs['input_ngram_ids'] = inputs['input_ngram_ids'].to(self.args.device)
                inputs['ngram_position_matrix'] = inputs['ngram_position_matrix'].to(self.args.device)
                inputs['ngram_attention_mask'] = inputs['ngram_attention_mask'].to(self.args.device)
                inputs['ngram_token_type_ids'] = inputs['ngram_token_type_ids'].to(self.args.device)

            with torch.no_grad():
                outputs = model(labels=labels, **inputs)
                loss, logits = outputs[:2]

            if preds is None:
                preds = logits.detach().cpu().numpy()
                eval_labels = labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                eval_labels = np.append(eval_labels, labels.detach().cpu().numpy(), axis=0)

        preds = np.argmax(preds, axis=1)
        acc = qic_metric(preds, eval_labels)
        logger.info("%s-%s acc: %s", args.task_name, args.model_name, acc)
        return acc

    def predict(self, test_dataset, model):
        args = self.args
        logger = self.logger
        test_dataloader = self.get_test_dataloader(test_dataset)
        num_examples = len(test_dataloader.dataset)
        model.to(args.device)

        preds = None

        logger.info("***** Running prediction *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(test_dataloader), desc='Prediction')
        for step, item in enumerate(test_dataloader):
            model.eval()

            text1 = item

            if self.args.model_type == 'zen':
                inputs = convert_examples_to_features(text1=text1, ngram_dict=self.ngram_dict,
                                                      tokenizer=self.tokenizer, max_seq_length=self.args.max_length,
                                                      return_tensors=True)
            else:
                inputs = self.tokenizer(text1, return_tensors='pt', padding='max_length',
                                        truncation='longest_first', max_length=self.args.max_length)
            if self.args.model_type == 'zen':
                inputs['input_ngram_ids'] = inputs['input_ngram_ids'].to(self.args.device)
                inputs['ngram_position_matrix'] = inputs['ngram_position_matrix'].to(self.args.device)
                inputs['ngram_attention_mask'] = inputs['ngram_attention_mask'].to(self.args.device)
                inputs['ngram_token_type_ids'] = inputs['ngram_token_type_ids'].to(self.args.device)

            inputs['input_ids'] = inputs['input_ids'].to(self.args.device)
            inputs['attention_mask'] = inputs['attention_mask'].to(self.args.device)
            inputs['token_type_ids'] = inputs['token_type_ids'].to(self.args.device)

            with torch.no_grad():
                outputs = model(**inputs)
                if self.args.model_type == 'zen':
                    logits = outputs
                else:
                    logits = outputs[0]

            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)

            pbar(step=step, info="")
        preds = np.argmax(preds, axis=1)
        qic_commit_prediction(dataset=test_dataset, preds=preds, output_dir=args.result_output_dir,
                              id2label=self.data_processor.id2label)

        return preds

    def _save_checkpoint(self, model, step):
        output_dir = os.path.join(self.args.output_dir, 'checkpoint-{}'.format(step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if self.args.model_type == 'zen':
            save_zen_model(output_dir, model=model, tokenizer=self.tokenizer,
                           ngram_dict=self.ngram_dict, args=self.args)
        else:
            model.save_pretrained(output_dir)
            torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))
            self.tokenizer.save_vocabulary(save_directory=output_dir)
        self.logger.info('Saving models checkpoint to %s', output_dir)

    def _save_best_checkpoint(self, best_step):
        model = self.model_class.from_pretrained(os.path.join(self.args.output_dir, f'checkpoint-{best_step}'),
                                                 num_labels=self.data_processor.num_labels)

        if self.args.model_type == 'zen':
            save_zen_model(self.args.output_dir, model=model, tokenizer=self.tokenizer,
                           ngram_dict=self.ngram_dict, args=self.args)
        else:
            model.save_pretrained(self.args.output_dir)
            torch.save(self.args, os.path.join(self.args.output_dir, 'training_args.bin'))
            self.tokenizer.save_vocabulary(save_directory=self.args.output_dir)
        self.logger.info('Saving models checkpoint to %s', self.args.output_dir)


class QQRTrainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            model_class,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(QQRTrainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
            model_class=model_class,
            ngram_dict=ngram_dict
        )

    def training_step(self, model, item):
        model.train()

        text1 = item[0]
        text2 = item[1]
        labels = item[2].to(self.args.device)

        if self.args.model_type == 'zen':
            inputs = convert_examples_to_features(text1=text1, text2=text2, ngram_dict=self.ngram_dict,
                                                  tokenizer=self.tokenizer, max_seq_length=self.args.max_length,
                                                  return_tensors=True)
        else:
            inputs = self.tokenizer(text1, text2, return_tensors='pt', padding='max_length',
                                    truncation='longest_first', max_length=self.args.max_length)

        if self.args.model_type == 'zen':
            inputs['input_ngram_ids'] = inputs['input_ngram_ids'].to(self.args.device)
            inputs['ngram_position_matrix'] = inputs['ngram_position_matrix'].to(self.args.device)
            inputs['ngram_attention_mask'] = inputs['ngram_attention_mask'].to(self.args.device)
            inputs['ngram_token_type_ids'] = inputs['ngram_token_type_ids'].to(self.args.device)

        inputs['input_ids'] = inputs['input_ids'].to(self.args.device)
        inputs['attention_mask'] = inputs['attention_mask'].to(self.args.device)
        inputs['token_type_ids'] = inputs['token_type_ids'].to(self.args.device)

        # default using 'Transformers' library models.
        outputs = model(labels=labels, **inputs)
        loss = outputs[0]
        loss.backward()

        return loss.detach()

    def evaluate(self, model):
        args = self.args
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)

        preds = None
        eval_labels = None

        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        for step, item in enumerate(eval_dataloader):
            model.eval()

            text1 = item[0]
            text2 = item[1]
            labels = item[2].to(args.device)

            if self.args.model_type == 'zen':
                inputs = convert_examples_to_features(text1=text1, text2=text2, ngram_dict=self.ngram_dict,
                                                      tokenizer=self.tokenizer, max_seq_length=self.args.max_length,
                                                      return_tensors=True)
            else:
                inputs = self.tokenizer(text1, text2, return_tensors='pt', padding='max_length',
                                        truncation='longest_first', max_length=self.args.max_length)

            if self.args.model_type == 'zen':
                inputs['input_ngram_ids'] = inputs['input_ngram_ids'].to(self.args.device)
                inputs['ngram_position_matrix'] = inputs['ngram_position_matrix'].to(self.args.device)
                inputs['ngram_attention_mask'] = inputs['ngram_attention_mask'].to(self.args.device)
                inputs['ngram_token_type_ids'] = inputs['ngram_token_type_ids'].to(self.args.device)

            inputs['input_ids'] = inputs['input_ids'].to(self.args.device)
            inputs['attention_mask'] = inputs['attention_mask'].to(self.args.device)
            inputs['token_type_ids'] = inputs['token_type_ids'].to(self.args.device)

            with torch.no_grad():
                outputs = model(labels=labels, **inputs)
                loss, logits = outputs[:2]

            if preds is None:
                preds = logits.detach().cpu().numpy()
                eval_labels = labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                eval_labels = np.append(eval_labels, labels.detach().cpu().numpy(), axis=0)

        preds = np.argmax(preds, axis=1)
        acc = qqr_metric(preds, eval_labels)
        logger.info("%s-%s acc: %s", args.task_name, args.model_name, acc)
        return acc

    def predict(self, test_dataset, model):
        args = self.args
        logger = self.logger
        test_dataloader = self.get_test_dataloader(test_dataset)
        num_examples = len(test_dataloader.dataset)
        model.to(args.device)

        preds = None

        logger.info("***** Running prediction *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(test_dataloader), desc='Prediction')
        for step, item in enumerate(test_dataloader):
            model.eval()

            text1 = item[0]
            text2 = item[1]

            if self.args.model_type == 'zen':
                inputs = convert_examples_to_features(text1=text1, text2=text2, ngram_dict=self.ngram_dict,
                                                      tokenizer=self.tokenizer, max_seq_length=self.args.max_length,
                                                      return_tensors=True)
            else:
                inputs = self.tokenizer(text1, text2, return_tensors='pt', padding='max_length',
                                        truncation='longest_first', max_length=self.args.max_length)

            if self.args.model_type == 'zen':
                inputs['input_ngram_ids'] = inputs['input_ngram_ids'].to(self.args.device)
                inputs['ngram_position_matrix'] = inputs['ngram_position_matrix'].to(self.args.device)
                inputs['ngram_attention_mask'] = inputs['ngram_attention_mask'].to(self.args.device)
                inputs['ngram_token_type_ids'] = inputs['ngram_token_type_ids'].to(self.args.device)

            inputs['input_ids'] = inputs['input_ids'].to(self.args.device)
            inputs['attention_mask'] = inputs['attention_mask'].to(self.args.device)
            inputs['token_type_ids'] = inputs['token_type_ids'].to(self.args.device)

            with torch.no_grad():
                outputs = model(**inputs)
                if self.args.model_type == 'zen':
                    logits = outputs
                else:
                    logits = outputs[0]

            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)

            pbar(step=step, info="")
        preds = np.argmax(preds, axis=1)
        qqr_commit_prediction(dataset=test_dataset, preds=preds, output_dir=args.result_output_dir,
                              id2label=self.data_processor.id2label)

    def _save_checkpoint(self, model, step):
        output_dir = os.path.join(self.args.output_dir, 'checkpoint-{}'.format(step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if self.args.model_type == 'zen':
            save_zen_model(output_dir, model=model, tokenizer=self.tokenizer,
                           ngram_dict=self.ngram_dict, args=self.args)
        else:
            model.save_pretrained(output_dir)
            torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))
            self.tokenizer.save_vocabulary(save_directory=output_dir)
        self.logger.info('Saving models checkpoint to %s', output_dir)

    def _save_best_checkpoint(self, best_step):
        model = self.model_class.from_pretrained(os.path.join(self.args.output_dir, f'checkpoint-{best_step}'),
                                                 num_labels=self.data_processor.num_labels)
        if self.args.model_type == 'zen':
            save_zen_model(self.args.output_dir, model=model, tokenizer=self.tokenizer,
                           ngram_dict=self.ngram_dict, args=self.args)
        else:
            model.save_pretrained(self.args.output_dir)
            torch.save(self.args, os.path.join(self.args.output_dir, 'training_args.bin'))
            self.tokenizer.save_vocabulary(save_directory=self.args.output_dir)
        self.logger.info('Saving models checkpoint to %s', self.args.output_dir)


class QTRTrainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            model_class,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(QTRTrainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
            model_class=model_class,
            ngram_dict=ngram_dict
        )

    def training_step(self, model, item):
        model.train()

        text1 = item[0]
        text2 = item[1]
        labels = item[2].to(self.args.device)
        if self.args.model_type == 'zen':
            inputs = convert_examples_to_features(text1=text1, text2=text2, ngram_dict=self.ngram_dict,
                                                  tokenizer=self.tokenizer, max_seq_length=self.args.max_length,
                                                  return_tensors=True)
        else:
            inputs = self.tokenizer(text1, text2, return_tensors='pt', padding='max_length',
                                    truncation='longest_first', max_length=self.args.max_length)

        if self.args.model_type == 'zen':
            inputs['input_ngram_ids'] = inputs['input_ngram_ids'].to(self.args.device)
            inputs['ngram_position_matrix'] = inputs['ngram_position_matrix'].to(self.args.device)
            inputs['ngram_attention_mask'] = inputs['ngram_attention_mask'].to(self.args.device)
            inputs['ngram_token_type_ids'] = inputs['ngram_token_type_ids'].to(self.args.device)

        inputs['input_ids'] = inputs['input_ids'].to(self.args.device)
        inputs['attention_mask'] = inputs['attention_mask'].to(self.args.device)
        inputs['token_type_ids'] = inputs['token_type_ids'].to(self.args.device)

        # default using 'Transformers' library models.
        outputs = model(labels=labels, **inputs)
        loss = outputs[0]
        loss.backward()

        return loss.detach()

    def evaluate(self, model):
        args = self.args
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)

        preds = None
        eval_labels = None

        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        for step, item in enumerate(eval_dataloader):
            model.eval()

            text1 = item[0]
            text2 = item[1]
            labels = item[2].to(args.device)

            if self.args.model_type == 'zen':
                inputs = convert_examples_to_features(text1=text1, text2=text2, ngram_dict=self.ngram_dict,
                                                      tokenizer=self.tokenizer, max_seq_length=self.args.max_length,
                                                      return_tensors=True)
            else:
                inputs = self.tokenizer(text1, text2, return_tensors='pt', padding='max_length',
                                        truncation='longest_first', max_length=self.args.max_length)

            if self.args.model_type == 'zen':
                inputs['input_ngram_ids'] = inputs['input_ngram_ids'].to(self.args.device)
                inputs['ngram_position_matrix'] = inputs['ngram_position_matrix'].to(self.args.device)
                inputs['ngram_attention_mask'] = inputs['ngram_attention_mask'].to(self.args.device)
                inputs['ngram_token_type_ids'] = inputs['ngram_token_type_ids'].to(self.args.device)

            inputs['input_ids'] = inputs['input_ids'].to(self.args.device)
            inputs['attention_mask'] = inputs['attention_mask'].to(self.args.device)
            inputs['token_type_ids'] = inputs['token_type_ids'].to(self.args.device)

            with torch.no_grad():
                outputs = model(labels=labels, **inputs)
                loss, logits = outputs[:2]

            if preds is None:
                preds = logits.detach().cpu().numpy()
                eval_labels = labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                eval_labels = np.append(eval_labels, labels.detach().cpu().numpy(), axis=0)

        preds = np.argmax(preds, axis=1)
        acc = qtr_metric(preds, eval_labels)
        logger.info("%s-%s acc: %s", args.task_name, args.model_name, acc)
        return acc

    def predict(self, test_dataset, model):
        args = self.args
        logger = self.logger
        test_dataloader = self.get_test_dataloader(test_dataset)
        num_examples = len(test_dataloader.dataset)
        model.to(args.device)

        preds = None

        logger.info("***** Running prediction *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(test_dataloader), desc='Prediction')
        for step, item in enumerate(test_dataloader):
            model.eval()

            text1 = item[0]
            text2 = item[1]

            if self.args.model_type == 'zen':
                inputs = convert_examples_to_features(text1=text1, text2=text2, ngram_dict=self.ngram_dict,
                                                      tokenizer=self.tokenizer, max_seq_length=self.args.max_length,
                                                      return_tensors=True)
            else:
                inputs = self.tokenizer(text1, text2, return_tensors='pt', padding='max_length',
                                        truncation='longest_first', max_length=self.args.max_length)

            if self.args.model_type == 'zen':
                inputs['input_ngram_ids'] = inputs['input_ngram_ids'].to(self.args.device)
                inputs['ngram_position_matrix'] = inputs['ngram_position_matrix'].to(self.args.device)
                inputs['ngram_attention_mask'] = inputs['ngram_attention_mask'].to(self.args.device)
                inputs['ngram_token_type_ids'] = inputs['ngram_token_type_ids'].to(self.args.device)

            inputs['input_ids'] = inputs['input_ids'].to(self.args.device)
            inputs['attention_mask'] = inputs['attention_mask'].to(self.args.device)
            inputs['token_type_ids'] = inputs['token_type_ids'].to(self.args.device)

            with torch.no_grad():
                outputs = model(**inputs)
                if self.args.model_type == 'zen':
                    logits = outputs
                else:
                    logits = outputs[0]

            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)

            pbar(step=step, info="")
        preds = np.argmax(preds, axis=1)
        qtr_commit_prediction(dataset=test_dataset, preds=preds, output_dir=args.result_output_dir,
                              id2label=self.data_processor.id2label)

        return preds

    def _save_checkpoint(self, model, step):
        output_dir = os.path.join(self.args.output_dir, 'checkpoint-{}'.format(step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if self.args.model_type == 'zen':
            save_zen_model(output_dir, model=model, tokenizer=self.tokenizer,
                           ngram_dict=self.ngram_dict, args=self.args)
        else:
            model.save_pretrained(output_dir)
            torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))
            self.tokenizer.save_vocabulary(save_directory=output_dir)
        self.logger.info('Saving models checkpoint to %s', output_dir)

    def _save_best_checkpoint(self, best_step):
        model = self.model_class.from_pretrained(os.path.join(self.args.output_dir, f'checkpoint-{best_step}'),
                                                 num_labels=self.data_processor.num_labels)
        if self.args.model_type == 'zen':
            save_zen_model(self.args.output_dir, model=model, tokenizer=self.tokenizer,
                           ngram_dict=self.ngram_dict, args=self.args)
        else:
            model.save_pretrained(self.args.output_dir)
            torch.save(self.args, os.path.join(self.args.output_dir, 'training_args.bin'))
            self.tokenizer.save_vocabulary(save_directory=self.args.output_dir)
        self.logger.info('Saving models checkpoint to %s', self.args.output_dir)


class CTCTrainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            model_class,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(CTCTrainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
            model_class=model_class,
            ngram_dict=ngram_dict
        )

    def training_step(self, model, item):
        model.train()

        input_ids = item[0].to(self.args.device)
        token_type_ids = item[1].to(self.args.device)
        attention_mask = item[2].to(self.args.device)
        labels = item[3].to(self.args.device)

        if self.args.model_type == 'zen':
            input_ngram_ids = item[4].to(self.args.device)
            ngram_attention_mask = item[5].to(self.args.device)
            ngram_token_type_ids = item[6].to(self.args.device)
            ngram_position_matrix = item[7].to(self.args.device)

        # default using 'Transformers' library models.
        if self.args.model_type == 'zen':
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            labels=labels, ngram_ids=input_ngram_ids, ngram_positions=ngram_position_matrix,
                            ngram_attention_mask=ngram_attention_mask, ngram_token_type_ids=ngram_token_type_ids)
        else:
            outputs = model(labels=labels, input_ids=input_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask)
        loss = outputs[0]
        loss.backward()

        return loss.detach()

    def evaluate(self, model):
        args = self.args
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)

        preds = None
        eval_labels = None

        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        for step, item in enumerate(eval_dataloader):
            model.eval()

            input_ids = item[0].to(self.args.device)
            token_type_ids = item[1].to(self.args.device)
            attention_mask = item[2].to(self.args.device)
            labels = item[3].to(self.args.device)

            if args.model_type == 'zen':
                input_ngram_ids = item[4].to(self.args.device)
                ngram_attention_mask = item[5].to(self.args.device)
                ngram_token_type_ids = item[6].to(self.args.device)
                ngram_position_matrix = item[7].to(self.args.device)

            with torch.no_grad():
                if self.args.model_type == 'zen':
                    outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                    labels=labels, ngram_ids=input_ngram_ids,
                                    ngram_positions=ngram_position_matrix,
                                    ngram_token_type_ids=ngram_token_type_ids,
                                    ngram_attention_mask=ngram_attention_mask)
                else:
                    outputs = model(labels=labels, input_ids=input_ids, token_type_ids=token_type_ids,
                                    attention_mask=attention_mask)

                loss, logits = outputs[:2]

            if preds is None:
                preds = logits.detach().cpu().numpy()
                eval_labels = labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                eval_labels = np.append(eval_labels, labels.detach().cpu().numpy(), axis=0)

        preds = np.argmax(preds, axis=1)
        p, r, f1, _ = ctc_metric(preds, eval_labels)
        logger.info("%s-%s precision: %s - recall: %s - f1 score: %s", args.task_name, args.model_name, p, r, f1)
        return f1

    def predict(self, test_dataset, model):
        args = self.args
        logger = self.logger
        test_dataloader = self.get_test_dataloader(test_dataset)
        num_examples = len(test_dataloader.dataset)
        model.to(args.device)

        preds = None

        logger.info("***** Running prediction *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(test_dataloader), desc='Prediction')
        for step, item in enumerate(test_dataloader):
            model.eval()

            input_ids = item[0].to(self.args.device)
            token_type_ids = item[1].to(self.args.device)
            attention_mask = item[2].to(self.args.device)

            if args.model_type == 'zen':
                input_ngram_ids = item[3].to(self.args.device)
                ngram_attention_mask = item[4].to(self.args.device)
                ngram_token_type_ids = item[5].to(self.args.device)
                ngram_position_matrix = item[6].to(self.args.device)

            with torch.no_grad():
                if self.args.model_type == 'zen':
                    outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                    ngram_ids=input_ngram_ids,
                                    ngram_positions=ngram_position_matrix,
                                    ngram_token_type_ids=ngram_token_type_ids,
                                    ngram_attention_mask=ngram_attention_mask)
                else:
                    outputs = model(input_ids=input_ids, token_type_ids=token_type_ids,
                                    attention_mask=attention_mask)

                if args.model_type == 'zen':
                    logits = outputs.detach()
                else:
                    logits = outputs[0].detach()

            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)

            pbar(step=step, info="")
        preds = np.argmax(preds, axis=1)
        ctc_commit_prediction(dataset=test_dataset, preds=preds, output_dir=args.result_output_dir,
                              id2label=self.data_processor.id2label)

        return preds

    def _save_checkpoint(self, model, step):
        output_dir = os.path.join(self.args.output_dir, 'checkpoint-{}'.format(step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if self.args.model_type == 'zen':
            save_zen_model(output_dir, model=model, tokenizer=self.tokenizer,
                           ngram_dict=self.ngram_dict, args=self.args)
        else:
            model.save_pretrained(output_dir)
            torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))
            self.tokenizer.save_vocabulary(save_directory=output_dir)
        self.logger.info('Saving models checkpoint to %s', output_dir)

    def _save_best_checkpoint(self, best_step):
        model = self.model_class.from_pretrained(os.path.join(self.args.output_dir, f'checkpoint-{best_step}'),
                                                 num_labels=self.data_processor.num_labels)
        if self.args.model_type == 'zen':
            save_zen_model(self.args.output_dir, model=model, tokenizer=self.tokenizer,
                           ngram_dict=self.ngram_dict, args=self.args)
        else:
            model.save_pretrained(self.args.output_dir)
            torch.save(self.args, os.path.join(self.args.output_dir, 'training_args.bin'))
            self.tokenizer.save_vocabulary(save_directory=self.args.output_dir)
        self.logger.info('Saving models checkpoint to %s', self.args.output_dir)


class ERTrainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            model_class,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(ERTrainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
            model_class=model_class,
            ngram_dict=ngram_dict
        )

        self.loss_fn = nn.BCELoss()

    def training_step(self, model, item):
        model.train()

        if self.args.model_type == 'zen':
            input_ids, token_type_ids, attention_mask, sub_start_label, sub_end_label, obj_start_label, \
            obj_end_label, input_ngram_ids, ngram_attention_mask, ngram_token_type_ids, ngram_position_matrix = item
        else:
            input_ids, token_type_ids, attention_mask, sub_start_label, sub_end_label, obj_start_label, obj_end_label = item
        input_ids = input_ids.to(self.args.device)
        token_type_ids = token_type_ids.to(self.args.device)
        attention_mask = attention_mask.to(self.args.device)
        sub_start_label = sub_start_label.to(self.args.device)
        sub_end_label = sub_end_label.to(self.args.device)
        obj_start_label = obj_start_label.to(self.args.device)
        obj_end_label = obj_end_label.to(self.args.device)

        if self.args.model_type == 'zen':
            input_ngram_ids = input_ngram_ids.to(self.args.device)
            ngram_token_type_ids = ngram_token_type_ids.to(self.args.device)
            ngram_attention_mask = ngram_attention_mask.to(self.args.device)
            ngram_position_matrix = ngram_position_matrix.to(self.args.device)

        if self.args.model_type == 'zen':
            sub_start_logits, sub_end_logits, obj_start_logits, obj_end_logits = model(input_ids, token_type_ids,
                                                                                       attention_mask,
                                                                                       input_ngram_ids=input_ngram_ids,
                                                                                       ngram_attention_mask=ngram_attention_mask,
                                                                                       ngram_position_matrix=ngram_position_matrix,
                                                                                       ngram_token_type_ids=ngram_token_type_ids)
        else:
            sub_start_logits, sub_end_logits, obj_start_logits, obj_end_logits = model(input_ids,
                                                                                       token_type_ids,
                                                                                       attention_mask)

        active_index = attention_mask.view(-1) == 1
        sub_start_loss = self.cal_loss(sub_start_logits, sub_start_label, active_index)
        sub_end_loss = self.cal_loss(sub_end_logits, sub_end_label, active_index)
        obj_start_loss = self.cal_loss(obj_start_logits, obj_start_label, active_index)
        obj_end_loss = self.cal_loss(obj_end_logits, obj_end_label, active_index)
        loss = sub_start_loss + sub_end_loss + obj_start_loss + obj_end_loss

        loss.backward()

        return loss.detach()

    def cal_loss(self, logits, labels, active_index):
        active_labels = labels.view(-1)[active_index]
        active_logits = logits.view(-1)[active_index]
        return self.loss_fn(active_logits.float()[1:-1], active_labels.float()[1:-1])

    def evaluate(self, model):
        args = self.args
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)

        sub_start_preds = []
        sub_end_preds = []
        obj_start_preds = []
        obj_end_preds = []

        sub_start_trues = []
        sub_end_trues = []
        obj_start_trues = []
        obj_end_trues = []

        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        for step, item in enumerate(eval_dataloader):
            model.eval()

            if self.args.model_type == 'zen':
                input_ids, token_type_ids, attention_mask, sub_start_label, sub_end_label, obj_start_label, \
                obj_end_label, input_ngram_ids, ngram_attention_mask,  ngram_token_type_ids, ngram_position_matrix = item
            else:
                input_ids, token_type_ids, attention_mask, sub_start_label, sub_end_label, obj_start_label, obj_end_label = item
            input_ids = input_ids.to(self.args.device)
            token_type_ids = token_type_ids.to(self.args.device)
            attention_mask = attention_mask.to(self.args.device)
            sub_start_label = sub_start_label.to(self.args.device)
            sub_end_label = sub_end_label.to(self.args.device)
            obj_start_label = obj_start_label.to(self.args.device)
            obj_end_label = obj_end_label.to(self.args.device)

            if self.args.model_type == 'zen':
                input_ngram_ids = input_ngram_ids.to(self.args.device)
                ngram_token_type_ids = ngram_token_type_ids.to(self.args.device)
                ngram_attention_mask = ngram_attention_mask.to(self.args.device)
                ngram_position_matrix = ngram_position_matrix.to(self.args.device)

            with torch.no_grad():
                if args.model_type == 'zen':
                    sub_start_logits, sub_end_logits, obj_start_logits, obj_end_logits = model(input_ids,
                                                                                               token_type_ids,
                                                                                               attention_mask,
                                                                                               input_ngram_ids=input_ngram_ids,
                                                                                               ngram_attention_mask=ngram_attention_mask,
                                                                                               ngram_position_matrix=ngram_position_matrix,
                                                                                               ngram_token_type_ids=ngram_token_type_ids)
                else:
                    sub_start_logits, sub_end_logits, obj_start_logits, obj_end_logits = model(input_ids,
                                                                                               token_type_ids,
                                                                                               attention_mask)

            active_index = attention_mask.view(-1) == 1
            sub_start_preds.extend((sub_start_logits.detach().view(-1) >= 0.5).cpu().long()[active_index])
            sub_end_preds.extend((sub_end_logits.detach().view(-1) >= 0.5).cpu().long()[active_index])
            obj_start_preds.extend((obj_start_logits.detach().view(-1) >= 0.5).cpu().long()[active_index])
            obj_end_preds.extend((obj_end_logits.detach().view(-1) >= 0.5).cpu()[active_index])

            sub_start_trues.extend(sub_start_label.detach().cpu().view(-1)[active_index].tolist())
            sub_end_trues.extend(sub_end_label.detach().cpu().view(-1)[active_index].tolist())
            obj_start_trues.extend(obj_start_label.detach().cpu().view(-1)[active_index].tolist())
            obj_end_trues.extend(obj_end_label.detach().cpu().view(-1)[active_index].tolist())

        s_start_p, s_start_r, s_start_f1, _ = er_metric(sub_start_preds, sub_start_trues)
        s_end_p, s_end_r, s_end_f1, _ = er_metric(sub_end_preds, sub_end_trues)
        o_start_p, o_start_r, o_start_f1, _ = er_metric(obj_start_preds, obj_start_trues)
        o_end_p, o_end_r, o_end_f1, _ = er_metric(obj_end_preds, obj_end_trues)
        f1 = (s_start_f1 + s_end_f1 + o_end_f1 + o_start_f1) / 4

        logger.info("%s-%s f1 score: %s", args.task_name, args.model_name, f1)
        return f1

    def predict(self, test_dataset, model):
        args = self.args
        logger = self.logger
        test_dataloader = self.get_test_dataloader(test_dataset, batch_size=1)
        num_examples = len(test_dataloader.dataset)
        model.to(args.device)

        logger.info("***** Running prediction *****")
        logger.info("Num samples %d", num_examples)
        with open(os.path.join(args.output_dir, 'CMeIE_test.json'), 'w', encoding='utf-8') as f:
            for step, item in enumerate(test_dataloader):
                model.eval()

                if args.model_type == 'zen':
                    input_ids, token_type_ids, attention_mask, input_ngram_ids,  ngram_attention_mask, ngram_token_type_ids, ngram_position_matrix = item
                else:
                    input_ids, token_type_ids, attention_mask = item
                input_ids = input_ids.to(self.args.device)
                token_type_ids = token_type_ids.to(self.args.device)
                attention_mask = attention_mask.to(self.args.device)

                if self.args.model_type == 'zen':
                    input_ngram_ids = input_ngram_ids.to(self.args.device)
                    ngram_token_type_ids = ngram_token_type_ids.to(self.args.device)
                    ngram_attention_mask = ngram_attention_mask.to(self.args.device)
                    ngram_position_matrix = ngram_position_matrix.to(self.args.device)

                with torch.no_grad():
                    if args.model_type == 'zen':
                        sub_start_logits, sub_end_logits, obj_start_logits, obj_end_logits = model(input_ids, token_type_ids,
                                                                                                   attention_mask,
                                                                                                   input_ngram_ids=input_ngram_ids,
                                                                                                   ngram_attention_mask=ngram_attention_mask,
                                                                                                   ngram_position_matrix=ngram_position_matrix,
                                                                                                   ngram_token_type_ids=ngram_token_type_ids)
                    else:
                        sub_start_logits, sub_end_logits, obj_start_logits, obj_end_logits = model(input_ids,
                                                                                                   token_type_ids,
                                                                                                   attention_mask)

                    text = test_dataset.texts[step]
                    text_start_id, text_end_id = 1, attention_mask.sum().int().item()  # end+1
                    text_mapping = TokenRematch().rematch(text, self.tokenizer.tokenize(text))

                    sub_arg_list = self.data_processor.extract_arg(sub_start_logits.view(-1), sub_end_logits.view(-1), text_start_id, text_end_id,
                                                                   text, text_mapping)
                    obj_arg_list = self.data_processor.extract_arg(obj_start_logits.view(-1), obj_end_logits.view(-1), text_start_id, text_end_id,
                                                                   text, text_mapping)
                    result = {'text': text, 'sub_list': sub_arg_list, 'obj_list': obj_arg_list}
                    json_data = json.dumps(result, ensure_ascii=False)
                    f.write(json_data + '\n')

    def _save_checkpoint(self, model, step):
        output_dir = os.path.join(self.args.output_dir, 'checkpoint-{}'.format(step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torch.save(model.state_dict(), os.path.join(output_dir, 'pytorch_model.pt'))
        self.logger.info('Saving models checkpoint to %s', output_dir)

        if self.args.model_type == 'zen':
            save_zen_model(output_dir, model.encoder, self.tokenizer, self.ngram_dict, self.args)
        else:
            model.encoder.save_pretrained(output_dir)
            self.tokenizer.save_vocabulary(save_directory=output_dir)

    def _save_best_checkpoint(self, best_step):
        pass


class RETrainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            model_class,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(RETrainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
            model_class=model_class,
            ngram_dict=ngram_dict
        )

    def training_step(self, model, item):
        model.train()

        if self.args.model_type == 'zen':
            input_ids, token_type_ids, attention_mask, flag, label, input_ngram_ids, \
            ngram_attention_mask, ngram_token_type_ids, ngram_position_matrix = item
        else:
            input_ids, token_type_ids, attention_mask, flag, label = item

        input_ids, token_type_ids, attention_mask, flag, label = input_ids.to(self.args.device), \
                                                                 token_type_ids.to(self.args.device), \
                                                                 attention_mask.to(self.args.device), \
                                                                 flag.to(self.args.device), label.to(self.args.device)

        if self.args.model_type == 'zen':
            input_ngram_ids = input_ngram_ids.to(self.args.device)
            ngram_position_matrix = ngram_position_matrix.to(self.args.device)
            ngram_attention_mask = ngram_attention_mask.to(self.args.device)
            ngram_token_type_ids = ngram_token_type_ids.to(self.args.device)

            loss, logits = model(input_ids, token_type_ids, attention_mask, flag, label,
                                 input_ngram_ids=input_ngram_ids, ngram_attention_mask=ngram_attention_mask,
                                 ngram_position_matrix=ngram_position_matrix, ngram_token_type_ids=ngram_token_type_ids)
        else:
            loss, logits = model(input_ids, token_type_ids, attention_mask, flag, label)
        loss.backward()

        return loss.detach()

    def evaluate(self, model):
        args = self.args
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)

        preds = None
        eval_labels = None

        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        for step, item in enumerate(eval_dataloader):
            model.eval()

            if self.args.model_type == 'zen':
                input_ids, token_type_ids, attention_mask, flag, label, input_ngram_ids, \
                ngram_attention_mask, ngram_token_type_ids, ngram_position_matrix = item
            else:
                input_ids, token_type_ids, attention_mask, flag, label = item

            input_ids, token_type_ids, attention_mask, flag, label = input_ids.to(self.args.device), \
                                                                     token_type_ids.to(self.args.device), \
                                                                     attention_mask.to(self.args.device), \
                                                                     flag.to(self.args.device), label.to(self.args.device)

            with torch.no_grad():
                if self.args.model_type == 'zen':
                    input_ngram_ids = input_ngram_ids.to(self.args.device)
                    ngram_position_matrix = ngram_position_matrix.to(self.args.device)
                    ngram_attention_mask = ngram_attention_mask.to(self.args.device)
                    ngram_token_type_ids = ngram_token_type_ids.to(self.args.device)

                    loss, logits = model(input_ids, token_type_ids, attention_mask, flag, label,
                                         input_ngram_ids=input_ngram_ids, ngram_attention_mask=ngram_attention_mask,
                                         ngram_position_matrix=ngram_position_matrix,
                                         ngram_token_type_ids=ngram_token_type_ids)
                else:
                    loss, logits = model(input_ids, token_type_ids, attention_mask, flag, label)

            if preds is None:
                preds = logits.detach().cpu().numpy()
                eval_labels = label.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                eval_labels = np.append(eval_labels, label.detach().cpu().numpy(), axis=0)

        preds = np.argmax(preds, axis=1)
        p, r, f1, _ = re_metric(preds, eval_labels)
        logger.info("%s-%s precision: %s - recall: %s - f1 score: %s", args.task_name, args.model_name, p, r, f1)
        return f1

    def predict(self, test_samples, model, re_dataset_class):
        args = self.args
        logger = self.logger
        model.to(args.device)

        logger.info("***** Running prediction *****")
        with open(os.path.join(args.result_output_dir, 'CMeIE_test.json'), 'w',
                  encoding="utf-8") as f:
            for data in test_samples:
                results, outputs = self.data_processor.build_text(data)
                spo_list = [re['spo_list'] for re in results]
                temp_re_dataset = re_dataset_class(outputs, data_processor=self.data_processor,
                                                   tokenizer=self.tokenizer, max_length=args.max_length, mode="test",
                                                   model_type=args.model_type, ngram_dict=self.ngram_dict)
                logits = []
                with torch.no_grad():
                    for item in temp_re_dataset:
                        if self.args.model_type == 'zen':
                            input_ids, token_type_ids, attention_mask, flag, input_ngram_ids, ngram_attention_mask, ngram_token_type_ids, ngram_position_matrix = item
                        else:
                            input_ids, token_type_ids, attention_mask, flag = item
                        input_ids, token_type_ids, attention_mask, flag = input_ids.to(args.device), \
                                                                          token_type_ids.to(args.device), \
                                                                          attention_mask.to(args.device), \
                                                                          flag.to(args.device)
                        if args.model_type == 'zen':
                            input_ngram_ids = input_ngram_ids.to(self.args.device)
                            ngram_position_matrix = ngram_position_matrix.to(self.args.device)
                            ngram_attention_mask = ngram_attention_mask.to(self.args.device)
                            ngram_token_type_ids = ngram_token_type_ids.to(self.args.device)

                            ngram_max_length = self.ngram_dict.max_ngram_in_seq

                            logit = model(input_ids=input_ids.view(1, -1), token_type_ids=token_type_ids.view(1, -1),
                                          attention_mask=attention_mask.view(1, -1), flag=flag.view(1, -1),
                                          input_ngram_ids=input_ngram_ids.view(1, -1), ngram_token_type_ids=ngram_token_type_ids.view(1, -1),
                                          ngram_attention_mask=ngram_attention_mask.view(1, -1),
                                          ngram_position_matrix=ngram_position_matrix.view(1, ngram_max_length, ngram_max_length))
                        else:
                            logit = model(input_ids=input_ids.view(1, -1), token_type_ids=token_type_ids.view(1, -1),
                                          attention_mask=attention_mask.view(1, -1),
                                          flag=flag.view(1, -1))  # batch, labels
                        logit = logit.argmax(dim=-1).squeeze(-1)  # batch,
                        logits.append(logit.detach().cpu().item())
                for i in range(len(temp_re_dataset)):
                    if logits[i] > 0:
                        spo_list[i]['predicate'] = self.data_processor.id2predicate[logits[i]]

                new_spo_list = []
                for spo in spo_list:
                    if 'predicate' in spo.keys():
                        combined = True
                        for text in data['text'].split("。"):
                            if spo['object'] in text and spo['subject'] in text:
                                combined = False
                                break
                        tmp = {}
                        tmp['Combined'] = combined
                        tmp['predicate'] = spo['predicate'].split('|')[0]
                        tmp['subject'] = spo['subject']
                        tmp['subject_type'] = self.data_processor.pre_sub_obj[spo['predicate']][0]
                        tmp['object'] = {'@value': spo['object']}
                        tmp['object_type'] = {'@value': self.data_processor.pre_sub_obj[spo['predicate']][1]}
                        new_spo_list.append(tmp)

                new_spo_list2 = []  # 去重
                for s in new_spo_list:
                    if s not in new_spo_list2:
                        new_spo_list2.append(s)

                for i in range(len(new_spo_list2)):
                    if 'object' not in new_spo_list2[i].keys():
                        del new_spo_list2[i]

                tmp_result = dict()
                tmp_result['text'] = data['text']
                tmp_result['spo_list'] = new_spo_list2
                json_data = json.dumps(tmp_result, ensure_ascii=False)
                f.write(json_data + '\n')

    def _save_checkpoint(self, model, step):
        output_dir = os.path.join(self.args.output_dir, 'checkpoint-{}'.format(step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torch.save(model.state_dict(), os.path.join(output_dir, 'pytorch_model.pt'))
        self.logger.info('Saving models checkpoint to %s', output_dir)
        if self.args.model_type == 'zen':
            save_zen_model(output_dir, model.encoder, self.tokenizer, self.ngram_dict, self.args)
        else:
            model.encoder.save_pretrained(output_dir)
            self.tokenizer.save_vocabulary(save_directory=output_dir)

    def _save_best_checkpoint(self, best_step):
        pass


class CDNForCLSTrainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            model_class,
            recall_orig_eval_samples=None,
            recall_orig_eval_samples_scores=None,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(CDNForCLSTrainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
            model_class=model_class,
            ngram_dict=ngram_dict
        )

        self.recall_orig_eval_samples = recall_orig_eval_samples
        self.recall_orig_eval_samples_scores = recall_orig_eval_samples_scores

    def training_step(self, model, item):
        model.train()

        text1 = item[0]
        text2 = item[1]
        labels = item[2].to(self.args.device)

        if self.args.model_type == 'zen':
            inputs = convert_examples_to_features(text1=text1, text2=text2, ngram_dict=self.ngram_dict,
                                                  tokenizer=self.tokenizer, max_seq_length=self.args.max_length,
                                                  return_tensors=True)
        else:
            inputs = self.tokenizer(text1, text2, return_tensors='pt', padding='max_length',
                                    truncation='longest_first', max_length=self.args.max_length)

        inputs['input_ids'] = inputs['input_ids'].to(self.args.device)
        inputs['attention_mask'] = inputs['attention_mask'].to(self.args.device)
        inputs['token_type_ids'] = inputs['token_type_ids'].to(self.args.device)

        if self.args.model_type == 'zen':
            inputs['input_ngram_ids'] = inputs['input_ngram_ids'].to(self.args.device)
            inputs['ngram_position_matrix'] = inputs['ngram_position_matrix'].to(self.args.device)
            inputs['ngram_attention_mask'] = inputs['ngram_attention_mask'].to(self.args.device)
            inputs['ngram_token_type_ids'] = inputs['ngram_token_type_ids'].to(self.args.device)

        outputs = model(labels=labels, **inputs)
        loss = outputs[0]
        loss.backward()

        return loss.detach()

    def evaluate(self, model):
        args = self.args
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)

        preds = None
        labels = None

        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluation')
        for step, item in enumerate(eval_dataloader):
            model.eval()

            text1 = item[0]
            text2 = item[1]
            label = item[2].to(args.device)

            if self.args.model_type == 'zen':
                inputs = convert_examples_to_features(text1=text1, text2=text2, ngram_dict=self.ngram_dict,
                                                      tokenizer=self.tokenizer, max_seq_length=self.args.max_length,
                                                      return_tensors=True)
            else:
                inputs = self.tokenizer(text1, text2, return_tensors='pt', padding='max_length',
                                        truncation='longest_first', max_length=self.args.max_length)

            inputs['input_ids'] = inputs['input_ids'].to(self.args.device)
            inputs['attention_mask'] = inputs['attention_mask'].to(self.args.device)
            inputs['token_type_ids'] = inputs['token_type_ids'].to(self.args.device)

            if self.args.model_type == 'zen':
                inputs['input_ngram_ids'] = inputs['input_ngram_ids'].to(self.args.device)
                inputs['ngram_position_matrix'] = inputs['ngram_position_matrix'].to(self.args.device)
                inputs['ngram_attention_mask'] = inputs['ngram_attention_mask'].to(self.args.device)
                inputs['ngram_token_type_ids'] = inputs['ngram_token_type_ids'].to(self.args.device)

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs

            if preds is None:
                preds = logits.detach().cpu().numpy()
                labels = label.cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu(), axis=0)
                labels = np.append(labels, label.detach().cpu().numpy(), axis=0)

            pbar(step, info="")

        preds = np.argmax(preds, axis=1)

        p, r, f1, _ = cdn_cls_metric(preds, labels)
        logger.info("%s-%s precision: %s - recall: %s - f1 score: %s", args.task_name, args.model_name, p, r, f1)
        return f1

    def predict(self, test_dataset, model):
        args = self.args
        logger = self.logger
        test_dataset.text1 = test_dataset.text1
        test_dataset.text2 = test_dataset.text2
        test_dataloader = self.get_test_dataloader(test_dataset)
        num_examples = len(test_dataloader.dataset)
        model.to(args.device)

        preds = None

        logger.info("***** Running prediction *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(test_dataloader), desc='Evaluation')
        for step, item in enumerate(test_dataloader):
            model.eval()

            text1 = item[0]
            text2 = item[1]

            if self.args.model_type == 'zen':
                inputs = convert_examples_to_features(text1=text1, text2=text2, ngram_dict=self.ngram_dict,
                                                      tokenizer=self.tokenizer, max_seq_length=self.args.max_length,
                                                      return_tensors=True)
            else:
                inputs = self.tokenizer(text1, text2, return_tensors='pt', padding='max_length',
                                        truncation='longest_first', max_length=self.args.max_length)

            inputs['input_ids'] = inputs['input_ids'].to(self.args.device)
            inputs['attention_mask'] = inputs['attention_mask'].to(self.args.device)
            inputs['token_type_ids'] = inputs['token_type_ids'].to(self.args.device)

            if self.args.model_type == 'zen':
                inputs['input_ngram_ids'] = inputs['input_ngram_ids'].to(self.args.device)
                inputs['ngram_position_matrix'] = inputs['ngram_position_matrix'].to(self.args.device)
                inputs['ngram_attention_mask'] = inputs['ngram_attention_mask'].to(self.args.device)
                inputs['ngram_token_type_ids'] = inputs['ngram_token_type_ids'].to(self.args.device)

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs

            if preds is None:
                preds = logits.detach().softmax(-1)[:, 1].cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().softmax(-1)[:, 1].cpu().numpy(), axis=0)

            pbar(step, info="")

        preds = preds.reshape(len(preds) // args.recall_k, args.recall_k)
        np.save(os.path.join(args.result_output_dir, f'cdn_test_preds.npy'), preds)
        return preds

    def _save_checkpoint(self, model, step):
        output_dir = os.path.join(self.args.output_dir, 'checkpoint-{}'.format(step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torch.save(model.state_dict(), os.path.join(output_dir, 'pytorch_model.pt'))
        self.logger.info('Saving models checkpoint to %s', output_dir)
        if self.args.model_type == 'zen':
            save_zen_model(output_dir, model.encoder, self.tokenizer, self.ngram_dict, self.args)
        else:
            model.encoder.save_pretrained(output_dir)
            self.tokenizer.save_vocabulary(save_directory=output_dir)

    def _save_best_checkpoint(self, best_step):
        pass


class CDNForNUMTrainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            model_class,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(CDNForNUMTrainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
            model_class=model_class,
            ngram_dict=ngram_dict
        )

    def training_step(self, model, item):
        model.train()

        text1 = item[0]
        labels = item[1].to(self.args.device)

        if self.args.model_type == 'zen':
            inputs = convert_examples_to_features(text1=text1, ngram_dict=self.ngram_dict,
                                                  tokenizer=self.tokenizer, max_seq_length=self.args.max_length,
                                                  return_tensors=True)
        else:
            inputs = self.tokenizer(text1, padding='max_length', max_length=self.args.max_length,
                                    truncation=True, return_tensors='pt')

        inputs['input_ids'] = inputs['input_ids'].to(self.args.device)
        inputs['attention_mask'] = inputs['attention_mask'].to(self.args.device)
        inputs['token_type_ids'] = inputs['token_type_ids'].to(self.args.device)

        if self.args.model_type == 'zen':
            inputs['input_ngram_ids'] = inputs['input_ngram_ids'].to(self.args.device)
            inputs['ngram_position_matrix'] = inputs['ngram_position_matrix'].to(self.args.device)
            inputs['ngram_attention_mask'] = inputs['ngram_attention_mask'].to(self.args.device)
            inputs['ngram_token_type_ids'] = inputs['ngram_token_type_ids'].to(self.args.device)

        outputs = model(labels=labels, **inputs)
        loss = outputs[0]
        loss.backward()

        return loss.detach()

    def evaluate(self, model):
        args = self.args
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)

        preds = None
        eval_labels = None

        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        for step, item in enumerate(eval_dataloader):
            model.eval()

            text1 = item[0]
            labels = item[1].to(args.device)
            if self.args.model_type == 'zen':
                inputs = convert_examples_to_features(text1=text1, ngram_dict=self.ngram_dict,
                                                      tokenizer=self.tokenizer, max_seq_length=self.args.max_length,
                                                      return_tensors=True)
            else:
                inputs = self.tokenizer(text1, padding='max_length', max_length=self.args.max_length,
                                        truncation=True, return_tensors='pt')
            inputs['input_ids'] = inputs['input_ids'].to(self.args.device)
            inputs['attention_mask'] = inputs['attention_mask'].to(self.args.device)
            inputs['token_type_ids'] = inputs['token_type_ids'].to(self.args.device)

            if self.args.model_type == 'zen':
                inputs['input_ngram_ids'] = inputs['input_ngram_ids'].to(self.args.device)
                inputs['ngram_position_matrix'] = inputs['ngram_position_matrix'].to(self.args.device)
                inputs['ngram_attention_mask'] = inputs['ngram_attention_mask'].to(self.args.device)
                inputs['ngram_token_type_ids'] = inputs['ngram_token_type_ids'].to(self.args.device)

            with torch.no_grad():
                outputs = model(labels=labels, **inputs)
                loss, logits = outputs[:2]

            if preds is None:
                preds = logits.detach().cpu().numpy()
                eval_labels = labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                eval_labels = np.append(eval_labels, labels.detach().cpu().numpy(), axis=0)

        preds = np.argmax(preds, axis=1)
        p, r, f1, _ = cdn_num_metric(preds, eval_labels)
        logger.info("%s-%s f1: %s", args.task_name, args.model_name, f1)
        return f1

    def predict(self, model, test_dataset, orig_texts, cls_preds, recall_labels, recall_scores):
        args = self.args
        logger = self.logger
        test_dataloader = self.get_test_dataloader(test_dataset)
        num_examples = len(test_dataloader.dataset)
        model.to(args.device)

        preds = None

        logger.info("***** Running prediction *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(test_dataloader), desc='Evaluation')
        for step, item in enumerate(test_dataloader):
            model.eval()

            text1 = item

            if self.args.model_type == 'zen':
                inputs = convert_examples_to_features(text1=text1, ngram_dict=self.ngram_dict,
                                                      tokenizer=self.tokenizer, max_seq_length=self.args.max_length,
                                                      return_tensors=True)
            else:
                inputs = self.tokenizer(text1, padding='max_length', max_length=self.args.max_length,
                                        truncation=True, return_tensors='pt')

            inputs['input_ids'] = inputs['input_ids'].to(self.args.device)
            inputs['attention_mask'] = inputs['attention_mask'].to(self.args.device)
            inputs['token_type_ids'] = inputs['token_type_ids'].to(self.args.device)

            if self.args.model_type == 'zen':
                inputs['input_ngram_ids'] = inputs['input_ngram_ids'].to(self.args.device)
                inputs['ngram_position_matrix'] = inputs['ngram_position_matrix'].to(self.args.device)
                inputs['ngram_attention_mask'] = inputs['ngram_attention_mask'].to(self.args.device)
                inputs['ngram_token_type_ids'] = inputs['ngram_token_type_ids'].to(self.args.device)

            with torch.no_grad():
                outputs = model(**inputs)

                if self.args.model_type == 'zen':
                    logits = outputs
                else:
                    logits = outputs[0]

            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)

            pbar(step, info="")
        preds = np.argmax(preds, axis=1)

        recall_labels = np.array(recall_labels['recall_label'])
        recall_scores = recall_scores
        cdn_commit_prediction(orig_texts, cls_preds, preds, recall_labels, recall_scores,
                              args.result_output_dir, self.data_processor.id2label)

    def _save_checkpoint(self, model, step):
        output_dir = os.path.join(self.args.output_dir, 'checkpoint-{}'.format(step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if self.args.model_type == 'zen':
            save_zen_model(output_dir, model=model, tokenizer=self.tokenizer,
                           ngram_dict=self.ngram_dict, args=self.args)
        else:
            model.save_pretrained(output_dir)
            torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))
            self.tokenizer.save_vocabulary(save_directory=output_dir)
        self.logger.info('Saving models checkpoint to %s', output_dir)

    def _save_best_checkpoint(self, best_step):
        model = self.model_class.from_pretrained(os.path.join(self.args.output_dir, f'checkpoint-{best_step}'),
                                                 num_labels=self.data_processor.num_labels_num)
        if not os.path.exists(os.path.join(self.args.output_dir, 'num')):
            os.mkdir(os.path.join(self.args.output_dir, 'num'))

        if self.args.model_type == 'zen':
            save_zen_model(os.path.join(self.args.output_dir, 'num'), model=model, tokenizer=self.tokenizer,
                           ngram_dict=self.ngram_dict, args=self.args)
        else:
            model.save_pretrained(os.path.join(self.args.output_dir, 'num'))
            torch.save(self.args, os.path.join(os.path.join(self.args.output_dir, 'num'), 'training_args.bin'))
            self.tokenizer.save_vocabulary(save_directory=os.path.join(self.args.output_dir, 'num'))
        self.logger.info('Saving models checkpoint to %s', os.path.join(self.args.output_dir, 'num'))
