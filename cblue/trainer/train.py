import os
import numpy as np
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader

from cblue.utils import seed_everything, ProgressBar
from cblue.metrics import sts_metric, qic_metric, qqr_metric, qtr_metric, \
    ctc_metric, ee_metric, er_metric, re_metric


class Trainer(object):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            train_dataset,
            eval_dataset,
            logger
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
                model.zero_grad()

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

        return global_step, best_step

    def evaluate(self, model):
        raise NotImplementedError

    def _save_checkpoint(self, model, step):
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


class EETrainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            train_dataset,
            eval_dataset,
            logger
    ):
        super(EETrainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger
        )

    def training_step(self, model, item):
        model.train()

        text1 = item[0]
        labels = item[1].to(self.args.device)

        inputs = self.tokenizer(text1, padding='max_length', max_length=self.args.max_length,
                                truncation=True, return_tensors='pt', add_special_tokens=False)
        inputs['input_ids'] = inputs['input_ids'].to(self.args.device)
        inputs['attention_mask'] = inputs['attention_mask'].to(self.args.device)
        inputs['token_type_ids'] = inputs['token_type_ids'].to(self.args.device)

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

            inputs = self.tokenizer(text1, return_tensors='pt', padding='max_length',
                                    truncation=True, max_length=self.args.max_length, add_special_tokens=False)
            inputs['input_ids'] = inputs['input_ids'].to(self.args.device)
            inputs['attention_mask'] = inputs['attention_mask'].to(self.args.device)
            inputs['token_type_ids'] = inputs['token_type_ids'].to(self.args.device)

            with torch.no_grad():
                outputs = model(labels=labels, **inputs)
                loss, logits = outputs[:2]
                active_index = inputs['attention_mask'].view(-1) == 1
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

    def _save_checkpoint(self, model, step):
        output_dir = os.path.join(self.args.output_dir, 'checkpoint-{}'.format(step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))
        self.logger.info('Saving models checkpoint to %s', output_dir)
        self.tokenizer.save_vocabulary(save_directory=output_dir)


class STSTrainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            train_dataset,
            eval_dataset,
            logger
    ):
        super(STSTrainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger
        )

    def training_step(self, model, item):
        model.train()

        text1 = item[0]
        text2 = item[1]
        labels = item[2].to(self.args.device)

        inputs = self.tokenizer(text1, text2, return_tensors='pt', padding='max_length',
                                truncation='longest_first', max_length=self.args.max_length)

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

            inputs = self.tokenizer(text1, text2, return_tensors='pt', padding='max_length',
                                    truncation='longest_first', max_length=self.args.max_length)
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
        p, r, f1, _ = sts_metric(preds, eval_labels)
        logger.info("%s-%s precision: %s - recall: %s - f1 score: %s", args.task_name, args.model_name, p, r, f1)
        return f1

    def _save_checkpoint(self, model, step):
        output_dir = os.path.join(self.args.output_dir, 'checkpoint-{}'.format(step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))
        self.logger.info('Saving models checkpoint to %s', output_dir)
        self.tokenizer.save_vocabulary(save_directory=output_dir)


class QICTrainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            train_dataset,
            eval_dataset,
            logger
    ):
        super(QICTrainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger
        )

    def training_step(self, model, item):
        model.train()

        text1 = item[0]
        labels = item[1].to(self.args.device)
        inputs = self.tokenizer(text1, padding='max_length', max_length=self.args.max_length,
                                truncation=True, return_tensors='pt')

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

            inputs = self.tokenizer(text1, return_tensors='pt', padding='max_length',
                                    truncation='longest_first', max_length=self.args.max_length)
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
        acc = qic_metric(preds, eval_labels)
        logger.info("%s-%s acc: %s", args.task_name, args.model_name, acc)
        return acc

    def _save_checkpoint(self, model, step):
        output_dir = os.path.join(self.args.output_dir, 'checkpoint-{}'.format(step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))
        self.logger.info('Saving models checkpoint to %s', output_dir)
        self.tokenizer.save_vocabulary(save_directory=output_dir)


class QQRTrainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            train_dataset,
            eval_dataset,
            logger
    ):
        super(QQRTrainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger
        )

    def training_step(self, model, item):
        model.train()

        text1 = item[0]
        text2 = item[1]
        labels = item[2].to(self.args.device)
        inputs = self.tokenizer(text1, text2, padding='max_length', max_length=self.args.max_length,
                                truncation='longest_first', return_tensors='pt')

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

            inputs = self.tokenizer(text1, text2, return_tensors='pt', padding='max_length',
                                    truncation='longest_first', max_length=self.args.max_length)
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

    def _save_checkpoint(self, model, step):
        output_dir = os.path.join(self.args.output_dir, 'checkpoint-{}'.format(step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))
        self.logger.info('Saving models checkpoint to %s', output_dir)
        self.tokenizer.save_vocabulary(save_directory=output_dir)


class QTRTrainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            train_dataset,
            eval_dataset,
            logger
    ):
        super(QTRTrainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger
        )

    def training_step(self, model, item):
        model.train()

        text1 = item[0]
        text2 = item[1]
        labels = item[2].to(self.args.device)
        inputs = self.tokenizer(text1, text2, padding='max_length', max_length=self.args.max_length,
                                truncation='longest_first', return_tensors='pt')

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

            inputs = self.tokenizer(text1, text2, return_tensors='pt', padding='max_length',
                                    truncation='longest_first', max_length=self.args.max_length)
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

    def _save_checkpoint(self, model, step):
        output_dir = os.path.join(self.args.output_dir, 'checkpoint-{}'.format(step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))
        self.logger.info('Saving models checkpoint to %s', output_dir)
        self.tokenizer.save_vocabulary(save_directory=output_dir)


class CTCTrainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            train_dataset,
            eval_dataset,
            logger
    ):
        super(CTCTrainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger
        )

    def training_step(self, model, item):
        model.train()

        text1 = item[0]
        labels = item[1].to(self.args.device)
        inputs = self.tokenizer(text1, padding='max_length', max_length=self.args.max_length,
                                truncation=True, return_tensors='pt')

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

            inputs = self.tokenizer(text1, return_tensors='pt', padding='max_length',
                                    truncation='longest_first', max_length=self.args.max_length)
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
        p, r, f1, _ = ctc_metric(preds, eval_labels)
        logger.info("%s-%s precision: %s - recall: %s - f1 score: %s", args.task_name, args.model_name, p, r, f1)
        return f1

    def _save_checkpoint(self, model, step):
        output_dir = os.path.join(self.args.output_dir, 'checkpoint-{}'.format(step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))
        self.logger.info('Saving models checkpoint to %s', output_dir)
        self.tokenizer.save_vocabulary(save_directory=output_dir)


class ERTrainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            train_dataset,
            eval_dataset,
            logger
    ):
        super(ERTrainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger
        )

        self.loss_fn = nn.BCELoss()

    def training_step(self, model, item):
        model.train()

        input_ids, token_type_ids, attention_mask, sub_start_label, sub_end_label, obj_start_label, obj_end_label = item
        input_ids = input_ids.to(self.args.device)
        token_type_ids = token_type_ids.to(self.args.device)
        attention_mask = attention_mask.to(self.args.device)
        sub_start_label = sub_start_label.to(self.args.device)
        sub_end_label = sub_end_label.to(self.args.device)
        obj_start_label = obj_start_label.to(self.args.device)
        obj_end_label = obj_end_label.to(self.args.device)

        sub_start_logits, sub_end_logits, obj_start_logits, obj_end_logits = model(input_ids, token_type_ids, attention_mask)

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

            input_ids, token_type_ids, attention_mask, sub_start_label, sub_end_label, obj_start_label, obj_end_label = item
            input_ids = input_ids.to(self.args.device)
            token_type_ids = token_type_ids.to(self.args.device)
            attention_mask = attention_mask.to(self.args.device)
            sub_start_label = sub_start_label.to(self.args.device)
            sub_end_label = sub_end_label.to(self.args.device)
            obj_start_label = obj_start_label.to(self.args.device)
            obj_end_label = obj_end_label.to(self.args.device)

            with torch.no_grad():
                sub_start_logits, sub_end_logits, obj_start_logits, obj_end_logits = model(input_ids, token_type_ids,
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

    def _save_checkpoint(self, model, step):
        output_dir = os.path.join(self.args.output_dir, 'checkpoint-{}'.format(step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torch.save(model.state_dict(), os.path.join(output_dir, 'pytorch_model.pt'))
        self.logger.info('Saving models checkpoint to %s', output_dir)


class RETrainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            train_dataset,
            eval_dataset,
            logger
    ):
        super(RETrainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger
        )

    def training_step(self, model, item):
        model.train()

        input_ids, token_type_ids, attention_mask, flag, label = item
        input_ids, token_type_ids, attention_mask, flag, label = input_ids.to(self.args.device), \
                                                                 token_type_ids.to(self.args.device), \
                                                                 attention_mask.to(self.args.device), \
                                                                 flag.to(self.args.device), label.to(self.args.device)
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

            input_ids, token_type_ids, attention_mask, flag, label = item
            input_ids, token_type_ids, attention_mask, flag, label = input_ids.to(self.args.device), \
                                                                     token_type_ids.to(self.args.device), \
                                                                     attention_mask.to(self.args.device), \
                                                                     flag.to(self.args.device), label.to(self.args.device)
            with torch.no_grad():
                loss, logits = model(input_ids, token_type_ids, attention_mask, flag, label)

            if preds is None:
                preds = logits.detach().cpu().numpy()
                eval_labels = label.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy())
                eval_labels = np.append(eval_labels, label.detach().cpu().numpy())

        preds = np.argmax(preds, axis=1)
        p, r, f1, _ = re_metric(preds, eval_labels)
        logger.info("%s-%s precision: %s - recall: %s - f1 score: %s", args.task_name, args.model_name, p, r, f1)
        return f1

    def _save_checkpoint(self, model, step):
        output_dir = os.path.join(self.args.output_dir, 'checkpoint-{}'.format(step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torch.save(model.state_dict(), os.path.join(output_dir, 'pytorch_model.pt'))
        self.logger.info('Saving models checkpoint to %s', output_dir)

