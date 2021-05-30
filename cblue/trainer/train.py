import os
import numpy as np
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader

from cblue.utils import seed_everything, ProgressBar
from cblue.metrics import sts_metric


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

        num_training_steps = len(train_dataloader) * args.epoch
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
        for i in range(args.epoch):
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
                    score = self.evaluate(model)
                    if score > best_score:
                        best_score = score
                        best_step = global_step
                        cnt_patience = 0
                        self._save_checkpoint(model, global_step)
                    else:
                        cnt_patience += 1
                        self.logger.info("Earlystopper counter: %s out of %s", cnt_patience, args.patience)
                        if cnt_patience >= self.args.patience:
                            break
            if cnt_patience >= args.patience:
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

        inputs = self.tokenizer(text1, text2, return_tentors='pt', padding='max_length',
                                truncation='longest_first', max_length=self.args.max_length)
        inputs['input_ids'] = inputs['input_ids'].to(self.args.device)
        inputs['attention_mask'] = inputs['attention_mask'].to(self.args.device)
        inputs['token_type_ids'] = inputs['token_type_ids'].to(self.args.device)

        # default using 'Transformers' library model.
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

        logger.info("***** Running training *****")
        logger.info("Num samples %d", num_examples)
        for step, item in enumerate(eval_dataloader):
            model.eval()

            text1 = item[0]
            text2 = item[1]
            labels = item[2].to(args.device)

            inputs = self.tokenizer(text1, text2, return_tentors='pt', padding='max_length',
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
        acc = sts_metric(preds, eval_labels)
        logger.info("%s-%s acc: %s", 'sts', args.model_name, acc)
        return acc

    def _save_checkpoint(self, model, step):
        output_dir = os.path.join(self.args.output_dir, 'checkpoint-{}'.format(step))
        if not os.path.join(output_dir):
            os.makedirs(output_dir)
        model.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))
        self.logger.info('Saving model checkpoint to %s', output_dir)
        self.tokenizer.save_vocabulary(save_directory=output_dir)
