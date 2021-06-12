import os
import sys
sys.path.append('.')
import argparse
import torch
import numpy as np
from transformers import BertTokenizer, BertModel, AlbertModel, BertForSequenceClassification, \
    AlbertForSequenceClassification

from cblue.models import CDNForCLSModel
from cblue.trainer import CDNForCLSTrainer, CDNForNUMTrainer
from cblue.utils import init_logger, seed_everything
from cblue.data import CDNDataset, CDNDataProcessor
from cblue.models import save_zen_model, ZenModel, ZenForSequenceClassification, ZenNgramDict


MODEL_CLASS = {
    'bert': (BertTokenizer, BertModel),
    'roberta': (BertTokenizer, BertModel),
    'albert': (BertTokenizer, AlbertModel),
    'zen': (BertTokenizer, ZenModel)
}

CLS_MODEL_CLASS = {
    'bert': BertForSequenceClassification,
    'roberta': BertForSequenceClassification,
    'albert': AlbertForSequenceClassification,
    'zen': ZenForSequenceClassification
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The task data directory.")
    parser.add_argument("--model_dir", default=None, type=str, required=True,
                        help="The directory of pretrained models")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="The type of selected pretrained models.")
    parser.add_argument("--model_name", default=None, type=str, required=True,
                        help="The path of selected pretrained models. (e.g. chinese-bert-wwm)")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of task to train")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The path of result data and models to be saved.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_predict", action='store_true',
                        help="Whether to run the models in inference mode on the test set.")
    parser.add_argument("--result_output_dir", default=None, type=str, required=True,
                        help="the directory of commit result to be saved")

    # For CDN task
    parser.add_argument("--recall_k", default=200, type=int,
                        help="the number of samples to be recalled.")
    parser.add_argument("--num_neg", default=3, type=int,
                        help="the number of negative samples to be sampled")
    parser.add_argument("--do_aug", default=1, type=int,
                        help="whether do data augment.")

    # models param
    parser.add_argument("--max_length", default=128, type=int,
                        help="the max length of sentence.")
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for, "
                             "E.g., 0.1 = 10% of training.")
    parser.add_argument("--earlystop_patience", default=2, type=int,
                        help="The patience of early stop")

    parser.add_argument('--logging_steps', type=int, default=10,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=1000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--seed', type=int, default=2021,
                        help="random seed for initialization")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = os.path.join(args.output_dir, args.task_name)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = os.path.join(args.output_dir, args.model_name)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    logger = init_logger(os.path.join(args.output_dir, f'{args.task_name}_{args.model_name}.log'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    seed_everything(args.seed)

    if 'albert' in args.model_name:
        args.model_type = 'albert'

    tokenizer_class, model_class = MODEL_CLASS[args.model_type]

    if args.do_train:
        logger.info('Training CLS model...')
        tokenizer = tokenizer_class.from_pretrained(os.path.join(args.model_dir, args.model_name))

        ngram_dict = None
        if args.model_type == 'zen':
            ngram_dict = ZenNgramDict(os.path.join(args.model_dir, args.model_name), tokenizer=tokenizer)

        data_processor = CDNDataProcessor(root=args.data_dir, recall_k=args.recall_k,
                                          negative_sample=args.num_neg)
        train_samples, recall_orig_train_samples, recall_orig_train_samples_scores = data_processor.get_train_sample(dtype='cls', do_augment=args.do_aug)
        eval_samples, recall_orig_eval_samples, recall_orig_train_samples_scores = data_processor.get_dev_sample(dtype='cls', do_augment=args.do_aug)
        if data_processor.recall:
            logger.info('first recall score: %s', data_processor.recall)

        train_dataset = CDNDataset(train_samples, data_processor, dtype='cls', mode='train')
        eval_dataset = CDNDataset(eval_samples, data_processor, dtype='cls', mode='eval')

        model = CDNForCLSModel(model_class, encoder_path=os.path.join(args.model_dir, args.model_name),
                               num_labels=data_processor.num_labels_cls)
        cls_model_class = CLS_MODEL_CLASS[args.model_type]
        trainer = CDNForCLSTrainer(args=args, model=model, data_processor=data_processor,
                                   tokenizer=tokenizer, train_dataset=train_dataset, eval_dataset=eval_dataset,
                                   logger=logger, recall_orig_eval_samples=recall_orig_eval_samples,
                                   model_class=cls_model_class, recall_orig_eval_samples_scores=recall_orig_train_samples_scores,
                                   ngram_dict=ngram_dict)

        global_step, best_step = trainer.train()

        model = CDNForCLSModel(model_class, encoder_path=os.path.join(args.output_dir, f'checkpoint-{best_step}'),
                               num_labels=data_processor.num_labels_cls)
        model.load_state_dict(torch.load(os.path.join(args.output_dir, f'checkpoint-{best_step}', 'pytorch_model.pt')))
        tokenizer = tokenizer_class.from_pretrained(os.path.join(args.output_dir, f'checkpoint-{best_step}'))
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'pytorch_model_cls.pt'))
        if not os.path.exists(os.path.join(args.output_dir, 'cls')):
            os.mkdir(os.path.join(args.output_dir, 'cls'))

        if args.model_type == 'zen':
            save_zen_model(os.path.join(args.output_dir, 'cls'), model.encoder, tokenizer, ngram_dict, args)
        else:
            model.encoder.save_pretrained(os.path.join(args.output_dir, 'cls'))

        tokenizer.save_vocabulary(save_directory=os.path.join(args.output_dir, 'cls'))
        logger.info('Saving models checkpoint to %s', os.path.join(args.output_dir, 'cls'))

        logger.info('Training NUM model...')
        args.logging_steps = 30
        args.save_steps = 30
        train_samples = data_processor.get_train_sample(dtype='num', do_augment=1)
        eval_samples = data_processor.get_dev_sample(dtype='num')
        train_dataset = CDNDataset(train_samples, data_processor, dtype='num', mode='train')
        eval_dataset = CDNDataset(eval_samples, data_processor, dtype='num', mode='eval')

        cls_model_class = CLS_MODEL_CLASS[args.model_type]
        model = cls_model_class.from_pretrained(os.path.join(args.model_dir, args.model_name),
                                                num_labels=data_processor.num_labels_num)
        trainer = CDNForNUMTrainer(args=args, model=model, data_processor=data_processor,
                                   tokenizer=tokenizer, train_dataset=train_dataset, eval_dataset=eval_dataset,
                                   logger=logger, model_class=cls_model_class, ngram_dict=ngram_dict)

        global_step, best_step = trainer.train()

    if args.do_predict:
        tokenizer = tokenizer_class.from_pretrained(os.path.join(args.output_dir, 'cls'))

        ngram_dict = None
        if args.model_type == 'zen':
            ngram_dict = ZenNgramDict(os.path.join(args.model_dir, args.model_name), tokenizer=tokenizer)

        data_processor = CDNDataProcessor(root=args.data_dir, recall_k=args.recall_k,
                                          negative_sample=args.num_neg)
        test_samples, recall_orig_test_samples, recall_orig_test_samples_scores = data_processor.get_test_sample(dtype='cls')

        test_dataset = CDNDataset(test_samples, data_processor, dtype='cls', mode='test')
        cls_model_class = CLS_MODEL_CLASS[args.model_type]

        model = CDNForCLSModel(model_class, encoder_path=os.path.join(args.output_dir, 'cls'),
                               num_labels=data_processor.num_labels_cls)
        model.load_state_dict(torch.load(os.path.join(args.output_dir, 'pytorch_model_cls.pt')))
        trainer = CDNForCLSTrainer(args=args, model=model, data_processor=data_processor,
                                   tokenizer=tokenizer, logger=logger,
                                   recall_orig_eval_samples=recall_orig_test_samples,
                                   model_class=cls_model_class, ngram_dict=ngram_dict)
        cls_preds = trainer.predict(test_dataset, model)

        # cls_preds = np.load(os.path.join(args.result_output_dir, 'cdn_test_preds.npy'))

        test_samples = data_processor.get_test_sample(dtype='num')
        orig_texts = data_processor.get_test_orig_text()
        test_dataset = CDNDataset(test_samples, data_processor, dtype='num', mode='test')
        model = cls_model_class.from_pretrained(os.path.join(args.output_dir, 'num'),
                                                num_labels=data_processor.num_labels_num)
        trainer = CDNForNUMTrainer(args=args, model=model, data_processor=data_processor,
                                   tokenizer=tokenizer, logger=logger,
                                   model_class=cls_model_class, ngram_dict=ngram_dict)
        trainer.predict(model, test_dataset, orig_texts, cls_preds, recall_orig_test_samples,
                        recall_orig_test_samples_scores)


if __name__ == '__main__':
    main()
