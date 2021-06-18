import os
import sys
sys.path.append('.')
import argparse
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AlbertForSequenceClassification, \
    BertForTokenClassification, AlbertForTokenClassification

from cblue.data import STSDataProcessor, STSDataset, QICDataset, QICDataProcessor, QQRDataset, \
    QQRDataProcessor, QTRDataset, QTRDataProcessor, CTCDataset, CTCDataProcessor, EEDataset, EEDataProcessor
from cblue.trainer import STSTrainer, QICTrainer, QQRTrainer, QTRTrainer, CTCTrainer, EETrainer
from cblue.utils import init_logger, seed_everything
from cblue.models import ZenConfig, ZenNgramDict, ZenForSequenceClassification, ZenForTokenClassification


TASK_DATASET_CLASS = {
    'ee': (EEDataset, EEDataProcessor),
    'ctc': (CTCDataset, CTCDataProcessor),
    'sts': (STSDataset, STSDataProcessor),
    'qqr': (QQRDataset, QQRDataProcessor),
    'qtr': (QTRDataset, QTRDataProcessor),
    'qic': (QICDataset, QICDataProcessor)
}

TASK_TRAINER = {
    'ee': EETrainer,
    'ctc': CTCTrainer,
    'sts': STSTrainer,
    'qic': QICTrainer,
    'qqr': QQRTrainer,
    'qtr': QTRTrainer
}

MODEL_CLASS = {
    'bert': (BertTokenizer, BertForSequenceClassification),
    'roberta': (BertTokenizer, BertForSequenceClassification),
    'albert': (BertTokenizer, AlbertForSequenceClassification),
    'zen': (BertTokenizer, ZenForSequenceClassification)
}

TOKEN_MODEL_CLASS = {
    'bert': (BertTokenizer, BertForTokenClassification),
    'roberta': (BertTokenizer, BertForTokenClassification),
    'albert': (BertTokenizer, AlbertForTokenClassification),
    'zen': (BertTokenizer, ZenForTokenClassification)
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

    if not os.path.exists(args.result_output_dir):
        os.mkdir(args.result_output_dir)

    logger = init_logger(os.path.join(args.output_dir, f'{args.task_name}_{args.model_name}.log'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    seed_everything(args.seed)

    if 'albert' in args.model_name:
        args.model_type = 'albert'

    tokenizer_class, model_class = MODEL_CLASS[args.model_type]
    dataset_class, data_processor_class = TASK_DATASET_CLASS[args.task_name]
    trainer_class = TASK_TRAINER[args.task_name]

    if args.task_name == 'ee':
        tokenizer_class, model_class = TOKEN_MODEL_CLASS[args.model_type]

    logger.info("Training/evaluation parameters %s", args)
    if args.do_train:
        tokenizer = tokenizer_class.from_pretrained(os.path.join(args.model_dir, args.model_name))

        # compatible with 'ZEN' model
        ngram_dict = None
        if args.model_type == 'zen':
            ngram_dict = ZenNgramDict(os.path.join(args.model_dir, args.model_name), tokenizer=tokenizer)

        data_processor = data_processor_class(root=args.data_dir)
        train_samples = data_processor.get_train_sample()
        eval_samples = data_processor.get_dev_sample()

        if args.task_name == 'ee' or args.task_name == 'ctc':
            train_dataset = dataset_class(train_samples, data_processor, tokenizer, mode='train',
                                          model_type=args.model_type, ngram_dict=ngram_dict, max_length=args.max_length)
            eval_dataset = dataset_class(eval_samples, data_processor, tokenizer, mode='eval',
                                         model_type=args.model_type, ngram_dict=ngram_dict, max_length=args.max_length)
        else:
            train_dataset = dataset_class(train_samples, data_processor, mode='train')
            eval_dataset = dataset_class(eval_samples, data_processor, mode='eval')

        model = model_class.from_pretrained(os.path.join(args.model_dir, args.model_name),
                                            num_labels=data_processor.num_labels)

        trainer = trainer_class(args=args, model=model, data_processor=data_processor,
                                tokenizer=tokenizer, train_dataset=train_dataset, eval_dataset=eval_dataset,
                                logger=logger, model_class=model_class, ngram_dict=ngram_dict)

        global_step, best_step = trainer.train()

    if args.do_predict:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)

        ngram_dict = None
        if args.model_type == 'zen':
            ngram_dict = ZenNgramDict(os.path.join(args.model_dir, args.model_name), tokenizer=tokenizer)

        data_processor = data_processor_class(root=args.data_dir)
        test_samples = data_processor.get_test_sample()

        if args.task_name == 'ee' or args.task_name == 'ctc':
            test_dataset = dataset_class(test_samples, data_processor, tokenizer, mode='test', ngram_dict=ngram_dict,
                                         max_length=args.max_length, model_type=args.model_type)
        else:
            test_dataset = dataset_class(test_samples, data_processor, mode='test')
            
        model = model_class.from_pretrained(args.output_dir, num_labels=data_processor.num_labels)
        trainer = trainer_class(args=args, model=model, data_processor=data_processor,
                                tokenizer=tokenizer, logger=logger, model_class=model_class, ngram_dict=ngram_dict)
        trainer.predict(test_dataset=test_dataset, model=model)


if __name__ == '__main__':
    main()
