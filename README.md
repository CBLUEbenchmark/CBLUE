[**English**](https://github.com/CBLUEbenchmark/CBLUE) | [**中文说明**](https://github.com/CBLUEbenchmark/CBLUE/blob/main/README_ZH.md) 

# CBLUE

AI (Artificial Intelligence) is playing an  indispensabe role in the biomedical field, helping improve medical technology. For further accelerating AI research in the biomedical field, we present **Chinese Biomedical Language Understanding Evaluation** (CBLUE), including datasets collected from real-world biomedical scenarios, baseline models,  and an online  platform for model evaluation, comparison and analysis.

## CBLUE Benchmark

We evaluate the current 11 Chinese pre-trained models on the eight biomedical language understanding tasks and report the baselines of these tasks.

| Model                                                        |  CMedEE  | CMedIE |   CDN    |   CTC    |   STS    |   QIC    |   QTR    |   QQR    | Avg. |
| ------------------------------------------------------------ | :------: | :----: | :------: | :------: | :------: | :------: | :------: | :------: | :--: |
| [BERT-base](https://github.com/ymcui/Chinese-BERT-wwm)       |   62.1   |  54.0  |   55.4   |   69.2   |   83.0   |   84.3   |   60.0   | **84.7** | 69.0 |
| [BERT-wwm-ext-base](https://github.com/ymcui/Chinese-BERT-wwm) |   61.7   |  54.0  |   55.4   |   70.1   |   83.9   |   84.5   |   60.9   |   84.4   | 69.4 |
| [ALBERT-tiny](https://github.com/brightmart/albert_zh)       |   50.5   |  30.4  |   50.2   |   45.4   |   79.7   |   75.8   |   55.5   |   79.8   | 58.4 |
| [ALBERT-xxlarge](https://huggingface.co/voidful/albert_chinese_xxlarge) |   61.8   |  47.6  |   37.5   |   58.6   |   84.8   |   84.8   |   62.2   |   83.1   | 65.2 |
| [RoBERTa-large](https://github.com/brightmart/roberta_zh)    |   62.1   |  54.4  |   56.5   | **70.9** |   84.7   |   84.2   |   60.9   |   82.9   | 69.6 |
| [RoBERTa-wwm-ext-base](https://github.com/ymcui/Chinese-BERT-wwm) |   62.4   |  53.7  |   56.4   |   69.4   |   83.7   | **85.5** |   60.3   |   82.7   | 69.3 |
| [RoBERTa-wwm-ext-large](https://github.com/ymcui/Chinese-BERT-wwm) |   61.8   |  55.9  |   55.7   |   69.0   |   85.2   |   85.3   |   62.8   |   84.4   | 70.0 |
| [PCL-MedBERT](https://code.ihub.org.cn/projects/1775)        |   60.6   |  49.1  |   55.8   |   67.8   |   83.8   |   84.3   |   59.3   |   82.5   | 67.9 |
| [ZEN](https://github.com/sinovation/ZEN)                     |   61.0   |  50.1  |   57.8   |   68.6   |   83.5   |   83.2   |   60.3   |   83.0   | 68.4 |
| [MacBERT-base](https://huggingface.co/hfl/chinese-macbert-base) |   60.7   |  53.2  |   57.7   |   67.7   |   84.4   |   84.9   |   59.7   |   84.0   | 69.0 |
| [MacBERT-large](https://huggingface.co/hfl/chinese-macbert-large) | **62.4** |  51.6  | **59.3** |   68.6   | **85.6** |   82.7   | **62.9** |   83.5   | 69.6 |
| Human                                                        |   67.0   |  66.0  |   65.0   |   78.0   |   93.0   |   88.0   |   71.0   |   89.0   | 77.1 |

## Baseline of tasks

We present the baseline models on the biomedical tasks and release corresponding codes for quick start.

#### Requirements

python3 / pytorch 1.7 / transformers 4.5.1 / jieba / gensim 

#### Data preparation

[Download dataset](https://tianchi.aliyun.com/dataset/dataDetail?dataId=95414)

The whole zip package includes the datasets of  8 biomedical NLU tasks (more detail in the following section). Every task includes the following files:

```text
├── {Task}
|  └── {Task}_train.json
|  └── {Task}_test.json
|  └── {Task}_dev.json
|  └── example_gold.json
|  └── example_pred.json
|  └── README.md
```

**Notice: a few tasks have additional files, e.g. it includes 'category.xlsx' file in the CHIP-CTC task.** 

You can download Chinese pre-trained models according to your need (download URLs are provided above). With [Huggingface-Transformers](https://huggingface.co/) , the models above could be easily accessed and loaded.

The reference directory:

```text
├── CBLUE         
|  └── baselines
|     └── run_classifier.py
|     └── ...
|  └── examples
|     └── run_qqr.sh
|     └── ...
|  └── cblue
|  └── CBLUEDatasets
|     └── KUAKE-QQR
|     └── ...
|  └── data
|     └── output
|     └── model_data
|        └── bert-base
|        └── ...
|     └── result_output
|        └── KUAKE-QQR_test.json
|        └── ...
```

#### Running examples

The shell files of training and evaluation for every task are provided in `examples/` , and could directly run.

Also, you can utilize the running codes in `baselines/` , and write your own shell files according to your need:

- `baselines/run_classifer.py`: support `{sts, qqr, qtr, qic, ctc, ee}` tasks;
- `baselines/run_cdn.py`: support `{cdn}` task;
- `baselines/run_ie.py`: support `{ie}` task.

**Training models**

Running shell files: `bash examples/run_{task}.sh`, and the contents of shell files are as follow:

```shell
DATA_DIR="CBLUEDatasets"

TASK_NAME="qqr"
MODEL_TYPE="bert"
MODEL_DIR="data/model_data"
MODEL_NAME="chinese-bert-wwm"
OUTPUT_DIR="data/output"
RESULT_OUTPUT_DIR="data/result_output"

MAX_LENGTH=128

python baselines/run_classifier.py \
    --data_dir=${DATA_DIR} \
    --model_type=${MODEL_TYPE} \
    --model_dir=${MODEL_DIR} \
    --model_name=${MODEL_NAME} \
    --task_name=${TASK_NAME} \
    --output_dir=${OUTPUT_DIR} \
    --result_output_dir=${RESULT_OUTPUT_DIR} \
    --do_train \
    --max_length=${MAX_LENGTH} \
    --train_batch_size=16 \
    --eval_batch_size=16 \
    --learning_rate=3e-5 \
    --epochs=3 \
    --warmup_proportion=0.1 \
    --earlystop_patience=3 \
    --logging_steps=250 \
    --save_steps=250 \
    --seed=2021
```

**Notice: the best checkpoint is saved in** `OUTPUT_DIR/MODEL_NAME/`.

- `MODEL_TYPE`: support `{bert, roberta, albert, zen}` model types;
- `MODEL_NAME`: support `{bert-base, bert-wwm-ext, albert-tiny, albert-xxlarge, zen, pcl-medbert, roberta-large, roberta-wwm-ext-base, roberta-wwm-ext-large, macbert-base, macbert-large}` Chinese pre-trained models.

The `MODEL_TYPE`-`MODEL_NAME` mappings are listed below.

| MODEL_TYPE | MODEL_NAME                                                   |
| :--------: | :----------------------------------------------------------- |
|   `bert`   | `bert-base`, `bert-wwm-ext`, `pcl-medbert`, `macbert-base`, `macbert-large` |
| `roberta`  | `roberta-large`, `roberta-wwm-ext-base`, `roberta-wwm-ext-large` |
|  `albert`  | `albert-tiny`, `albert-xxlarge`                              |
|   `zen`    | `zen`                                                        |

**Inference & generation of results**

Running shell files: `base examples/run_{task}.sh predict`, and the contents of shell files are as follows:

```shell
DATA_DIR="CBLUEDatasets"

TASK_NAME="qqr"
MODEL_TYPE="bert"
MODEL_DIR="data/model_data"
MODEL_NAME="chinese-bert-wwm"
OUTPUT_DIR="data/output"
RESULT_OUTPUT_DIR="data/result_output"

MAX_LENGTH=128

python baselines/run_classifier.py \
    --data_dir=${DATA_DIR} \
    --model_type=${MODEL_TYPE} \
    --model_name=${MODEL_NAME} \
    --model_dir=${MODEL_DIR} \
    --task_name=${TASK_NAME} \
    --output_dir=${OUTPUT_DIR} \
    --result_output_dir=${RESULT_OUTPUT_DIR} \
    --do_predict \
    --max_length=${MAX_LENGTH} \
    --eval_batch_size=16 \
    --seed=2021
```

**Notice: the result of prediction** `{TASK_NAME}_test.json` **will be generated in** `RESULT_OUTPUT_DIR` .

#### Commit results

Compressing `RESULT_OUTPUT_DIR` as `.zip` file and committing the file, you will get the score of evaluation on these biomedical NLU tasks, and your ranking! [Commit your results!](https://tianchi.aliyun.com/dataset/dataDetail?dataId=95414)

![commit](resources/img/commit.png)

## Introduction of tasks

## Training setup

**Unified hyper-parameters**

|       Param       | Value |
| :---------------: | :---: |
| warmup_proportion |  0.1  |
|   weight_decay    | 0.01  |
|   adam_epsilon    | 1e-8  |
|   max_grad_norm   |  1.0  |

**CMeEE**

| Model                 | epoch | batch_size | max_length | learning_rate |
| --------------------- | :---: | :--------: | :--------: | :-----------: |
| bert-base             |   5   |     32     |    128     |     4e-5      |
| bert-wwm-ext          |   5   |     32     |    128     |     4e-5      |
| roberta-wwm-ext       |   5   |     32     |    128     |     4e-5      |
| roberta-wwm-ext-large |   5   |     12     |     65     |     2e-5      |
| roberta-large         |   5   |     12     |     65     |     2e-5      |
| albert-tiny           |  10   |     32     |    128     |     5e-5      |
| albert-xxlarge        |   5   |     12     |     65     |     1e-5      |
| PCL-MedBERT           |   5   |     32     |    128     |     4e-5      |

**CMeIE-ER**

| Model                 | epoch | batch_size | max_length | learning_rate |
| --------------------- | :---: | :--------: | :--------: | :-----------: |
| bert-base             |   7   |     32     |    128     |     5e-5      |
| bert-wwm-ext          |   7   |     32     |    128     |     5e-5      |
| roberta-wwm-ext       |   7   |     32     |    128     |     4e-5      |
| roberta-wwm-ext-large |   7   |     16     |     80     |     4e-5      |
| roberta-large         |   7   |     16     |     80     |     2e-5      |
| albert-tiny           |  10   |     32     |    128     |     4e-5      |
| albert-xxlarge        |   7   |     16     |     80     |     1e-5      |
| PCL-MedBERT           |   7   |     32     |    128     |     4e-5      |

**CMeIE-RE**

| Model                 | epoch | batch_size | max_length | learning_rate |
| --------------------- | :---: | :--------: | :--------: | :-----------: |
| bert-base             |   8   |     32     |    128     |     5e-5      |
| bert-wwm-ext          |   8   |     32     |    128     |     5e-5      |
| roberta-wwm-ext       |   8   |     32     |    128     |     4e-5      |
| roberta-wwm-ext-large |   8   |     16     |     80     |     4e-5      |
| roberta-large         |   8   |     16     |     80     |     2e-5      |
| albert-tiny           |  10   |     32     |    128     |     4e-5      |
| albert-xxlarge        |   8   |     16     |     80     |     1e-5      |
| PCL-MedBERT           |   8   |     32     |    128     |     4e-5      |

**CHIP-CTC**

| Model                 | epoch | batch_size | max_length | learning_rate |
| --------------------- | :---: | :--------: | :--------: | :-----------: |
| bert-base             |   5   |     32     |    128     |     5e-5      |
| bert-wwm-ext          |   5   |     32     |    128     |     5e-5      |
| roberta-wwm-ext       |   5   |     32     |    128     |     4e-5      |
| roberta-wwm-ext-large |   5   |     20     |     50     |     3e-5      |
| roberta-large         |   5   |     20     |     50     |     4e-5      |
| albert-tiny           |  10   |     32     |    128     |     4e-5      |
| albert-xxlarge        |   5   |     20     |     50     |     1e-5      |
| PCL-MedBERT           |   5   |     32     |    128     |     4e-5      |

**CHIP-CDN-cls**

| Param               | Value |
| ------------------- | ----- |
| recall_k            | 200   |
| num_negative_sample | 10    |

| Model                 | epoch | batch_size | max_length | learning_rate |
| --------------------- | :---: | :--------: | :--------: | :-----------: |
| bert-base             |   3   |     32     |    128     |     4e-5      |
| bert-wwm-ext          |   3   |     32     |    128     |     5e-5      |
| roberta-wwm-ext       |   3   |     32     |    128     |     4e-5      |
| roberta-wwm-ext-large |   3   |     32     |     40     |     4e-5      |
| roberta-large         |   3   |     32     |     40     |     4e-5      |
| albert-tiny           |   3   |     32     |    128     |     4e-5      |
| albert-xxlarge        |   3   |     32     |     40     |     1e-5      |
| PCL-MedBERT           |   3   |     32     |    128     |     4e-5      |

**CHIP-CDN-num**

| Model                 | epoch | batch_size | max_length | learning_rate |
| --------------------- | :---: | :--------: | :--------: | :-----------: |
| bert-base             |  20   |     32     |    128     |     4e-5      |
| bert-wwm-ext          |  20   |     32     |    128     |     5e-5      |
| roberta-wwm-ext       |  20   |     32     |    128     |     4e-5      |
| roberta-wwm-ext-large |  20   |     12     |     40     |     4e-5      |
| roberta-large         |  20   |     12     |     40     |     4e-5      |
| albert-tiny           |  20   |     32     |    128     |     4e-5      |
| albert-xxlarge        |  20   |     12     |     40     |     1e-5      |
| PCL-MedBERT           |  20   |     32     |    128     |     4e-5      |

**CHIP-STS**

| Model                 | epoch | batch_size | max_length | learning_rate |
| --------------------- | :---: | :--------: | :--------: | :-----------: |
| bert-base             |   3   |     16     |     40     |     3e-5      |
| bert-wwm-ext          |   3   |     16     |     40     |     3e-5      |
| roberta-wwm-ext       |   3   |     16     |     40     |     4e-5      |
| roberta-wwm-ext-large |   3   |     16     |     40     |     4e-5      |
| roberta-large         |   3   |     16     |     40     |     2e-5      |
| albert-tiny           |   3   |     16     |     40     |     5e-5      |
| albert-xxlarge        |   3   |     16     |     40     |     1e-5      |
| PCL-MedBERT           |   3   |     16     |     40     |     2e-5      |

**KUAKE-QIC**

| Model                 | epoch | batch_size | max_length | learning_rate |
| --------------------- | :---: | :--------: | :--------: | :-----------: |
| bert-base             |   3   |     16     |     50     |     2e-5      |
| bert-wwm-ext          |   3   |     16     |     50     |     2e-5      |
| roberta-wwm-ext       |   3   |     16     |     50     |     2e-5      |
| roberta-wwm-ext-large |   3   |     16     |     50     |     2e-5      |
| roberta-large         |   3   |     16     |     50     |     3e-5      |
| albert-tiny           |   3   |     16     |     50     |     5e-5      |
| albert-xxlarge        |   3   |     16     |     50     |     1e-5      |
| PCL-MedBERT           |   3   |     16     |     50     |     2e-5      |

**KUAKE-QTR**

| Model                 | epoch | batch_size | max_length | learning_rate |
| --------------------- | :---: | :--------: | :--------: | :-----------: |
| bert-base             |   3   |     16     |     40     |     4e-5      |
| bert-wwm-ext          |   3   |     16     |     40     |     2e-5      |
| roberta-wwm-ext       |   3   |     16     |     40     |     3e-5      |
| roberta-wwm-ext-large |   3   |     16     |     40     |     2e-5      |
| roberta-large         |   3   |     16     |     40     |     2e-5      |
| albert-tiny           |   3   |     16     |     40     |     5e-5      |
| albert-xxlarge        |   3   |     16     |     40     |     1e-5      |
| PCL-MedBERT           |   3   |     16     |     40     |     3e-5      |
**KUAKE-QQR**

| Model                 | epoch | batch_size | max_length | learning_rate |
| --------------------- | :---: | :--------: | :--------: | :-----------: |
| bert-base             |   3   |     16     |     30     |     3e-5      |
| bert-wwm-ext          |   3   |     16     |     30     |     3e-5      |
| roberta-wwm-ext       |   3   |     16     |     30     |     3e-5      |
| roberta-wwm-ext-large |   3   |     16     |     30     |     3e-5      |
| roberta-large         |   3   |     16     |     30     |     2e-5      |
| albert-tiny           |   3   |     16     |     30     |     5e-5      |
| albert-xxlarge        |   3   |     16     |     30     |     3e-5      |
| PCL-MedBERT           |   3   |     16     |     30     |     2e-5      |

