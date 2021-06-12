import os
import math
import torch
from random import shuffle
from .file_utils import WEIGHTS_NAME, CONFIG_NAME
from .ngram_utils import NGRAM_DICT_NAME


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_examples_to_features(text1, max_seq_length, tokenizer, ngram_dict, text2=None, return_tensors=False):
    inputs = {'input_ids': [], 'attention_mask': [], 'token_type_ids': [],
              'input_ngram_ids': [], 'ngram_attention_mask': [], 'ngram_token_type_ids': [],
              'ngram_position_matrix': []}

    for idx in range(len(text1)):
        tokens_a = tokenizer.tokenize(text1[idx])

        tokens_b = None
        if text2:
            tokens_b = tokenizer.tokenize(text2[idx])
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        token_type_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            token_type_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        attention_mask += padding
        token_type_ids += padding

        ngram_matches = []
        for p in range(2, 8):
            for q in range(0, len(tokens) - p + 1):
                character_segment = tokens[q:q + p]
                # j is the starting position of the word
                # i is the length of the current word
                character_segment = tuple(character_segment)
                if character_segment in ngram_dict.ngram_to_id_dict:
                    ngram_index = ngram_dict.ngram_to_id_dict[character_segment]
                    ngram_matches.append([ngram_index, q, p, character_segment])

        shuffle(ngram_matches)
        max_word_in_seq_proportion = math.ceil((len(tokens) / max_seq_length) * ngram_dict.max_ngram_in_seq)
        if len(ngram_matches) > max_word_in_seq_proportion:
            ngram_matches = ngram_matches[:max_word_in_seq_proportion]
        ngram_ids = [ngram[0] for ngram in ngram_matches]
        ngram_positions = [ngram[1] for ngram in ngram_matches]
        ngram_lengths = [ngram[2] for ngram in ngram_matches]
        ngram_tuples = [ngram[3] for ngram in ngram_matches]
        ngram_token_type_ids = [0 if position < (len(tokens_a) + 2) else 1 for position in ngram_positions]

        import numpy as np
        ngram_attention_mask = np.zeros(ngram_dict.max_ngram_in_seq, dtype=np.bool)
        ngram_attention_mask[:len(ngram_ids)] = 1

        ngram_positions_matrix = np.zeros(shape=(max_seq_length, ngram_dict.max_ngram_in_seq), dtype=np.int32)
        for i in range(len(ngram_ids)):
            ngram_positions_matrix[ngram_positions[i]:ngram_positions[i] + ngram_lengths[i], i] = 1.0

        padding = [0] * (ngram_dict.max_ngram_in_seq - len(ngram_ids))
        ngram_ids += padding
        ngram_lengths += padding
        ngram_token_type_ids += padding

        inputs['input_ids'].append(input_ids)
        inputs['attention_mask'].append(attention_mask)
        inputs['token_type_ids'].append(token_type_ids)
        inputs['input_ngram_ids'].append(ngram_ids)
        inputs['ngram_token_type_ids'].append(ngram_token_type_ids)
        inputs['ngram_attention_mask'].append(ngram_attention_mask)
        inputs['ngram_position_matrix'].append(ngram_positions_matrix)

    if return_tensors:
        inputs['input_ids'] = torch.tensor(inputs['input_ids'])
        inputs['attention_mask'] = torch.tensor(inputs['attention_mask'])
        inputs['token_type_ids'] = torch.tensor(inputs['token_type_ids'])
        inputs['input_ngram_ids'] = torch.tensor(inputs['input_ngram_ids'])
        inputs['ngram_token_type_ids'] = torch.tensor(inputs['ngram_token_type_ids'])
        inputs['ngram_attention_mask'] = torch.tensor(inputs['ngram_attention_mask'])
        inputs['ngram_position_matrix'] = torch.tensor(inputs['ngram_position_matrix'])

    return inputs


def convert_examples_to_features_for_tokens(text, max_seq_length=128, tokenizer=None, ngram_dict=None,
                                            return_tensors=False):
    inputs = {'input_ids': [], 'attention_mask': [], 'token_type_ids': [],
              'input_ngram_ids': [], 'ngram_attention_mask': [], 'ngram_token_type_ids': [],
              'ngram_position_matrix': [], 'valid_ids': [], 'label_mask': []}

    text_list = text

    tokens = []
    if isinstance(text_list, list):
        for i, word in enumerate(text_list):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
    else:
        tokens = tokenizer.tokenize(text_list)

    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]

    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
    ntokens.append("[SEP]")
    segment_ids.append(0)
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)
    label_mask = [1] * len(label_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        label_ids.append(0)
        label_mask.append(0)
    while len(label_ids) < max_seq_length:
        label_ids.append(0)
        label_mask.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    assert len(label_mask) == max_seq_length

    ngram_matches = []
    for p in range(2, 8):
        for q in range(0, len(tokens) - p + 1):
            character_segment = tokens[q:q + p]
            # j is the starting position of the ngram
            # i is the length of the current ngram
            character_segment = tuple(character_segment)
            if character_segment in ngram_dict.ngram_to_id_dict:
                ngram_index = ngram_dict.ngram_to_id_dict[character_segment]
                ngram_matches.append([ngram_index, q, p, character_segment])

    shuffle(ngram_matches)

    max_ngram_in_seq_proportion = math.ceil((len(tokens) / max_seq_length) * ngram_dict.max_ngram_in_seq)
    if len(ngram_matches) > max_ngram_in_seq_proportion:
        ngram_matches = ngram_matches[:max_ngram_in_seq_proportion]

    ngram_ids = [ngram[0] for ngram in ngram_matches]
    ngram_positions = [ngram[1] for ngram in ngram_matches]
    ngram_lengths = [ngram[2] for ngram in ngram_matches]
    ngram_tuples = [ngram[3] for ngram in ngram_matches]
    ngram_seg_ids = [0 if position < (len(tokens) + 2) else 1 for position in ngram_positions]

    import numpy as np
    ngram_mask_array = np.zeros(ngram_dict.max_ngram_in_seq, dtype=np.bool)
    ngram_mask_array[:len(ngram_ids)] = 1

    # record the masked positions
    ngram_positions_matrix = np.zeros(shape=(max_seq_length, ngram_dict.max_ngram_in_seq), dtype=np.int32)
    for i in range(len(ngram_ids)):
        ngram_positions_matrix[ngram_positions[i]:ngram_positions[i] + ngram_lengths[i], i] = 1.0

    # Zero-pad up to the max ngram in seq length.
    padding = [0] * (ngram_dict.max_ngram_in_seq - len(ngram_ids))
    ngram_ids += padding
    ngram_lengths += padding
    ngram_seg_ids += padding

    assert len(ngram_ids) == ngram_dict.max_ngram_in_seq
    assert len(ngram_mask_array) == ngram_dict.max_ngram_in_seq
    assert len(ngram_positions_matrix) == ngram_dict.max_ngram_in_seq
    assert len(ngram_seg_ids) == ngram_dict.max_ngram_in_seq

    inputs['input_ids'] = input_ids
    inputs['attention_mask'] = input_mask
    inputs['token_type_ids'] = segment_ids
    inputs['input_ngram_ids'] = ngram_ids
    inputs['ngram_attention_mask'] = ngram_mask_array
    inputs['ngram_token_type_ids'] = ngram_seg_ids
    inputs['ngram_position_matrix'] = ngram_positions_matrix

    if return_tensors:
        inputs['input_ids'] = torch.tensor(inputs['input_ids'])
        inputs['attention_mask'] = torch.tensor(inputs['attention_mask'])
        inputs['token_type_ids'] = torch.tensor(inputs['token_type_ids'])
        inputs['input_ngram_ids'] = torch.tensor(inputs['input_ngram_ids'])
        inputs['ngram_token_type_ids'] = torch.tensor(inputs['ngram_token_type_ids'])
        inputs['ngram_attention_mask'] = torch.tensor(inputs['ngram_attention_mask'])
        inputs['ngram_position_matrix'] = torch.tensor(inputs['ngram_position_matrix'])

    return inputs


def save_zen_model(save_zen_model_path, model, tokenizer, ngram_dict, args):
    # Save a trained model, configuration and tokenizer
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(save_zen_model_path, WEIGHTS_NAME)
    output_config_file = os.path.join(save_zen_model_path, CONFIG_NAME)
    output_ngram_dict_file = os.path.join(save_zen_model_path, NGRAM_DICT_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(save_zen_model_path)
    ngram_dict.save(output_ngram_dict_file)
    output_args_file = os.path.join(save_zen_model_path, 'training_args.bin')
    torch.save(args, output_args_file)
