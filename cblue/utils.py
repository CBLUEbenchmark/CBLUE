import json


def load_json(input_file):
    with open(input_file, 'r') as f:
        samples = json.load(f)
    return samples


def load_dict(dict_path):
    """load_dict"""
    vocab = {}
    for line in open(dict_path, 'r', encoding='utf-8'):
        key, value = line.strip('\n').split('\t')
        vocab[int(key)] = value
    return vocab


def write_dict(dict_path, dict_data):
    with open(dict_path, "w", encoding="utf-8") as f:
        for key, value in dict_data.items():
            f.writelines("{}\t{}\n".format(key, value))


def str_q2b(text):
    ustring = text
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif 65281 <= inside_code <= 65374:
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring
