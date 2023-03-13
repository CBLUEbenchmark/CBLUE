import json
import sys
from collections import defaultdict
from format_checker import FormatChecker

class IMCS_NER(FormatChecker):
    def load_file(self, filename):
        data = {}
        with open(filename) as input_file:
            json_content = json.load(input_file)
            for key, dialog in json_content.items():
                for sentence_id, value in dialog.items():
                    composed_key = '_'.join([key, sentence_id])
                    data[composed_key] = value
        return data

    def check_format(self, submission_filename, source_filename):
        with open(submission_filename) as input_file, open(source_filename) as source_file:
            source_json_content = json.load(source_file)
            text_map = defaultdict(dict)
            for dialog_id, value in source_json_content.items():
                sentence_list = value['dialogue']
                for sentence in sentence_list:
                    sentence_id = sentence['sentence_id']
                    text = sentence['sentence']
                    text_map[dialog_id][sentence_id] = text

            json_content = json.load(input_file)
            for dialog_id, value in text_map.items():
                for sentence_id, text in value.items():
                    bio_text = json_content[dialog_id][sentence_id]
                    assert self.check_field_type(bio_text, str), 'The BIO result should be str. The defect dialog is "{}", and the defect sentence_id is "{}" '.format(dialog_id, sentence_id)
                    bio_tag_list = bio_text.split()
                    assert len(text) == len(bio_tag_list), 'The BIO tag must be equal to the original text length. Expected: {}, acutal: {}. The defect dialog is "{}", and the defect sentence_id is "{}".'.format(len(text), len(bio_tag_list), dialog_id, sentence_id)
                    for tag in bio_tag_list:
                        assert self.check_predefined_list(tag), 'The BIO tag must be in the predefined list. The defect dialog is "{}", and the defect sentence_id is "{}", and the defected tag is "{}" '.format(dialog_id, sentence_id, tag)

if __name__ == '__main__':

    raw_filename = sys.argv[1]
    submission_filename = sys.argv[2]
    source_filename = sys.argv[3]


    SUBMISSION_FILENAME = 'IMCS-V2-NER_test.json'
    NER_TAG_LIST = ['O', 'B-Symptom', 'I-Symptom', 'B-Drug', 'I-Drug', 'B-Drug_Category', 'I-Drug_Category', 'B-Medical_Examination', 'I-Medical_Examination', 'B-Operation', 'I-Operation']
    checker = IMCS_NER(SUBMISSION_FILENAME, NER_TAG_LIST)
    checker.check_filename(submission_filename)
    checker.check_record_number(raw_filename, submission_filename)
    checker.check_format(submission_filename, source_filename)

    print("Format Check Success!")





