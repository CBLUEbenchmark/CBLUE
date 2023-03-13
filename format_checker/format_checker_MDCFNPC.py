import json
import sys
from format_checker import FormatChecker

class MDCFNPC(FormatChecker):
    def load_file(self, filename):
        data = {}
        with open(filename) as input_file:
            for line in input_file:
                json_content = json.loads(line.strip())
                dialog_id = json_content['dialog_id']
                dialog_info = json_content['dialog_info']
                for sentence in dialog_info:
                    sentence_id = sentence['sentence_id']
                    ner_list = sentence['ner']
                    for ner in ner_list:
                        start_idx, end_idx = ner['range']
                        attr = ner['attr']
                        composed_key = '_'.join([str(dialog_id), str(sentence_id), str(start_idx), str(end_idx)])
                        data[composed_key] = attr
        return data

    def check_format(self, submission_filename):
        with open(submission_filename) as input_file:
            for line in input_file:
                json_content = json.loads(line.strip())
                dialog_id = json_content['dialog_id']
                dialog_info = json_content['dialog_info']
                for sentence in dialog_info:
                    sentence_id = sentence['sentence_id']
                    ner_list = sentence['ner']
                    for ner in ner_list:
                        assert self.check_predefined_list(ner['attr']), '"attr" should be in predefined list. The defect dialog is "{}", and the defect sentence_id is "{}", and the defect ner is "{}".'.format(dialog_id, sentence_id, ner)

if __name__ == '__main__':

    source_filename = sys.argv[1]
    submission_filename = sys.argv[2]

    SUBMISSION_FILENAME = 'CHIP-MDCFNPC_test.jsonl'
    ATTR_LIST = ['阴性', '阳性', '其他', '不标注']
    checker = MDCFNPC(SUBMISSION_FILENAME, ATTR_LIST)
    checker.check_filename(submission_filename)
    checker.check_record_number(source_filename, submission_filename)
    checker.check_format(submission_filename)

    print("Format Check Success!")





