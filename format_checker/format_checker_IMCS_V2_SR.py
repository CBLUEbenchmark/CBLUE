import json
import sys
from collections import defaultdict
from format_checker import FormatChecker

class IMCS_SR(FormatChecker):
    def load_file(self, filename):
        data = {}
        with open(filename) as input_file:
            json_content = json.load(input_file)
            for key, dialog in json_content.items():
                for sentence_id, value in dialog.items():
                    composed_key = '_'.join([key, sentence_id])
                    data[composed_key] = value
        return data

    def check_format(self, submission_filename):
        with open(submission_filename) as input_file:
            json_content = json.load(input_file)
            for dialog_id, dialog in json_content.items():
                for sentence_id, symptom_map in dialog.items():
                    for symptom, value in symptom_map.items():
                        assert self.check_predefined_list(value), 'Symptom value must be in the predefined list. The defect dialog is "{}", and the defect sentence_id is "{}", and the defect key is "{}", and the defect value is "{}"'.format(dialog_id, sentence_id, symptom, value)

                        # # optional:
                        # # check key is in the "mappings.json"
                        # norm_map = json.load(open('mappings.json'))[0]
                        # norm_list = list(norm_map.keys())
                        # assert symptom in norm_list, 'symptom must be in the normalized symptom dictionary. The defect dialog is "{}", and the defect sentence_id is "{}", and the defect key is "{}"'.format(dialog_id, sentence_id, symptom)


if __name__ == '__main__':

    raw_filename = sys.argv[1]
    submission_filename = sys.argv[2]

    SUBMISSION_FILENAME = 'IMCS-V2-SR_test.json'
    SR_LIST = ['0', '1', '2']
    checker = IMCS_SR(SUBMISSION_FILENAME, SR_LIST)
    checker.check_filename(submission_filename)
    checker.check_record_number(raw_filename, submission_filename)
    checker.check_format(submission_filename)

    print("Format Check Success!")





