import json
import sys
from collections import defaultdict
from format_checker import FormatChecker

class IMCS_SR(FormatChecker):
    def load_file(self, filename):
        data = {}
        with open(filename) as input_file:
            json_content = json.load(input_file)
            for dialog_id, result in json_content.items():
                data[dialog_id] = result
        return data

    def check_format(self, submission_filename):
        with open(submission_filename) as input_file:
            json_content = json.load(input_file)
            for dialog_id, result in json_content.items():
                for key, value in result.items():
                    assert self.check_predefined_list(value), 'Symptom value must be in the predefined list. The defect dialog is "{}", and the defect key is "{}", and the defect result is "{}"'.format(dialog_id, key, value)

                    # # optional:
                    # # check key is in the "symptom_norm.csv"
                    # norm_list = open('symptom_norm.csv').read().strip().split('\n')
                    # norm_list = norm_list[1:]  # remove the first dummy words
                    # assert key in norm_list, 'symptom must be in the normalized symptom dictionary. The defect dialog is "{}", and the defect key is "{}" '.format(dialog_id, key)

if __name__ == '__main__':

    raw_filename = sys.argv[1]
    submission_filename = sys.argv[2]

    SUBMISSION_FILENAME = 'IMCS-SR_test.json'
    SR_LIST = ['0', '1', '2']
    checker = IMCS_SR(SUBMISSION_FILENAME, SR_LIST)
    checker.check_filename(submission_filename)
    checker.check_record_number(raw_filename, submission_filename)
    checker.check_format(submission_filename)

    print("Format Check Success!")





