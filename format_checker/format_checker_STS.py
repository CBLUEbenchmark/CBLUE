import json
import sys
from format_checker import FormatChecker

class STS(FormatChecker):
    def load_file(self, filename):
        data = {}
        with open(filename) as input_file:
            json_content = json.load(input_file)
            for block in json_content:
                id = block['id']
                label = block['label']
                data[id] = label
        return data

    def check_format(self, submission_filename):
        with open(submission_filename) as input_file:
            json_content = json.load(input_file)
            for block in json_content:
                record_id = block['id']
                label = block['label']
                assert self.check_predefined_list(label), '"label" should be in predefined list. The defect record is "{}", and the defect label is "{}".'.format(record_id, label)


if __name__ == '__main__':

    source_filename = sys.argv[1]
    submission_filename = sys.argv[2]

    SUBMISSION_FILENAME = 'CHIP-STS_test.json'
    LABEL_LIST = ['0', '1']
    checker = STS(SUBMISSION_FILENAME, LABEL_LIST)
    checker.check_filename(submission_filename)
    checker.check_record_number(source_filename, submission_filename)
    checker.check_format(submission_filename)

    print("Format Check Success!")





