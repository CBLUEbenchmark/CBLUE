import json
import sys
from collections import defaultdict
from format_checker import FormatChecker

class IMCS_MRG(FormatChecker):
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
                assert self.check_field_type(result['report'], str), '"report" must be str. The defect dialog_id is: "{}"'.format(dialog_id)

if __name__ == '__main__':

    raw_filename = sys.argv[1]
    submission_filename = sys.argv[2]

    SUBMISSION_FILENAME = 'IMCS-MRG_test.json'
    checker = IMCS_MRG(SUBMISSION_FILENAME)
    checker.check_filename(submission_filename)
    checker.check_record_number(raw_filename, submission_filename)
    checker.check_format(submission_filename)

    print("Format Check Success!")





