import json
import sys
from format_checker import FormatChecker

class CDN(FormatChecker):
    def load_file(self, filename):
        data = {}
        with open(filename) as input_file:
            json_content = json.load(input_file)
            for block in json_content:
                text = block['text']
                normalized_result = block['normalized_result']
                data[text] = normalized_result
        return data

    def check_format(self, submission_filename):
        with open(submission_filename) as input_file:
            json_content = json.load(input_file)
            for block in json_content:
                text = block['text']
                normalized_result = block['normalized_result']
                assert self.check_field_type(normalized_result, str), '"normalized_result" should be str. The defect record is "{}", and the defect normalized_result is "{}".'.format(text, normalized_result)


if __name__ == '__main__':

    source_filename = sys.argv[1]
    submission_filename = sys.argv[2]

    SUBMISSION_FILENAME = 'CHIP-CDN_test.json'
    checker = CDN(SUBMISSION_FILENAME)
    checker.check_filename(submission_filename)
    checker.check_record_number(source_filename, submission_filename)
    checker.check_format(submission_filename)

    print("Format Check Success!")





