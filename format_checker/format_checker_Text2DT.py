import json
import sys
from format_checker import FormatChecker

class Text2DT(FormatChecker):
    def load_file(self, filename):
        data = {}
        with open(filename) as input_file:
            json_content = json.load(input_file)
            for block in json_content:
                text = block['text']
                tree = block['tree']
                data[text] = tree
        return data

    def check_format(self, submission_filename):
        with open(submission_filename) as input_file:
            json_content = json.load(input_file)
            for block in json_content:
                text = block['text']
                tree = block['tree']
                for rule in tree:
                    assert self.check_required_fields(rule), '{} are required. The defect record is "{}", and the defect rule is {}.'.format(self.required_field_list, text, rule)
                    assert self.check_field_type(rule['triples'], list), '"triples" must be list, The defect record is "{}", and the defect rule is {}.'.format(text, rule)
                    assert rule['role'] in ['D', 'C'], '"role" must be "C" or "D". The defect record is "{}", and the defect rule is {}.'.format(text, rule)
                    assert self.check_predefined_list(rule['logical_rel']), '"logical_rel" must be in predefined list. The defect record is "{}", and the defect rule is {}.'.format(text, rule)

if __name__ == '__main__':

    source_filename = sys.argv[1]
    submission_filename = sys.argv[2]

    SUBMISSION_FILENAME = 'Text2DT_test.json'
    LOGICAL_REL_LIST = ['and', 'or', 'null']
    REQUIRED_FIELD_LIST = ['role', 'triples', 'logical_rel']

    checker = Text2DT(SUBMISSION_FILENAME, LOGICAL_REL_LIST, REQUIRED_FIELD_LIST)
    checker.check_filename(submission_filename)
    checker.check_record_number(source_filename, submission_filename)
    checker.check_format(submission_filename)

    print("Format Check Success!")





