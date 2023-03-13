import json
import sys
from format_checker import FormatChecker

class MedDG(FormatChecker):

    def load_file(self, filename):
        data = {}
        with open(filename) as input_file:
            json_content = json.load(input_file)
            for block in json_content:
                id = block['id']
                output = block['output']
                data[id] = output
        return data

    def check_format(self, submission_filename):
        with open(submission_filename) as input_file:
            json_content = json.load(input_file)
            for block in json_content:
                id = block['id']
                output = block['output']
                assert self.check_field_type(output, str), '"output" must be str. The defect record is "{}", and the defect output is "{}".'.format(id, output)

                # optional:
                # "output" must contain at least one entity given in the entity_list.txt, which is from MedDG.zip.
                # required_entity_list = open('entity_list.txt').read().strip().split('\n')
                # assert any([key in output for key in required_entity_list]), '"output" must contain at least one entity given in the MedDG_entity_list.txt. The defect record is "{}", and the defect output is "{}".'.format(id, output)


if __name__ == '__main__':

    source_filename = sys.argv[1]
    submission_filename = sys.argv[2]

    SUBMISSION_FILENAME = 'MedDG_test.json'

    checker = MedDG(SUBMISSION_FILENAME)
    checker.check_filename(submission_filename)
    checker.check_record_number(source_filename, submission_filename)
    checker.check_format(submission_filename)

    print("Format Check Success!")





