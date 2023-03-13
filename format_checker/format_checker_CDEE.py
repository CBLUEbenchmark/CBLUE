import json
import sys
from format_checker import FormatChecker

class CDEE(FormatChecker):
    def load_file(self, filename):
        data = {}
        with open(filename) as input_file:
            json_content = json.load(input_file)
            for block in json_content:
                id = block['id']
                event = block['event']
                data[id] = event
        return data

    def check_format(self, submission_filename):
        with open(submission_filename) as input_file:
            json_content = json.load(input_file)
            for block in json_content:
                record_id = block['id']
                for event in block['event']:
                    assert self.check_required_fields(event), '{} are required. The defect record is "{}", and the defect event is {}.'.format(self.required_field_list, record_id, event)
                    assert self.check_field_type(event['core_name'], str), '"core_name" must be string. The defect record is "{}", and the defect event is {}.'.format(record_id, event)
                    assert self.check_field_type(event['character'], list), '"character" must be list. The defect record is "{}", and the defect event is {}.'.format(record_id, event)
                    assert self.check_field_type(event['anatomy_list'], list), '"anatomy_list" must be list. The defect record is "{}", and the defect event is {}.'.format(record_id, event)
                    assert self.check_predefined_list(event['tendency']), '"tendency" should be in the predefined tendency list. The defect record is "{}", and the defect event is {}.'.format(record_id, event)


if __name__ == '__main__':

    source_filename = sys.argv[1]
    submission_filename = sys.argv[2]

    SUBMISSION_FILENAME = 'CHIP-CDEE_test.json'
    TENDENCY_LIST = ['不确定', '否定', '']
    REQUIRED_FIELD_LIST = ['core_name', 'tendency', 'character', 'anatomy_list']
    checker = CDEE(SUBMISSION_FILENAME, TENDENCY_LIST, REQUIRED_FIELD_LIST)
    checker.check_filename(submission_filename)
    checker.check_record_number(source_filename, submission_filename)
    checker.check_format(submission_filename)

    print("Format Check Success!")





