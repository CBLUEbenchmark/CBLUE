import json
import sys
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
                assert self.check_required_fields(result), '{} are required. The defect dialog_id is: "{}"'.format(self.required_field_list, dialog_id)

                assert self.check_field_type(result['主诉'], str), '"主诉" must be str. The defect dialog_id is: "{}"'.format(dialog_id)
                assert self.check_field_type(result['现病史'], str), '"现病史" must be str. The defect dialog_id is: "{}"'.format(dialog_id)
                assert self.check_field_type(result['辅助检查'], str), '"辅助检查" must be str. The defect dialog_id is: "{}"'.format(dialog_id)
                assert self.check_field_type(result['既往史'], str), '"既往史" must be str. The defect dialog_id is: "{}"'.format(dialog_id)
                assert self.check_field_type(result['诊断'], str), '"诊断" must be str. The defect dialog_id is: "{}"'.format(dialog_id)
                assert self.check_field_type(result['建议'], str), '"建议" must be str. The defect dialog_id is: "{}"'.format(dialog_id)

if __name__ == '__main__':

    raw_filename = sys.argv[1]
    submission_filename = sys.argv[2]

    SUBMISSION_FILENAME = 'IMCS-V2-MRG_test.json'
    MRG_LIST = ['主诉', '现病史', '辅助检查', '既往史', '诊断', '建议']
    checker = IMCS_MRG(SUBMISSION_FILENAME, [], MRG_LIST)
    checker.check_filename(submission_filename)
    checker.check_record_number(raw_filename, submission_filename)
    checker.check_format(submission_filename)

    print("Format Check Success!")





