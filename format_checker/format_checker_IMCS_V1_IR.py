import json
import sys
from collections import defaultdict
from format_checker import FormatChecker

class IMCS_IR(FormatChecker):
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
                for sentence_id, value in dialog.items():
                    assert self.check_predefined_list(value), 'Prediction value must be in the predefined list. The defect dialog is "{}", and the defect sentence_id is "{}", and the defect result is "{}".'.format(dialog_id, sentence_id, value)


if __name__ == '__main__':

    raw_filename = sys.argv[1]
    submission_filename = sys.argv[2]

    SUBMISSION_FILENAME = 'IMCS-IR_test.json'
    IR_LIST = ['Request-Symptom', 'Inform-Symptom',
               'Request-Etiology', 'Inform-Etiology',
               'Request-Basic_Information', 'Inform-Basic_Information',
               'Request-Existing_Examination_and_Treatment', 'Inform-Existing_Examination_and_Treatment',
               'Request-Drug_Recommendation', 'Inform-Drug_Recommendation',
               'Request-Medical_Advice', 'Inform-Medical_Advice',
               'Request-Precautions', 'Inform-Precautions',
               'Diagnose',
               'Other']
    checker = IMCS_IR(SUBMISSION_FILENAME, IR_LIST)
    checker.check_filename(submission_filename)
    checker.check_record_number(raw_filename, submission_filename)
    checker.check_format(submission_filename)

    print("Format Check Success!")





