import json
import sys
from format_checker import FormatChecker

class CTC(FormatChecker):
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

    SUBMISSION_FILENAME = 'CHIP-CTC_test.json'
    LABEL_LIST = [
        'Disease', 'Symptom', 'Sign', 'Pregnancy-related Activity',
        'Neoplasm Status', 'Non-Neoplasm Disease Stage', 'Allergy Intolerance', 'Organ or Tissue Status',
        'Life Expectancy', 'Oral related', 'Pharmaceutical Substance or Drug', 'Therapy or Surgery',
        'Device', 'Nursing', 'Diagnostic', 'Laboratory Examinations',
        'Risk Assessment', 'Receptor Status', 'Age', 'Special Patient Characteristic',
        'Literacy', 'Gender', 'Education', 'Address',
        'Ethnicity', 'Consent', 'Enrollment in other studies', 'Researcher Decision',
        'Capacity', 'Ethical Audit', 'Compliance with Protocol', 'Addictive Behavior',
        'Bedtime', 'Exercise', 'Diet', 'Alcohol Consumer',
        'Sexual related', 'Smoking Status', 'Blood Donation', 'Encounter',
        'Disabilities', 'Healthy', 'Data Accessible', 'Multiple'
    ]

    checker = CTC(SUBMISSION_FILENAME, LABEL_LIST)
    checker.check_filename(submission_filename)
    checker.check_record_number(source_filename, submission_filename)
    checker.check_format(submission_filename)

    print("Format Check Success!")





