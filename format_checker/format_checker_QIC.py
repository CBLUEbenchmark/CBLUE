import json
import sys
from format_checker import FormatChecker

class QIC(FormatChecker):
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

    SUBMISSION_FILENAME = 'KUAKE-QIC_test.json'
    LABEL_LIST = ['病情诊断', '病因分析', '治疗方案', '就医建议', '指标解读', '疾病表述', '后果表述', '注意事项', '功效作用', '医疗费用', '其他']

    checker = QIC(SUBMISSION_FILENAME, LABEL_LIST)
    checker.check_filename(submission_filename)
    checker.check_record_number(source_filename, submission_filename)
    checker.check_format(submission_filename)

    print("Format Check Success!")





