import json
import sys
from format_checker import FormatChecker

class IR(FormatChecker):
    def load_file(self, filename):
        data = {}
        with open(filename) as input_file:
            for line in input_file:
                line = line.strip().split('\t')
                id = line[0]
                data[id] = line[1] if len(line) == 2 else ''
        return data

    def check_format(self, submission_filename):
        with open(submission_filename) as input_file:
            for line in input_file:
                line = line.strip().split('\t')
                record_id = line[0]
                assert len(line) == 2, 'The output format is "id\tc1,c2,c3,c4,c5,c6,c7,c8,c9,c10". The defect record_id is "{}".'.format(record_id)
                candidate_list = line[1].split(',')
                assert len(candidate_list) == 10, '10 candidates are required. The defect record_id is "{}".'.format(record_id)


if __name__ == '__main__':

    source_filename = sys.argv[1]
    submission_filename = sys.argv[2]

    SUBMISSION_FILENAME = 'KUAKE-IR_test.tsv'

    checker = IR(SUBMISSION_FILENAME)
    checker.check_filename(submission_filename)
    checker.check_record_number(source_filename, submission_filename)
    checker.check_format(submission_filename)

    print("Format Check Success!")





