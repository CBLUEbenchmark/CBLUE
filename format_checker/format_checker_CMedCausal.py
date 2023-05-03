import json
import sys
from format_checker import FormatChecker


class CMedCausal(FormatChecker):
    def load_file(self, filename):
        data = {}
        with open(filename) as input_file:
            json_content = json.load(input_file)
            for record in json_content:
                text = record['text']
                relation_of_mention_list = record['relation_of_mention']
                data[text] = relation_of_mention_list
        return data

    def check_format(self, submission_filename):

        def nested_check_format(text, spo, check_tail_flag=True):
            required_field_list = ['mention', 'start_idx', 'end_idx']
            assert all([key in spo['head'] for key in required_field_list]), '{} are required. The defect record is "{}", and the defect spo is {}.'.format(required_field_list, text, spo)
            assert self.check_field_type(spo['head']['start_idx'], int), '"start_idx" should be int. The defect record is "{}", and the defect spo is {}.'.format(text, spo)
            assert self.check_field_type(spo['head']['end_idx'], int), '"end_idx" should be int. The defect record is "{}", and the defect spo is {}.'.format(text, spo)

            if check_tail_flag:
                assert all([key in spo['tail'] for key in required_field_list]), '{} are required. The defect record is "{}", and the defect spo is {}.'.format(required_field_list, text, spo)
                assert self.check_field_type(spo['tail']['start_idx'], int), '"start_idx" should be int. The defect record is "{}", and the defect spo is {}.'.format(text, spo)
                assert self.check_field_type(spo['tail']['end_idx'], int), '"end_idx" should be int. The defect record is "{}", and the defect spo is {}.'.format(text, spo)

                assert spo['relation'] in [1, 3], 'relation type should be causal relation(1) or hypernym relation(3), The defect record is "{}", and the defect spo is {}.'.format(text, spo)

        with open(submission_filename) as input_file:
            json_content = json.load(input_file)
            for record in json_content:
                text = record['text']
                for spo in record['relation_of_mention']:
                    # check spo:
                    assert self.check_required_fields(spo), '{} are required. The defect record is "{}", and the defect spo is {}.'.format(self.required_field_list, text, spo)
                    assert self.check_field_type(spo['head'], dict), '"head" should be dict. The defect record is "{}", and the defect spo is {}.'.format(text, spo)
                    assert self.check_field_type(spo['tail'], dict), '"tail" should be dict. The defect record is "{}", and the defect spo is {}.'.format(text, spo)
                    assert self.check_predefined_list(spo['relation']), '"predicate" should be in the predefined relation list. The defect record is "{}", and the defect spo is {}.'.format(text, spo)
                    # check subject & object:
                    if spo['relation'] in [1, 3]:
                        assert 'type' in spo['tail'] and spo['tail']['type'] == 'mention', '"type" must be in tail and tail["type"] must be "mention" for causal(1) & hypernym(3) relation. The defect record is "{}", and the defect spo is {}.'.format(text, spo)
                        nested_check_format(text, spo)
                    else:
                        assert self.check_required_fields(spo['tail']), '{} are required for spo["tail"] in conditional relation. The defect record is "{}", and the defect spo is {}.'.format(self.required_field_list, text, spo['tail'])
                        # check spo['head'] only
                        nested_check_format(text, spo, False)
                        # check spo['tail']:
                        assert 'type' in spo['tail'] and spo['tail']['type'] == 'relation', '"type" must be in tail and tail["type"] must be "relation" for conditional relation(2). The defect record is "{}", and the defect spo is {}.'.format(text, spo)
                        nested_check_format(text, spo['tail'])

if __name__ == '__main__':

    source_filename = sys.argv[1]
    submission_filename = sys.argv[2]

    SUBMISSION_FILENAME = 'CMedCausal_test.json'
    RELATION_TYPE_LIST = [
        1, 2, 3
    ]
    REQUIRED_FIELD_LIST = ['head', 'relation', 'tail']
    checker = CMedCausal(SUBMISSION_FILENAME, RELATION_TYPE_LIST, REQUIRED_FIELD_LIST)
    checker.check_filename(submission_filename)
    checker.check_record_number(source_filename, submission_filename)
    checker.check_format(submission_filename)

    print("Format Check Success!")





