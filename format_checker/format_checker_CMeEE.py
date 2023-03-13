import json
import sys
from format_checker import FormatChecker


class CMeEE(FormatChecker):

    def load_file(self, filename):
        data = {}
        with open(filename) as input_file:
            json_content = json.load(input_file)
            for block in json_content:
                text = block['text']
                entity = block['entities']
                data[text] = entity
        return data

    def check_format(self, submission_filename):
        with open(submission_filename) as input_file:
            json_content = json.load(input_file)
            for block in json_content:
                text = block['text']
                # check required fields & field type:
                for entity in block['entities']:
                    assert self.check_required_fields(entity), '{} are required. The defect record is "{}", and the defect entity is {}.'.format(self.required_field_list, text, entity)
                    assert self.check_field_type(entity['start_idx'], int), '"start_idx" should be int. The defect record is "{}", and the defect entity is {}.'.format(text, entity)
                    assert self.check_field_type(entity['end_idx'], int), '"end_idx" should be int. The defect record is "{}", and the defect entity is {}.'.format(text, entity)
                    assert self.check_predefined_list(entity['type']), '"type" should be in the predefined entity list. The defect record is "{}", and the defect entity is {}.'.format(text, entity)

                # optional: check nested entities, the outter entity type must be "sym":
                # entity_list = block['entities']
                # for source_idx, source_entity in enumerate(entity_list):
                #     for target_idx, target_entity in enumerate(entity_list):
                #         if target_idx != source_idx:
                #             if self._is_contain(source_entity, target_entity):
                #                 assert source_entity['type'] == 'sym', 'Outter entity for nested type must be of "sym" type. The defect record is {}. The outter entity is {}, and the inner entity is {}.'.format(text, source_entity, target_entity)

    def _is_contain(self, entity_1, entity_2):
        entity_1_start_idx, entity_1_end_idx = entity_1['start_idx'], entity_1['end_idx']
        entity_2_start_idx, entity_2_end_idx = entity_2['start_idx'], entity_2['end_idx']
        return entity_1_start_idx <= entity_2_start_idx and entity_2_end_idx <= entity_1_end_idx


if __name__ == '__main__':

    source_filename = sys.argv[1]
    submission_filename = sys.argv[2]

    SUBMISSION_FILENAME = 'CMeEE_test.json'
    ENTITY_TYPE_LIST = ['dis', 'sym', 'dru', 'equ', 'pro', 'bod', 'ite', 'mic', 'dep']
    REQUIRED_FIELD_LIST = ['start_idx', 'end_idx', 'type']
    checker = CMeEE(SUBMISSION_FILENAME, ENTITY_TYPE_LIST, REQUIRED_FIELD_LIST)
    checker.check_filename(submission_filename)
    checker.check_record_number(source_filename, submission_filename)
    checker.check_format(submission_filename)

    print("Format Check Success!")





