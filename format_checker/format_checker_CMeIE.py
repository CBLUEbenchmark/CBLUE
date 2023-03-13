import json
import sys
from format_checker import FormatChecker


class CMeIE(FormatChecker):
    def load_file(self, filename):
        data = {}
        with open(filename) as input_file:
            for line in input_file:
                record = json.loads(line.strip())
                text = record['text']
                spo_list = record['spo_list']
                data[text] = spo_list
        return data

    def check_format(self, submission_filename):
        with open(submission_filename) as input_file:
            for line in input_file:
                record = json.loads(line)
                text = record['text']
                for spo in record['spo_list']:
                    # check spo:
                    assert self.check_required_fields(spo), '{} are required. The defect record is "{}", and the defect spo is {}.'.format(self.required_field_list, text, spo)
                    assert self.check_field_type(spo['subject'], str), '"subject" should be str. The defect record is "{}", and the defect spo is {}.'.format(text, spo)
                    assert self.check_field_type(spo['object'], dict), '"object" should be dict. The defect record is "{}", and the defect spo is {}.'.format(text, spo)
                    assert self.check_predefined_list(spo['predicate']), '"predicate" should be in the predefined relation list. The defect record is "{}", and the defect spo is {}.'.format(text, spo)
                    # check spo['object']
                    assert '@value' in spo['object'], '"@value" is required in spo["object"]. The defect record is "{}", and the defect spo is {}.'.format(text, spo)
                    assert self.check_field_type(spo['object']['@value'], str), '"@value" should be str. The defect record is "{}", and the defect spo is {}.'.format(text, spo)


if __name__ == '__main__':

    source_filename = sys.argv[1]
    submission_filename = sys.argv[2]

    SUBMISSION_FILENAME = 'CMeIE_test.jsonl'
    RELATION_TYPE_LIST = [
        "预防", "阶段", "就诊科室", "同义词",
        "辅助治疗", "化疗", "放射治疗", "手术治疗",
        "实验室检查", "影像学检查", "辅助检查", "组织学检查",
        "内窥镜检查", "筛查", "多发群体", "发病率",
        "发病年龄", "多发地区", "发病性别倾向", "死亡率",
        "多发季节", "传播途径", "并发症", "病理分型",
        "相关（导致）", "鉴别诊断", "相关（转化）", "相关（症状）",
        "临床表现", "治疗后症状", "侵及周围组织转移的症状", "病因",
        "高危因素", "风险评估因素", "病史", "遗传因素",
        "发病机制", "病理生理", "药物治疗", "发病部位",
        "转移部位", "外侵部位", "预后状况", "预后生存率"
    ]
    REQUIRED_FIELD_LIST = ['subject', 'predicate', 'object']
    checker = CMeIE(SUBMISSION_FILENAME, RELATION_TYPE_LIST, REQUIRED_FIELD_LIST)
    checker.check_filename(submission_filename)
    checker.check_record_number(source_filename, submission_filename)
    checker.check_format(submission_filename)

    print("Format Check Success!")





