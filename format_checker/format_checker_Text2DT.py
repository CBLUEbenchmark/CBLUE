import json
import sys
from format_checker import FormatChecker

class Text2DT(FormatChecker):
    def load_file(self, filename):
        data = {}
        with open(filename) as input_file:
            json_content = json.load(input_file)
            for block in json_content:
                text = block['text']
                tree = block['tree']
                data[text] = tree
        return data

    def check_format(self, submission_filename):
        EMPTY_DECISION_NODE = '{"role":"D","triples":[],"logical_rel":"null"}'
        with open(submission_filename) as input_file:
            json_content = json.load(input_file)
            for block in json_content:
                text = block['text']
                tree = block['tree']
                #print('processing: ', text)
                for rule in tree:
                    assert self.check_required_fields(rule), '{} are required. The defect record is "{}", and the defect rule is {}.'.format(self.required_field_list, text, rule)
                    assert self.check_field_type(rule['triples'], list), '"triples" must be list, The defect record is "{}", and the defect rule is {}.'.format(text, rule)
                    for triple in rule['triples']:
                        # (sub, rel, obj)
                        triple_rel = triple[1]
                        assert triple_rel in ['临床表现', '治疗药物', '用法用量', '治疗方案', '禁用药物', '基本情况'], '"triple" relation must be in predefined list. The defect record is "{}", and the defect rule is {}.'.format(text, rule)
                    assert rule['role'] in ['D', 'C'], '"role" must be "C" or "D". The defect record is "{}", and the defect rule is {}.'.format(text, rule)
                    assert self.check_predefined_list(rule['logical_rel']), '"logical_rel" must be in predefined list. The defect record is "{}", and the defect rule is {}.'.format(text, rule)
                # check if it is a valid pre-order binary tree:
                assert self._check_valid_preorder_tree(tree), 'A valid pre-order binary tree is not satisfied. Each leaf node must be a Decision node, and each Condition node must have two child nodes(left-child and right-child), and EMPTY Decision node "{}" is required if necessary. The defect record is "{}"'.format(EMPTY_DECISION_NODE, text)

    def _check_valid_preorder_tree(self, tree):
        nodelist = []
        for i in range(len(tree)):
            nodelist.append(tree[i]["role"])

        # 将符合诊疗决策树的节点前序序列转化为代表诊疗决策树结构的节点矩阵，matrix[i][j]='F'/'L'/'R'表示第j个节点是第i个节点的父/左子/右子节点
        node_matrix = [[0 for i in range(len(nodelist))] for j in range(len(nodelist))]

        while nodelist[0] != 'D':
            raw_nodelist = nodelist.copy()
            for i in range(len(nodelist)):
                if nodelist[i] == 'C':
                    flag, leaf1, leaf2 = 0, 0, 0
                    for j in range(i + 1, len(nodelist)):
                        if nodelist[j] == 'D' and flag == 0:
                            flag = 1
                            leaf1 = j
                        elif nodelist[j] == 'X':
                            continue
                        elif nodelist[j] == 'D' and flag == 1:
                            # print(i)
                            leaf2 = j
                            nodelist[i] = 'D'
                            node_matrix[leaf1][i] = 'F'
                            node_matrix[leaf2][i] = 'F'
                            node_matrix[i][leaf1] = 'L'
                            node_matrix[i][leaf2] = 'R'
                            for k in range(i + 1, leaf2 + 1):
                                nodelist[k] = 'X'
                            flag = 2
                            break
                        elif nodelist[j] == 'C':
                            break
                    if flag == 2:
                        break
            # After each iteration, the nodelist should change, or it will go to dead-loop
            if raw_nodelist == nodelist:
                return False

        return True


if __name__ == '__main__':

    source_filename = sys.argv[1]
    submission_filename = sys.argv[2]

    SUBMISSION_FILENAME = 'Text2DT_test.json'
    LOGICAL_REL_LIST = ['and', 'or', 'null']
    REQUIRED_FIELD_LIST = ['role', 'triples', 'logical_rel']

    checker = Text2DT(SUBMISSION_FILENAME, LOGICAL_REL_LIST, REQUIRED_FIELD_LIST)
    checker.check_filename(submission_filename)
    checker.check_record_number(source_filename, submission_filename)
    checker.check_format(submission_filename)

    print("Format Check Success!")
    
