
class FormatChecker:

    def __init__(self, filename, predefined_list=[], required_field_list=[]):
        self.filename = filename
        self.predefined_list = predefined_list
        self.required_field_list = required_field_list

    def load_file(self, filename):
        # related to "check_record_number" function
        pass

    def check_filename(self, filename):
        assert self.filename == filename, 'Submission filename should be ' + self.filename

    def check_record_number(self, source_filename, submission_filename):
        # please refer to the "load_file" function for the detailed logic
        standard_result = self.load_file(source_filename)
        predict_result = self.load_file(submission_filename)
        intersection = list(set(standard_result.keys()) & set(predict_result.keys()))
        difference = list(set(standard_result.keys()) - set(intersection))
        assert len(intersection) == len(standard_result), 'Record number check fail. Expected {}, Actual {}. For example, Record "{}" is not in the submission file.'.format(len(standard_result), len(intersection), difference[0])

    def check_required_fields(self, element):
        return all([key in element for key in self.required_field_list])

    def check_field_type(self, field, check_type):
        return isinstance(field, check_type)

    def check_predefined_list(self, element):
        return element in self.predefined_list

    def check_format(self, submission_filename):
        pass

