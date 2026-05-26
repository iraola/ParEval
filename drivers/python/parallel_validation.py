from cpp.parallel_validation import Validator


class PyCOMPSSValidator(Validator):

    def __init__(self):
        super().__init__("pycompss")

    def validate(self, source: str) -> bool:
        return "@task" in source
