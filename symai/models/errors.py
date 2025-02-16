from pydantic import BaseModel

class ExceptionWithUsage(Exception):
    def __init__(self, message, usage):
        super().__init__(message)
        self.usage = usage

class TypeValidationError(Exception):
    def __init__(self, violations, *args):
        super().__init__(*args)
        if violations and not isinstance(violations, list):
            violations = [violations]
        self.violations = violations

    def __str__(self):
        return "\n".join(self.violations)

    def __repr__(self):
        return f"TypeValidationError({self.violations})"

class SemanticValidationError(Exception):
    def __init__(self, task: str, result: str, violations: list[str], *args):
        super().__init__(*args)
        self.task = task
        self.result = result
        if violations and not isinstance(violations, list):
            violations = [violations]
        self.violations = violations

    def __str__(self) -> str:
        violations = "\n".join(self.violations)
        return (
            "[[Task]]\n"
            f"{self.task}\n"
            "\n[[Result]]\n"
            f"{self.result}\n"
            "\n[[Violations]]\n"
            f"{violations}\n"
        )

    def __repr__(self):
        return f"SemanticValidationError({self.task}, {self.result}, {self.violations})"
