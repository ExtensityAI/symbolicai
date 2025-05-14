from pydantic import BaseModel

class ExceptionWithUsage(Exception):
    def __init__(self, message, usage):
        super().__init__(message)
        self.usage = usage

class TypeValidationError(Exception):
    def __init__(self, prompt: str, result: str, violations: list[str], *args):
        super().__init__(*args)
        self.prompt = prompt
        self.result = result
        if violations and not isinstance(violations, list):
            violations = [violations]
        self.violations = violations

    def __str__(self) -> str:
        violations = "\n".join(self.violations)
        return (
            "[[Prompt]]\n"
            f"{self.prompt}"
            "\n\n"
            "[[Result]]\n"
            f"{self.result}"
            "\n\n"
            "[[Violations]]\n"
            f"{violations}"
        )

    def __repr__(self):
        return f"TypeValidationError({self.prompt}, {self.result}, {self.violations})"
