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
