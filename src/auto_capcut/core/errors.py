class AutoCapCutError(Exception):
    """Expected, user-facing application error."""


class ValidationError(AutoCapCutError):
    pass


class SRTParseError(ValidationError):
    pass


class EffectDirectionError(ValidationError):
    pass


class DrawParseError(ValidationError):
    """Invalid draw animation effect file."""


class SceneValidationError(ValidationError):
    """Invalid draw scene definition."""


class DrawRenderError(AutoCapCutError):
    """A draw animation could not be rendered."""
