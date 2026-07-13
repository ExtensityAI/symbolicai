class IntegrationError(Exception):
    """Base class for every error raised by an integration client."""


class TransportError(IntegrationError):
    """Raised when the integration could not be reached at all.

    Connection, DNS, TLS, and process-spawn failures, and timeouts — anything that
    went wrong before a well-formed response existed.
    """


class ResponseError(IntegrationError):
    """Raised when the integration replied but the reply could not be decoded or validated."""
