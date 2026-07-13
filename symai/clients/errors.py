class ClientError(Exception):
    """Base class for every error raised by a provider client."""


class TransportError(ClientError):
    """Raised when the provider could not be reached at all.

    Connection, DNS, TLS, and process-spawn failures, and timeouts — anything that
    went wrong before a well-formed response existed.
    """


class ResponseError(ClientError):
    """Raised when the provider replied but its response could not be decoded or validated."""
