def __getattr__(name: str) -> str:
    """Expose the installed distribution version as `symai.__version__`.

    The version is read from installed package metadata rather than restated here, so
    `pyproject.toml` stays its only source and the two cannot drift. The read is lazy:
    importing `symai` does no filesystem work and binds no public name.

    Raises:
        importlib.metadata.PackageNotFoundError: if `symbolicai` is not installed, such
            as when the checkout is placed on `sys.path` directly instead of installed.
    """
    if name == "__version__":
        # Deferred so that importing symai reads no package metadata from disk.
        from importlib.metadata import version  # noqa: PLC0415

        return version("symbolicai")

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
