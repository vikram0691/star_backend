__version__ = "0.0.0+unknown"

try:
    from ._vesrion import version as __version__
except ImportError:
    try:
        from importlib.metadata import version, PackageNotFoundError
        __version__ = version(__package__ or __name__)
    except ImportError:
        pass

__all__ = ["__version__"]