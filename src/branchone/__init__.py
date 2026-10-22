"""BranchOne: independent multimodal candidate scoring."""

__version__ = "0.3.0"
__all__ = ["BranchOne"]


def __getattr__(name):
    if name == "BranchOne":
        from .model import BranchOne

        return BranchOne
    raise AttributeError(name)
