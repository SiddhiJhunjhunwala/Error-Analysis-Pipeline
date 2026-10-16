"""Shared state for one run: config, spreadsheet, normalizer, and the labels (loaded on first use)."""
from .config import resolve_path
from .text import Normalizer


class Context:
    def __init__(self, cfg, sh):
        self.cfg = cfg
        self.sh = sh
        self.normalizer = Normalizer(cfg["part1"])
        self.out_dir = resolve_path(cfg, cfg["run"]["output_dir"])
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._labels = None

    def labels(self, verbose=False):
        """Categories + manual labels (Part 2). Loaded once per run; verbose prints the full check."""
        from .labels import load_labels
        if self._labels is None or verbose:
            self._labels = load_labels(self, verbose=verbose)
        return self._labels
