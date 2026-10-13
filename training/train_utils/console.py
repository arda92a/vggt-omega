"""Console output that stays readable in a terminal and in a log collector.

A terminal gets one progress bar that disappears when the loop ends. A log collector (docker,
CI, `> file`) has no cursor control, so a bar would write thousands of overwritten lines into
the middle of other log output. There the loop prints one line every `every` steps instead.
"""

import logging
import sys
from typing import Mapping

from tqdm.auto import tqdm


class Progress:
    def __init__(self, total: int, desc: str, enabled: bool = True, every: int = 100) -> None:
        self.total = total
        self.desc = desc
        self.every = max(1, every)
        self.bar = None
        self.lines = False
        if enabled and sys.stderr.isatty():
            self.bar = tqdm(total=total, desc=desc, dynamic_ncols=True, leave=False, mininterval=0.5)
        else:
            self.lines = enabled

    def step(self, done: int, values: Mapping[str, float]) -> None:
        """Call after `done` steps finished. `values` are running averages to show."""
        text = " | ".join(f"{name} {value:.3f}" for name, value in values.items())
        if self.bar is not None:
            self.bar.update(1)
            self.bar.set_postfix_str(text, refresh=False)
        elif self.lines and (done % self.every == 0 or done == self.total):
            logging.info(f"{self.desc} {done}/{self.total} | {text}")

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()
            self.bar = None
