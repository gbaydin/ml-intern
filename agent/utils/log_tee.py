"""
TeeFile — mirror all stdout/stderr writes to a timestamped log file.

Usage:
    from agent.utils.log_tee import install_log_tee
    cleanup = install_log_tee()   # call cleanup() on exit to flush & close
"""

import io
import os
import re
import sys
from datetime import datetime
from pathlib import Path

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\x1b\].*?\x07|\r")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


class TeeFile(io.TextIOBase):
    """Wraps an original stream and copies every write to a log file (stripped
    of ANSI escapes).  Proxies all attributes the original has so that
    Rich Console, prompt_toolkit, and plain ``print()`` all keep working."""

    def __init__(self, original: io.TextIOBase, log_file: io.TextIOWrapper):
        self._original = original
        self._log = log_file

    # ── Core write path ──────────────────────────────────────────────
    def write(self, s: str) -> int:
        n = self._original.write(s)
        try:
            clean = _strip_ansi(s)
            if clean:
                self._log.write(clean)
                self._log.flush()
        except Exception:
            pass
        return n

    def flush(self) -> None:
        self._original.flush()
        try:
            self._log.flush()
        except Exception:
            pass

    # ── Proxy everything else to the original stream ─────────────────
    def fileno(self) -> int:
        return self._original.fileno()

    def isatty(self) -> bool:
        return self._original.isatty()

    @property
    def encoding(self):
        return getattr(self._original, "encoding", "utf-8")

    @property
    def errors(self):
        return getattr(self._original, "errors", "strict")

    @property
    def name(self):
        return getattr(self._original, "name", "<tee>")

    def readable(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def close(self) -> None:
        try:
            self._log.flush()
            self._log.close()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self._original, name)


def install_log_tee(log_dir: str = "logs") -> callable:
    """Replace sys.stdout and sys.stderr with TeeFile wrappers.

    Returns a cleanup callable that restores the original streams and
    closes the log file.  The log is written to ``<log_dir>/log-<timestamp>.txt``.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(log_dir, f"log-{timestamp}.txt")
    log_file = open(log_path, "a", encoding="utf-8", buffering=1)

    log_file.write(f"=== ML-Intern session started {datetime.now().isoformat()} ===\n\n")
    log_file.flush()

    orig_stdout = sys.stdout
    orig_stderr = sys.stderr

    sys.stdout = TeeFile(orig_stdout, log_file)
    sys.stderr = TeeFile(orig_stderr, log_file)

    def cleanup():
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        try:
            log_file.write(f"\n=== ML-Intern session ended {datetime.now().isoformat()} ===\n")
            log_file.flush()
            log_file.close()
        except Exception:
            pass

    print(f"Logging to {log_path}", file=orig_stderr)
    return cleanup
