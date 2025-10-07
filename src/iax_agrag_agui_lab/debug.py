import json
import logging
from typing import Any, List

try:
    from uvicorn.logging import DefaultFormatter as _BaseFormatter
except ImportError:  # pragma: no cover - uvicorn may not be present during tests
    _BaseFormatter = logging.Formatter


def _format_json_lines(value: Any, indent: int = 0) -> List[str]:
    prefix = " " * indent
    if isinstance(value, dict):
        lines: List[str] = []
        for key, child in value.items():
            child_lines = _format_json_lines(child, indent + 2)
            if len(child_lines) == 1:
                lines.append(f"{prefix}{key}: {child_lines[0].lstrip()}")
            else:
                lines.append(f"{prefix}{key}:")
                lines.extend(child_lines)
        return lines
    if isinstance(value, list):
        lines = []
        for item in value:
            item_lines = _format_json_lines(item, indent + 2)
            if len(item_lines) == 1:
                lines.append(f"{prefix}- {item_lines[0].lstrip()}")
            else:
                lines.append(f"{prefix}-")
                lines.extend(item_lines)
        return lines or [f"{prefix}-"]
    if isinstance(value, str):
        pieces = value.splitlines()
        if len(pieces) == 1:
            return [f"{prefix}{pieces[0]}"]
        return [f"{prefix}{pieces[0]}"] + [f"{prefix}{piece}" for piece in pieces[1:]]
    return [f"{prefix}{value}"]


class JsonAwareFormatter(_BaseFormatter):
    def format(
        self, record: logging.LogRecord
    ) -> str:  # pragma: no cover - formatting logic
        original_msg, original_args = record.msg, record.args
        try:
            pretty = self._maybe_pretty_json(record)
            if pretty:
                record.msg = pretty
                record.args = ()
            return super().format(record)
        finally:
            record.msg, record.args = original_msg, original_args

    def _maybe_pretty_json(self, record: logging.LogRecord) -> str | None:
        message = record.getMessage()
        if not isinstance(message, str):
            return None
        stripped = message.strip()
        if not stripped or (stripped[0] not in "{[" or stripped[-1] not in "}]"):
            return None
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        lines = _format_json_lines(parsed)
        return "\n".join(lines)


_LOGGING_INITIALISED = False


def configure_console_logging() -> None:
    global _LOGGING_INITIALISED
    if _LOGGING_INITIALISED:
        return
    formatter = (
        JsonAwareFormatter("%(levelprefix)s %(message)s", use_colors=False)
        if _BaseFormatter is not logging.Formatter
        else JsonAwareFormatter("%(levelname)s %(message)s")
    )
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
    else:
        for handler in root_logger.handlers:
            handler.setFormatter(formatter)
    for logger_name in ("generic", "adk", "adk_agui_middleware"):
        logger = logging.getLogger(logger_name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        else:
            for handler in logger.handlers:
                handler.setFormatter(formatter)
        logger.propagate = False
    _LOGGING_INITIALISED = True