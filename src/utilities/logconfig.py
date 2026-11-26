"""Trace logging for luminous: a deterministic record of ray traversals.

The log file IS a table - the same layout for the file, the terminal and the
tests. Header row, then one aligned row per record, at offsets declared in
`LOG_COLUMNS`:

    TYPE         from             target                 action
    ORIGIN       CAM_1            CAM_1                  init_trace
    LEG          CAM_1            START                  reverse_leg
    TERMINATION  MIRROR_1         INFINITY               ray_escaped

`from:` answers "where did this ray leg originate most recently?" and `target:`
answers "which surface or object did this iteration occur on?". Together with
TYPE and `action` those four fields identify a single decision in the path
trace.

`DEFAULT_LOG_FORMAT` is `%(message)s`: the `asctime` / `name` / `levelname`
prefix is boilerplate in a single-purpose trace file (every line is DEBUG, every
line comes from `luminous.src.*`) and a timestamp is non-deterministic, which
makes byte-for-byte log assertions impossible. Pass `VERBOSE_LOG_FORMAT` to get
the clock back for live debugging, accepting that its prefix breaks the offsets.

The log file is truncated when the logger is set up, so the file is exactly the
trace of the Scene that created it, not an append log of every run.
"""

import logging
import re
import sys

import numpy as np

# --- TYPE vocabulary --------------------------------------------------------
# The `TYPE:` token separates what kind of thing happened. `ORIGIN`/`LEG`/
# `TERMINATION` describe the ray life-cycle, `CONTEXT` records a geometry query
# (which light is visible from here), `INTERSECT` records the traversal a ray
# fork took at a surface, and `SHADERS` records a detector model being scheduled
# for that same surface. `CONTEXT`/`SHADERS` pairs share a leg and differ only in
# `action:`, which is what makes a log readable as a sequence.

ORIGIN = "ORIGIN"            # the root vertex of a trace, at a detector (or source)
LEG = "LEG"                  # entry into one recursion - one ray leg
CONTEXT = "CONTEXT"          # visibility / connectivity query result
INTERSECT = "INTERSECT"      # a ray fork was taken at a surface
SHADERS = "SHADERS"          # a detector model was scheduled
TERMINATION = "TERMINATION"  # a leg ended without forking further

# --- traversal tags (the `action:` of an INTERSECT record) ------------------

R_EXT = "R_EXT"  # reflection off the outside surface of a volume
R_INT = "R_INT"  # reflection off the interior wall of a volume (ray stays inside)
T_IN = "T_IN"    # transmission into a volume
T_OUT = "T_OUT"  # transmission out of a volume

# `target:` sentinel for a TERMINATION record where no surface is involved.
INFINITY_TARGET = "INFINITY"

# --- record layout ---------------------------------------------------------
# The one format for the file, the terminal and the tests. The rules that keep a
# row's bytes dependent only on that row are in the module docstring; they are not
# restated below.

PATH_SEPARATOR = "->"

# (column, width), left to right. Every width is the longest token the column can
# meet: `TERMINATION` (11), `SUBSURF-REFLECT` (15), `transmission_model` (18), `n`
# up to 250000 (6). The sentinel tokens `scene.py` emits were shortened to fit -
# `SURF-REFLECT`, `SUBSURF-REFLECT`, `TRANS-IN`, `TRANS-OUT`, `bounces_exhausted` -
# because a column is only as narrow as its longest word. `from`/`target` are 16
# because a `node_id` is developer-issued and bounded only by `NODE_ID_PATTERN`, so
# no length can be promised; 16 covers the ids the examples and tests use.
LOG_COLUMNS = (
    ("TYPE", 11),
    ("from", 16),
    ("target", 16),
    ("action", 18),
    ("hop", 3),
    ("b", 2),
    ("d", 1),
    ("n", 6),
    ("src", 3),
)

PATH_COLUMN = "path"
MESSAGE = "MESSAGE"
LOG_GUTTER = "  "
TRACE_TYPES = (ORIGIN, LEG, CONTEXT, INTERSECT, SHADERS, TERMINATION, MESSAGE)
PACKAGE_ROOT = (__package__ or "").split(".")[0]

# Compact record: the state line is the whole line. Deterministic, so it can be
# asserted against verbatim.
DEFAULT_LOG_FORMAT = "%(message)s"

# Opt-in formatter for live debugging, where wall-clock time is useful. Its
# prefix breaks the column offsets, so use it for the eye only, never for a log
# a test asserts against.
VERBOSE_LOG_FORMAT = "%(asctime)s.%(msecs)03d %(levelname).1s %(message)s"
VERBOSE_LOG_DATE_FORMAT = "%H:%M:%S"


def _cell(value) -> str:
    """One column's text: an absent column is blank, a number is its digits.

    0 is written as '0' - `src: 0` is shadow and `d: 0` is a depth of zero - so a
    zero never reads as an omitted field.
    """
    return "" if value is None else str(value)


def format_row(values) -> str:
    """Lay one record out at the declared offsets, and `rstrip` it.

    Args:
        values (dict): column name -> value. An absent column is blank; a 0 is '0'.

    A value wider than its column is written in full, per the module docstring's
    rules; trailing padding is stripped, so a row that ends early is simply shorter.
    """
    cells = [_cell(values.get(name)).ljust(width) for name, width in LOG_COLUMNS]
    line = LOG_GUTTER.join(cells) + LOG_GUTTER + _cell(values.get(PATH_COLUMN))
    return line.rstrip()


# First line of every table: the column names at the offsets they name.
HEADER_LINE = format_row({name: name for name, _ in LOG_COLUMNS} | {PATH_COLUMN: PATH_COLUMN})


def type_of(line: str) -> str:
    """The TYPE token of a laid-out row, '' when the line carries none."""
    return line[:LOG_COLUMNS[0][1]].strip()


def is_state_line(line: str) -> bool:
    """True when `line` is a record already laid out at the declared offsets."""
    return type_of(line) in TRACE_TYPES


def setup_logging(level=logging.CRITICAL, log_format=DEFAULT_LOG_FORMAT, log_file=None):
    """Configure the package root logger, and by propagation every `luminous.*` child.

    Parameters:
        level (int): Level for this logger. `logging.DEBUG` turns the trace
            records on; `logging.CRITICAL` (the default) silences them.
        log_format (str): Record format. Defaults to `DEFAULT_LOG_FORMAT`
            ('%(message)s'), which keeps every line on the declared column
            offsets. `VERBOSE_LOG_FORMAT` adds a clock at the cost of them.
        log_file (str, optional): Path for a file handler, opened in 'w' mode so
            the file holds only this run's trace. Default is None (stream only).

    Every handler writes the one table format: `HEADER_LINE` before the first row,
    then one aligned row per record. There is no second presentation to turn on -
    the file, the terminal and a test's expectation are the same bytes.
    """
    logger = logging.getLogger(PACKAGE_ROOT)
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()

    handlers = [TableStreamHandler()]

    if log_file:
        handlers.append(TableFileHandler(log_file, mode="w", encoding="utf-8"))

    formatter = logging.Formatter(log_format, datefmt=VERBOSE_LOG_DATE_FORMAT)
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False

    return logger


# --- node identity ----------------------------------------------------------
# A row names `from` and `target` by a `node_id`, so an id is not decoration: it is
# the vocabulary of the file. The table gives `path:` a grammar, and ids are written
# into it, so the characters that carry structure there cannot appear in an id: `>`,
# `(` and `)` delimit hops, `=` pairs with `key=value`, `|` separates the columns the
# CLI readers print, a space is indistinguishable from column padding, and a newline
# splits one record into two. Letter-initial keeps an id distinguishable from the
# numeric cells a reader slices out of the same row.
# Ids are length-unbounded, per the `from`/`target` note above: a row carrying a
# longer id stops aligning from that column on, and loses no data.
NODE_ID_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_\-]*")


def validate_node_id(value) -> str:
    """Return `value` as a printable node id, or raise at the site that invented it.

    Called by the `Element`, `Source` and `Detector` constructors, which are the only
    doors a `node_id` enters the framework through: nothing mints a name on the
    developer's behalf, so the tokens in a log are the tokens of the program that
    wrote it. Validation sits beside the format it protects, and fails at
    construction - the only place an omission can be fixed - rather than in a record
    nobody reads.
    """
    if not isinstance(value, str):
        raise TypeError(f"node_id must be a string, got {type(value).__name__}")
    if not NODE_ID_PATTERN.fullmatch(value):
        raise ValueError(
            f"invalid node_id {value!r}: must match {NODE_ID_PATTERN.pattern} - "
            "letter-initial, then letters, digits, '_' and '-'"
        )
    return value


def state_line(log_type, origin, target, action, hop=None, b=None, d=None, n=None, src=None, path=None) -> str:
    """Build one trace row: TYPE, from, target, action, then the budget counters.

    The four tokens are strings: `Scene` hands over a `node_id` for a scene object
    and a literal for everything else (`INFINITY_TARGET`, a `recursion_enum`). Each
    keyword lands in the column that shares its name and a None keyword is omitted;
    the vocabulary is closed, so a keyword naming no column raises at the call site.
    The counters are short keys so the row stays legible:

        hop (int)  self.counter, the ordinal of this recursion
        b   (int)  bounce count at this vertex
        d   (int)  transmission depth at this vertex
        n   (int)  rays in the mask this record describes (filled by `trace(mask=...)`)
        src (int)  sources in direct view from this vertex
        path(str)  the breadcrumb, last and unbounded (see `extend_path`)

    Prefer `trace()` over calling this directly: it guards on the log level.
    """
    values = {"TYPE": log_type, "from": origin, "target": target, "action": action,
              "hop": hop, "b": b, "d": d, "n": n, "src": src, "path": path}
    return format_row({key: value for key, value in values.items() if value is not None})


def trace(logger, log_type, origin, target, action, mask=None, **fields) -> None:
    """Emit one trace record, at zero cost when `logger` is not at DEBUG.

    `Scene` calls this at every decision point of a path trace. Because the
    default log level is CRITICAL, the guard matters: `state_line()` builds a
    string, and a `mask` reduction is numpy work, so building the record
    unconditionally would tax every trace for lines nobody reads.

    `mask` is the ray mask the record describes; its size is appended as the
    `n:` field, which keeps call sites short (`mask=hit` rather than
    `n=int(np.count_nonzero(hit))`) and defers the count behind the same guard.
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return
    if mask is not None:
        fields["n"] = int(np.count_nonzero(mask))
    logger.debug(state_line(log_type, origin, target, action, **fields))


def extend_path(path: str, tag: str, node_id: str) -> str:
    """Append one hop to a breadcrumb: extend_path('CAM1', T_IN, 'LENS1') -> 'CAM1->T_IN(LENS1)'.

    Exactly ONE hop is appended per recursion, so the LAST hop of a breadcrumb is
    the surface the ray departed from most recently. Hops are tagged with the
    traversal that produced them (R_EXT / R_INT / T_IN / T_OUT), which makes the
    breadcrumb self-describing when it is logged on a LEG record.
    """
    return f"{path}{PATH_SEPARATOR}{tag}({node_id})"


def most_recent_origin(path: str) -> str:
    """The node the ray originated from most recently, as a bare `node_id`.

    'CAM1->R_EXT(MIRROR1)->R_EXT(MIRROR2)' -> 'MIRROR2'. At the START vertex no
    hop has been appended yet and the breadcrumb is the root's own id, which
    is returned unchanged. The traversal tag is deliberately dropped: which
    traversal launched this leg is already recorded by the PRECEDING INTERSECT
    record's `action:` field, so keeping it here is duplication.
    """
    last_hop = path.split(PATH_SEPARATOR)[-1]
    if last_hop.endswith(")"):
        # A hop is always `tag(id)`: `extend_path` is the only writer of a breadcrumb
        # and a `node_id` cannot contain '(', so the tags' parentheses are unambiguous.
        return last_hop[last_hop.index("(") + 1:last_hop.rindex(")")]
    return last_hop


def read_log_lines(log_file) -> list:
    """Read `log_file` and return its trace rows, in trace order, header excluded.

    For full-sequence assertions: a test pastes the known-good table in verbatim
    and compares line-for-line, so a failure prints the whole trace next to the
    whole expectation, row-aligned. `read_log_table` is these rows as one string.
    """
    with open(log_file, encoding="utf-8") as log:
        return [line.rstrip("\n") for line in log
                if line.strip() and line.rstrip() != HEADER_LINE]


def read_log_table(log_file, types=None) -> str:
    """Render `log_file` as the table: `HEADER_LINE`, then the file's own rows.

    Args:
        log_file: the trace log written by `Scene(log_file=...)`.
        types (list, optional): filter to these TYPE tokens, e.g.
            `read_log_table("luminous.log", types=[TERMINATION])`. A filter drops
            the free-form `MESSAGE` rows; ask for them by name to see them.

    Returns:
        The table as a string; "" when no row matches.
    """
    rows = read_log_lines(log_file)
    if types is not None:
        wanted = set(types)
        rows = [row for row in rows if type_of(row) in wanted]
    if not rows:
        return ""
    return "\n".join([HEADER_LINE] + rows)


class _TableFormatMixin:
    """Writes `HEADER_LINE` before the first row, and keeps every line on the grid.

    A line that is not a trace record - `detector.py`'s data-size summary, a
    warning from anywhere - has no columns to sit in, so it is parked in the
    `path` column under `MESSAGE`. The file stays table-shaped on every line
    rather than gaining a ragged free-form region, and `types=` filters it away.
    """

    def emit(self, record) -> None:
        try:
            message = self.format(record)
            if not is_state_line(message):
                message = format_row({"TYPE": MESSAGE, PATH_COLUMN: message})
            if not self._header_written:
                self._header_written = True
                self.stream.write(HEADER_LINE + "\n")
            record.msg, record.args = message, None
            super().emit(record)
        except Exception:
            self.handleError(record)


class TableStreamHandler(_TableFormatMixin, logging.StreamHandler):
    """Streams the trace to the terminal as the table format, one row per record."""

    def __init__(self, stream=None) -> None:
        super().__init__(stream)
        self._header_written = False


class TableFileHandler(_TableFormatMixin, logging.FileHandler):
    """Writes the trace to a file as the table format, one row per record.

    Written as the records are produced, so the file is readable during a run and
    complete after a crash - the layout needs no look-ahead.
    """

    def __init__(self, filename, mode="a", encoding=None, delay=False, errors=None) -> None:
        super().__init__(filename, mode, encoding, delay, errors)
        self._header_written = False


def _cli(argv) -> int:
    """Entry point for `python -m src.utilities.logconfig <log_file> [--types ...]`."""
    import argparse
    import os

    parser = argparse.ArgumentParser(
        prog="logconfig",
        description="Render a luminous trace log as an aligned ASCII table.")
    parser.add_argument("log_file", nargs="?", default="luminous.log",
                        help="the trace log written by Scene(log_file=...)")
    parser.add_argument("--types", nargs="*", default=None,
                        help="filter to these TYPE tokens, e.g. --types %s %s"
                             % (TERMINATION, SHADERS))
    args = parser.parse_args(argv)

    if not os.path.exists(args.log_file):
        print("no such log file: %s\npass a path, e.g. `python -m src.utilities.logconfig "
              "results/luminous.log`" % args.log_file, file=sys.stderr)
        return 2

    table = read_log_table(args.log_file, types=args.types)
    if not table:
        print("no trace records in %s (empty log?)" % args.log_file, file=sys.stderr)
        return 1

    print(table)
    return 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))