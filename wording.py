"""Finds every string the site says, so that all of it can be rewritten.

Nothing here is a list of strings to maintain.  The site's own source is read
and the user-visible text picked out of it by how it is *used* -- text between
tags, ``textContent = ...``, ``say(...)``, ``SolverError(...)`` and so on --
so a string written the ordinary way is editable the moment it exists, with no
step in between.

Each occurrence is found at a character offset, so two identical strings in two
places are two separate things to edit.  Rewriting therefore happens to the
served bytes of ``index.html`` and ``app.js`` rather than to the rendered page:
by the time text reaches the DOM there is no telling the two apart.

An occurrence's id is ``<file>:<hash of the text>:<n>``, where n counts from 0
among identical strings *in that file*.  Line numbers and offsets both move
whenever anything above them is edited, so neither can key an override meant to
survive the code being worked on; this survives everything short of reordering
two identical strings within one file.

Values baked into a string -- ``${count}`` in the script, ``{row}`` in the
server -- are held to be part of it: an edit may move one but not drop it.
"""

import hashlib
import json
import os
import re
import threading
import time

from wordle_data import write_json

HERE = os.path.dirname(os.path.abspath(__file__))
OVERRIDES_PATH = os.path.join(HERE, "wording.json")

# punctuation, markup and lone symbols are not worth offering
SKIP = re.compile(r"^[\s&;#\-—–·…\d/|+.,:()\[\]{}<>=*^%$@!?~`'\"\\]*$")
ENTITY = re.compile(r"&[a-zA-Z]+;|&#\d+;")

VOID_TAGS = {"br", "img", "input", "link", "meta", "hr", "source", "wbr"}


def worth_offering(text):
    """True if there is a word in there to rewrite.

    ``&mdash;`` and ``&larr; &rarr;`` are the page drawing a symbol, not saying
    something, and a panel full of them is a panel that is harder to read.
    """
    if not text.strip():
        return False
    return not SKIP.match(ENTITY.sub("", text).strip())


def tidy(text):
    """A string as it reads, rather than as it is laid out in the source: prose
    wrapped over several lines is one line of text on the page."""
    return re.sub(r"\s*\n\s*", " ", text).strip()


class Occurrence:
    """One place one string is written."""

    __slots__ = ("id", "file", "start", "end", "text", "line", "context",
                 "kind", "quote", "slots")

    def __init__(self, file, start, end, text, line, context, kind, quote=""):
        self.file = file
        # the span is where it lives in the file; the text is how it reads
        self.start = start
        self.end = end
        self.text = tidy(text)
        self.line = line
        self.context = context
        self.kind = kind
        self.quote = quote
        self.slots = tuple(slots_in(text, quote))
        self.id = ""

    def as_dict(self, override=None):
        return {
            "id": self.id,
            "file": self.file,
            "line": self.line,
            "context": self.context,
            "kind": self.kind,
            "default": self.text,
            "value": override if override is not None else self.text,
            "edited": override is not None,
            "slots": list(self.slots),
        }


def slots_in(text, quote):
    """The values baked into a string, which an edit has to keep."""
    if quote == "`":
        return re.findall(r"\$\{[^{}]*\}", text)
    if quote in ("'", '"'):
        return re.findall(r"\{[a-zA-Z_][\w .\[\]'\"()+-]*\}", text)
    return []


def _digest(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def _line_of(source, offset):
    return source.count("\n", 0, offset) + 1


# ----------------------------------------------------------------------- HTML

_TAG = re.compile(r"<(/?)([A-Za-z][\w-]*)((?:[^>\"']|\"[^\"]*\"|'[^']*')*)>")
_HTML_TEXT = re.compile(r">([^<>]+)<")
_HTML_ATTR = re.compile(r'\b(title|placeholder|aria-label)="([^"]*)"')
_HTML_SKIPPED = re.compile(r"<(script|style)\b.*?</\1>|<!--.*?-->", re.S)


def _html_context(source, offset):
    """Where in the page this text sits: nearest labeled ancestor, then tag."""
    stack = []
    for match in _TAG.finditer(source, 0, offset):
        closing, tag, attrs = match.group(1), match.group(2).lower(), match.group(3)
        if closing:
            for index in range(len(stack) - 1, -1, -1):
                if stack[index][0] == tag:
                    del stack[index:]
                    break
        elif tag not in VOID_TAGS and not attrs.rstrip().endswith("/"):
            found = re.search(r'\bid="([^"]+)"', attrs)
            stack.append((tag, found.group(1) if found else None))
    labels = [name for _, name in stack if name]
    tag = stack[-1][0] if stack else ""
    parts = []
    if labels:
        parts.append("#" + labels[-1])
    if tag:
        parts.append("<%s>" % tag)
    return " ".join(parts)


# what an attribute's text is for, since none of it is painted on the page
ATTRIBUTE_KINDS = {
    "aria-label": "read aloud, never shown",
    "title": "hover text",
    "placeholder": "gray text inside the box",
}


def _attribute_owner(source, offset):
    """The element the attribute is written on.

    Not ``_html_context``: that walks the tags that have *finished* before this
    point, and the tag holding the attribute is still open, so it would name the
    parent instead.
    """
    start = source.rfind("<", 0, offset)
    if start < 0:
        return ""
    end = source.find(">", offset)
    opening = source[start:end if end > 0 else offset]
    tag = re.match(r"<([A-Za-z][\w-]*)", opening)
    found = re.search(r'\bid="([^"]+)"', opening)
    parts = []
    if found:
        parts.append("#" + found.group(1))
    if tag:
        parts.append("<%s>" % tag.group(1))
    return " ".join(parts)


def scan_html(source, name):
    blanked = _HTML_SKIPPED.sub(lambda m: " " * len(m.group(0)), source)
    found = []
    for match in _HTML_TEXT.finditer(blanked):
        raw = match.group(1)
        text = raw.strip()
        if not worth_offering(text):
            continue
        start = match.start(1) + (len(raw) - len(raw.lstrip()))
        found.append(Occurrence(name, start, start + len(text), text,
                                _line_of(source, start),
                                _html_context(source, match.start()), "markup"))
    for match in _HTML_ATTR.finditer(blanked):
        text = match.group(2).strip()
        if not worth_offering(text):
            continue
        found.append(Occurrence(name, match.start(2), match.end(2), text,
                                _line_of(source, match.start(2)),
                                _attribute_owner(source, match.start()),
                                ATTRIBUTE_KINDS.get(match.group(1), "attribute")))
    return found


# ----------------------------------------------------------------- JS and PY

def _literals(source):
    """(body start, body end, text, quote) for every quoted literal.

    Hand-rolled rather than a regex: backticked text runs over several lines and
    holds the longest prose on the site, which a line-bounded pattern misses.
    """
    out = []
    index, length = 0, len(source)
    while index < length:
        char = source[index]
        if char in "\"'`":
            end = index + 1
            while end < length:
                if source[end] == "\\":
                    end += 2
                    continue
                if source[end] == char:
                    break
                if char != "`" and source[end] == "\n":
                    break
                end += 1
            if end < length and source[end] == char:
                out.append((index + 1, end, source[index + 1:end], char))
                index = end + 1
                continue
        index += 1
    return out


def _blanked_comments(source, style):
    """The source with comments replaced by spaces, so offsets still line up."""
    pattern = r"//[^\n]*|/\*(?:.|\n)*?\*/" if style == "js" else r"#[^\n]*"
    return re.sub(pattern, lambda m: " " * len(m.group(0)), source)


def _docstrings(source):
    return [(m.start(), m.end()) for m in re.finditer(r'"""(?:.|\n)*?"""', source)]


# a literal is user-visible if it is being assigned to something that shows it
_JS_LEAD = re.compile(
    r"(?:textContent|innerHTML|\.title|\.placeholder|\.label)\s*=\s*$"
    r"|say\(\s*$"
    r"|\bnote\s*=\s*$"
    r"|\b(?:name|value|why|label|units|summary|description|solve|play)\s*:\s*$"
)
# ...or if it is an arm of a choice between two such strings, which is only
# safe to assume on a line that is doing one of the above
_JS_ARM = re.compile(r"[?:]\s*$")
_JS_TEXTY = re.compile(r"textContent|\.title|say\(|\bnote\b"
                       r"|(?:name|value|units|why|label)\s*:")

_PY_LEAD = re.compile(
    r"SolverError\(\s*$"
    r"|self\.progress\s*=\s*$"
    r"|send_error_json\(\s*$"
    r"|Game\([^()]*,\s*$"
)

_PREFIX = re.compile(r"[fruFRUbB]{0,2}$")


def _js_context(source, offset):
    head = source[:offset]
    names = re.findall(r"^(?:async\s+)?function\s+(\w+)|^(?:const|let|var)\s+(\w+)\s*=",
                       head, re.M)
    for first, second in reversed(names):
        if first or second:
            return (first or second) + "()"
    return ""


def _py_context(source, offset):
    head = source[:offset]
    names = re.findall(r"^\s*def\s+(\w+)|^class\s+(\w+)", head, re.M)
    for first, second in reversed(names):
        if first or second:
            return (first + "()") if first else second
    return ""


def scan_code(source, name, style):
    lead = _JS_LEAD if style == "js" else _PY_LEAD
    blanked = _blanked_comments(source, style)
    skip = _docstrings(source) if style == "py" else []
    context = _js_context if style == "js" else _py_context
    found = []
    for start, end, text, quote in _literals(blanked):
        real = source[start:end]
        if not worth_offering(real):
            continue
        if any(a <= start < b for a, b in skip):
            continue
        before = _PREFIX.sub("", source[max(0, start - 60):start - 1])
        if not lead.search(before):
            if style != "js" or not _JS_ARM.search(before):
                continue
            line_start = source.rfind("\n", 0, start)
            if not _JS_TEXTY.search(source[line_start:start]):
                continue
        kind = "script" if style == "js" else "server"
        if style == "js" and re.search(r"\.title\s*=\s*$", before):
            kind = ATTRIBUTE_KINDS["title"]   # also never shown until hovered
        found.append(Occurrence(name, start, end, real, _line_of(source, start),
                                context(source, start), kind, quote))
    return found


# -------------------------------------------------------------------- the scan

SOURCES = [
    ("web/index.html", scan_html, None),
    ("web/app.js", scan_code, "js"),
    ("web.py", scan_code, "py"),
]


def read(name):
    with open(os.path.join(HERE, name), "r", encoding="utf-8") as handle:
        return handle.read()


def scan_file(name, scanner, style):
    source = read(name)
    found = scanner(source, name) if style is None else scanner(source, name, style)
    found.sort(key=lambda o: o.start)
    seen = {}
    for occurrence in found:
        digest = _digest(occurrence.text)
        index = seen.get(digest, 0)
        seen[digest] = index + 1
        occurrence.id = "%s:%s:%d" % (name, digest, index)
    return found


def scan():
    """Every occurrence in every source, in file then document order."""
    out = []
    for name, scanner, style in SOURCES:
        out.extend(scan_file(name, scanner, style))
    return out


# --------------------------------------------------------------------- writing

def escape_html(text):
    return text.replace("&", "&amp;").replace("<", "&lt;")


def escape_js(text, quote):
    text = text.replace("\\", "\\\\").replace(quote, "\\" + quote)
    if quote != "`":
        return text.replace("\n", " ")
    # ${...} is left alone: it is how the value gets in
    return text


def rewrite(source, occurrences, overrides, style, wrap=False):
    """Apply overrides to a file's text, back to front so offsets hold.

    ``wrap`` lays long prose out over several lines at the indentation it found,
    which matters only when writing back to the source: nobody wants a 300
    column line in the markup, and ``tidy`` folds it back to one string when the
    file is read again.
    """
    for occurrence in sorted(occurrences, key=lambda o: o.start, reverse=True):
        replacement = overrides.get(occurrence.id)
        if replacement is None or replacement == occurrence.text:
            continue
        if style == "html":
            replacement = escape_html(replacement)
            if wrap and occurrence.kind == "markup":
                replacement = _wrapped(source, occurrence.start, replacement)
        else:
            replacement = escape_js(replacement, occurrence.quote)
        source = source[:occurrence.start] + replacement + source[occurrence.end:]
    return source


def _wrapped(source, start, text, width=78):
    """``text`` over as many lines as it needs, indented to match its line."""
    if len(text) <= width:
        return text
    line_start = source.rfind("\n", 0, start) + 1
    indent = re.match(r"[ \t]*", source[line_start:start]).group(0) or "  "
    lines, current = [], ""
    for word in text.split():
        if current and len(indent) + len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = (current + " " + word).strip()
    if current:
        lines.append(current)
    return ("\n" + indent).join(lines)


class Wording:
    """The overrides on disk, and the scan they apply to."""

    def __init__(self, path=OVERRIDES_PATH):
        self.path = path
        self.lock = threading.Lock()
        self.overrides = {}
        self._messages = None
        self.load()

    def load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                saved = json.load(handle)
        except (OSError, ValueError):
            saved = {}
        with self.lock:
            self.overrides = {k: v for k, v in saved.items() if isinstance(v, str)}
            self._messages = None

    def _write(self):
        tmp = self.path + ".tmp"
        write_json(tmp, self.overrides, indent=2, sort_keys=True)
        os.replace(tmp, self.path)
        self._messages = None

    # ------------------------------------------------------------------- reads

    def manifest(self):
        """Everything the editor shows: every string, where it is written, and
        what it currently says."""
        found = scan()
        with self.lock:
            overrides = dict(self.overrides)
        known = {o.id for o in found}
        return {
            "entries": [o.as_dict(overrides.get(o.id)) for o in found],
            "files": [name for name, _, _ in SOURCES],
            # edits whose string has since been rewritten or removed in the
            # source, so they no longer apply to anything
            "orphans": sorted(key for key in overrides if key not in known),
            "count": len(overrides),
        }

    def served(self, name):
        """A source file with the overrides applied to it."""
        source = read(name)
        with self.lock:
            overrides = dict(self.overrides)
        if not overrides:
            return source
        scanner, style = next((s, t) for n, s, t in SOURCES if n == name)
        found = scanner(source, name) if style is None else scanner(source, name, style)
        seen = {}
        for occurrence in sorted(found, key=lambda o: o.start):
            digest = _digest(occurrence.text)
            index = seen.get(digest, 0)
            seen[digest] = index + 1
            occurrence.id = "%s:%s:%d" % (name, digest, index)
        return rewrite(source, found, overrides,
                       "html" if name.endswith(".html") else "js")

    def message(self, text):
        """An override for a string this process is about to send.

        The server's own literals were compiled long before any edit, so they
        are matched on the way out rather than at the point they were written.
        That means matching against the finished message, values and all: a
        literal like ``Row {row} is empty`` becomes a pattern, and whatever it
        captured is put back into the rewritten version.

        Matching is by content, so a message written twice in web.py is one
        thing to edit rather than two.  Everything the *page* says goes through
        ``served`` instead, which is per-occurrence.
        """
        with self.lock:
            if not self.overrides:
                return text
            if self._messages is None:
                self._messages = self._message_table()
            plain, patterned = self._messages

        if text in plain:
            return plain[text]
        for pattern, names, replacement in patterned:
            found = pattern.match(text)
            if not found:
                continue
            out = replacement
            for name, value in zip(names, found.groups()):
                out = out.replace(name, value, 1)
            return out
        return text

    def _message_table(self):
        """``({exact: text}, [(pattern, slot names, replacement), ...])``."""
        plain, patterned = {}, []
        for occurrence in scan_file("web.py", scan_code, "py"):
            replacement = self.overrides.get(occurrence.id)
            if replacement is None:
                continue
            if not occurrence.slots:
                plain[occurrence.text] = replacement
                continue
            # the literal, with each {value} standing in for anything
            pattern = "".join(
                "(.+?)" if part in occurrence.slots else re.escape(part)
                for part in re.split(r"(\{[^{}]*\})", occurrence.text) if part)
            try:
                patterned.append((re.compile("^" + pattern + "$", re.S),
                                  [p for p in re.split(r"(\{[^{}]*\})", occurrence.text)
                                   if p in occurrence.slots],
                                  replacement))
            except re.error:
                continue
        return plain, patterned

    # ------------------------------------------------------------------ writes

    def save(self, edits):
        """``{id: text}``; a value equal to the original clears the override."""
        if not isinstance(edits, dict):
            raise ValueError("expected an object of id: text")
        known = {o.id: o for o in scan()}
        cleaned = {}
        for key, value in edits.items():
            occurrence = known.get(key)
            if occurrence is None:
                raise ValueError("no such text: %s" % key)
            if not isinstance(value, str):
                raise ValueError("%s: expected text" % key)
            value = re.sub(r"\s*\n\s*", " ", value.replace("\r\n", "\n")).strip()
            if not value:
                raise ValueError("%r: cannot be emptied" % occurrence.text[:40])
            missing = [slot for slot in occurrence.slots if slot not in value]
            if missing:
                raise ValueError("%r must still contain %s"
                                 % (occurrence.text[:40], ", ".join(missing)))
            invented = [slot for slot in slots_in(value, occurrence.quote)
                        if slot not in occurrence.slots]
            if invented:
                raise ValueError("%r has no value called %s"
                                 % (occurrence.text[:40], ", ".join(invented)))
            cleaned[key] = value

        with self.lock:
            for key, value in cleaned.items():
                if value == known[key].text:
                    self.overrides.pop(key, None)
                else:
                    self.overrides[key] = value
            self._write()
            return len(self.overrides)

    def bake(self):
        """Write the rewrites into the source files themselves, and forget them.

        Overrides sit beside the code and are matched to it by wording, which is
        what makes them fragile the moment that wording is worked on.  Writing
        them in ends that: the source says what it says, and there is nothing
        left to keep in step.

        ``wording_history.json`` keeps the old and new phrases.
        """
        with self.lock:
            overrides = dict(self.overrides)
        if not overrides:
            return {"written": [], "phrases": 0}

        stamp = time.strftime("%Y%m%d-%H%M%S")
        written, record = [], []

        for name, scanner, style in SOURCES:
            found = scan_file(name, scanner, style)
            mine = [o for o in found if o.id in overrides
                    and overrides[o.id] != o.text]
            if not mine:
                continue
            source = read(name)
            updated = rewrite(source, mine, overrides,
                              "html" if name.endswith(".html") else "js", wrap=True)
            # newline="" so baking a phrase does not also rewrite every line
            with open(os.path.join(HERE, name), "w", encoding="utf-8",
                      newline="") as handle:
                handle.write(updated)
            written.append(name)
            record.extend({"file": name, "line": o.line, "context": o.context,
                           "was": o.text, "now": overrides[o.id]} for o in mine)

        self._log(stamp, record)
        # every override now describes wording that is no longer in the source,
        # so keeping them would only leave a list of orphans behind
        with self.lock:
            self.overrides.clear()
            self._write()
        # not "count": the manifest this gets merged with has one of those
        return {"written": written, "phrases": len(record)}

    def _log(self, stamp, record):
        path = os.path.join(HERE, "wording_history.json")
        try:
            with open(path, "r", encoding="utf-8") as handle:
                history = json.load(handle)
            if not isinstance(history, list):
                history = []
        except (OSError, ValueError):
            history = []
        history.append({"applied": stamp, "changes": record})
        write_json(path, history, indent=2)

    def forget(self, keys=None):
        with self.lock:
            if keys is None:
                self.overrides.clear()
            else:
                for key in keys:
                    self.overrides.pop(key, None)
            self._write()
            return len(self.overrides)
