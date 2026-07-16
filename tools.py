from typing import List, Optional, Dict, Any
import logging
import os
import fnmatch
import re
import mimetypes
import time

logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.DEBUG)


# Tracks files that were read via the Read tool in this process
READ_HISTORY: Dict[str, float] = {}  # file_path -> last_read_timestamp


# Helpful docs on all Claude Code tools
# https://blog.thepete.net/claude-code-tools/#read


# Register your tools with: name -> (callable, json-schema-like params)
# Each callable must accept **kwargs and return a serializable dict or string.
def tool_write(file_path: str, content: str) -> Dict[str, Any]:
    # EXAMPLE TOOL: Writes a file. Adjust path safety for your environment!

    #logger.info(f"Tool Write called with file_path={file_path}, content length={len(content)}")

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return {"ok": True, "file_path": file_path, "bytes": len(content)}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def tool_edit(
    file_path: str,
    old_string: str,
    new_string: str,
    replace_all: Optional[bool] = False,
    encoding: str = "utf-8",
) -> Dict[str, Any]:
    """
    Performs exact string replacements in files.

    Enforced rules:
    - file_path must be absolute and point to an existing regular file
    - Must have been read via the Read tool at least once in this process (READ_HISTORY)
    - old_string and new_string must be non-empty and different
    - If replace_all is False:
        - exactly one occurrence must exist, otherwise the edit FAILS
      If replace_all is True:
        - all occurrences are replaced; if zero found, returns ok=False with error

    Notes for the caller (Claude):
    - When editing using Read tool output (cat -n format), never include the line number prefix in old_string/new_string.
      Only include the content after the tab.
    """
    try:
        # Validate inputs
        if not isinstance(file_path, str) or not file_path:
            return {"ok": False, "error": "file_path (string) is required"}
        if not os.path.isabs(file_path):
            return {"ok": False, "error": "file_path must be an absolute path"}
        if not os.path.exists(file_path):
            return {"ok": False, "error": f"File not found: {file_path}"}
        if not os.path.isfile(file_path):
            return {"ok": False, "error": f"Not a regular file: {file_path}"}

        if not isinstance(old_string, str) or old_string == "":
            return {"ok": False, "error": "old_string (non-empty string) is required"}
        if not isinstance(new_string, str) or new_string == "":
            return {"ok": False, "error": "new_string (non-empty string) is required"}
        if old_string == new_string:
            return {"ok": False, "error": "new_string must be different from old_string"}

        # Enforce "must have used Read at least once in the conversation"
        # We approximate this by requiring that this process has recorded a Read of the file.
        last_read = READ_HISTORY.get(file_path)
        if last_read is None:
            return {
                "ok": False,
                "error": (
                    "Edit denied: you must use the Read tool on this file at least once before editing. "
                    "Please call Read and then retry the edit."
                ),
            }

        # Read the file content
        try:
            with open(file_path, "r", encoding=encoding, errors="strict") as fh:
                original = fh.read()
        except UnicodeDecodeError:
            # Retry with replace to avoid failure, but warn
            with open(file_path, "r", encoding=encoding, errors="replace") as fh:
                original = fh.read()

        occurrences = original.count(old_string)

        if replace_all:
            if occurrences == 0:
                return {
                    "ok": False,
                    "error": "No occurrences of old_string found in file. Nothing to replace.",
                    "occurrences": 0,
                }
            new_content = original.replace(old_string, new_string)
            updated = (new_content != original)
            if not updated:
                return {"ok": False, "error": "Replacement produced no change (unexpected)."}
            # Write back
            with open(file_path, "w", encoding=encoding, errors="replace") as fh:
                fh.write(new_content)
            return {
                "ok": True,
                "file_path": file_path,
                "mode": "replace_all",
                "occurrences_replaced": occurrences,
                "bytes_before": len(original.encode(encoding, errors="replace")),
                "bytes_after": len(new_content.encode(encoding, errors="replace")),
            }
        else:
            # Must be exactly one occurrence
            if occurrences == 0:
                return {
                    "ok": False,
                    "error": "old_string not found in file. Provide more context or use replace_all if appropriate.",
                    "occurrences": 0,
                }
            if occurrences > 1:
                return {
                    "ok": False,
                    "error": (
                        "Edit failed: old_string matched multiple locations. "
                        "Provide a longer old_string with more surrounding context or set replace_all=true."
                    ),
                    "occurrences": occurrences,
                }
            # Replace exactly once
            new_content = original.replace(old_string, new_string, 1)
            if new_content == original:
                return {"ok": False, "error": "Replacement produced no change (unexpected)."}
            with open(file_path, "w", encoding=encoding, errors="replace") as fh:
                fh.write(new_content)
            return {
                "ok": True,
                "file_path": file_path,
                "mode": "single",
                "occurrences_replaced": 1,
                "bytes_before": len(original.encode(encoding, errors="replace")),
                "bytes_after": len(new_content.encode(encoding, errors="replace")),
            }

    except Exception as e:
        return {"ok": False, "error": str(e)}

def tool_read(
    file_path: str,
    offset: Optional[int] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Read a file and return lines in `cat -n` format.

    Behavior (as specified):
    - file_path must be an absolute path
    - By default reads up to 2000 lines starting from the beginning (offset=0, limit=2000)
    - Optional offset (start line number) and limit (number of lines)
    - Lines longer than 2000 characters are truncated
    - Returns cat -n formatted text with line numbers starting at 1
    - If file exists but empty, returns a system reminder instead of content
    - For binary/visual files (e.g., images, PDFs, notebooks) returns a descriptive note
    """
    try:
        # Enforce absolute path
        if not isinstance(file_path, str) or not file_path:
            return {"ok": False, "error": "file_path (string) is required"}
        if not os.path.isabs(file_path):
            return {"ok": False, "error": "file_path must be an absolute path"}

        # Defaults: offset=0, limit=2000
        off = 0 if offset is None else max(0, int(offset))
        lim = 2000 if limit is None else max(0, int(limit))

        if not os.path.exists(file_path):
            # It is okay to read a file that does not exist; an error will be returned
            return {"ok": False, "error": f"File not found: {file_path}"}
        if not os.path.isfile(file_path):
            return {"ok": False, "error": f"Not a regular file: {file_path}"}

        size_bytes = os.path.getsize(file_path)

        # Heuristic: decide text vs binary based on MIME type and decodability
        mime, _ = mimetypes.guess_type(file_path)
        is_probably_text = False
        if mime is None:
            # Unknown; try a small decode probe
            try:
                with open(file_path, "rb") as fh:
                    probe = fh.read(4096)
                probe.decode("utf-8")
                is_probably_text = True
            except Exception:
                is_probably_text = False
        else:
            # Treat common texty types as text
            is_probably_text = mime.startswith("text/") or mime in (
                "application/json",
                "application/xml",
                "application/javascript",
            )

        # Special-casing of known "visual"/binary types
        visual_exts = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff"}
        pdf_exts = {".pdf"}
        ipynb_exts = {".ipynb"}
        _, ext = os.path.splitext(file_path.lower())

        if ((mime and not is_probably_text) or ext in visual_exts or ext in pdf_exts or ext in ipynb_exts):
            # Return a descriptive note for visual/binary content
            kind = "binary"
            if ext in visual_exts:
                kind = "image"
            elif ext in pdf_exts:
                kind = "pdf"
            elif ext in ipynb_exts:
                kind = "notebook"

            note_lines = [
                f"[Read] {kind.upper()} file: {file_path}",
                f"[Read] MIME: {mime or 'application/octet-stream'}; Size: {size_bytes} bytes",
                "[Read] Contents are visual/binary and will be interpreted by the client.",
            ]
            output_text = "\n".join(note_lines)

            READ_HISTORY[file_path] = time.time()
            return {
                "ok": True,
                "file_path": file_path,
                "bytes": size_bytes,
                "type": kind,
                "content": output_text,
                "format": "note",
            }

        # Stream the file as text, line-by-line, to handle very large files
        total_lines = 0
        lines_returned = 0
        truncated_lines = 0

        # cat -n numbers lines starting at 1 for the whole file
        # We will still emit numbers based on the true line index, even when offset>0
        output_chunks: List[str] = []
        max_line_len = 2000

        with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
            for idx, raw_line in enumerate(fh, start=1):
                total_lines += 1

                # Only emit if within the offset/limit window
                if idx <= off:
                    continue
                if lim is not None and lines_returned >= lim:
                    break

                line = raw_line.rstrip("\n")
                if len(line) > max_line_len:
                    line = line[:max_line_len]
                    truncated_lines += 1

                # cat -n style: right-aligned width 6 + tab
                output_chunks.append(f"{idx:>6}\t{line}")
                lines_returned += 1

        # Handle empty file: system reminder instead of content
        if total_lines == 0:
            READ_HISTORY[file_path] = time.time()
            return {
                "ok": True,
                "file_path": file_path,
                "bytes": size_bytes,
                "content": "[System reminder] File exists but is empty.",
                "format": "cat-n",
                "total_lines": 0,
                "offset": off,
                "limit": lim,
                "truncated_lines": 0,
            }

        READ_HISTORY[file_path] = time.time()
        return {
            "ok": True,
            "file_path": file_path,
            "bytes": size_bytes,
            "content": "\n".join(output_chunks),
            "format": "cat-n",
            "total_lines": total_lines,
            "offset": off,
            "limit": lim,
            "truncated_lines": truncated_lines,
        }

    except Exception as e:
        return {"ok": False, "error": str(e)}


def tool_delete(
    file_path: str,
    allow_missing: Optional[bool] = False,
) -> Dict[str, Any]:
    """
    Deletes a file from the local filesystem.

    Behavior:
    - file_path must be an absolute path
    - Deletes regular files and symlinks (removes the link, not the target)
    - Does NOT delete directories
    - If allow_missing is True and the file doesn't exist, returns ok=True, removed=False
    - Returns metadata including whether it was a symlink and the size (best-effort)

    Returns:
      {
        "ok": True,
        "file_path": "<path>",
        "removed": True|False,
        "was_symlink": True|False,
        "bytes": <int|None>,
      }
    or on error:
      {"ok": False, "error": "<message>"}
    """
    try:
        if not isinstance(file_path, str) or not file_path:
            return {"ok": False, "error": "file_path (string) is required"}
        if not os.path.isabs(file_path):
            return {"ok": False, "error": "file_path must be an absolute path"}

        # Not found handling
        if not os.path.exists(file_path) and not os.path.islink(file_path):
            if allow_missing:
                return {
                    "ok": True,
                    "file_path": file_path,
                    "removed": False,
                    "was_symlink": False,
                    "bytes": None,
                    "note": "File did not exist",
                }
            return {"ok": False, "error": f"File not found: {file_path}"}

        # Disallow directories
        if os.path.isdir(file_path) and not os.path.islink(file_path):
            return {"ok": False, "error": "Refuses to delete directories. Only files and symlinks are allowed."}

        was_symlink = os.path.islink(file_path)

        # Best-effort size (for symlinks, lstat size; for files, actual size)
        try:
            if was_symlink:
                st = os.lstat(file_path)
                size_bytes = st.st_size
            else:
                size_bytes = os.path.getsize(file_path)
        except Exception:
            size_bytes = None

        # Try to make file writable if needed
        try:
            os.chmod(file_path, 0o666)
        except Exception:
            pass

        # Remove file or symlink
        os.remove(file_path)

        return {
            "ok": True,
            "file_path": file_path,
            "removed": True,
            "was_symlink": was_symlink,
            "bytes": size_bytes,
        }

    except PermissionError as e:
        return {"ok": False, "error": f"Permission denied: {e}"}
    except IsADirectoryError:
        return {"ok": False, "error": "Refuses to delete directories. Only files and symlinks are allowed."}
    except FileNotFoundError:
        if allow_missing:
            return {
                "ok": True,
                "file_path": file_path,
                "removed": False,
                "was_symlink": False,
                "bytes": None,
                "note": "File did not exist",
            }
        return {"ok": False, "error": f"File not found: {file_path}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ========================
# ==========GREP==========
# ========================


# Subset map for --type like ripgrep; extend as needed
_RG_TYPE_MAP: Dict[str, List[str]] = {
    "js": [".js", ".mjs", ".cjs", ".jsx"],
    "ts": [".ts"],
    "tsx": [".tsx"],
    "py": [".py"],
    "rust": [".rs"],
    "rs": [".rs"],
    "go": [".go"],
    "java": [".java"],
    "c": [".c", ".h"],
    "cpp": [".cpp", ".cxx", ".cc", ".hpp", ".hh", ".hxx"],
    "cs": [".cs"],
    "php": [".php"],
    "rb": [".rb"],
    "kt": [".kt", ".kts"],
    "swift": [".swift"],
    "json": [".json"],
    "yaml": [".yaml", ".yml"],
    "toml": [".toml"],
    "md": [".md", ".markdown"],
    "html": [".html", ".htm"],
    "css": [".css", ".scss", ".sass"],
    "sh": [".sh", ".bash"],
    # add more on demand
}


def _expand_brace_glob(glob_pattern: str) -> List[str]:
    """
    Expand simple brace sets like .{ts,tsx} into [".ts", ".tsx"].
    If no braces, returns [glob_pattern].
    """
    if "{" not in glob_pattern or "}" not in glob_pattern:
        return [glob_pattern]
    # Very simple expansion: find first {...}
    try:
        pre, rest = glob_pattern.split("{", 1)
        body, post = rest.split("}", 1)
        alts = [pre + alt + post for alt in body.split(",")]
        return alts
    except Exception:
        return [glob_pattern]


def _iter_files(start_path: str) -> List[str]:
    if os.path.isfile(start_path):
        return [os.path.abspath(start_path)]
    paths: List[str] = []
    for dirpath, _, filenames in os.walk(start_path):
        for fname in filenames:
            paths.append(os.path.join(dirpath, fname))
    return paths


def _passes_glob(path: str, root: str, glob_pat: Optional[str]) -> bool:
    if not glob_pat:
        return True
    rel = os.path.relpath(path, root).replace("\\", "/")
    # support brace sets by checking all expanded globs
    for gp in _expand_brace_glob(glob_pat):
        if fnmatch.fnmatch(rel, gp) or fnmatch.fnmatch(os.path.basename(rel), gp):
            return True
    return False


def _passes_type(path: str, type_name: Optional[str]) -> bool:
    if not type_name:
        return True
    exts = _RG_TYPE_MAP.get(type_name.lower())
    if not exts:
        # Unknown type => conservative: do not include
        return False
    _, ext = os.path.splitext(path.lower())
    return ext in exts


def _compile_pattern(pattern: str, ignore_case: bool, multiline: bool) -> re.Pattern:
    flags = 0
    if ignore_case:
        flags |= re.IGNORECASE
    if multiline:
        flags |= re.DOTALL  # dot matches newline
    return re.compile(pattern, flags)


def _line_start_indices(text: str) -> List[int]:
    """Return list of indices where each line starts to compute line numbers quickly."""
    idxs = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            idxs.append(i + 1)
    return idxs


def _pos_to_line_col(pos: int, line_starts: List[int]) -> int:
    """Binary search to map byte/char index to 1-based line number."""
    lo, hi = 0, len(line_starts) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if line_starts[mid] <= pos:
            lo = mid + 1
        else:
            hi = mid - 1
    return hi + 1  # 1-based line


def tool_grep(**kwargs) -> Dict[str, Any]:
    """
    Grep: ripgrep-like search tool implemented in Python.

    Accepted kwargs (with ripgrep-like names):
      - pattern: str (required)
      - path: Optional[str] (defaults to os.getcwd())
      - glob: Optional[str] (e.g., "*.ts" or ".{ts,tsx}")
      - output_mode: Optional[str]: "content" | "files_with_matches" | "count" (default "files_with_matches")
      - -B: Optional[int]  (context before, content mode only)
      - -A: Optional[int]  (context after, content mode only)
      - -C: Optional[int]  (symmetric context, content mode only)
      - -n: Optional[bool] (line numbers in content mode; default True)
      - -i: Optional[bool] (case insensitive)
      - type: Optional[str] (file type filter)
      - head_limit: Optional[int] (limit returned entries/lines)
      - offset: Optional[int] (skip first N entries/lines)
      - multiline: Optional[bool] (dot matches newline)
    """
    try:
        # Extract and validate inputs (allow hyphenated keys)
        def take(name: str, default=None):
            return kwargs.get(name, default)

        pattern = take("pattern")
        if not isinstance(pattern, str) or not pattern:
            return {"ok": False, "error": "pattern (string) is required"}

        start_path = take("path") or os.getcwd()
        if not os.path.exists(start_path):
            return {"ok": False, "error": f"path does not exist: {start_path}"}

        glob_pat = take("glob")
        output_mode = take("output_mode") or "files_with_matches"
        if output_mode not in ("content", "files_with_matches", "count"):
            return {
                "ok": False,
                "error": "output_mode must be one of: content, files_with_matches, count",
            }

        # Context options (content mode)
        c_before = take("-B")
        c_after = take("-A")
        c_symmetric = take("-C")
        show_line_numbers = take("-n", True)
        ignore_case = bool(take("-i", False))
        type_name = take("type")
        head_limit = take("head_limit")
        offset = int(take("offset", 0) or 0)
        multiline = bool(take("multiline", False))

        # Normalize integers
        def _as_int(v, default=None):
            if v is None:
                return default
            try:
                return int(v)
            except Exception:
                return default

        c_before = _as_int(c_before, 0)
        c_after = _as_int(c_after, 0)
        c_symmetric = _as_int(c_symmetric, None)
        if c_symmetric is not None:
            c_before = c_symmetric
            c_after = c_symmetric

        head_limit = _as_int(head_limit, None)
        if head_limit is not None and head_limit < 0:
            head_limit = None
        if offset < 0:
            offset = 0

        rx = _compile_pattern(pattern, ignore_case=ignore_case, multiline=multiline)

        files = _iter_files(start_path)
        files_scanned = 0
        files_with_hits: List[str] = []
        per_file_counts: List[tuple[str, int]] = []
        content_lines: List[str] = []
        total_matches = 0

        for fpath in files:
            files_scanned += 1
            # Type/glob filters
            if not _passes_type(fpath, type_name):
                continue
            root_for_rel = (
                start_path if os.path.isdir(start_path) else os.path.dirname(start_path)
            )
            if not _passes_glob(fpath, root_for_rel, glob_pat):
                continue

            # Read file
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                    text = fh.read()
            except Exception:
                continue

            hits_in_file = 0
            if multiline:
                # Match across whole text
                line_starts = _line_start_indices(text)
                for m in rx.finditer(text):
                    hits_in_file += 1
                    total_matches += 1
                    line_num = _pos_to_line_col(m.start(), line_starts)
                    # Build output for content mode
                    if output_mode == "content":
                        # Context windows are by line around the starting line of the match
                        start_line = max(1, line_num - c_before)
                        end_line = min(len(line_starts), line_num + c_after)
                        # Emit before
                        for ln in range(start_line, line_num):
                            if show_line_numbers:
                                content_lines.append(
                                    f"{fpath}-{ln}-{text[line_starts[ln - 1] : text.find('\\n', line_starts[ln - 1]) if text.find('\\n', line_starts[ln - 1]) != -1 else len(text)].rstrip()}"
                                )
                            else:
                                content_lines.append(
                                    f"{fpath}-"
                                    + text[
                                        line_starts[ln - 1] : text.find(
                                            "\n", line_starts[ln - 1]
                                        )
                                        if text.find("\n", line_starts[ln - 1]) != -1
                                        else len(text)
                                    ].rstrip()
                                )
                        # Emit match line
                        match_line_end = text.find("\n", line_starts[line_num - 1])
                        if match_line_end == -1:
                            match_line_end = len(text)
                        line_text = text[
                            line_starts[line_num - 1] : match_line_end
                        ].rstrip()
                        if show_line_numbers:
                            content_lines.append(f"{fpath}:{line_num}:{line_text}")
                        else:
                            content_lines.append(f"{fpath}:{line_text}")
                        # Emit after
                        for ln in range(line_num + 1, end_line + 1):
                            if show_line_numbers:
                                content_lines.append(
                                    f"{fpath}+{ln}+{text[line_starts[ln - 1] : text.find('\\n', line_starts[ln - 1]) if text.find('\\n', line_starts[ln - 1]) != -1 else len(text)].rstrip()}"
                                )
                            else:
                                content_lines.append(
                                    f"{fpath}+"
                                    + text[
                                        line_starts[ln - 1] : text.find(
                                            "\n", line_starts[ln - 1]
                                        )
                                        if text.find("\n", line_starts[ln - 1]) != -1
                                        else len(text)
                                    ].rstrip()
                                )
            else:
                # Line-by-line search
                lines = text.splitlines()
                for idx, line in enumerate(lines, start=1):
                    matched = rx.search(line) is not None
                    if not matched:
                        continue
                    hits_in_file += 1
                    total_matches += 1
                    if output_mode == "content":
                        # Before context
                        for ln in range(max(1, idx - c_before), idx):
                            if show_line_numbers:
                                content_lines.append(f"{fpath}-{ln}-{lines[ln - 1]}")
                            else:
                                content_lines.append(f"{fpath}-{lines[ln - 1]}")
                        # Match line
                        if show_line_numbers:
                            content_lines.append(f"{fpath}:{idx}:{line}")
                        else:
                            content_lines.append(f"{fpath}:{line}")
                        # After context
                        for ln in range(idx + 1, min(len(lines), idx + c_after) + 1):
                            if show_line_numbers:
                                content_lines.append(f"{fpath}+{ln}+{lines[ln - 1]}")
                            else:
                                content_lines.append(f"{fpath}+{lines[ln - 1]}")

            if hits_in_file > 0:
                files_with_hits.append(fpath)
                per_file_counts.append((fpath, hits_in_file))

        # Build output according to mode
        if output_mode == "files_with_matches":
            entries = [p for p in files_with_hits]
            # Apply offset/head_limit window
            entries_window = entries[offset:] if offset else entries
            if head_limit is not None:
                entries_window = entries_window[:head_limit]
            output_str = "\n".join(entries_window)

            return {
                "ok": True,
                "output_mode": "files_with_matches",
                "output": output_str,
                "stats": {
                    "files_scanned": files_scanned,
                    "files_with_matches": len(files_with_hits),
                    "matches": total_matches,
                    "returned": len(entries_window),
                    "offset": offset,
                    "head_limit": head_limit,
                },
            }

        elif output_mode == "count":
            entries = [f"{p}:{cnt}" for p, cnt in per_file_counts if cnt > 0]
            entries_window = entries[offset:] if offset else entries
            if head_limit is not None:
                entries_window = entries_window[:head_limit]
            output_str = "\n".join(entries_window)

            return {
                "ok": True,
                "output_mode": "count",
                "output": output_str,
                "stats": {
                    "files_scanned": files_scanned,
                    "files_with_matches": len(files_with_hits),
                    "matches": total_matches,
                    "returned": len(entries_window),
                    "offset": offset,
                    "head_limit": head_limit,
                },
            }

        else:  # content
            # Apply window across all rendered lines
            entries = content_lines
            entries_window = entries[offset:] if offset else entries
            if head_limit is not None:
                entries_window = entries_window[:head_limit]
            output_str = "\n".join(entries_window)

            return {
                "ok": True,
                "output_mode": "content",
                "output": output_str,
                "stats": {
                    "files_scanned": files_scanned,
                    "files_with_matches": len(files_with_hits),
                    "matches": total_matches,
                    "lines_rendered": len(entries),
                    "returned": len(entries_window),
                    "offset": offset,
                    "head_limit": head_limit,
                    "line_numbers": bool(show_line_numbers),
                    "context": {"-B": c_before, "-A": c_after},
                    "multiline": multiline,
                },
            }

    except re.error as e:
        return {"ok": False, "error": f"Invalid regex: {e}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}
