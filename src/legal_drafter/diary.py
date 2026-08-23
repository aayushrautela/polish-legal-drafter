from __future__ import annotations

import fcntl
import json
import os
import re
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

ENTRY_RE = re.compile(r"^# ENTRY\s+(.*?)\s*$")
END_RE = re.compile(r"^# ENDENTRY seq=(\d+)\s*$")

DEFAULT_MAX_BYTES = 2 * 1024 * 1024  # rotate to the other buffer at ~2 MiB


def capture_context(repo_root: str | Path | None = None) -> dict:
    """Auto-capture environment context for an entry.

    Always includes local date/time; adds git commit awareness (hash, branch,
    dirty flag, short message) when run inside a git repository. Failures are
    swallowed and recorded as ``unknown`` so diary writes never break.
    """
    now = datetime.now()
    ctx: dict = {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "tz": now.astimezone().strftime("%z"),
    }
    try:
        root = str(repo_root) if repo_root else "."
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root, capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=root, capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root, capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip()
        msg = subprocess.run(
            ["git", "log", "-1", "--format=%s"],
            cwd=root, capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip()
        ctx["git"] = head[:12]
        ctx["branch"] = branch
        ctx["dirty"] = "1" if dirty else "0"
        ctx["commit_msg"] = msg
    except Exception:
        ctx["git"] = "unknown"
        ctx["branch"] = "unknown"
        ctx["dirty"] = "unknown"
    return ctx


def _format_kv(d: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in d.items() if v != "" and v is not None)


@dataclass(frozen=True)
class Entry:
    seq: int
    ts: str
    kind: str
    text: str
    meta: dict = None

    def __post_init__(self) -> None:
        if self.meta is None:
            object.__setattr__(self, "meta", {})


class DiaryError(Exception):
    pass


class Diary:
    """Append-only, crash-resilient work diary with lossless A/B rotation.

    The diary never rewrites or deletes history. Every write is an immutable
    append. Two physical buffers (``diary.a.log`` / ``diary.b.log``) alternate
    as the active target so a single file does not grow without bound, while
    the inactive buffer is preserved forever. A small manifest (the only file
    that is ever overwritten, and atomically at that) records which buffer is
    active and the next sequence number. Because entries are self-describing
    and carry their sequence number, the manifest can always be rebuilt by
    scanning both buffers.
    """

    def __init__(self, root: str | Path, max_bytes: int = DEFAULT_MAX_BYTES) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self._lock = threading.RLock()
        self._manifest_path = self.root / "manifest.json"
        self._active_path = self.root / "active"
        self._a_path = self.root / "diary.a.log"
        self._b_path = self.root / "diary.b.log"
        self._ensure_manifest()

    @property
    def buffers(self) -> dict[str, Path]:
        return {"A": self._a_path, "B": self._b_path}

    def _read_manifest(self) -> dict:
        if self._manifest_path.exists():
            try:
                data = json.loads(self._manifest_path.read_text(encoding="utf-8"))
                if data.get("active") in ("A", "B") and isinstance(data.get("seq"), int):
                    return data
            except (json.JSONDecodeError, OSError):
                pass
        return self._rebuild_manifest()

    def _rebuild_manifest(self) -> dict:
        seq = 0
        seen_a = self._a_path.exists()
        seen_b = self._b_path.exists()
        if seen_a or seen_b:
            entries = self._parse(self._a_path) + self._parse(self._b_path)
            if entries:
                seq = max(e.seq for e in entries) + 1
        active = "A"
        if seen_b and not seen_a:
            active = "B"
        manifest = {"active": active, "seq": seq}
        self._write_manifest(manifest)
        return manifest

    def _write_manifest(self, manifest: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.root), prefix=".manifest.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(json.dumps(manifest, ensure_ascii=False, indent=2))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self._manifest_path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def _ensure_manifest(self) -> None:
        with self._lock:
            self._read_manifest()

    def _parse(self, path: Path) -> list[Entry]:
        if not path.exists():
            return []
        entries: list[Entry] = []
        cur: Optional[dict] = None
        buf: list[str] = []
        try:
            for raw in path.read_text(encoding="utf-8").splitlines():
                m = ENTRY_RE.match(raw)
                if m:
                    kv = dict(re.findall(r"(\S+)=(\S*)", m.group(1)))
                    cur = {
                        "seq": int(kv.get("seq", "0")),
                        "ts": kv.get("ts", ""),
                        "kind": kv.get("kind", "note"),
                        "meta": {k: v for k, v in kv.items() if k not in ("seq", "ts", "kind")},
                    }
                    buf = []
                    continue
                if cur is not None:
                    em = END_RE.match(raw)
                    if em and int(em.group(1)) == cur["seq"]:
                        text = "\n".join(buf)
                        if text.endswith("\n"):
                            text = text[:-1]
                        entries.append(Entry(
                            seq=cur["seq"], ts=cur["ts"], kind=cur["kind"],
                            text=text, meta=cur["meta"],
                        ))
                        cur = None
                        buf = []
                    else:
                        buf.append(raw)
        except OSError as exc:
            raise DiaryError(f"Failed to read diary buffer {path}: {exc}") from exc
        if cur is not None:
            text = "\n".join(buf)
            entries.append(Entry(
                seq=cur["seq"], ts=cur["ts"], kind=cur["kind"],
                text=text, meta=cur["meta"],
            ))
        return entries

    def _active_file(self, active: str) -> Path:
        return self.buffers[active]

    def append(self, kind: str, text: str, repo_root: str | Path | None = None) -> int:
        """Append an immutable entry and return its sequence number.

        Auto-captures local date/time and git commit awareness on every entry.
        """
        kind = re.sub(r"\s+", "_", kind.strip()) or "note"
        text = text.rstrip("\n")
        with self._lock:
            manifest = self._read_manifest()
            active = manifest["active"]
            target = self._active_file(active)
            try:
                size = target.stat().st_size
            except OSError:
                size = 0
            if size >= self.max_bytes:
                active = "B" if active == "A" else "A"
                target = self._active_file(active)
                manifest["active"] = active
                self._write_manifest(manifest)
            seq = manifest["seq"]
            ts = datetime.now(timezone.utc).isoformat()
            ctx = capture_context(repo_root)
            header = _format_kv({"seq": seq, "ts": ts, "kind": kind, **ctx})
            block = f"# ENTRY {header}\n{text}\n# ENDENTRY seq={seq}\n"
            self._atomic_append(target, block)
            manifest["seq"] = seq + 1
            self._write_manifest(manifest)
            return seq

    def _atomic_append(self, path: Path, block: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(block)
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def summarize(self, text: str) -> int:
        """Append a summary entry. Never replaces or deletes prior entries."""
        return self.append("summary", text)

    def commit_snapshot(self, note: str = "", repo_root: str | Path | None = None) -> int:
        """Record an explicit commit-awareness milestone entry."""
        ctx = capture_context(repo_root)
        body = (note + "\n\n" if note else "") + (
            f"commit: {ctx.get('git')}\n"
            f"branch: {ctx.get('branch')}\n"
            f"dirty:  {ctx.get('dirty')}\n"
            f"msg:    {ctx.get('commit_msg')}"
        )
        return self.append("commit", body.rstrip("\n"))

    def entries(self) -> list[Entry]:
        """Return all entries across both buffers, ordered by sequence."""
        with self._lock:
            all_entries = self._parse(self._a_path) + self._parse(self._b_path)
        all_entries.sort(key=lambda e: e.seq)
        return all_entries

    def tail(self, n: int = 20) -> list[Entry]:
        return self.entries()[-n:]

    def read_text(self) -> str:
        """Concatenated human-readable view of the whole diary."""
        out: list[str] = []
        for e in self.entries():
            meta = " ".join(f"{k}={v}" for k, v in (e.meta or {}).items())
            out.append(
                f"# ENTRY seq={e.seq} ts={e.ts} kind={e.kind} {meta}\n{e.text}\n# ENDENTRY seq={e.seq}"
            )
        return "\n\n".join(out)

    def compact(self, summary: str) -> int:
        """Lossless 'compaction': archive a digest as a new entry.

        History is never deleted. This only records a high-level digest so the
        AI can orient itself without re-reading the entire diary.
        """
        return self.append("digest", summary)

    def stats(self) -> dict:
        with self._lock:
            manifest = self._read_manifest()
        entries = self.entries()
        return {
            "root": str(self.root),
            "active_buffer": manifest["active"],
            "next_seq": manifest["seq"],
            "entry_count": len(entries),
            "a_bytes": self._safe_size(self._a_path),
            "b_bytes": self._safe_size(self._b_path),
        }

    @staticmethod
    def _safe_size(path: Path) -> int:
        try:
            return path.stat().st_size
        except OSError:
            return 0

    def __iter__(self) -> Iterator[Entry]:
        return iter(self.entries())
