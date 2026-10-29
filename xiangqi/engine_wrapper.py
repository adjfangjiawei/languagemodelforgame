"""
Cross-platform (Windows .exe & Linux/WSL2) UCI wrapper around Pikafish NNUE.
Automatically selects the matching OS binary and loads `pikafish.nnue` from the engine directory.
"""

from __future__ import annotations

import os
import platform
import stat
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_default_engine_path() -> str:
    env_path = os.environ.get("PIKAFISH_BIN", "").strip()
    if env_path and os.path.exists(env_path):
        return env_path

    is_win = platform.system().lower().startswith("win")
    bin_name = "Pikafish-Windows-x86-64-universal.exe" if is_win else "Pikafish-Linux-x86-64-universal"

    candidate_dirs = [
        os.path.join(PROJECT_ROOT, "engine"),
        "/tmp/pikafish_assets",
    ]
    for d in candidate_dirs:
        cand = os.path.join(d, bin_name)
        nnue = os.path.join(d, "pikafish.nnue")
        if os.path.exists(cand) and os.path.exists(nnue):
            if not is_win:
                try:
                    st = os.stat(cand).st_mode
                    if not (st & stat.S_IXUSR):
                        os.chmod(cand, st | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                except OSError:
                    pass
            return cand

    return os.path.join(PROJECT_ROOT, "engine", bin_name)


@dataclass
class MoveEval:
    uci: str
    score_cp: int
    is_mate: bool = False
    mate_in: Optional[int] = None
    pv: List[str] = field(default_factory=list)


class PikafishEngine:
    """Persistent cross-platform UCI interface to Pikafish with NNUE."""

    def __init__(
        self,
        engine_path: Optional[str] = None,
        hash_mb: int = 32,
        threads: int = 1,
    ):
        self.engine_path = engine_path or resolve_default_engine_path()
        if not os.path.exists(self.engine_path):
            raise FileNotFoundError(
                f"Pikafish engine binary not found at: {self.engine_path}\n"
                f"Ensure `engine/` contains Pikafish binary and `pikafish.nnue`."
            )
        self.engine_dir = os.path.dirname(os.path.abspath(self.engine_path))
        self.hash_mb = hash_mb
        self.threads = threads
        self._current_multipv: Optional[int] = None

        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        self.proc = subprocess.Popen(
            [self.engine_path],
            cwd=self.engine_dir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creationflags,
        )
        self._send("uci")
        self._wait_for("uciok")
        self._send(f"setoption name Hash value {self.hash_mb}")
        self._send(f"setoption name Threads value {self.threads}")
        self._send("isready")
        self._wait_for("readyok")

    def _send(self, cmd: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _wait_for(self, token: str) -> None:
        assert self.proc.stdout is not None
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(f"Pikafish exited unexpectedly while waiting for {token}")
            if token in line:
                return

    def evaluate_multipv(self, fen: str, depth: int = 8, multipv: int = 5) -> List[MoveEval]:
        assert self.proc.stdout is not None
        if self._current_multipv != multipv:
            self._send(f"setoption name MultiPV value {multipv}")
            self._current_multipv = multipv

        self._send(f"position fen {fen}")
        self._send(f"go depth {depth}")

        by_rank: Dict[int, tuple[int, MoveEval]] = {}
        bestmove_uci: Optional[str] = None

        while True:
            raw = self.proc.stdout.readline()
            if not raw:
                raise RuntimeError(f"Pikafish EOF during search on FEN: {fen}")
            line = raw.strip()
            if line.startswith("info ") and " score " in line and " pv " in line:
                parsed = self._parse_info_line(line)
                if parsed is not None:
                    d_val, rank, mv_eval = parsed
                    prev = by_rank.get(rank)
                    if prev is None or d_val >= prev[0]:
                        by_rank[rank] = (d_val, mv_eval)
            elif line.startswith("bestmove"):
                parts = line.split()
                if len(parts) >= 2 and parts[1] != "(none)":
                    bestmove_uci = parts[1]
                break

        results = [by_rank[k][1] for k in sorted(by_rank.keys())]
        seen = set()
        unique_results: List[MoveEval] = []
        for item in results:
            if item.uci not in seen:
                seen.add(item.uci)
                unique_results.append(item)

        if not unique_results and bestmove_uci:
            unique_results.append(MoveEval(uci=bestmove_uci, score_cp=0, pv=[bestmove_uci]))

        unique_results.sort(key=lambda m: m.score_cp, reverse=True)
        return unique_results

    @staticmethod
    def _parse_info_line(line: str) -> Optional[tuple[int, int, MoveEval]]:
        tokens = line.split()
        try:
            d_idx = tokens.index("depth")
            depth_val = int(tokens[d_idx + 1])
            rank = int(tokens[tokens.index("multipv") + 1]) if "multipv" in tokens else 1
            s_idx = tokens.index("score")
            s_type = tokens[s_idx + 1]
            s_raw = int(tokens[s_idx + 2])
            pv_idx = tokens.index("pv")
            pv_moves = tokens[pv_idx + 1 :]
            if not pv_moves or len(pv_moves[0]) != 4:
                return None

            is_mate = s_type == "mate"
            mate_in: Optional[int] = None
            if is_mate:
                mate_in = s_raw
                score_cp = (10000 - s_raw) if s_raw > 0 else (-10000 + abs(s_raw))
            else:
                score_cp = max(-9500, min(9500, s_raw))

            return depth_val, rank, MoveEval(
                uci=pv_moves[0],
                score_cp=score_cp,
                is_mate=is_mate,
                mate_in=mate_in,
                pv=pv_moves,
            )
        except (ValueError, IndexError):
            return None

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self._send("quit")
                self.proc.wait(timeout=3)
            except Exception:
                self.proc.kill()

    def __enter__(self) -> "PikafishEngine":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
