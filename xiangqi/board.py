"""
Complete Chinese Chess (Xiangqi / 中国象棋) board representation, exact board-state hashing (`state_key`),
NNUE legality validation, and 100% bijective UCI <-> Chinese standard move notation.
"""

from __future__ import annotations

from collections import Counter
from typing import List, Optional, Tuple

START_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"

RED_NUMS = ["", "一", "二", "三", "四", "五", "六", "七", "八", "九"]
BLACK_NUMS = ["", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

PIECE_TO_ZH = {
    "K": "帅", "A": "仕", "B": "相", "E": "相", "N": "马", "H": "马", "R": "车", "C": "炮", "P": "兵",
    "k": "将", "a": "士", "b": "象", "e": "象", "n": "马", "h": "马", "r": "车", "c": "炮", "p": "卒",
}

CANONICAL_PIECE = {"E": "B", "H": "N", "e": "b", "h": "n"}


def mirror_uci(uci: str) -> str:
    """Horizontally mirror a 4-char UCI move across center file 'e' (col -> 8 - col)."""
    c1 = 8 - (ord(uci[0]) - ord("a"))
    r1 = uci[1]
    c2 = 8 - (ord(uci[2]) - ord("a"))
    r2 = uci[3]
    return f"{chr(ord('a') + c1)}{r1}{chr(ord('a') + c2)}{r2}"


class XiangqiBoard:
    """Xiangqi board with legal move generation, exact transposition key, and Chinese standard notation."""

    def __init__(self, fen: str = START_FEN):
        self.grid: List[List[str]] = [["." for _ in range(9)] for _ in range(10)]
        self.turn: str = "w"
        self.halfmove: int = 0
        self.fullmove: int = 1
        self._load_fen(fen)

    def copy(self) -> "XiangqiBoard":
        b = XiangqiBoard.__new__(XiangqiBoard)
        b.grid = [row[:] for row in self.grid]
        b.turn = self.turn
        b.halfmove = self.halfmove
        b.fullmove = self.fullmove
        return b

    def _load_fen(self, fen: str) -> None:
        parts = fen.strip().split()
        ranks = parts[0].split("/")
        if len(ranks) != 10:
            raise ValueError(f"Invalid Xiangqi FEN (expected 10 ranks): {fen}")
        for idx, rank_str in enumerate(ranks):
            row = 9 - idx
            col = 0
            for ch in rank_str:
                if ch.isdigit():
                    col += int(ch)
                else:
                    self.grid[row][col] = CANONICAL_PIECE.get(ch, ch)
                    col += 1
            if col != 9:
                raise ValueError(f"Invalid rank width in FEN: {rank_str}")
        if len(parts) >= 2:
            self.turn = "w" if parts[1].lower() in ("w", "r") else "b"
        if len(parts) >= 5 and parts[4].isdigit():
            self.halfmove = int(parts[4])
        if len(parts) >= 6 and parts[5].isdigit():
            self.fullmove = int(parts[5])

    def grid_fen(self) -> str:
        ranks = []
        for row in range(9, -1, -1):
            empty = 0
            s = []
            for col in range(9):
                p = self.grid[row][col]
                if p == ".":
                    empty += 1
                else:
                    if empty > 0:
                        s.append(str(empty))
                        empty = 0
                    s.append(p)
            if empty > 0:
                s.append(str(empty))
            ranks.append("".join(s))
        return "/".join(ranks)

    def state_key(self) -> str:
        """Exact transposition equivalence key (piece layout + side to move, ignoring clocks)."""
        return f"{self.grid_fen()} {self.turn}"

    def to_fen(self) -> str:
        return f"{self.grid_fen()} {self.turn} - - {self.halfmove} {self.fullmove}"

    def mirrored(self) -> "XiangqiBoard":
        b = self.copy()
        for r in range(10):
            b.grid[r] = list(reversed(b.grid[r]))
        return b

    def is_nnue_valid(self) -> bool:
        """Strict Xiangqi piece count, square legality, and king-safety check required by Pikafish NNUE."""
        red_adv = {(3, 0), (5, 0), (4, 1), (3, 2), (5, 2)}
        blk_adv = {(3, 9), (5, 9), (4, 8), (3, 7), (5, 7)}
        red_ele = {(2, 0), (6, 0), (0, 2), (4, 2), (8, 2), (2, 4), (6, 4)}
        blk_ele = {(2, 9), (6, 9), (0, 7), (4, 7), (8, 7), (2, 5), (6, 5)}

        counts: Counter[str] = Counter()
        for r in range(10):
            for c in range(9):
                p = self.grid[r][c]
                if p == ".":
                    continue
                counts[p] += 1
                if p == "K" and not (3 <= c <= 5 and 0 <= r <= 2):
                    return False
                if p == "k" and not (3 <= c <= 5 and 7 <= r <= 9):
                    return False
                if p == "A" and (c, r) not in red_adv:
                    return False
                if p == "a" and (c, r) not in blk_adv:
                    return False
                if p == "B" and (c, r) not in red_ele:
                    return False
                if p == "b" and (c, r) not in blk_ele:
                    return False
                if p == "P" and (r < 3 or (r in (3, 4) and c % 2 != 0)):
                    return False
                if p == "p" and (r > 6 or (r in (5, 6) and c % 2 != 0)):
                    return False

        if counts["K"] != 1 or counts["k"] != 1:
            return False
        for ch in ("A", "a", "B", "b", "R", "r", "N", "n", "C", "c"):
            if counts[ch] > 2:
                return False
        if counts["P"] > 5 or counts["p"] > 5:
            return False
        opp = "b" if self.turn == "w" else "w"
        if self.is_in_check(opp):
            return False
        return True

    @staticmethod
    def parse_uci(uci: str) -> Tuple[int, int, int, int]:
        uci = uci.strip()
        if len(uci) != 4:
            raise ValueError(f"Invalid UCI move: {uci}")
        c1 = ord(uci[0]) - ord("a")
        r1 = ord(uci[1]) - ord("0")
        c2 = ord(uci[2]) - ord("a")
        r2 = ord(uci[3]) - ord("0")
        return c1, r1, c2, r2

    @staticmethod
    def format_uci(c1: int, r1: int, c2: int, r2: int) -> str:
        return f"{chr(ord('a') + c1)}{r1}{chr(ord('a') + c2)}{r2}"

    def push_uci(self, uci: str) -> str:
        c1, r1, c2, r2 = self.parse_uci(uci)
        moving = self.grid[r1][c1]
        if moving == ".":
            raise ValueError(f"No piece at {uci[:2]} on FEN {self.to_fen()}")
        captured = self.grid[r2][c2]
        self.grid[r2][c2] = moving
        self.grid[r1][c1] = "."
        self.halfmove = 0 if captured != "." else (self.halfmove + 1)
        if self.turn == "b":
            self.fullmove += 1
        self.turn = "b" if self.turn == "w" else "w"
        return captured

    def _find_king(self, color: str) -> Optional[Tuple[int, int]]:
        target = "K" if color == "w" else "k"
        r_range = range(0, 3) if color == "w" else range(7, 10)
        for r in r_range:
            for c in range(3, 6):
                if self.grid[r][c] == target:
                    return (c, r)
        for r in range(10):
            for c in range(9):
                if self.grid[r][c] == target:
                    return (c, r)
        return None

    def is_in_check(self, color: str) -> bool:
        kp = self._find_king(color)
        if kp is None:
            return True
        kc, kr = kp
        enemy_is_red = color == "b"

        def is_enemy(p: str, kind_upper: str) -> bool:
            return p != "." and (p.isupper() == enemy_is_red) and (p.upper() == kind_upper)

        for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            screens = 0
            c, r = kc + dc, kr + dr
            while 0 <= c < 9 and 0 <= r < 10:
                p = self.grid[r][c]
                if p != ".":
                    if screens == 0:
                        if is_enemy(p, "R") or is_enemy(p, "K"):
                            return True
                        screens = 1
                    else:
                        if is_enemy(p, "C"):
                            return True
                        break
                c += dc
                r += dr

        for nc_off, nr_off in ((1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)):
            nc, nr = kc + nc_off, kr + nr_off
            if 0 <= nc < 9 and 0 <= nr < 10 and is_enemy(self.grid[nr][nc], "N"):
                leg_c = nc + (-nc_off // 2 if abs(nc_off) == 2 else 0)
                leg_r = nr + (-nr_off // 2 if abs(nr_off) == 2 else 0)
                if self.grid[leg_r][leg_c] == ".":
                    return True

        if enemy_is_red:
            if kr - 1 >= 0 and self.grid[kr - 1][kc] == "P":
                return True
            if kr >= 5:
                for dc in (-1, 1):
                    if 0 <= kc + dc < 9 and self.grid[kr][kc + dc] == "P":
                        return True
        else:
            if kr + 1 < 10 and self.grid[kr + 1][kc] == "p":
                return True
            if kr <= 4:
                for dc in (-1, 1):
                    if 0 <= kc + dc < 9 and self.grid[kr][kc + dc] == "p":
                        return True
        return False

    def generate_pseudo_legal_moves(self) -> List[str]:
        moves: List[str] = []
        red_turn = self.turn == "w"

        def own(p: str) -> bool:
            return p != "." and (p.isupper() == red_turn)

        for r in range(10):
            for c in range(9):
                p = self.grid[r][c]
                if not own(p):
                    continue
                kind = p.upper()
                if kind == "R":
                    for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nc, nr = c + dc, r + dr
                        while 0 <= nc < 9 and 0 <= nr < 10:
                            q = self.grid[nr][nc]
                            if q == ".":
                                moves.append(self.format_uci(c, r, nc, nr))
                            else:
                                if not own(q):
                                    moves.append(self.format_uci(c, r, nc, nr))
                                break
                            nc += dc
                            nr += dr
                elif kind == "C":
                    for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nc, nr = c + dc, r + dr
                        jumped = False
                        while 0 <= nc < 9 and 0 <= nr < 10:
                            q = self.grid[nr][nc]
                            if not jumped:
                                if q == ".":
                                    moves.append(self.format_uci(c, r, nc, nr))
                                else:
                                    jumped = True
                            else:
                                if q != ".":
                                    if not own(q):
                                        moves.append(self.format_uci(c, r, nc, nr))
                                    break
                            nc += dc
                            nr += dr
                elif kind == "N":
                    for dc, dr, lc, lr in (
                        (1, 2, 0, 1), (-1, 2, 0, 1), (1, -2, 0, -1), (-1, -2, 0, -1),
                        (2, 1, 1, 0), (2, -1, 1, 0), (-2, 1, -1, 0), (-2, -1, -1, 0),
                    ):
                        nc, nr = c + dc, r + dr
                        if 0 <= nc < 9 and 0 <= nr < 10 and self.grid[r + lr][c + lc] == "." and not own(self.grid[nr][nc]):
                            moves.append(self.format_uci(c, r, nc, nr))
                elif kind == "B":
                    r_min, r_max = (0, 4) if red_turn else (5, 9)
                    for dc, dr in ((2, 2), (2, -2), (-2, 2), (-2, -2)):
                        nc, nr = c + dc, r + dr
                        if 0 <= nc < 9 and r_min <= nr <= r_max and self.grid[r + dr // 2][c + dc // 2] == "." and not own(self.grid[nr][nc]):
                            moves.append(self.format_uci(c, r, nc, nr))
                elif kind == "A":
                    r_min, r_max = (0, 2) if red_turn else (7, 9)
                    for dc, dr in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                        nc, nr = c + dc, r + dr
                        if 3 <= nc <= 5 and r_min <= nr <= r_max and not own(self.grid[nr][nc]):
                            moves.append(self.format_uci(c, r, nc, nr))
                elif kind == "K":
                    r_min, r_max = (0, 2) if red_turn else (7, 9)
                    for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nc, nr = c + dc, r + dr
                        if 3 <= nc <= 5 and r_min <= nr <= r_max and not own(self.grid[nr][nc]):
                            moves.append(self.format_uci(c, r, nc, nr))
                elif kind == "P":
                    fwd = 1 if red_turn else -1
                    crossed = (r >= 5) if red_turn else (r <= 4)
                    nr = r + fwd
                    if 0 <= nr < 10 and not own(self.grid[nr][c]):
                        moves.append(self.format_uci(c, r, c, nr))
                    if crossed:
                        for dc in (-1, 1):
                            nc = c + dc
                            if 0 <= nc < 9 and not own(self.grid[r][nc]):
                                moves.append(self.format_uci(c, r, nc, r))
        return moves

    def generate_legal_moves(self) -> List[str]:
        legal = []
        mover = self.turn
        for uci in self.generate_pseudo_legal_moves():
            c1, r1, c2, r2 = self.parse_uci(uci)
            moving = self.grid[r1][c1]
            captured = self.grid[r2][c2]
            self.grid[r2][c2] = moving
            self.grid[r1][c1] = "."
            in_chk = self.is_in_check(mover)
            self.grid[r1][c1] = moving
            self.grid[r2][c2] = captured
            if not in_chk:
                legal.append(uci)
        return legal

    def uci_to_zh(self, uci: str) -> str:
        c1, r1, c2, r2 = self.parse_uci(uci)
        p = self.grid[r1][c1]
        if p == ".":
            raise ValueError(f"Empty square at {uci[:2]} on FEN {self.to_fen()}")

        red = p.isupper()
        nums = RED_NUMS if red else BLACK_NUMS
        file1 = nums[9 - c1] if red else nums[c1 + 1]
        file2 = nums[9 - c2] if red else nums[c2 + 1]
        zh_piece = PIECE_TO_ZH[p]
        kind = p.upper()

        same_file_rows = [r for r in range(10) if self.grid[r][c1] == p]
        same_file_rows.sort(reverse=red)

        if kind in ("R", "N", "C") and len(same_file_rows) >= 2:
            idx = same_file_rows.index(r1)
            prefix = f"{'前' if idx == 0 else '后'}{zh_piece}"
        elif kind == "P" and len(same_file_rows) >= 2:
            other_multi = [
                col for col in range(9)
                if col != c1 and sum(1 for r in range(10) if self.grid[r][col] == p) >= 2
            ]
            idx = same_file_rows.index(r1)
            if len(same_file_rows) == 2:
                pos_tag = "前" if idx == 0 else "后"
            elif len(same_file_rows) == 3:
                pos_tag = ["前", "中", "后"][idx]
            else:
                pos_tag = RED_NUMS[idx + 1] if red else BLACK_NUMS[idx + 1]
            prefix = f"{pos_tag}{zh_piece}{file1}" if other_multi else f"{pos_tag}{zh_piece}"
        else:
            prefix = f"{zh_piece}{file1}"

        if r1 == r2:
            action, arg = "平", file2
        else:
            going_forward = (r2 > r1) if red else (r2 < r1)
            action = "进" if going_forward else "退"
            arg = file2 if kind in ("N", "B", "A") else nums[abs(r2 - r1)]

        return f"{prefix}{action}{arg}"

    def zh_to_uci(self, zh_move: str) -> Optional[str]:
        cleaned = zh_move.strip().replace(" ", "")
        for k, v in {"帥": "帅", "將": "将", "馬": "马", "車": "车", "砲": "炮", "包": "炮"}.items():
            cleaned = cleaned.replace(k, v)
        for i, fw in enumerate("０１２３４５６７８９"):
            cleaned = cleaned.replace(fw, str(i))

        legal = self.generate_legal_moves()
        for uci in legal:
            if self.uci_to_zh(uci) == cleaned:
                return uci
        norm_table = str.maketrans({"士": "仕", "象": "相"})
        cleaned_norm = cleaned.translate(norm_table)
        for uci in legal:
            if self.uci_to_zh(uci).translate(norm_table) == cleaned_norm:
                return uci
        return None

    # Compatibility aliases
    uci_to_chinese = uci_to_zh
    chinese_to_uci = zh_to_uci

