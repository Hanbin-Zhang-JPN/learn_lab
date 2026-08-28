"""文書読み込みと、見出し・文境界を守るチャンク化を実装する。"""  # このファイルの責務を示す。

from __future__ import annotations  # 型注釈の評価を遅らせる。

import hashlib  # 文書とチャンクの安定 ID を作る。
import re  # Markdown 見出しと文末を検出する。
from dataclasses import dataclass  # 内部の位置付き文字列を表す。
from pathlib import Path  # ファイルとフォルダを安全に扱う。

from .model import Chunk, Document  # 共通データ型を読み込む。


@dataclass(frozen=True)  # 分割途中の文を不変にする。
class _Unit:  # 元文書位置を持つ一つの文を表す。
    text: str  # 前後空白を除いた本文を持つ。
    start: int  # 元文書内の開始位置を持つ。
    end: int  # 元文書内の終了位置を持つ。


@dataclass(frozen=True)  # 見出し単位の領域を不変にする。
class _Section:  # 一つの Markdown 節を表す。
    heading: str  # 節の見出しを持つ。
    text: str  # 節の本文を持つ。
    offset: int  # 節本文の開始位置を持つ。


class TextChunker:  # 意味の境界を優先して文書を分割する。
    def __init__(self, max_chars: int = 700, overlap_chars: int = 120) -> None:  # 分割幅を受け取る。
        if max_chars < 100:  # 極端に短い設定を防ぐ。
            raise ValueError("max_chars は 100 以上にしてください。")  # 修正可能な説明を返す。
        if overlap_chars < 0 or overlap_chars >= max_chars:  # 重複幅を検査する。
            raise ValueError("overlap_chars は 0 以上 max_chars 未満にしてください。")  # 無限分割を防ぐ。
        self.max_chars = max_chars  # 最大文字数を保存する。
        self.overlap_chars = overlap_chars  # 引き継ぐ文字数を保存する。

    def chunk(self, document: Document) -> list[Chunk]:  # 一文書を検索単位へ分ける。
        chunks: list[Chunk] = []  # 完成したチャンクをためる。
        ordinal = 0  # 文書内の通し番号を初期化する。
        for section in self._sections(document.text):  # 見出し単位で処理する。
            units = self._units(section)  # 節を位置付き文へ分ける。
            current: list[_Unit] = []  # 現在のチャンク候補を作る。
            for unit in units:  # 文書順に各文を処理する。
                pieces = self._split_long(unit) if len(unit.text) > self.max_chars else [unit]  # 長文だけ固定窓へ分ける。
                for piece in pieces:  # 通常文または長文片を追加する。
                    proposed = self._joined_size(current + [piece])  # 追加後の長さを測る。
                    if current and proposed > self.max_chars:  # 上限を越える直前か確認する。
                        chunks.append(self._make_chunk(document, section.heading, ordinal, current))  # 現在分を確定する。
                        ordinal += 1  # 次の番号へ進める。
                        current = self._tail(current)  # 境界文脈を次へ渡す。
                        if current and self._joined_size(current + [piece]) > self.max_chars:  # 重複後も収まるか確認する。
                            current = []  # 上限を優先して重複を外す。
                    current.append(piece)  # 新しい文を現在分へ加える。
            if current:  # 節末に未確定分があるか確認する。
                chunks.append(self._make_chunk(document, section.heading, ordinal, current))  # 最後の分を確定する。
                ordinal += 1  # 通し番号を更新する。
        return chunks  # 文書順のチャンク列を返す。

    def _sections(self, text: str) -> list[_Section]:  # Markdown 見出しで領域を分ける。
        sections: list[_Section] = []  # 完成した節をためる。
        heading = "本文"  # 見出し前の既定名を決める。
        parts: list[str] = []  # 現在節の行をためる。
        body_start = 0  # 現在節の開始位置を持つ。
        cursor = 0  # 元文書上の走査位置を持つ。
        for line in text.splitlines(keepends=True):  # 改行を残して各行を読む。
            match = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", line)  # Markdown 見出しを探す。
            if match:  # 見出し行を処理する。
                body = "".join(parts)  # 直前の本文を結合する。
                if body.strip():  # 空節を除外する。
                    sections.append(_Section(heading, body, body_start))  # 直前の節を保存する。
                heading = match.group(1).strip()  # 新しい見出しを保存する。
                parts = []  # 本文格納先を空にする。
                body_start = cursor + len(line)  # 見出し直後を開始位置にする。
            else:  # 通常の本文行を処理する。
                if not parts:  # 最初の本文行か確認する。
                    body_start = cursor  # 実際の開始位置を保存する。
                parts.append(line)  # 行を現在節へ加える。
            cursor += len(line)  # 元文書位置を次へ進める。
        body = "".join(parts)  # 最後の本文を結合する。
        if body.strip():  # 最後の節が空でないか確認する。
            sections.append(_Section(heading, body, body_start))  # 最後の節を保存する。
        return sections  # 文書順の節を返す。

    def _units(self, section: _Section) -> list[_Unit]:  # 節を日本語と英語の文末で分ける。
        pattern = re.compile(r".+?(?:[。！？!?](?:[」』】）)])?\s*|\n\s*\n|$)", re.DOTALL)  # 文末と空行を認識する。
        units: list[_Unit] = []  # 位置付き文をためる。
        for match in pattern.finditer(section.text):  # 文らしい範囲を順に探す。
            raw = match.group(0)  # 元の空白を含む文字列を得る。
            clean = raw.strip()  # 外側の空白を除く。
            if not clean:  # 空白だけか確認する。
                continue  # 空の文を除外する。
            left = len(raw) - len(raw.lstrip())  # 左側で除いた長さを測る。
            start = section.offset + match.start() + left  # 元文書の開始位置を求める。
            units.append(_Unit(clean, start, start + len(clean)))  # 文と位置を保存する。
        return units  # 文書順の文を返す。

    def _split_long(self, unit: _Unit) -> list[_Unit]:  # 長い一文を重複窓へ分ける。
        pieces: list[_Unit] = []  # 分割片をためる。
        step = self.max_chars - self.overlap_chars  # 一回で進む幅を決める。
        local_start = 0  # 長文内の開始位置を初期化する。
        while local_start < len(unit.text):  # 文末まで窓を動かす。
            local_end = min(local_start + self.max_chars, len(unit.text))  # 現在窓の終了位置を決める。
            text = unit.text[local_start:local_end]  # 窓内の文字列を取り出す。
            start = unit.start + local_start  # 元文書の位置へ戻す。
            pieces.append(_Unit(text, start, start + len(text)))  # 位置付き分割片を保存する。
            if local_end == len(unit.text):  # 文末へ着いたか確認する。
                break  # 余分な窓を作らず終了する。
            local_start += step  # 重複を残して次へ進める。
        return pieces  # 分割片を返す。

    def _tail(self, units: list[_Unit]) -> list[_Unit]:  # 次チャンクへ残す末尾文を選ぶ。
        if self.overlap_chars == 0:  # 重複なし設定を確認する。
            return []  # 空の引き継ぎを返す。
        selected: list[_Unit] = []  # 選んだ末尾文をためる。
        size = 0  # 現在の重複長を初期化する。
        for unit in reversed(units):  # 後ろの文から選ぶ。
            if selected and size + len(unit.text) > self.overlap_chars:  # 目標幅を越えるか確認する。
                break  # 現在の選択で止める。
            selected.append(unit)  # 末尾文を追加する。
            size += len(unit.text)  # 重複長を更新する。
            if size >= self.overlap_chars:  # 目標へ達したか確認する。
                break  # 十分な文脈で終了する。
        return list(reversed(selected))  # 元の文書順へ戻す。

    @staticmethod  # インスタンス状態を変えない計算と示す。
    def _joined_size(units: list[_Unit]) -> int:  # 改行込みの結合長を求める。
        return sum(len(unit.text) for unit in units) + max(0, len(units) - 1)  # 本文と改行を合計する。

    @staticmethod  # インスタンス状態を変えない生成と示す。
    def _make_chunk(document: Document, heading: str, ordinal: int, units: list[_Unit]) -> Chunk:  # 文列をチャンクへ変える。
        text = "\n".join(unit.text for unit in units)  # 文境界を改行で残す。
        start = units[0].start  # 最初の文の開始位置を使う。
        end = units[-1].end  # 最後の文の終了位置を使う。
        seed = f"{document.doc_id}\0{ordinal}\0{start}\0{text}"  # ID の再現可能な材料を作る。
        chunk_id = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:20]  # 衝突しにくい短い ID を作る。
        return Chunk(chunk_id, document.doc_id, document.source, document.title, heading, ordinal, start, end, text)  # 完成チャンクを返す。


def load_documents(path: Path) -> list[Document]:  # ファイルまたはフォルダから文書を読む。
    base = path.resolve()  # 入力を絶対パスにする。
    if not base.exists():  # 入力の存在を確認する。
        raise FileNotFoundError(f"文書パスが見つかりません: {base}")  # 原因を具体的に示す。
    files = [base] if base.is_file() else sorted(item for item in base.rglob("*") if item.suffix.lower() in {".md", ".txt"})  # 対応ファイルを安定順で集める。
    root = base.parent if base.is_file() else base  # 相対出典の基準を決める。
    documents: list[Document] = []  # 読み込んだ文書をためる。
    for file_path in files:  # 各ファイルを順番に読む。
        text = file_path.read_text(encoding="utf-8-sig")  # BOM を吸収して UTF-8 を読む。
        if not text.strip():  # 空ファイルか確認する。
            continue  # 空文書を索引へ入れない。
        source = file_path.relative_to(root).as_posix()  # OS 非依存の相対出典を作る。
        title_match = re.search(r"^\s*#\s+(.+?)\s*$", text, flags=re.MULTILINE)  # 最初の大見出しを探す。
        title = title_match.group(1).strip() if title_match else file_path.stem  # 見出しまたはファイル名をタイトルにする。
        seed = f"{source}\0{text}"  # パスと内容を ID の材料にする。
        doc_id = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:20]  # 文書の安定 ID を作る。
        documents.append(Document(doc_id, source, title, text))  # 読み込み結果を保存する。
    if not documents:  # 有効な入力がないか確認する。
        raise ValueError(".md または .txt の文書が見つかりません。")  # 対応形式を明確に伝える。
    return documents  # 文書列を返す。
