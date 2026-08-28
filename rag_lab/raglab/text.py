"""文書読み込みと、見出し・文境界を尊重したチャンク化を行う。"""  # このファイルの責務を示す。

from __future__ import annotations  # 型注釈を実行時評価から切り離す。

import hashlib  # 文書とチャンクの安定 ID を作る。
import re  # 見出しと文末を小さな規則で検出する。
from dataclasses import dataclass  # 内部用の位置付きテキストを簡潔に表す。
from pathlib import Path  # OS に依存しないパス操作を行う。

from .model import Chunk, Document  # 共通データ構造を読み込む。


@dataclass(frozen=True)  # 分割途中のテキスト片を不変にする。
class _Unit:  # 一つの文または長文の一部を表す。
    text: str  # 前後の空白を除いた内容を持つ。
    start: int  # 元文書内の開始位置を持つ。
    end: int  # 元文書内の終了位置を持つ。


@dataclass(frozen=True)  # 見出し単位の領域を不変にする。
class _Section:  # Markdown の一つの節を表す。
    heading: str  # 最も近い見出しを持つ。
    text: str  # 節の本文を持つ。
    offset: int  # 節本文が始まる元文書位置を持つ。


class TextChunker:  # 意味の切れ目を優先して文書を分割する。
    def __init__(self, max_chars: int = 500, overlap_chars: int = 100) -> None:  # サイズ設定を受け取る。
        if max_chars < 80:  # 極端に短いチャンクを防ぐ。
            raise ValueError("max_chars は 80 以上にしてください。")  # 誤設定を明確に伝える。
        if overlap_chars < 0 or overlap_chars >= max_chars:  # 重複幅の範囲を検証する。
            raise ValueError("overlap_chars は 0 以上 max_chars 未満にしてください。")  # 無限分割を防ぐ。
        self.max_chars = max_chars  # 最大文字数を保存する。
        self.overlap_chars = overlap_chars  # 文脈重複の目標文字数を保存する。

    def chunk(self, document: Document) -> list[Chunk]:  # 一文書をチャンク列へ変換する。
        chunks: list[Chunk] = []  # 完成チャンクの格納先を用意する。
        ordinal = 0  # 文書内の通し番号を初期化する。
        for section in self._sections(document.text):  # 見出し単位で処理する。
            units = self._sentence_units(section)  # 節を位置付きの文へ分ける。
            current: list[_Unit] = []  # 現在組み立てているチャンクを用意する。
            for unit in units:  # 各文を順番に追加する。
                if len(unit.text) > self.max_chars:  # 一文だけで上限を超える場合を扱う。
                    if current:  # 先に通常文がたまっているか確認する。
                        chunks.append(self._make_chunk(document, section.heading, ordinal, current))  # たまった文を確定する。
                        ordinal += 1  # 次の通し番号へ進める。
                        current = []  # 作業中の文を空に戻す。
                    for piece in self._split_long_unit(unit):  # 長文を安全な窓に分割する。
                        chunks.append(self._make_chunk(document, section.heading, ordinal, [piece]))  # 各窓をチャンク化する。
                        ordinal += 1  # 次の通し番号へ進める。
                    continue  # 長文の処理後は次の文へ進む。
                proposed = self._joined_length(current + [unit])  # 追加後の見込み文字数を測る。
                if current and proposed > self.max_chars:  # 上限を超える直前で確定する。
                    chunks.append(self._make_chunk(document, section.heading, ordinal, current))  # 現在のチャンクを保存する。
                    ordinal += 1  # 次の通し番号へ進める。
                    current = self._overlap_tail(current)  # 末尾の文を次チャンクへ引き継ぐ。
                    if current and self._joined_length(current + [unit]) > self.max_chars:  # 重複で上限を超えないか調べる。
                        current = []  # 収まらない場合は重複より上限を優先する。
                current.append(unit)  # 新しい文を作業中チャンクへ加える。
            if current:  # 節末に未確定の文があるか確認する。
                chunks.append(self._make_chunk(document, section.heading, ordinal, current))  # 最後のチャンクを保存する。
                ordinal += 1  # 通し番号を一つ進める。
        return chunks  # 文書順を保ったチャンク列を返す。

    def _sections(self, text: str) -> list[_Section]:  # Markdown 見出しで本文を区切る。
        sections: list[_Section] = []  # 節の格納先を用意する。
        heading = "本文"  # 見出しがない領域の表示名を決める。
        body_start = 0  # 現在の節本文の開始位置を持つ。
        cursor = 0  # 各行の元文書位置を追跡する。
        body_parts: list[str] = []  # 現在の節の行をためる。
        for line in text.splitlines(keepends=True):  # 改行を保持したまま行を走査する。
            match = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", line)  # Markdown 見出しを検出する。
            if match:  # 見出し行だった場合を扱う。
                body = "".join(body_parts)  # 直前までの本文を結合する。
                if body.strip():  # 空の節を索引へ入れない。
                    sections.append(_Section(heading=heading, text=body, offset=body_start))  # 直前の節を保存する。
                heading = match.group(1).strip()  # 新しい見出し名を保存する。
                body_parts = []  # 新しい節の本文格納先を空にする。
                body_start = cursor + len(line)  # 見出し直後を本文開始位置にする。
            else:  # 通常の本文行を扱う。
                if not body_parts:  # 節の最初の本文行か確認する。
                    body_start = cursor  # 実際の開始位置を記録する。
                body_parts.append(line)  # 本文行を節へ追加する。
            cursor += len(line)  # 元文書上の位置を次の行へ進める。
        body = "".join(body_parts)  # 最後の節本文を結合する。
        if body.strip():  # 最後の節が空でないか確認する。
            sections.append(_Section(heading=heading, text=body, offset=body_start))  # 最後の節を保存する。
        return sections  # 文書順の節を返す。

    def _sentence_units(self, section: _Section) -> list[_Unit]:  # 節を文末または空行で分ける。
        pattern = re.compile(r".+?(?:[。！？!?](?:[」』】）)])?\s*|\n\s*\n|$)", re.DOTALL)  # 日本語と英語の文末を認識する。
        units: list[_Unit] = []  # 位置付き文の格納先を用意する。
        for match in pattern.finditer(section.text):  # 文らしい範囲を順番に探す。
            raw = match.group(0)  # 元の空白を含む範囲を得る。
            stripped = raw.strip()  # 検索に不要な外側の空白を除く。
            if not stripped:  # 空白だけの範囲を除外する。
                continue  # 次の候補へ進む。
            left_trim = len(raw) - len(raw.lstrip())  # 左側で除いた文字数を測る。
            start = section.offset + match.start() + left_trim  # 元文書上の正確な開始位置を求める。
            end = start + len(stripped)  # 元文書上の終了位置を求める。
            units.append(_Unit(text=stripped, start=start, end=end))  # 文を位置情報付きで保存する。
        return units  # 文書順の文を返す。

    def _split_long_unit(self, unit: _Unit) -> list[_Unit]:  # 長すぎる一文を重複窓で分ける。
        pieces: list[_Unit] = []  # 分割片の格納先を用意する。
        step = self.max_chars - self.overlap_chars  # 次の窓まで進む文字数を決める。
        local_start = 0  # 長文内の開始位置を初期化する。
        while local_start < len(unit.text):  # 文末まで窓を移動する。
            local_end = min(local_start + self.max_chars, len(unit.text))  # 現在窓の終了位置を求める。
            piece_text = unit.text[local_start:local_end].strip()  # 窓の外側空白を除く。
            if piece_text:  # 空の分割片を除外する。
                absolute_start = unit.start + local_start  # 元文書上の開始位置へ戻す。
                pieces.append(_Unit(piece_text, absolute_start, absolute_start + len(piece_text)))  # 分割片を保存する。
            if local_end == len(unit.text):  # 文末へ到達したか確認する。
                break  # 最後の窓を重複生成せず終了する。
            local_start += step  # 重複を残して次の窓へ進む。
        return pieces  # 長文から作った窓を返す。

    def _overlap_tail(self, units: list[_Unit]) -> list[_Unit]:  # 次チャンクへ残す末尾文を選ぶ。
        if self.overlap_chars == 0:  # 重複なし設定を先に扱う。
            return []  # 空の引き継ぎを返す。
        selected: list[_Unit] = []  # 末尾文の格納先を用意する。
        size = 0  # 現在の重複文字数を初期化する。
        for unit in reversed(units):  # 後ろの文から調べる。
            if selected and size + len(unit.text) > self.overlap_chars:  # 目標幅を超える前で止める。
                break  # 選択を終了する。
            selected.append(unit)  # この文を重複範囲へ加える。
            size += len(unit.text)  # 重複文字数を更新する。
            if size >= self.overlap_chars:  # 目標幅へ達したか確認する。
                break  # 十分な文脈を確保できたので終了する。
        return list(reversed(selected))  # 元の文書順へ戻して返す。

    @staticmethod  # インスタンス状態を使わない計算として示す。
    def _joined_length(units: list[_Unit]) -> int:  # 改行込みの結合文字数を求める。
        return sum(len(unit.text) for unit in units) + max(0, len(units) - 1)  # 本文と区切り改行を合計する。

    @staticmethod  # インスタンス状態を使わない生成処理として示す。
    def _make_chunk(document: Document, heading: str, ordinal: int, units: list[_Unit]) -> Chunk:  # 文列をチャンクへ変える。
        chunk_text = "\n".join(unit.text for unit in units)  # 文の境界が見える形で本文を結合する。
        start = units[0].start  # 最初の文の開始位置を採用する。
        end = units[-1].end  # 最後の文の終了位置を採用する。
        seed = f"{document.doc_id}:{ordinal}:{start}:{chunk_text}"  # ID 用の再現可能な材料を作る。
        chunk_id = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]  # 衝突しにくい短い ID を作る。
        return Chunk(chunk_id, document.doc_id, document.source, heading, ordinal, start, end, chunk_text)  # 完成値を返す。


def load_documents(path: Path) -> list[Document]:  # ファイルまたはフォルダから文書を読む。
    base = path.resolve()  # 相対表示の基準となる絶対パスを得る。
    if not base.exists():  # 入力パスが存在するか確認する。
        raise FileNotFoundError(f"文書パスが見つかりません: {base}")  # 原因の分かる例外を返す。
    files = [base] if base.is_file() else sorted(p for p in base.rglob("*") if p.suffix.lower() in {".md", ".txt"})  # 対応ファイルを安定順で集める。
    root = base.parent if base.is_file() else base  # 出典表示の基準フォルダを決める。
    documents: list[Document] = []  # 読み込んだ文書の格納先を用意する。
    for file_path in files:  # 各ファイルを順番に読む。
        text = file_path.read_text(encoding="utf-8-sig")  # BOM の有無を吸収して UTF-8 を読む。
        if not text.strip():  # 空ファイルを索引から除く。
            continue  # 次のファイルへ進む。
        source = file_path.relative_to(root).as_posix()  # OS に依存しない相対出典名を作る。
        seed = f"{source}\0{text}"  # パスと内容を ID の材料にする。
        doc_id = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]  # 文書の安定 ID を作る。
        documents.append(Document(doc_id=doc_id, source=source, text=text))  # 文書を保存する。
    if not documents:  # 有効な文書が一件もないか確認する。
        raise ValueError(".md または .txt の文書が見つかりません。")  # 入力条件を明確に伝える。
    return documents  # 読み込んだ文書列を返す。
