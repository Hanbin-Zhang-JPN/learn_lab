"""検索根拠を回答へ変える抽出型と OpenAI 互換型の生成器を実装する。"""  # このファイルの責務を示す。

from __future__ import annotations  # 型注釈を実行時評価から切り離す。

import json  # HTTP の要求本文と応答本文を扱う。
import re  # チャンクから回答候補の文を切り出す。
import urllib.request  # 外部 Python 依存なしで HTTP API を呼ぶ。
from collections import Counter  # 質問と文の特徴重複を数える。

from .model import SearchHit  # 検索結果の共通型を読み込む。
from .vector import analyze, normalize_text  # 検索と同じ特徴抽出を使う。


def build_context(hits: list[SearchHit], max_chars: int = 6000) -> tuple[str, list[SearchHit]]:  # 根拠ラベル付きコンテキストを作る。
    parts: list[str] = []  # プロンプト断片の格納先を用意する。
    used: list[SearchHit] = []  # 実際に収まった検索結果を記録する。
    size = 0  # 現在のコンテキスト文字数を追跡する。
    for number, hit in enumerate(hits, start=1):  # 検索順位どおりに根拠を追加する。
        label = f"S{number}"  # 回答から参照できる短い根拠番号を作る。
        part = f"[{label}] 出典={hit.chunk.source} 見出し={hit.chunk.heading}\n{hit.chunk.text}"  # メタデータと本文を一組にする。
        if parts and size + len(part) > max_chars:  # 二件目以降が予算を超えるか確認する。
            break  # 途中で切らず、直前の完全な根拠まで使う。
        parts.append(part[:max_chars] if not parts else part)  # 一件目だけは長すぎても安全に切る。
        used.append(hit)  # 使用した検索結果を記録する。
        size += len(parts[-1])  # 実際に追加した文字数を更新する。
    return "\n\n".join(parts), used  # コンテキストと対応する検索結果を返す。


class ExtractiveAnswerer:  # LLM なしでも根拠付き回答を返す。
    def answer(self, query: str, hits: list[SearchHit], max_sentences: int = 4) -> str:  # 関連文を抽出して回答にする。
        if not hits:  # 根拠が一件もない場合を扱う。
            return "索引内に、この質問へ答えるための根拠が見つかりませんでした。"  # 推測せず不明を返す。
        query_counts = Counter(analyze(query))  # 質問の特徴量を数える。
        candidates: list[tuple[float, int, str]] = []  # 文スコア、根拠番号、本文をためる。
        for number, hit in enumerate(hits, start=1):  # 上位チャンクから候補文を作る。
            sentences = [piece.strip() for piece in re.findall(r".+?(?:[。！？!?](?:[」』】）)])?|$)", hit.chunk.text, flags=re.DOTALL) if piece.strip()]  # 文末を保って本文を分ける。
            for sentence in sentences:  # 各文と質問の重なりを測る。
                sentence_counts = Counter(analyze(sentence))  # 文側の特徴量を数える。
                overlap = sum(min(count, sentence_counts.get(term, 0)) for term, count in query_counts.items())  # 共通特徴の回数を求める。
                if overlap == 0:  # 無関係な文を除く。
                    continue  # 次の候補文へ進む。
                length_penalty = max(1.0, len(sentence_counts) ** 0.5)  # 長文だけが有利になることを防ぐ。
                score = overlap / length_penalty + hit.score  # 文の局所一致とチャンク関連度を足す。
                candidates.append((score, number, sentence))  # 根拠番号とともに保存する。
        candidates.sort(key=lambda item: item[0], reverse=True)  # 関連度の高い文を先に並べる。
        selected: list[tuple[int, str]] = []  # 重複を除いた回答文を格納する。
        seen: set[str] = set()  # 正規化済み文で重複を検出する。
        for _, number, sentence in candidates:  # 高得点順に文を選ぶ。
            key = normalize_text(sentence)  # 表記差をならした重複キーを作る。
            if key in seen:  # 同じ文が重複チャンクにないか確認する。
                continue  # 重複文は回答へ入れない。
            seen.add(key)  # この文を選択済みとして記録する。
            selected.append((number, sentence))  # 文と根拠番号を回答へ加える。
            if len(selected) >= max_sentences:  # 必要な文数へ達したか確認する。
                break  # 抽出を終了する。
        if not selected:  # チャンクはあるが文単位の一致がない場合を扱う。
            selected.append((1, hits[0].chunk.text[:300].strip()))  # 最上位根拠を短く提示する。
        lines = [f"- {sentence} [S{number}]" for number, sentence in selected]  # 各文へ根拠番号を付ける。
        return "検索した文書から、次の根拠が見つかりました。\n\n" + "\n".join(lines)  # 抽出回答を返す。


class OpenAICompatibleAnswerer:  # OpenAI 互換 Chat Completions API で文章回答を作る。
    def __init__(self, endpoint: str, model: str, api_key: str = "", timeout: float = 60.0) -> None:  # 接続設定を受け取る。
        self.endpoint = endpoint  # chat/completions の完全 URL を保持する。
        self.model = model  # API 側で使うモデル名を保持する。
        self.api_key = api_key  # 必要な場合だけ認証キーを保持する。
        self.timeout = timeout  # 応答待ち上限秒を保持する。

    def answer(self, query: str, hits: list[SearchHit]) -> str:  # 検索根拠だけを使う回答を依頼する。
        if not hits:  # 根拠が一件もない場合を扱う。
            return "索引内に、この質問へ答えるための根拠が見つかりませんでした。"  # LLM に推測させず終了する。
        context, _ = build_context(hits)  # 根拠番号付きのコンテキストを作る。
        system = "あなたは根拠厳守の RAG 回答器です。与えられた根拠だけで日本語回答し、各主張の末尾に [S1] の形式で出典を付けてください。根拠にない場合は不明と答えてください。"  # 幻覚を抑える役割指示を作る。
        user = f"質問:\n{query}\n\n根拠:\n{context}"  # 質問と検索結果を明確に区切る。
        payload = {"model": self.model, "temperature": 0, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}  # 再現性を優先した要求を作る。
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")  # JSON を UTF-8 バイト列へ変える。
        headers = {"Content-Type": "application/json"}  # JSON API のヘッダーを用意する。
        if self.api_key:  # 認証キーが設定されているか確認する。
            headers["Authorization"] = f"Bearer {self.api_key}"  # Bearer 認証ヘッダーを追加する。
        request = urllib.request.Request(self.endpoint, data=body, headers=headers, method="POST")  # HTTP POST 要求を組み立てる。
        with urllib.request.urlopen(request, timeout=self.timeout) as response:  # 指定時間内で API を呼び出す。
            result = json.loads(response.read().decode("utf-8"))  # 応答 JSON を辞書へ変える。
        try:  # API 形式の違いを分かりやすい例外へ変換する。
            return str(result["choices"][0]["message"]["content"]).strip()  # OpenAI 互換形式から回答本文を返す。
        except (KeyError, IndexError, TypeError) as error:  # 必要な応答項目がない場合を捕捉する。
            raise ValueError("LLM API の応答が OpenAI Chat Completions 形式ではありません。") from error  # 接続設定の確認を促す。
