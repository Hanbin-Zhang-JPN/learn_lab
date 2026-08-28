# rag_core — 原理を読める実用日本語 RAG

`rag_core` は、手元の Markdown / text 文書を検索し、検証可能な出典付き回答を返す Python 製 RAG です。

目的は「便利なライブラリをつないだだけの見えない RAG」ではなく、文書分割、特徴抽出、候補検索、順位付け、重複抑制、引用、評価までを、短い日本語コメント付きの具体的なソースコードとして理解・変更できるようにすることです。

実行時の外部 Python パッケージはありません。Python 3.11 以上の標準ライブラリだけで動きます。自然な文章生成が必要な場合だけ、任意の OpenAI Chat Completions 互換 LLM を接続できます。LLM を接続しなくても、索引、検索、評価、HTTP API、抽出型回答はすべて動作します。

## まず試す

```bash
cd rag_core
python3 -m ragcore demo
```

自分の文書を使う基本手順は次の三行です。

```bash
python3 -m ragcore index --docs docs --out store/rag.db
python3 -m ragcore search --index store/rag.db -q "返品の期限は？"
python3 -m ragcore ask --index store/rag.db -q "返品の期限と条件は？"
```

インストールは必須ではありません。`ragcore` という短いコマンドを登録したい場合だけ、任意で次を実行します。

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
ragcore demo
```

## この RAG が実際に行うこと

```text
登録時
  .md / .txt
      ↓ UTF-8 読み込み
  見出し・文境界を守ったチャンク
      ↓ 日本語文字 bigram / trigram、英数字語
  特徴頻度、文書頻度、転置リスト
      ↓ 一時 DB を完成後に原子置換
  store/rag.db

質問時
  質問
      ↓ 同じ文字正規化と特徴抽出
  希少特徴 → SQLite 転置索引 → 少数候補
      ├─ BM25 順位
      ├─ TF-IDF cosine 順位
      └─ 直接語句一致順位
             ↓ RRF で順位融合
             ↓ MMR で重複抑制
  引用番号 [S1] 付きの根拠
      ├─ 標準の抽出型回答
      └─ 任意の LLM による自然文回答
```

RAG は LLM 自体へ社内知識を再学習させる方式ではありません。質問時に必要な根拠だけを取り出し、質問と一緒に回答器へ渡します。そのため、検索結果、出典、回答不能判定、品質評価を一つの経路として扱っています。

## 現代的・実用的にした点

### 1. 質問ごとの全件走査を避ける

特徴からチャンク ID を引く `postings` 転置表を使います。最初に珍しい一致特徴で候補を絞り、重い BM25 と cosine の計算は候補にだけ行います。

既定では、全チャンクの 20% 以下に現れる特徴が質問にあれば、その希少特徴を候補生成へ使います。希少特徴がなければ既知特徴全体へ戻るため、短く一般的な質問も空振りさせません。精密採点の上限は `--candidates` で変更できます。

検索後には処理量が表示されます。

```text
検索統計: {'total_chunks': 5000, 'matched_chunks': 119, 'scored_chunks': 119}
```

### 2. 一種類の検索信号に依存しない

- BM25: 型番、数字、固有名詞、正確な表現を強く拾います。
- TF-IDF cosine: 質問と文書の特徴比率が似ているかを測ります。
- 直接語句一致: 質問全体がタイトル、見出し、本文にある場合を補助します。
- RRF: 単位の違う生スコアを無理に足さず、順位を `1 / (60 + rank)` で融合します。
- MMR: 上位候補同士の Jaccard 重複率を罰として引き、同じ内容ばかり返ることを抑えます。

タイトルと見出しの特徴頻度は本文の 2 倍として扱います。重要な見出しを優先しながら、本文中の根拠も失わないための明示的な値です。

### 3. 日本語を外部 tokenizer なしで検索する

日本語には通常、英語のような空白区切りがありません。この実装は辞書や形態素解析器を使わず、連続した日本語から 2 文字と 3 文字の部分列を作ります。

`返品条件` の例:

```text
j:返品条件
c2:返品  c2:品条  c2:条件
c3:返品条  c3:品条件
```

全角 / 半角、互換文字、英字の大小は Unicode NFKC と小文字化でそろえます。実際の特徴は次のコマンドで観察できます。

```bash
python3 -m ragcore inspect -t "返品条件"
```

### 4. 壊れかけの索引を公開しない

索引作成は同じフォルダの一時 SQLite DB へ全表を書き、完成後に `os.replace()` で置き換えます。索引作成中に処理が失敗しても、以前の完成済み DB を途中状態で上書きしません。

### 5. 根拠と回答を分離する

標準の抽出型回答は、上位チャンク内で質問特徴と重なる文を選び、必ず `[S1]` 形式の引用番号を付けます。一致根拠がなければ推測せず「根拠が見つからない」と答えます。

任意 LLM を使う場合も、検索済み文書を `trust="untrusted-data"` の source 要素として囲みます。文書内の「以前の指示を無視せよ」などを命令として扱わないよう system 指示を加え、応答中の引用番号が実在するかを検査します。これは防御を強くしますが、LLM の prompt injection を完全に防ぐ保証ではありません。

### 6. 品質を感覚ではなく数値で確認する

`eval/cases.jsonl` に質問と期待出典を保存し、次を計測します。

- Recall@k: 期待した出典を上位 k 件で何割回収したか。
- MRR: 最初の正解が何位に現れたか。
- latency p50 / p95: 検索時間の中央値と遅い側 95% の境界。

```bash
python3 -m ragcore eval \
  --index store/rag.db \
  --cases eval/cases.jsonl \
  -k 3
```

評価ケースは一行一 JSON です。

```json
{"query":"返品できる期限は？","expected_sources":["shop.md"]}
```

文書や検索設定を変える前後で同じ評価を実行すると、改善と劣化を比較できます。

## フォルダ構成

名前を短くし、用途を一目で判別できるようにしています。

```text
rag_core/
├── README.md            目的、原理、操作、制約
├── pyproject.toml       Python 版と任意 CLI 登録
├── docs/                すぐ試せる架空の知識文書
├── eval/
│   └── cases.jsonl      検索品質の正解セット
├── ragcore/
│   ├── model.py         Document / Chunk / SearchHit
│   ├── text.py          読み込み、見出し解析、チャンク化
│   ├── terms.py         正規化、日本語 n-gram
│   ├── search.py        SQLite 転置索引と順位付け
│   ├── answer.py        抽出回答と任意 LLM 接続
│   ├── evaluate.py      Recall、MRR、検索時間
│   ├── api.py           /health、/search、/ask
│   └── cli.py           全コマンドの入口
├── scripts/
│   └── bench.py         合成データの性能計測
├── store/               生成した rag.db の置き場所
└── tests/               標準 unittest
```

Python の実行行には、処理を追いやすい短い日本語コメントをほぼ一行ずつ付けています。数式と保存構造も、外部ライブラリの関数名だけで済ませずソース内へ直接書いています。

## SQLite を使う目的と方法

SQLite は外部パッケージではなく Python 標準ライブラリ `sqlite3` から使用します。ここでは検索順位を SQLite の非公開機能へ任せていません。目的は次の三つだけです。

1. 文書と索引を一ファイルへ永続化する。
2. 特徴に一致するチャンク ID だけを転置表から読む。
3. 一時 DB と原子置換で更新中の破損を避ける。

保存表は四つです。

| 表 | 内容 |
|---|---|
| `meta` | 形式番号、件数、平均長、作成時刻、内容指紋 |
| `terms` | 特徴、document frequency、TF-IDF の IDF |
| `chunks` | 本文、出典、見出し、原文位置、ベクトル長 |
| `postings` | 特徴が現れる chunk ID、本文頻度、見出し頻度 |

候補取得後の BM25、TF-IDF cosine、RRF、MMR は [`ragcore/search.py`](ragcore/search.py) の Python コードで計算します。DB の中身は標準 `sqlite3` コマンドや Python から直接確認できます。

## チャンク化

実装は [`ragcore/text.py`](ragcore/text.py) です。

1. Markdown の `#` 〜 `######` を節境界として認識します。
2. `。！？!?` と空行を文境界として扱います。
3. 文を壊さない範囲で既定 700 文字以下へまとめます。
4. 次のチャンクへ末尾の文を既定約 120 文字引き継ぎます。
5. 一文自体が長い場合だけ、重なり付き固定窓へ分けます。
6. `source`、`heading`、`start`、`end`、`ordinal` を残します。

```bash
python3 -m ragcore index \
  --docs your_docs \
  --chunk-size 900 \
  --overlap 150 \
  --out store/rag.db
```

短すぎるチャンクは必要な条件を失い、長すぎるチャンクは無関係な語を混ぜます。実文書では「一つの質問へ答える条件が同じチャンクに入るか」を評価セットで確認して調整してください。

## 検索の調整

```bash
python3 -m ragcore search \
  --index store/rag.db \
  -q "型番 AX-204 の保証期間は？" \
  -k 5 \
  --candidates 1200 \
  --mmr-lambda 0.78 \
  --min-signal-ratio 0.12 \
  --source-prefix "manual/"
```

- `top-k`: 最後に返す根拠数です。
- `candidates`: 転置索引の後で精密採点する最大件数です。増やすと再現率を守りやすくなりますが遅くなります。
- `mmr-lambda`: 1 に近いほど質問への関連性、0 に近いほど結果の多様性を重視します。
- `min-signal-ratio`: 最上位 BM25 / cosine の指定割合にも届かない弱い一致を除きます。
- `source-prefix`: 指定パスで検索対象を限定します。

`source-prefix` は検索範囲の機能であり、認証そのものではありません。複数利用者の機密情報を扱う場合は、HTTP 要求で利用者に自由入力させず、認証済みの所属情報からサーバー側で決定してください。

機械処理では `--json` を使用できます。

```bash
python3 -m ragcore search --index store/rag.db -q "営業時間" --json
```

## 任意 LLM の接続

自然な統合回答が必要な場合だけ使用します。検索品質を改善する機能ではなく、取得済み根拠を文章へまとめるための接続です。

Python SDK は導入せず、標準 `urllib.request` で OpenAI Chat Completions 互換 endpoint へ HTTP POST します。

```bash
export RAG_LLM_URL="http://localhost:11434/v1/chat/completions"
export RAG_LLM_MODEL="your-local-model"
python3 -m ragcore ask \
  --index store/rag.db \
  -q "返品条件を簡潔にまとめて" \
  --provider llm
```

認証が必要な接続先では `RAG_LLM_API_KEY` を設定します。キーは索引や出力へ保存しません。外部 endpoint へ送信される内容は、質問、上位チャンクの本文、出典名です。機密文書を外部へ送ってよいか、接続前に必ず確認してください。

## HTTP API

```bash
python3 -m ragcore serve --index store/rag.db --host 127.0.0.1 --port 8080
```

生存確認:

```bash
curl http://127.0.0.1:8080/health
```

検索:

```bash
curl -X POST http://127.0.0.1:8080/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"返品期限は？","top_k":3}'
```

抽出回答:

```bash
curl -X POST http://127.0.0.1:8080/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"返品期限と条件は？","top_k":3}'
```

API は既定で `127.0.0.1` のみに待ち受け、要求本文を 64 KiB に制限します。認証、TLS、rate limit は内蔵していません。外部公開する場合は、認証と TLS を持つ reverse proxy の内側へ置いてください。

## テスト

```bash
python3 -m unittest discover -s tests -v
```

テストは次を確認します。

- 見出し、文字上限、重複、元文書位置。
- 正しい出典の最上位検索。
- 引用番号を持つ抽出回答。
- 出典 prefix による検索範囲。
- 転置索引の候補削減統計。
- SQLite の可視な四表。
- 未知質問の空結果。
- Recall@k と MRR の集計。

## 性能計測

```bash
python3 scripts/bench.py --chunks 5000 --queries 100
```

このワークスペースで 2026-08-28 に実行した一例です。

```text
chunks       : 5000
index seconds: 0.704
index MiB    : 22.82
top1 accuracy: 1.000
latency p50  : 5.447 ms
latency p95  : 5.841 ms
last stats   : {'total_chunks': 5000, 'matched_chunks': 119, 'scored_chunks': 119}
```

これは識別子を含む合成文書の結果であり、すべての実データ性能を保証する値ではありません。CPU、SSD、文書長、語の共通度、質問、候補上限で変わります。`scripts/bench.py` を自分の環境で再実行し、実データでは `eval` の品質も同時に確認してください。

## 外部要素の一覧

| 要素 | 必須 | 使用目的 | 具体的方法 |
|---|---:|---|---|
| Python 3.11+ | はい | 全処理の実行 | 標準ライブラリだけを使用 |
| `sqlite3` | はい | 永続化と転置候補取得 | Python 同梱 SQLite に四つの単純な表を保存 |
| setuptools | いいえ | `ragcore` コマンドの登録 | `pip install -e .` を選んだ時だけ使用 |
| OpenAI 互換 LLM | いいえ | 検索済み根拠の自然文統合 | `urllib.request` で Chat Completions JSON を POST |

RAG の中心処理に LangChain、LlamaIndex、ベクトル DB、形態素解析器、埋め込み SDK は使っていません。

## 適用範囲と限界

この実装が向く用途:

- 社内規程、FAQ、製品マニュアル、運用手順など、語句と数字が重要な日本語文書。
- 数百〜数万チャンク程度の単一マシン検索。
- 検索理由と保存構造を監査、学習、改造したい用途。
- ネットワークなしで検索と抽出回答を動かしたい用途。

現在の限界:

- 入力は抽出済み UTF-8 の `.md` / `.txt` です。PDF、Office、画像 OCR は前処理が必要です。
- 学習済み埋め込みを使わないため、文字がほぼ重ならない同義語検索は弱いです。
- 索引更新は安全な全体再構築です。非常に大きい文書集合の差分更新は未実装です。
- SQLite は単一マシン用です。数百万チャンク、複数ノード、高頻度更新には専用の分散検索基盤が必要です。
- 抽出型回答は根拠に忠実ですが、複数文書を自然に要約する能力は限定的です。
- LLM の出力検査は引用番号の存在確認であり、すべての主張の真偽を自動証明するものではありません。

意味検索が本当に必要になった場合は、`analyze(text)` と並ぶ明示的な境界として `embed(text) -> list[float]` を追加し、同じ chunk ID へ対応付ける設計が可能です。その際は、使用モデル名、版、ベクトル次元、正規化方法、送信データ、保存場所、費用、評価差を README と評価結果へ記録してください。外部モデルを入れること自体ではなく、どの問題を改善したかを Recall と実質問で確認することが重要です。

## ライセンスとサンプルデータ

同梱 `docs/` は動作確認用に作成した架空サービスの文書です。実在する会社、商品、規約を表しません。コードは [`LICENSE`](LICENSE) の MIT License で利用できます。
