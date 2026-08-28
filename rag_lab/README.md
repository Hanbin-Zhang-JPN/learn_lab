# rag_lab — 中身を追える日本語 RAG

`rag_lab` は、RAG（Retrieval-Augmented Generation）の「文書をどう分け、どう数値へ変え、どう近さを測り、どう回答へ渡すか」を、実行できる Python コードで学ぶための小さなリポジトリです。

実行時に必要な外部 Python パッケージはありません。Python 3.11 以上の標準ライブラリだけで、文書読み込み、チャンク化、TF-IDF ベクトル化、コサイン類似度、BM25、順位融合、重複抑制、索引保存、根拠付き回答まで動きます。

`python3 -m raglab ...` の実行には packaging tool も不要です。任意で `pip install -e .` を行い `raglab` コマンドを登録するときだけ、build backend として setuptools を使います。setuptools の目的は CLI のインストールだけで、検索・ベクトル計算・回答処理には使いません。

## まず動かす

```bash
cd rag_lab
python3 -m raglab demo
```

自分の `.md` / `.txt` 文書を検索する場合は、次の三段階です。

```bash
python3 -m raglab index --docs docs --out store/index.json
python3 -m raglab search --index store/index.json -q "返品の期限は？"
python3 -m raglab ask --index store/index.json -q "返品の期限と条件は？"
python3 -m raglab inspect -t "返品条件" --index store/index.json
```

テストは次の一行で実行できます。

```bash
python3 -m unittest discover -s tests -v
```

## RAG の全体像

```text
登録時:
  .md / .txt
      ↓ 読み込み
  見出し・文境界を保つチャンク
      ↓ analyze()
  文字 n-gram と単語特徴
      ↓ fit()
  TF-IDF 疎ベクトル + BM25 統計
      ↓ save()
  store/index.json

質問時:
  質問
    ├─ TF-IDF → コサイン類似度順位 ─┐
    └─ 字面特徴 → BM25 順位 ────────┤
                                      ↓ RRF
                                  統合順位
                                      ↓ MMR
                              重複を抑えた根拠
                                      ↓
                         抽出回答 または任意の LLM
                                      ↓
                              [S1] 付き回答
```

RAG の本質は、LLM 自体へ知識を再学習させることではありません。質問に関係する小さな根拠を検索し、その根拠を質問と一緒に回答器へ渡すことです。このため、検索品質と出典管理は生成モデルと同じくらい重要です。

## 1. チャンク化は何をしているか

実装は [`raglab/text.py`](raglab/text.py) にあります。

1. Markdown の `#` 〜 `######` を節境界として認識します。
2. 節内を `。！？!?` と空行で文単位へ分けます。
3. 文を壊さない範囲で `max_chars` 以下へまとめます。
4. 次のチャンクへ末尾の文を約 `overlap_chars` だけ引き継ぎます。
5. 一文自体が長すぎる場合だけ、重なり付き固定窓へ分けます。
6. `source`、`heading`、`start`、`end`、`ordinal` をメタデータとして残します。

既定値は 500 文字、重複 100 文字です。短すぎるチャンクは意味が不足し、長すぎるチャンクは無関係な語が混ざります。重複は境界にまたがる説明を救いますが、多すぎると同じ根拠ばかり返ります。実データでは「質問に必要な一まとまりが一チャンクに入るか」を評価して調整します。

本格的な tokenizer が使える環境では、文字数ではなくモデル固有の token 数で上限を管理します。この実装は処理を完全に見える状態へ保つため文字数を採用しています。

## 2. ベクトル化は何をしているか

実装は [`raglab/vector.py`](raglab/vector.py) にあります。

日本語は英語のように空白で単語が分かれません。形態素解析器を導入せず部分一致を可能にするため、次の特徴を作ります。

- 英数字: 単語全体 `w:rag` と、長い語の 3 文字列
- 日本語: 文字 bigram と trigram
- 日本語列全体: 完全一致を強く拾う補助特徴

たとえば `返品条件` から `返-品`、`品-条`、`条-件` に相当する bigram と、`返品条`、`品条件` に相当する trigram ができます。実際のキーには種類を区別する `c2:`、`c3:` が付きます。

各特徴の TF-IDF 重みは、おおよそ次の形です。

```text
tf(t, d)  = 1 + log(文書 d 内の特徴 t の回数)
idf(t)    = log((1 + 全チャンク数) / (1 + t を含むチャンク数)) + 1
weight    = tf × idf
vector    = weight を L2 長で割って正規化
```

全特徴を巨大な配列にせず、値が存在する座標だけを `dict[str, float]` に保存します。これが疎ベクトルです。`store/index.json` を開くと、特徴名、IDF、各チャンクの座標値をそのまま確認できます。

## 3. ベクトル類似度はどう計算するか

質問も、登録時に学習した同じ IDF を使ってベクトル化します。質問ベクトル `q` とチャンクベクトル `d` の近さにはコサイン類似度を使います。

```text
cos(q, d) = (q · d) / (||q|| × ||d||)
```

この実装では両ベクトルを先に長さ 1 へ正規化しているため、検索時は共通座標の積を足すだけです。値が 1 に近いほど特徴の向きが似ており、0 は既知特徴の重なりがないことを表します。計算は [`cosine_similarity()`](raglab/vector.py) の数行だけです。

## 4. なぜコサイン類似度だけではないか

実用検索では一種類の信号だけに頼ると弱点が出ます。このプロジェクトは次を組み合わせます。

- TF-IDF + cosine: 文全体の特徴比率が似たチャンクを拾います。
- BM25: 固有名詞や正確な字面の一致を、文書長と語の希少性を補正して拾います。
- RRF（Reciprocal Rank Fusion）: 異なる尺度の生スコアを無理に足さず、`1 / (60 + 順位)` を足して順位を融合します。
- MMR（Maximal Marginal Relevance）: 質問への関連性を保ちながら、すでに選んだチャンクと似すぎる候補へ罰を与えます。
- 相対しきい値: 最上位のコサインまたは BM25 信号の既定 15% にも届かない、偶然の文字一致を除きます。

主な実装は [`raglab/index.py`](raglab/index.py) にあります。CLI の検索結果には `hybrid`、`cosine`、`bm25` をすべて表示するため、なぜ上位になったかを観察できます。

## 5. 回答生成は二方式

### 外部不要の抽出型

既定の `--provider extractive` は、上位チャンクから質問特徴と重なる文を選び、`[S1]` 形式の根拠番号を付けます。自然な要約能力は限定的ですが、ネットワーク、API キー、追加パッケージなしで全経路を確認できます。

### 任意の OpenAI 互換 LLM

自然な統合回答が必要な場合だけ、`--provider llm` を使えます。Python SDK は導入せず、標準ライブラリ `urllib.request` で OpenAI Chat Completions 互換 HTTP endpoint へ送ります。これは「検索」ではなく、検索済み根拠を自然文へまとめる目的だけに使います。

```bash
export RAG_LLM_URL="http://localhost:11434/v1/chat/completions"
export RAG_LLM_MODEL="your-local-model"
python3 -m raglab ask -q "返品の期限と例外は？" --provider llm
```

認証が必要なサービスでは `RAG_LLM_API_KEY` を環境変数へ設定します。キーは索引やログへ保存しません。送信されるものは質問と取得チャンク本文です。機密文書を外部 endpoint へ送ってよいかは、利用前に必ず確認してください。

外部 LLM サーバーはこのリポジトリの依存物ではなく、任意の接続先です。未設定でも抽出型回答まで完全に動作します。

## フォルダ構成

```text
rag_lab/
├── README.md           目的、数式、使い方、限界
├── pyproject.toml      Python 版と CLI 名。外部依存は空
├── docs/               すぐ試せる架空サービスの知識文書
├── raglab/
│   ├── model.py        Document / Chunk / SearchHit
│   ├── text.py         読み込み、見出し解析、チャンク化
│   ├── vector.py       n-gram、TF-IDF、コサイン類似度
│   ├── index.py        BM25、RRF、MMR、JSON 保存
│   ├── answer.py       抽出回答、任意 LLM 接続
│   └── cli.py          index / search / ask / demo
├── store/              生成した索引の置き場所
└── tests/              各段階の標準 unittest
```

Python ファイルには、学習時に処理を追いやすいよう、ほぼすべての実行行へ短い日本語コメントを付けています。

## 調整できる値

```bash
python3 -m raglab index \
  --docs your_docs \
  --chunk-size 700 \
  --overlap 120 \
  --out store/index.json

python3 -m raglab search \
  --index store/index.json \
  -q "質問" \
  --top-k 5 \
  --vector-weight 1.0 \
  --bm25-weight 1.2 \
  --mmr-lambda 0.75 \
  --min-signal-ratio 0.15
```

- `vector-weight`: 類似表現を拾う順位の寄与です。
- `bm25-weight`: 固有名詞・型番・正確な用語を拾う順位の寄与です。
- `mmr-lambda`: 1 に近いほど関連度、0 に近いほど候補の多様性を優先します。
- `min-signal-ratio`: 最上位結果に対して弱すぎる候補を落とします。0 なら足切りません。

値は感覚だけで決めず、「質問、期待する出典、期待順位」を持つ小さな評価セットを作り、Recall@k や MRR と回答正確性を比較するのが実務的です。

## 現代的な本番 RAG との差と拡張点

この実装は数百〜数千程度のチャンクをローカルで理解・検索する用途には使えますが、大規模本番環境の完成形ではありません。

- 意味検索: TF-IDF は同義語に弱いため、本番では埋め込みモデルの dense vector を追加します。埋め込み API を使うなら、`TfidfVectorizer` と同じ境界に `embed(text) -> list[float]` を置き、同じチャンク ID に対応させます。
- 大規模検索: 現在は全ベクトルを線形走査します。大量データでは HNSW などの ANN 索引やベクトル DB を使います。
- 再ランキング: 上位 20〜100 件を cross-encoder または LLM reranker で並べ直すと、質問と根拠の細かな関係を評価できます。
- 更新: 現在は文書集合全体を再索引します。本番では内容 hash により追加・変更・削除だけを反映します。
- 品質管理: 検索 Recall、根拠忠実性、回答不能判定、速度、費用を継続計測します。
- 安全性: 文書アクセス権、個人情報除去、prompt injection 対策、監査ログが必要です。
- PDF / Office: この教材は抽出済み UTF-8 テキストを入力にします。PDF のレイアウト解析や OCR は別工程です。

「埋め込みモデルを呼べば RAG」ではありません。適切なチャンク、検索の複数信号、メタデータ、引用、回答不能、評価までを一つの経路として設計することが、実際に使える RAG の要点です。

## ライセンスとデータ

同梱の `docs/` は動作確認用に作成した架空サービスの説明です。実在する会社や規約を表しません。プロジェクト本体は学習・改変用のサンプルとして利用できます。
