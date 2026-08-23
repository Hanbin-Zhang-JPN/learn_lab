"""Create a small, transparent Japanese story corpus.

There is no downloaded corpus here. Every ingredient is visible below. The
random generator recombines them so the model learns a pattern rather than one
fixed story.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterator


NAMES = [
    "葵", "凛", "澪", "結衣", "陽菜", "楓", "紬", "灯", "桜", "遥",
    "蓮", "湊", "樹", "律", "悠真", "颯太", "朝陽", "海斗", "直樹", "蒼",
    "美月", "七海", "小春", "千尋", "琴音", "真央", "晴香", "和奏", "雫", "杏",
    "拓海", "大和", "伊織", "春樹", "奏太", "隼人", "優斗", "陸", "朔", "新",
]

PLACES = [
    "鎌倉", "京都", "奈良", "金沢", "小樽", "松本", "高山", "尾道", "長崎", "熊本",
    "函館", "日光", "川越", "倉敷", "横浜", "神戸", "広島", "仙台", "秋田", "青森",
    "札幌", "富山", "福井", "伊豆", "箱根", "熱海", "高知", "松山", "別府", "阿蘇",
    "屋久島", "宮島", "直島", "佐渡", "淡路島", "琵琶湖", "上高地", "白川郷", "飛騨", "出雲",
]

STYLES = ("恋愛", "文芸", "ユーモア", "恐怖")

STYLE_ALIASES = {
    "恋愛": "恋愛", "爱情": "恋愛", "恋爱": "恋愛",
    "文芸": "文芸", "文艺": "文芸", "文学": "文芸",
    "ユーモア": "ユーモア", "幽默": "ユーモア", "喜剧": "ユーモア",
    "恐怖": "恐怖", "ホラー": "恐怖",
}

# Each style has its own vocabulary and rhythm. Keeping these parts in plain
# sight makes the conditioning mechanism inspectable instead of mysterious.
STYLE_PARTS = {
    "恋愛": {
        "openings": [
            "雨宿りした軒先で、昔と変わらない声に呼び止められた",
            "駅の売店で、最後に残った同じ菓子へ誰かと同時に手を伸ばした",
            "古い喫茶店で、向かいの席に置かれた短い手紙を見つけた",
            "夕暮れの橋で、毎朝すれ違う人が立ち止まるのを見た",
            "図書館で、いつも借りる本に小さなしおりが挟まっているのに気づいた",
            "花屋の前で、風に飛ばされた注文票を拾った",
            "祭りの帰り道で、はぐれた誰かの鈴を見つけた",
            "海辺のベンチで、隣に座った人から温かい缶茶を渡された",
        ],
        "middles": [
            "その人は一本の傘を差し出し、遠回りして帰ろうと言った",
            "二人は照れ隠しに天気の話をしながら、閉店まで同じ席にいた",
            "しおりの裏には、次の休日に会いたいと丁寧な字で書かれていた",
            "言いかけてやめた言葉の代わりに、そっと手が差し出された",
            "昔交わした約束を、相手もまだ覚えていることが分かった",
            "帰る方向は反対なのに、相手はもう少し歩こうと笑った",
            "人混みの中で離れないよう、二人は同じ袖をつかんだ",
            "沈黙は気まずくなく、波の音だけが二人の間を行き来した",
        ],
        "endings": [
            "別れ際、次は偶然ではなく会おうと約束した。",
            "雨は上がっていたが、二人は傘を閉じずに歩いた。",
            "帰りの電車で届いた短い便りを、何度も読み返した。",
            "町の灯がともるころ、握った手の温度だけが確かに残った。",
            "言えなかった名前を呼ぶと、その人は振り向いて笑った。",
            "翌朝、待ち合わせの時刻だけが手帳に増えていた。",
            "遠回りした道は、不思議なくらい短く感じられた。",
            "二人の影は、川沿いの道で一つになったり離れたりした。",
        ],
    },
    "文芸": {
        "openings": [
            "朝の川沿いを歩き、水面に揺れる家々を眺めた",
            "古本屋で、余白ばかりの薄い本を手に取った",
            "港の待合室で、宛名の消えた絵はがきを拾った",
            "冬の公園で、誰も座らない濡れた椅子を見つめた",
            "坂の上から、夕日で輪郭をなくしていく町を見下ろした",
            "閉店前の写真館で、知らない家族の古い写真を見つけた",
            "始発の窓に映る自分が、少し遅れて瞬きをするのを見た",
            "雨上がりの路地で、石畳に残る空の色を踏んで歩いた",
        ],
        "middles": [
            "風が吹くたび、町は一度ほどけて別の形に結び直された",
            "紙の匂いの奥から、忘れていた夏の午後が静かに戻ってきた",
            "遠い鐘の音が、言葉にならない記憶の縁をなぞった",
            "人のいない窓辺にも、暮らしの続きだけが淡く残っていた",
            "昨日と今日の境目は、濡れた舗道の上で曖昧になった",
            "読めない文字ほど、自分へ向けられた便りのように思えた",
            "長く伸びた影を見て、時間にも帰る場所があるのだと思った",
            "季節の変わる音は、古い戸が閉まる音によく似ていた",
        ],
        "endings": [
            "その日から、見慣れた道が少しだけ遠く見えた。",
            "本を閉じると、夕方の鐘が遠くで一度鳴った。",
            "答えは見つからなかったが、探していた時間は戻ってきた。",
            "空になった椅子には、雨の匂いだけが残っていた。",
            "振り返らずに歩くと、町は静かに夜へ沈んでいった。",
            "写真を棚へ戻し、自分の今日をもう一度始めた。",
            "窓を開けると、知らない朝が同じ顔で待っていた。",
            "足元の水たまりが消えるまで、しばらくそこに立っていた。",
        ],
    },
    "ユーモア": {
        "openings": [
            "駅前で、どうしても動かない自動販売機と話し合いを始めた",
            "道を尋ねた相手が、町でいちばん迷子になりやすい郵便配達員だった",
            "食堂でカレーを頼むと、店主から先に味の感想を聞かれた",
            "公園で、立入禁止の札を守っている猫に呼び止められた",
            "貸自転車を借りると、かごに手書きの運転免許試験が入っていた",
            "宿の朝食で、自分の卵焼きだけ妙に立派な皿に載っていた",
            "商店街の福引で、使い道の分からない大きな鍵を当てた",
            "駅員に近道を聞くと、なぜか準備運動から始めるよう言われた",
        ],
        "middles": [
            "そこへ猫が来て足元のボタンを押し、温かいお茶を二本受け取った",
            "二人で地図を逆さに持ち、目的地とは反対へ自信満々に歩いた",
            "困っていると、隣の客が今日だけ自分が店主だと名乗り出た",
            "説明を聞くほど話は複雑になり、見物人だけが増えていった",
            "正しい答えを選ぶたび、自転車のベルが残念そうに鳴った",
            "理由を尋ねると、卵焼きが今朝の料理長だと小声で教えられた",
            "鍵の持ち主を探すうち、町中の人が鍵を当てたことが分かった",
            "準備運動を終えたころ、乗るはずの電車が静かに出発した",
        ],
        "endings": [
            "一本は、どうやら猫の分だった。",
            "目的地より先に、評判の団子屋へ着いた。",
            "結局、三人で代金を払い、誰が店主かは分からないままだった。",
            "最後には、説明していた本人まで列の後ろに並んだ。",
            "合格証には『歩いたほうが速い』と書いてあった。",
            "敬意を表して食べると、いつもの卵焼きの味がした。",
            "夕方、その鍵で福引会場の倉庫だけが無事に開いた。",
            "駅員は深くうなずき、これで次の電車には間に合うと言った。",
        ],
    },
    "恐怖": {
        "openings": [
            "終電を逃し、誰もいないホームで次の列車を待った",
            "細い路地で、自分のほかに足音が一つ多いことに気づいた",
            "古い宿で、使われていないはずの隣室から水音を聞いた",
            "閉館前の資料館で、自分と同じ名前の日記を見つけた",
            "夜の公園で、誰も乗っていないブランコが急に止まるのを見た",
            "山道の電話箱で、鳴り続ける受話器を取った",
            "海辺の階段で、濡れた足跡が陸へ向かって続くのを見つけた",
            "無人の写真館で、暗室から呼ばれる自分の名前を聞いた",
        ],
        "middles": [
            "やがて時刻表にない列車が止まり、扉の奥から名前を呼ぶ声がした",
            "立ち止まると音も止まり、歩き出すと一歩遅れてまた続いた",
            "戸を開けても部屋は空で、濡れた畳に手形だけが残っていた",
            "最後の頁には、今夜ここへ来ることまで細かく記されていた",
            "背を向けた途端、鎖のきしむ音がすぐ後ろまで近づいた",
            "受話器の向こうで、数秒後の自分の声が帰れと繰り返した",
            "足跡は階段の途中で消え、肩に冷たい水滴が落ちた",
            "現像液の中で、まだ撮っていない自分の写真が浮かび上がった",
        ],
        "endings": [
            "時計の針は、同じ一分を繰り返していた。",
            "家へ着くと、玄関に濡れた靴が一足増えていた。",
            "朝になっても、隣室の鍵だけは内側から掛かっていた。",
            "読み終えた頁の下に、いま書いたばかりの署名があった。",
            "振り返ると、ブランコには自分の上着が座っていた。",
            "電話を切ったあと、ポケットの中でも同じ呼び出し音が鳴った。",
            "波の音に混じって、耳元でただいまと囁く声がした。",
            "写真の裏には、明日の日付が書かれていた。",
        ],
    },
}


def clean_condition(value: str, label: str) -> str:
    value = "".join(value.strip().splitlines())
    if not value:
        raise ValueError(f"{label}不能为空")
    if len(value) > 16:
        raise ValueError(f"{label}请控制在 16 个字符以内")
    return value


def normalize_style(style: str) -> str:
    """Accept Chinese/Japanese labels, then store one canonical Japanese token."""
    style = clean_condition(style, "故事风格")
    try:
        return STYLE_ALIASES[style]
    except KeyError as error:
        choices = " / ".join(STYLES)
        raise ValueError(f"故事风格请选择：{choices}") from error


def story_prompt(name: str, place: str, style: str) -> str:
    """The generated story starts here, while the style remains a condition."""
    name = clean_condition(name, "姓名")
    place = clean_condition(place, "地点")
    style = normalize_style(style)
    return f"名前:{name}\n場所:{place}\n作風:{style}\n物語:{place}で、{name}は"


def make_story(rng: random.Random, name: str, place: str, style: str) -> str:
    style = normalize_style(style)
    parts = STYLE_PARTS[style]
    return (
        f"{place}で、{name}は{rng.choice(parts['openings'])}。"
        f"{rng.choice(parts['middles'])}。{rng.choice(parts['endings'])}"
    )


def records(count: int, seed: int = 20260823) -> Iterator[dict[str, str]]:
    rng = random.Random(seed)
    for index in range(count):
        name = rng.choice(NAMES)
        place = rng.choice(PLACES)
        style = STYLES[index % len(STYLES)]
        yield {
            "name": name,
            "place": place,
            "style": style,
            "story": make_story(rng, name, place, style),
        }


def training_text(record: dict[str, str]) -> str:
    return (
        f"名前:{record['name']}\n場所:{record['place']}\n作風:{record['style']}\n"
        f"物語:{record['story']}"
    )


def build_dataset(path: str | Path, count: int = 8_000, seed: int = 20260823) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records(count, seed):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def load_records(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="生成透明的日文小故事训练集")
    parser.add_argument("--output", default="data/stories.jsonl")
    parser.add_argument("--count", type=int, default=8_000)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()
    path = build_dataset(args.output, args.count, args.seed)
    first = load_records(path)[0]
    print(f"已写入 {args.count:,} 条故事：{path}")
    print("第一条示例：")
    print(f"作風：{first['style']}")
    print(first["story"])


if __name__ == "__main__":
    main()
