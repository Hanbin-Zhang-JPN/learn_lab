'use client';

import { FormEvent, useEffect, useRef, useState } from 'react';
import { BrowserStoryModel } from '../lib/browser-model';

const demoStories: Record<string, string[]> = {
  恋愛: [
    '雨宿りした店先で、昔と変わらない声に呼び止められた。振り向くと、会えなかった人が一本の傘を差し出していた。二人は肩を寄せ、遠回りして帰った。',
    '古い喫茶店の窓辺で、向かいの席に置かれた手紙を見つけた。差出人は、いつも同じ電車で見かける人だった。店を出るころ、その人が外で静かに待っていた。',
  ],
  文芸: [
    '朝の川沿いを歩きながら、水面に揺れる家々を眺めた。風が吹くたび、町は一度ほどけて、また別の形に結び直された。その日から、見慣れた道が少し違って見えた。',
    '古本屋で、余白ばかりの薄い本を手に取った。最後の頁にだけ、昨日見た雲の形が描かれていた。本を閉じると、夕方の鐘が遠くで一度鳴った。',
  ],
  ユーモア: [
    '駅前で、どうしても動かない自動販売機と十分も話し合った。そこへ猫が来て足元のボタンを押すと、温かいお茶が二本出てきた。一本は、どうやら猫の分だった。',
    '道を尋ねた相手が、町でいちばん迷子になりやすい郵便配達員だった。二人で地図を逆さに持って歩いた結果、目的地より先に評判の団子屋へ着いた。',
  ],
  恐怖: [
    '終電を逃し、誰もいないホームで次の列車を待った。やがて時刻表にない列車が止まり、開いた扉の奥から自分の名前を呼ぶ声がした。時計の針は、同じ一分を繰り返していた。',
    '細い路地で、足音が一つ多いことに気づいた。立ち止まると音も止まり、歩き出すとまた続いた。家へ着くと、玄関には濡れた靴が一足、きれいに揃えてあった。',
  ],
};

function demoStory(name: string, place: string, style: string): string {
  const stories = demoStories[style] ?? demoStories.文芸;
  const key = Array.from(name + place + style).reduce(
    (sum, char) => sum + (char.codePointAt(0) ?? 0),
    0,
  );
  return `${place}で、${name}は${stories[key % stories.length]}`;
}

export default function Home() {
  const [name, setName] = useState('');
  const [place, setPlace] = useState('');
  const [style, setStyle] = useState('');
  const [story, setStory] = useState('');
  const [ready, setReady] = useState(false);
  const [working, setWorking] = useState(false);
  const modelRef = useRef<BrowserStoryModel | null>(null);

  useEffect(() => {
    BrowserStoryModel.load()
      .then((model) => {
        modelRef.current = model;
      })
      .catch(() => {
        modelRef.current = null;
      })
      .finally(() => setReady(true));
  }, []);

  async function createStory(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const safeName = name.trim().slice(0, 16);
    const safePlace = place.trim().slice(0, 16);
    if (!safeName || !safePlace || !style || !ready || working) return;

    setWorking(true);
    setStory('');
    try {
      if (modelRef.current) {
        setStory(await modelRef.current.generate(safeName, safePlace, style));
      } else {
        await new Promise((resolve) => setTimeout(resolve, 280));
        setStory(demoStory(safeName, safePlace, style));
      }
    } catch {
      setStory(demoStory(safeName, safePlace, style));
    } finally {
      setWorking(false);
    }
  }

  return (
    <main className="page-shell">
      <div className="composer">
        <header>
          <p className="brand">Monogatari</p>
          <h1>名前、場所、作風を<br />教えてください。</h1>
          <p className="intro">三つの言葉から、短い物語をつくります。</p>
        </header>

        <form onSubmit={createStory}>
          <div className="fields">
            <label>
              <span>人の名前</span>
              <input
                autoFocus
                autoComplete="off"
                maxLength={16}
                placeholder="葵"
                value={name}
                onChange={(event) => setName(event.target.value)}
              />
            </label>
            <label>
              <span>場所</span>
              <input
                autoComplete="off"
                maxLength={16}
                placeholder="鎌倉"
                value={place}
                onChange={(event) => setPlace(event.target.value)}
              />
            </label>
            <label>
              <span>作風</span>
              <select value={style} onChange={(event) => setStyle(event.target.value)}>
                <option value="">選んでください</option>
                <option value="恋愛">恋愛</option>
                <option value="文芸">文芸</option>
                <option value="ユーモア">ユーモア</option>
                <option value="恐怖">恐怖</option>
              </select>
            </label>
          </div>
          <button type="submit" disabled={!ready || working || !name.trim() || !place.trim() || !style}>
            {!ready ? '準備しています' : working ? '書いています' : '物語をつくる'}
          </button>
        </form>

        {story && (
          <article className="story" aria-live="polite">
            <p>{story}</p>
          </article>
        )}
      </div>
    </main>
  );
}
