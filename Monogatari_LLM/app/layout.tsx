import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  metadataBase: new URL('https://monogatari-llm.chang-hanpin0927.chatgpt.site'),
  title: 'ものがたり — 名前、場所、作風から短い話',
  description: '名前と場所、四つの作風から短い日本語の物語をつくります。',
  openGraph: {
    title: 'ものがたり — 名前、場所、作風から短い話',
    description: '名前と場所、四つの作風から短い日本語の物語をつくります。',
    type: 'website',
    locale: 'ja_JP',
    url: 'https://monogatari-llm.chang-hanpin0927.chatgpt.site',
    images: [{ url: '/og.png', width: 1200, height: 630, alt: 'Monogatari_LLM — ことばが物語になるまで。' }],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'ものがたり — 名前、場所、作風から短い話',
    description: '名前と場所、四つの作風から短い日本語の物語をつくります。',
    images: ['/og.png'],
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="ja">
      <body>{children}</body>
    </html>
  );
}
