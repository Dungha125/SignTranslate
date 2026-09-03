import { useCallback, useEffect, useState } from 'react'
import { Copy, Delete, Eraser } from 'lucide-react'
import { appendSentence, getSentence, popSentence, resetSentence } from '../lib/api'

/**
 * Ghép nhiều lượt nhận dạng thành một câu.
 * Trạng thái nằm ở Redis (theo tài khoản) nên đóng tab mở lại vẫn còn.
 */
export function useSentence() {
  const [tokens, setTokens] = useState([])
  const [text, setText] = useState('')

  const apply = useCallback((d) => {
    setTokens(d.tokens || [])
    setText(d.text || '')
  }, [])

  useEffect(() => {
    getSentence().then(apply).catch(() => {})
  }, [apply])

  const append = useCallback(
    (pred) => appendSentence(pred.gloss, pred.score ?? 0).then(apply).catch(() => {}),
    [apply],
  )
  const pop = useCallback(() => popSentence().then(apply).catch(() => {}), [apply])
  const reset = useCallback(() => resetSentence().then(apply).catch(() => {}), [apply])

  return { tokens, text, append, pop, reset }
}

export default function SentenceBar({ sentence }) {
  const { tokens, text, pop, reset } = sentence
  const [copied, setCopied] = useState(false)

  const copy = () => {
    navigator.clipboard?.writeText(text).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1600)
    })
  }

  return (
    <section className="card">
      <div className="card-head">
        <h3>Câu đang dựng</h3>
        <span className="chip mono">{tokens.length} từ</span>
        <div className="grow" />
        <button className="btn btn-quiet btn-sm" onClick={copy} disabled={!text}>
          <Copy size={13} /> {copied ? 'Đã chép' : 'Chép'}
        </button>
        <button className="btn btn-quiet btn-sm" onClick={pop} disabled={!tokens.length} title="Xoá từ cuối">
          <Delete size={13} />
        </button>
        <button className="btn btn-quiet btn-sm" onClick={reset} disabled={!tokens.length} title="Xoá cả câu">
          <Eraser size={13} />
        </button>
      </div>
      <div className="card-body">
        {tokens.length === 0 ? (
          <p className="small dim">
            Bấm <em>Thêm vào câu</em> ở kết quả để ghép nhiều ký hiệu thành một câu hoàn chỉnh.
          </p>
        ) : (
          <>
            <p style={{ fontFamily: 'var(--font-display)', fontSize: '1.35rem', lineHeight: 1.35 }}>
              {text}
            </p>
            <div className="row gap-2 wrap-row" style={{ marginTop: 12 }}>
              {tokens.map((t, i) => (
                <span key={`${t.gloss}-${i}`} className="chip">
                  {t.gloss}
                  {t.score > 0 && <span className="mono tiny dim">{(t.score * 100).toFixed(0)}%</span>}
                </span>
              ))}
            </div>
          </>
        )}
      </div>
    </section>
  )
}
