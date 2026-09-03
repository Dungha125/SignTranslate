import { useState } from 'react'
import { Check, Plus, X, Zap } from 'lucide-react'
import { sendFeedback } from '../lib/api'

/** Kết quả nhận dạng + phản hồi đúng/sai + thêm vào câu. */
export default function ResultCard({ result, error, loading, onAppend }) {
  const [fb, setFb] = useState(null)          // null | 'ok' | 'wrong' | 'sent'
  const [wrongGloss, setWrongGloss] = useState('')

  if (loading) {
    return (
      <section className="card">
        <div className="card-head"><h3>Kết quả</h3></div>
        <div className="empty stack gap-3" style={{ alignItems: 'center' }}>
          <span className="spinner spinner-lg" style={{ color: 'var(--accent)' }} />
          <span className="small">Đang trích skeleton và suy luận…</span>
        </div>
      </section>
    )
  }

  if (error) {
    return (
      <section className="card">
        <div className="card-head"><h3>Kết quả</h3></div>
        <div className="card-body">
          <div style={{ background: 'var(--rose-soft)', borderRadius: 'var(--r-md)', padding: '13px 15px' }}>
            <div style={{ color: 'var(--rose)', fontWeight: 600, marginBottom: 3 }}>Không dịch được</div>
            <div className="small muted" style={{ wordBreak: 'break-word' }}>{error}</div>
          </div>
        </div>
      </section>
    )
  }

  if (!result) return null

  const preds = result.predictions || []
  const top = preds[0]
  const rest = preds.slice(1, 6)
  const confident = (top?.confidence_pct ?? 0) >= 45

  const submit = async (correct) => {
    if (!result.entry_id) return
    try {
      await sendFeedback(result.entry_id, correct, correct ? null : wrongGloss)
      setFb('sent')
    } catch {
      setFb('sent')
    }
  }

  return (
    <section className="card rise">
      <div className="card-head">
        <h3>Kết quả</h3>
        <span className="chip">{result.display_name}</span>
        <div className="grow" />
        {result.cached && (
          <span className="chip chip-steel" title="Lấy lại từ cache Redis">
            <Zap size={11} /> cache
          </span>
        )}
        <span className="chip mono">{result.elapsed_ms} ms</span>
      </div>

      {top && (
        <div className="card-body" style={{ borderBottom: '1px solid var(--line)' }}>
          <div className="eyebrow" style={{ marginBottom: 8 }}>Nhận dạng</div>
          <div className="row gap-3 wrap-row" style={{ alignItems: 'baseline' }}>
            <span className="mark">{top.gloss}</span>
            <span
              className="chip"
              style={{
                background: confident ? 'var(--accent-soft)' : 'var(--amber-soft)',
                color: confident ? 'var(--accent-ink)' : 'var(--amber)',
                borderColor: 'transparent',
              }}
            >
              <span className="mono">{top.confidence_pct.toFixed(1)}%</span>
              {!confident && ' · độ tin cậy thấp'}
            </span>
          </div>

          <div className="meter" style={{ marginTop: 14 }}>
            <i style={{ width: `${Math.min(100, top.confidence_pct)}%` }} />
          </div>

          <div className="row gap-2 wrap-row" style={{ marginTop: 16 }}>
            {onAppend && (
              <button className="btn btn-ghost btn-sm" onClick={() => onAppend(top)}>
                <Plus size={13} /> Thêm vào câu
              </button>
            )}
            {result.entry_id && fb !== 'sent' && (
              <>
                <button className="btn btn-ghost btn-sm" onClick={() => submit(true)} title="Kết quả đúng">
                  <Check size={13} /> Đúng
                </button>
                <button className="btn btn-ghost btn-sm" onClick={() => setFb('wrong')} title="Kết quả sai">
                  <X size={13} /> Sai
                </button>
              </>
            )}
            {fb === 'sent' && <span className="chip chip-accent"><Check size={11} /> Đã ghi nhận</span>}
          </div>

          {fb === 'wrong' && (
            <div className="row gap-2" style={{ marginTop: 10 }}>
              <input
                className="field"
                placeholder="Từ đúng là gì?"
                value={wrongGloss}
                onChange={(e) => setWrongGloss(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && submit(false)}
                style={{ maxWidth: 260 }}
                autoFocus
              />
              <button className="btn btn-primary btn-sm" onClick={() => submit(false)}>Gửi</button>
              <button className="btn btn-quiet btn-sm" onClick={() => setFb(null)}>Bỏ qua</button>
            </div>
          )}
        </div>
      )}

      {rest.length > 0 && (
        <div className="card-body">
          <div className="eyebrow" style={{ marginBottom: 10 }}>Khả năng khác</div>
          <div className="stack gap-2">
            {rest.map((p) => (
              <div key={p.rank} className="row gap-3">
                <span className="mono dim tiny" style={{ width: 16 }}>{p.rank}</span>
                <span className="small" style={{ minWidth: 110 }}>{p.gloss}</span>
                <div className="grow">
                  <div className="meter meter-quiet" style={{ height: 4 }}>
                    <i style={{ width: `${Math.min(100, (p.confidence_pct / (top?.confidence_pct || 100)) * 100)}%` }} />
                  </div>
                </div>
                <span className="mono tiny dim" style={{ width: 44, textAlign: 'right' }}>
                  {p.confidence_pct.toFixed(1)}%
                </span>
                {onAppend && (
                  <button className="btn btn-quiet btn-sm" onClick={() => onAppend(p)} title="Thêm vào câu">
                    <Plus size={12} />
                  </button>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </section>
  )
}
