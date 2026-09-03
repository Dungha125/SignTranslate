import { useCallback, useEffect, useState } from 'react'
import { Check, RefreshCw, Trash2, X, Zap } from 'lucide-react'
import { clearHistory, errMessage, getHistory, sendFeedback, timeAgo } from '../lib/api'
import PageHead from './PageHead'

export default function HistoryView() {
  const [entries, setEntries] = useState([])
  const [filter, setFilter] = useState('all')     // all | unrated | wrong
  const [note, setNote] = useState(null)

  const refresh = useCallback(() => {
    getHistory({ limit: 100 })
      .then((d) => setEntries(d.entries || []))
      .catch((e) => setNote(errMessage(e)))
  }, [])

  useEffect(() => { refresh() }, [refresh])

  const rate = async (entry, correct, trueGloss) => {
    try {
      const updated = await sendFeedback(entry.id, correct, trueGloss)
      setEntries((es) => es.map((e) => (e.id === entry.id ? updated : e)))
    } catch (e) {
      setNote(errMessage(e))
    }
  }

  const wipe = async () => {
    if (!window.confirm('Xoá toàn bộ lịch sử dịch?')) return
    try { await clearHistory(); setEntries([]) } catch (e) { setNote(errMessage(e)) }
  }

  const shown = entries.filter((e) => {
    if (filter === 'unrated') return !e.feedback
    if (filter === 'wrong') return e.feedback && !e.feedback.correct
    return true
  })

  return (
    <>
      <PageHead
        title="Lịch sử dịch"
        sub="Lưu trong Redis (7 ngày). Đánh dấu đúng/sai để xây dữ liệu đánh giá thực tế và tìm những từ model hay nhầm."
        action={
          <div className="row gap-2">
            <button className="btn btn-ghost btn-sm" onClick={refresh}><RefreshCw size={13} /> Làm mới</button>
            <button className="btn btn-danger btn-sm" onClick={wipe} disabled={!entries.length}>
              <Trash2 size={13} /> Xoá hết
            </button>
          </div>
        }
      />

      {note && <p className="small" style={{ color: 'var(--rose)' }}>{note}</p>}

      <div className="row gap-2">
        {[['all', 'Tất cả'], ['unrated', 'Chưa đánh giá'], ['wrong', 'Bị sai']].map(([id, label]) => (
          <button
            key={id}
            className="btn btn-sm"
            onClick={() => setFilter(id)}
            style={{
              border: `1px solid ${filter === id ? 'var(--accent)' : 'var(--line)'}`,
              background: filter === id ? 'var(--accent-soft)' : 'var(--surface)',
              color: filter === id ? 'var(--accent-ink)' : 'var(--ink-2)',
            }}
          >
            {label}
          </button>
        ))}
        <div className="grow" />
        <span className="small dim mono">{shown.length}/{entries.length}</span>
      </div>

      {shown.length === 0 ? (
        <div className="card empty">
          <h3>Chưa có lượt dịch nào</h3>
          <p className="small">Sang tab <strong>Dịch</strong> và thử một video — mọi lượt sẽ được ghi lại ở đây.</p>
        </div>
      ) : (
        <div className="stack gap-2">
          {shown.map((e) => <Row key={e.id} entry={e} onRate={rate} />)}
        </div>
      )}
    </>
  )
}

function Row({ entry, onRate }) {
  const [asking, setAsking] = useState(false)
  const [trueGloss, setTrueGloss] = useState('')
  const fb = entry.feedback

  return (
    <article className="card card-pad">
      <div className="row gap-3 wrap-row">
        <div style={{ minWidth: 150 }}>
          <div style={{ fontFamily: 'var(--font-display)', fontSize: '1.2rem' }}>{entry.top_gloss || '—'}</div>
          <div className="tiny dim mono">{(entry.top_score * 100).toFixed(1)}%</div>
        </div>

        <div className="grow row gap-2 wrap-row">
          {(entry.predictions || []).slice(1, 4).map((p) => (
            <span key={p.rank} className="chip tiny">{p.gloss} <span className="mono dim">{p.confidence_pct.toFixed(0)}</span></span>
          ))}
        </div>

        <div className="row gap-2 wrap-row">
          <span className="chip tiny">{entry.source}</span>
          {entry.cached && <span className="chip chip-steel tiny"><Zap size={10} /> cache</span>}
          <span className="chip tiny mono">{entry.elapsed_ms} ms</span>
          <span className="tiny dim">{timeAgo(entry.created_at)}</span>
        </div>

        <div className="row gap-2">
          {fb ? (
            <span className={`chip ${fb.correct ? 'chip-accent' : 'chip-rose'}`}>
              {fb.correct ? <><Check size={11} /> đúng</> : <><X size={11} /> {fb.true_gloss || 'sai'}</>}
            </span>
          ) : (
            <>
              <button className="btn btn-quiet btn-sm" onClick={() => onRate(entry, true)} title="Đúng"><Check size={14} /></button>
              <button className="btn btn-quiet btn-sm" onClick={() => setAsking((a) => !a)} title="Sai"><X size={14} /></button>
            </>
          )}
        </div>
      </div>

      {asking && !fb && (
        <div className="row gap-2" style={{ marginTop: 10 }}>
          <input
            className="field"
            placeholder="Từ đúng là gì?"
            value={trueGloss}
            onChange={(ev) => setTrueGloss(ev.target.value)}
            onKeyDown={(ev) => ev.key === 'Enter' && onRate(entry, false, trueGloss)}
            style={{ maxWidth: 260 }}
            autoFocus
          />
          <button className="btn btn-primary btn-sm" onClick={() => onRate(entry, false, trueGloss)}>Gửi</button>
        </div>
      )}

      {entry.filename && <div className="tiny dim truncate" style={{ marginTop: 6 }}>{entry.filename}</div>}
    </article>
  )
}
