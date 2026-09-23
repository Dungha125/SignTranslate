import { useCallback, useEffect, useMemo, useState } from 'react'
import { Check, RefreshCw, Save } from 'lucide-react'
import { enrollFrames, errMessage, getGallery, getModels, getVocab, rebuildGallery } from '../lib/api'
import { useWebcam } from '../hooks/useWebcam'
import WebcamPanel from './WebcamPanel'
import PageHead from './PageHead'

const FALLBACK = { capture_buffer_frames: 32, capture_min_frames: 16, capture_interval_ms: 100, capture_jpeg_quality: 0.9 }

/**
 * Enroll: thêm mẫu cá nhân vào gallery kNN.
 * Đây là cách nhanh nhất để nâng độ chính xác trên webcam của chính bạn —
 * mô hình không đổi, chỉ có tập tham chiếu giàu thêm.
 */
export default function EnrollTab({ modelId }) {
  const [cfg, setCfg] = useState(FALLBACK)
  const [vocab, setVocab] = useState([])
  const [gallery, setGallery] = useState(null)
  const [gloss, setGloss] = useState('')
  const [q, setQ] = useState('')
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState(null)

  const cam = useWebcam({
    maxFrames: cfg.capture_buffer_frames,
    intervalMs: cfg.capture_interval_ms,
    quality: cfg.capture_jpeg_quality,
  })

  const refreshGallery = useCallback(() => {
    getGallery(modelId).then(setGallery).catch(() => setGallery(null))
  }, [modelId])

  useEffect(() => {
    getModels()
      .then((ms) => {
        const m = ms.find((x) => x.id === modelId)
        if (m) setCfg({ ...FALLBACK, ...m })
      })
      .catch(() => {})
    getVocab(modelId).then((d) => setVocab(d.glosses || [])).catch(() => setVocab([]))
    refreshGallery()
  }, [modelId, refreshGallery])

  const shown = useMemo(() => {
    const s = q.trim().toLowerCase()
    return (s ? vocab.filter((g) => g.toLowerCase().includes(s)) : vocab).slice(0, 200)
  }, [vocab, q])

  const enrolled = gallery?.by_gloss?.[gloss] || 0

  const save = async () => {
    if (!gloss) { setMsg({ bad: true, text: 'Chọn một từ trước khi lưu.' }); return }
    const frames = cam.frames.current
    if (frames.length < cfg.capture_min_frames) {
      setMsg({ bad: true, text: `Cần ít nhất ${cfg.capture_min_frames} frame, mới ghi ${frames.length}.` })
      return
    }
    setBusy(true)
    try {
      const r = await enrollFrames({ frames_b64: frames, gloss, model_id: modelId })
      setMsg({ bad: false, text: `Đã lưu mẫu cho “${gloss}”. Gallery hiện có ${r.total_refs} tham chiếu.` })
      cam.reset()
      refreshGallery()
    } catch (e) {
      setMsg({ bad: true, text: errMessage(e) })
    } finally { setBusy(false) }
  }

  const rebuild = async () => {
    if (!window.confirm('Dựng lại gallery từ tập train + val? Quá trình này có thể mất vài phút.')) return
    setBusy(true)
    setMsg({ bad: false, text: 'Đang dựng lại gallery…' })
    try {
      await rebuildGallery(modelId)
      setMsg({ bad: false, text: 'Đã dựng lại gallery.' })
      refreshGallery()
    } catch (e) {
      setMsg({ bad: true, text: errMessage(e) })
    } finally { setBusy(false) }
  }

  return (
    <>
      <PageHead
        title="Cá nhân hoá nhận dạng"
        sub="Ghi 2–3 mẫu cho mỗi từ bằng chính webcam và ánh sáng bạn sẽ dùng. Mẫu vào thẳng gallery kNN nên có tác dụng ngay, không cần huấn luyện lại model."
        action={
          <button className="btn btn-ghost btn-sm" onClick={rebuild} disabled={busy}>
            <RefreshCw size={13} /> Dựng lại từ train+val
          </button>
        }
      />

      {msg && (
        <p
          className="small"
          style={{
            background: msg.bad ? 'var(--rose-soft)' : 'var(--accent-soft)',
            color: msg.bad ? 'var(--rose)' : 'var(--accent-ink)',
            padding: '10px 14px', borderRadius: 'var(--r-md)',
          }}
        >
          {msg.text}
        </p>
      )}

      {gallery && (
        <div className="grid-4">
          <Stat label="Tham chiếu" value={gallery.total_refs} />
          <Stat label="Từ đã phủ" value={gallery.num_classes_covered} />
          <Stat label="Từ trong bộ" value={vocab.length} />
          <Stat label={gloss ? `Mẫu cho “${gloss}”` : 'Từ đang chọn'} value={gloss ? enrolled : '—'} />
        </div>
      )}

      <div className="grid-2" style={{ alignItems: 'start' }}>
        <section className="card">
          <div className="card-head">
            <h3>Chọn từ</h3>
            <div className="grow" />
            <input
              className="field"
              placeholder="Tìm từ…"
              value={q}
              onChange={(e) => setQ(e.target.value)}
              style={{ maxWidth: 180, padding: '6px 10px', fontSize: '.8rem' }}
            />
          </div>
          <div className="card-body scroll-y" style={{ maxHeight: 420 }}>
            <div className="row gap-2 wrap-row">
              {shown.map((g) => {
                const n = gallery?.by_gloss?.[g] || 0
                const active = g === gloss
                return (
                  <button
                    key={g}
                    onClick={() => setGloss(g)}
                    className="chip"
                    style={{
                      cursor: 'pointer',
                      borderColor: active ? 'var(--accent)' : 'var(--line)',
                      background: active ? 'var(--accent-soft)' : 'var(--surface-2)',
                      color: active ? 'var(--accent-ink)' : 'var(--ink-2)',
                      fontWeight: active ? 600 : 500,
                    }}
                  >
                    {g}
                    {n > 0 && <span className="mono tiny dim">{n}</span>}
                    {n >= 2 && <Check size={10} />}
                  </button>
                )
              })}
              {!shown.length && <p className="small dim">Không có từ nào khớp.</p>}
            </div>
          </div>
        </section>

        <WebcamPanel
          cam={cam}
          title={gloss ? `Ghi mẫu: ${gloss}` : 'Ghi mẫu'}
          maxFrames={cfg.capture_buffer_frames}
          minFrames={cfg.capture_min_frames}
          hint={`Thực hiện ký hiệu trong khoảng ${((cfg.capture_buffer_frames * cfg.capture_interval_ms) / 1000).toFixed(0)} giây. Ghi 2–3 lần cho mỗi từ để kNN ổn định.`}
          actions={
            <button className="btn btn-primary grow" onClick={save} disabled={busy || !gloss}>
              {busy ? <span className="spinner" /> : <Save size={14} />}
              {gloss ? `Lưu vào gallery` : 'Chọn từ trước'}
            </button>
          }
        />
      </div>
    </>
  )
}

function Stat({ label, value }) {
  return (
    <div className="card card-pad">
      <div className="eyebrow">{label}</div>
      <div className="stat-value" style={{ marginTop: 4 }}>{value}</div>
    </div>
  )
}
