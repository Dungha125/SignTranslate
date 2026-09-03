import { useCallback, useEffect, useMemo, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { Film, Layers, Search, Upload, X } from 'lucide-react'
import {
  compareModels,
  errMessage,
  getModels,
  getVocab,
  translateFrames,
  translateVideo,
} from '../lib/api'
import { useWebcam } from '../hooks/useWebcam'
import WebcamPanel from './WebcamPanel'
import ResultCard from './ResultCard'
import SentenceBar, { useSentence } from './SentenceBar'
import PageHead from './PageHead'

const FALLBACK_CAPTURE = { capture_buffer_frames: 32, capture_min_frames: 16, capture_interval_ms: 100, capture_jpeg_quality: 0.9 }

export default function TranslateView({ modelId, onModelId }) {
  const [models, setModels] = useState([])
  const [source, setSource] = useState('upload')     // upload | webcam
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)
  const [compare, setCompare] = useState(null)
  const sentence = useSentence()

  useEffect(() => {
    getModels().then(setModels).catch(() => setModels([]))
  }, [])

  const cfg = useMemo(() => {
    const m = models.find((x) => x.id === modelId)
    return { ...FALLBACK_CAPTURE, ...(m || {}) }
  }, [models, modelId])

  const cam = useWebcam({
    maxFrames: cfg.capture_buffer_frames,
    intervalMs: cfg.capture_interval_ms,
    quality: cfg.capture_jpeg_quality,
  })

  const ok = (r) => { setResult(r); setError(null); setCompare(null) }
  const bad = (e) => { setError(errMessage(e)); setResult(null); setCompare(null) }

  const runFrames = async () => {
    const frames = cam.frames.current
    if (frames.length < cfg.capture_min_frames) {
      bad(`Cần ít nhất ${cfg.capture_min_frames} frame, mới ghi được ${frames.length}.`)
      return
    }
    setLoading(true)
    try { ok(await translateFrames(frames, modelId)) } catch (e) { bad(e) } finally { setLoading(false) }
  }

  const runCompare = async () => {
    const frames = cam.frames.current
    const ids = models.filter((m) => m.loaded).map((m) => m.id).slice(0, 4)
    if (frames.length < 8 || ids.length < 2) {
      bad('Cần ít nhất 2 model đã nạp và một đoạn ghi hợp lệ.')
      return
    }
    setLoading(true)
    try {
      setResult(null); setError(null)
      setCompare(await compareModels(frames, ids))
    } catch (e) { bad(e) } finally { setLoading(false) }
  }

  return (
    <>
      <PageHead
        title="Dịch ngôn ngữ ký hiệu"
        sub="Tải video lên hoặc ghi trực tiếp từ webcam. Kết quả gồm 10 khả năng xếp hạng, có thể ghép dần thành câu."
      />

      <ModelBar models={models} modelId={modelId} onChange={onModelId} />

      <div className="row gap-2">
        <SourceToggle value={source} onChange={(v) => { setSource(v); setResult(null); setError(null); setCompare(null) }} />
      </div>

      <div className="grid-2" style={{ alignItems: 'start' }}>
        <div className="stack gap-4">
          {source === 'upload' ? (
            <UploadPanel modelId={modelId} loading={loading} setLoading={setLoading} onResult={ok} onError={bad} />
          ) : (
            <WebcamPanel
              cam={cam}
              title="Ghi từ webcam"
              maxFrames={cfg.capture_buffer_frames}
              minFrames={cfg.capture_min_frames}
              hint={`Thực hiện ký hiệu trong khoảng ${((cfg.capture_buffer_frames * cfg.capture_interval_ms) / 1000).toFixed(0)} giây, giữ hai tay và khuôn mặt trong khung hình.`}
              actions={
                <>
                  <button className="btn btn-primary grow" onClick={runFrames} disabled={loading}>
                    {loading ? <span className="spinner" /> : <Search size={14} />} Dịch
                  </button>
                  <button className="btn btn-ghost" onClick={runCompare} disabled={loading} title="Chạy qua mọi model đã nạp">
                    <Layers size={14} /> So sánh
                  </button>
                </>
              }
            />
          )}
          <SentenceBar sentence={sentence} />
        </div>

        <div className="stack gap-4">
          <ResultCard result={result} error={error} loading={loading} onAppend={sentence.append} />
          {compare && <CompareCard data={compare} />}
          {!result && !error && !loading && !compare && <VocabCard modelId={modelId} />}
        </div>
      </div>
    </>
  )
}

/* ─────────────────────────────────────────────────────────────────────── */
function SourceToggle({ value, onChange }) {
  const opts = [
    { id: 'upload', label: 'Tải video', icon: Upload },
    { id: 'webcam', label: 'Webcam', icon: Film },
  ]
  return (
    <div className="row" style={{ background: 'var(--surface-2)', borderRadius: 'var(--r-md)', padding: 3, gap: 2 }}>
      {opts.map((o) => {
        const active = value === o.id
        return (
          <button
            key={o.id}
            onClick={() => onChange(o.id)}
            className="btn btn-sm"
            style={{
              background: active ? 'var(--surface)' : 'transparent',
              color: active ? 'var(--ink)' : 'var(--ink-3)',
              boxShadow: active ? 'var(--shadow-sm)' : 'none',
              fontWeight: active ? 600 : 500,
            }}
          >
            <o.icon size={13} /> {o.label}
          </button>
        )
      })}
    </div>
  )
}

function ModelBar({ models, modelId, onChange }) {
  if (!models.length) return null
  return (
    <div className="card card-pad row gap-3 wrap-row">
      <span className="eyebrow" style={{ flexShrink: 0 }}>Model</span>
      <div className="row gap-2 wrap-row grow">
        {models.map((m) => {
          const active = m.id === modelId
          return (
            <button
              key={m.id}
              onClick={() => m.loaded && onChange(m.id)}
              disabled={!m.loaded}
              title={m.loaded ? m.description : 'Checkpoint chưa được nạp'}
              className="btn btn-sm"
              style={{
                border: `1px solid ${active ? 'var(--accent)' : 'var(--line)'}`,
                background: active ? 'var(--accent-soft)' : 'var(--surface)',
                color: active ? 'var(--accent-ink)' : 'var(--ink-2)',
                opacity: m.loaded ? 1 : 0.45,
                fontWeight: active ? 600 : 500,
              }}
            >
              <span className={`dot ${m.loaded ? 'dot-ok' : 'dot-warn'}`} />
              {m.display_name}
              {m.loaded && m.num_classes > 0 && (
                <span className="mono tiny dim">{m.num_classes}</span>
              )}
            </button>
          )
        })}
      </div>
    </div>
  )
}

function UploadPanel({ modelId, loading, setLoading, onResult, onError }) {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [keep, setKeep] = useState(false)

  const onDrop = useCallback((accepted) => {
    if (!accepted.length) return
    setFile(accepted[0])
    setPreview(URL.createObjectURL(accepted[0]))
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'video/*': ['.mp4', '.webm', '.mov', '.avi', '.mkv'] },
    maxFiles: 1,
    multiple: false,
  })

  const run = async () => {
    if (!file) return
    setLoading(true)
    try {
      onResult(await translateVideo(file, modelId, { saveToDataset: keep }))
    } catch (e) {
      onError(e)
    } finally {
      setLoading(false)
    }
  }

  const clear = () => {
    if (preview) URL.revokeObjectURL(preview)
    setFile(null)
    setPreview(null)
  }

  return (
    <section className="card">
      <div className="card-head">
        <h3>Tải video lên</h3>
        {file && (
          <button className="btn btn-quiet btn-sm" style={{ marginLeft: 'auto' }} onClick={clear}>
            <X size={13} /> Bỏ chọn
          </button>
        )}
      </div>
      <div className="card-body stack gap-3">
        {!file ? (
          <div {...getRootProps()} className="dropzone" data-active={isDragActive}>
            <input {...getInputProps()} />
            <Upload size={26} strokeWidth={1.4} style={{ color: 'var(--ink-3)', marginBottom: 10 }} />
            <div style={{ fontWeight: 500, marginBottom: 4 }}>
              {isDragActive ? 'Thả video vào đây' : 'Kéo thả video hoặc bấm để chọn'}
            </div>
            <div className="tiny dim">mp4 · webm · mov · avi · mkv</div>
          </div>
        ) : (
          <>
            <video src={preview} controls style={{ width: '100%', maxHeight: 320, borderRadius: 'var(--r-md)', background: '#101211' }} />
            <div className="row gap-2 small muted">
              <Film size={14} />
              <span className="truncate grow">{file.name}</span>
              <span className="mono tiny dim">{(file.size / 1048576).toFixed(1)} MB</span>
            </div>
          </>
        )}

        <label className="row gap-2 small muted" style={{ cursor: 'pointer' }}>
          <input type="checkbox" checked={keep} onChange={(e) => setKeep(e.target.checked)} />
          Lưu video này vào kho dữ liệu (MinIO) sau khi dịch
        </label>

        <button className="btn btn-primary btn-lg btn-block" onClick={run} disabled={!file || loading}>
          {loading ? <><span className="spinner" /> Đang phân tích…</> : <><Search size={15} /> Dịch video</>}
        </button>
      </div>
    </section>
  )
}

function CompareCard({ data }) {
  return (
    <section className="card rise">
      <div className="card-head">
        <h3>So sánh model</h3>
        <span className="chip">{data.results.length} model</span>
      </div>
      <div className="card-body stack gap-4">
        {data.consensus?.length > 0 && (
          <div>
            <div className="eyebrow" style={{ marginBottom: 8 }}>Đồng thuận</div>
            <div className="row gap-2 wrap-row">
              {data.consensus.map((c, i) => (
                <span key={c.gloss} className={`chip ${i === 0 ? 'chip-accent' : ''}`}>
                  {c.gloss} <span className="mono tiny">{(c.score * 100).toFixed(0)}</span>
                </span>
              ))}
            </div>
          </div>
        )}
        {data.results.map((r) => (
          <div key={r.model_id}>
            <div className="row gap-2" style={{ marginBottom: 6 }}>
              <span className="small" style={{ fontWeight: 600 }}>{r.display_name}</span>
              <div className="grow" />
              <span className="mono tiny dim">{r.elapsed_ms} ms</span>
            </div>
            {r.error ? (
              <p className="tiny" style={{ color: 'var(--rose)' }}>{r.error}</p>
            ) : (
              <div className="row gap-2 wrap-row">
                {r.predictions.slice(0, 3).map((p, i) => (
                  <span key={p.rank} className="chip" style={i === 0 ? { fontWeight: 600 } : undefined}>
                    {p.gloss} <span className="mono tiny dim">{p.confidence_pct.toFixed(0)}%</span>
                  </span>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </section>
  )
}

function VocabCard({ modelId }) {
  const [glosses, setGlosses] = useState([])
  const [q, setQ] = useState('')

  useEffect(() => {
    if (!modelId) return
    getVocab(modelId).then((d) => setGlosses(d.glosses || [])).catch(() => setGlosses([]))
  }, [modelId])

  const shown = useMemo(() => {
    const s = q.trim().toLowerCase()
    return s ? glosses.filter((g) => g.toLowerCase().includes(s)) : glosses
  }, [glosses, q])

  if (!glosses.length) return null

  return (
    <section className="card">
      <div className="card-head">
        <h3>Bộ từ vựng</h3>
        <span className="chip mono">{glosses.length}</span>
        <div className="grow" />
        <input
          className="field"
          placeholder="Tìm từ…"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          style={{ maxWidth: 190, padding: '6px 10px', fontSize: '.8rem' }}
        />
      </div>
      <div className="card-body scroll-y" style={{ maxHeight: 330 }}>
        <div className="row gap-2 wrap-row">
          {shown.map((g) => <span key={g} className="chip">{g}</span>)}
          {!shown.length && <p className="small dim">Không có từ nào khớp “{q}”.</p>}
        </div>
      </div>
    </section>
  )
}
