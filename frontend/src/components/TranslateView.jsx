import { useCallback, useEffect, useMemo, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { Film, Search, Upload, X } from 'lucide-react'
import { errMessage, getModels, getVocab, translateFrames, translateVideo } from '../lib/api'
import { useWebcam } from '../hooks/useWebcam'
import WebcamPanel from './WebcamPanel'
import ResultCard from './ResultCard'
import SentenceBar, { useSentence } from './SentenceBar'
import PageHead from './PageHead'

const FALLBACK_CAPTURE = {
  capture_buffer_frames: 48,
  capture_min_frames: 20,
  capture_interval_ms: 90,
  capture_jpeg_quality: 0.9,
}

export default function TranslateView({ modelId }) {
  const [cfg, setCfg] = useState(FALLBACK_CAPTURE)
  const [source, setSource] = useState('upload')     // upload | webcam
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)
  const sentence = useSentence()

  useEffect(() => {
    getModels()
      .then((ms) => {
        const m = ms.find((x) => x.id === modelId) || ms[0]
        if (m) setCfg({ ...FALLBACK_CAPTURE, ...m })
      })
      .catch(() => {})
  }, [modelId])

  const cam = useWebcam({
    maxFrames: cfg.capture_buffer_frames,
    intervalMs: cfg.capture_interval_ms,
    quality: cfg.capture_jpeg_quality,
  })

  const ok = (r) => { setResult(r); setError(null) }
  const bad = (e) => { setError(typeof e === 'string' ? e : errMessage(e)); setResult(null) }

  const runFrames = async () => {
    const frames = cam.frames.current
    if (frames.length < cfg.capture_min_frames) {
      bad(`Cần ít nhất ${cfg.capture_min_frames} frame, mới ghi được ${frames.length}.`)
      return
    }
    setLoading(true)
    try { ok(await translateFrames(frames, modelId)) } catch (e) { bad(e) } finally { setLoading(false) }
  }

  const seconds = ((cfg.capture_buffer_frames * cfg.capture_interval_ms) / 1000).toFixed(0)

  return (
    <>
      <PageHead
        title="Dịch ngôn ngữ ký hiệu"
        sub="Tải video lên hoặc ghi trực tiếp từ webcam. Kết quả gồm 10 khả năng xếp hạng, có thể ghép dần thành câu."
      />

      <SourceToggle
        value={source}
        onChange={(v) => { setSource(v); setResult(null); setError(null) }}
      />

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
              hint={`Thực hiện ký hiệu trong khoảng ${seconds} giây, giữ hai tay và khuôn mặt trong khung hình.`}
              actions={
                <button className="btn btn-primary grow" onClick={runFrames} disabled={loading}>
                  {loading ? <span className="spinner" /> : <Search size={14} />} Dịch
                </button>
              }
            />
          )}
          <SentenceBar sentence={sentence} />
        </div>

        <div className="stack gap-4">
          <ResultCard result={result} error={error} loading={loading} onAppend={sentence.append} />
          {!result && !error && !loading && <VocabCard modelId={modelId} />}
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
    <div className="row">
      <div className="subtabs" style={{ background: 'var(--surface-2)', border: '1px solid var(--line)', borderRadius: 99, padding: 4 }}>
        {opts.map((o) => (
          <button
            key={o.id}
            onClick={() => onChange(o.id)}
            className="tab"
            data-active={value === o.id}
          >
            <o.icon size={13} /> {o.label}
          </button>
        ))}
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
            <div className="stage">
              <video src={preview} controls style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
            </div>
            <div className="row gap-2 small muted">
              <Film size={14} />
              <span className="truncate grow">{file.name}</span>
              <span className="mono tiny dim">{(file.size / 1048576).toFixed(1)} MB</span>
            </div>
          </>
        )}

        <label className="row gap-2 small muted" style={{ cursor: 'pointer' }}>
          <input type="checkbox" checked={keep} onChange={(e) => setKeep(e.target.checked)} />
          Lưu video này vào kho dữ liệu sau khi dịch
        </label>

        <button className="btn btn-primary btn-lg btn-block" onClick={run} disabled={!file || loading}>
          {loading ? <><span className="spinner" /> Đang phân tích…</> : <><Search size={15} /> Dịch video</>}
        </button>
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
        <h3>Từ model nhận được</h3>
        <span className="chip mono">{glosses.length}</span>
        <div className="grow" />
        <input
          className="field"
          placeholder="Tìm từ…"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          style={{ maxWidth: 180, padding: '6px 10px', fontSize: '.8rem' }}
        />
      </div>
      <div className="card-body scroll-y" style={{ maxHeight: 340 }}>
        <div className="row gap-2 wrap-row">
          {shown.map((g) => <span key={g} className="chip">{g}</span>)}
          {!shown.length && <p className="small dim">Không có từ nào khớp “{q}”.</p>}
        </div>
      </div>
      <div className="card-pad tiny dim" style={{ borderTop: '1px solid var(--line)', padding: '10px 20px' }}>
        Muốn xem cách thực hiện từng từ? Sang tab <strong>Từ vựng</strong> để xem clip mẫu và tự luyện.
      </div>
    </section>
  )
}
