import { useCallback, useEffect, useState } from 'react'
import axios from 'axios'
import { GraduationCap, Play } from 'lucide-react'
import { errMessage, getModels } from '../lib/api'
import { useWebcam } from '../hooks/useWebcam'
import WebcamPanel from './WebcamPanel'
import PageHead from './PageHead'

const FALLBACK = { capture_buffer_frames: 50, capture_min_frames: 40, capture_interval_ms: 100, capture_jpeg_quality: 0.92 }

/**
 * Học từ mới: ghi một mẫu rồi fine-tune (A→B→C) ngay trên máy chủ.
 * Chỉ những model hỗ trợ continual learning mới bật được (curivsl, hsp_bimamba).
 */
export default function LearnTab({ modelId }) {
  const [cfg, setCfg] = useState(FALLBACK)
  const [supported, setSupported] = useState(true)
  const [label, setLabel] = useState('')
  const [vocab, setVocab] = useState([])
  const [jobId, setJobId] = useState(null)
  const [job, setJob] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  const cam = useWebcam({
    maxFrames: cfg.capture_buffer_frames,
    intervalMs: cfg.capture_interval_ms,
    quality: cfg.capture_jpeg_quality,
  })

  useEffect(() => {
    getModels()
      .then((ms) => {
        const m = ms.find((x) => x.id === modelId)
        if (m) {
          setCfg({ ...FALLBACK, ...m })
          setSupported(['curivsl', 'hsp_bimamba'].includes(m.type))
        }
      })
      .catch(() => {})
    axios
      .get('/api/learn/vocab', { params: { model_id: modelId } })
      .then((r) => setVocab(r.data.labels || []))
      .catch(() => setVocab([]))
  }, [modelId])

  const train = useCallback(async () => {
    if (!label.trim()) { setError('Nhập nhãn cho từ mới.'); return }
    const frames = cam.frames.current
    if (frames.length < cfg.capture_min_frames) {
      setError(`Cần ít nhất ${cfg.capture_min_frames} frame, mới ghi ${frames.length}.`)
      return
    }
    setBusy(true); setError(null); setJob(null)
    try {
      const { data } = await axios.post(
        '/api/learn/train',
        { frames_b64: frames, label: label.trim(), model_id: modelId },
        { timeout: 60_000 },
      )
      setJobId(data.job_id)
    } catch (e) {
      setError(errMessage(e))
    } finally { setBusy(false) }
  }, [label, cam, cfg, modelId])

  useEffect(() => {
    if (!jobId) return undefined
    let alive = true
    const poll = async () => {
      try {
        const { data } = await axios.get(`/api/learn/status/${jobId}`)
        if (!alive) return
        setJob(data)
        if (data.status === 'done') {
          axios.get('/api/learn/vocab', { params: { model_id: modelId } })
            .then((r) => setVocab(r.data.labels || []))
            .catch(() => {})
        }
      } catch { /* job có thể chưa sẵn sàng */ }
    }
    poll()
    const t = setInterval(poll, 1500)
    return () => { alive = false; clearInterval(t) }
  }, [jobId, modelId])

  const running = job && !['done', 'error'].includes(job.status)

  return (
    <>
      <PageHead
        title="Học từ mới"
        sub="Ghi một mẫu và fine-tune model ngay trên máy chủ theo lộ trình A→B→C, giữ lại các từ đã học bằng distillation."
      />

      {!supported && (
        <p className="small" style={{ background: 'var(--amber-soft)', color: 'var(--amber)', padding: '10px 14px', borderRadius: 'var(--r-md)' }}>
          Model <strong>{modelId}</strong> chưa hỗ trợ huấn luyện trực tuyến. Với LT-SignDiff, hãy dùng tab
          <strong> Enroll</strong> — nhanh hơn nhiều và không cần huấn luyện lại.
        </p>
      )}
      {error && (
        <p className="small" style={{ background: 'var(--rose-soft)', color: 'var(--rose)', padding: '10px 14px', borderRadius: 'var(--r-md)' }}>
          {error}
        </p>
      )}

      <div className="grid-2" style={{ alignItems: 'start' }}>
        <WebcamPanel
          cam={cam}
          title="Ghi mẫu"
          maxFrames={cfg.capture_buffer_frames}
          minFrames={cfg.capture_min_frames}
          hint={`Thực hiện ký hiệu trong khoảng ${((cfg.capture_buffer_frames * cfg.capture_interval_ms) / 1000).toFixed(0)} giây, giữ hai tay và thân trong khung.`}
        />

        <div className="stack gap-4">
          <section className="card">
            <div className="card-head">
              <GraduationCap size={15} style={{ color: 'var(--ink-3)' }} />
              <h3>Nhãn</h3>
              <div className="grow" />
              {vocab.length > 0 && <span className="chip mono">{vocab.length} từ</span>}
            </div>
            <div className="card-body stack gap-3">
              <div>
                <label className="label">Từ cần dạy</label>
                <input
                  className="field"
                  value={label}
                  onChange={(e) => setLabel(e.target.value)}
                  placeholder="ví dụ: cảm ơn"
                  list="learn-vocab"
                />
                <datalist id="learn-vocab">
                  {vocab.slice(0, 500).map((v) => <option key={v} value={v} />)}
                </datalist>
                <p className="tiny dim" style={{ marginTop: 5 }}>
                  Nhập từ mới, hoặc chọn từ đã có để bổ sung thêm mẫu.
                </p>
              </div>

              <button
                className="btn btn-primary btn-lg btn-block"
                onClick={train}
                disabled={busy || running || !supported || cam.state !== 'done'}
              >
                {busy || running ? <><span className="spinner" /> Đang huấn luyện…</> : <><Play size={14} /> Bắt đầu huấn luyện</>}
              </button>
            </div>
          </section>

          {job && <JobCard job={job} onReset={() => { setJobId(null); setJob(null); setLabel(''); cam.reset() }} />}
        </div>
      </div>
    </>
  )
}

function JobCard({ job, onReset }) {
  const pct = Math.round((job.progress ?? 0) * 100)
  const done = job.status === 'done'
  const failed = job.status === 'error'

  return (
    <section className="card rise">
      <div className="card-head">
        <h3>Tiến trình</h3>
        <span className={`chip ${done ? 'chip-accent' : failed ? 'chip-rose' : 'chip-steel'}`}>{job.status}</span>
        <div className="grow" />
        {(done || failed) && <button className="btn btn-quiet btn-sm" onClick={onReset}>Ghi mẫu khác</button>}
      </div>
      <div className="card-body stack gap-3">
        <div className="meter"><i style={{ width: `${pct}%` }} /></div>
        <div className="row gap-2 small muted">
          <span className="grow">{job.stage || job.message || '—'}</span>
          <span className="mono">{pct}%</span>
        </div>
        {job.log && (
          <pre
            className="mono tiny scroll-y"
            style={{ maxHeight: 180, background: 'var(--surface-2)', padding: 12, borderRadius: 'var(--r-md)', whiteSpace: 'pre-wrap' }}
          >
            {job.log}
          </pre>
        )}
      </div>
    </section>
  )
}
