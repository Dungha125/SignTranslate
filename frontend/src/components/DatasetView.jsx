import { useCallback, useEffect, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import {
  Database, Download, HardDrive, Play, RefreshCw, Trash2, Upload, X,
} from 'lucide-react'
import {
  datasetClips, datasetDelete, datasetImport, datasetPatch, datasetStats,
  datasetUpload, datasetWebcamClip, errMessage, formatBytes, getStorageHealth, mediaUrl, timeAgo,
} from '../lib/api'
import { useWebcam } from '../hooks/useWebcam'
import WebcamPanel from './WebcamPanel'
import PageHead from './PageHead'

const SPLITS = ['unassigned', 'train', 'val', 'test']

export default function DatasetView() {
  const [stats, setStats] = useState(null)
  const [storage, setStorage] = useState(null)
  const [clips, setClips] = useState([])
  const [total, setTotal] = useState(0)
  const [filters, setFilters] = useState({ q: '', split: '', source: '' })
  const [busy, setBusy] = useState(false)
  const [note, setNote] = useState(null)
  const [playing, setPlaying] = useState(null)
  const [tab, setTab] = useState('browse')     // browse | add

  const refresh = useCallback(async () => {
    try {
      const params = { limit: 60 }
      if (filters.q) params.q = filters.q
      if (filters.split) params.split = filters.split
      if (filters.source) params.source = filters.source
      const [page, s, h] = await Promise.all([
        datasetClips(params),
        datasetStats(),
        getStorageHealth().catch(() => null),
      ])
      setClips(page.clips || [])
      setTotal(page.total || 0)
      setStats(s)
      setStorage(h)
    } catch (e) {
      setNote({ kind: 'bad', text: errMessage(e) })
    }
  }, [filters])

  useEffect(() => { refresh() }, [refresh])

  const remove = async (clip) => {
    if (!window.confirm(`Xoá clip “${clip.gloss}” khỏi kho? Video sẽ bị xoá khỏi object store.`)) return
    setBusy(true)
    try {
      await datasetDelete(clip.clip_id)
      setNote({ kind: 'ok', text: 'Đã xoá clip.' })
      await refresh()
    } catch (e) {
      setNote({ kind: 'bad', text: errMessage(e) })
    } finally { setBusy(false) }
  }

  const changeSplit = async (clip, split) => {
    try {
      await datasetPatch(clip.clip_id, { split })
      setClips((cs) => cs.map((c) => (c.clip_id === clip.clip_id ? { ...c, split } : c)))
      datasetStats().then(setStats).catch(() => {})
    } catch (e) {
      setNote({ kind: 'bad', text: errMessage(e) })
    }
  }

  const importCorpus = async () => {
    if (!window.confirm('Nạp 2 clip cho mỗi từ từ corpus VSL nằm trên đĩa máy chủ (tối đa 120 clip)?')) return
    setBusy(true)
    setNote({ kind: 'info', text: 'Đang nạp corpus, có thể mất vài phút…' })
    try {
      const r = await datasetImport({ limit_per_gloss: 2, max_clips: 120 })
      // Máy chủ production không mang theo thư mục Dataset/ nên đây là lỗi hay gặp nhất.
      if (r.error) setNote({ kind: 'bad', text: `${r.error} — corpus chỉ có trên máy phát triển.` })
      else setNote({ kind: 'ok', text: `Đã nạp ${r.imported} clip (${r.skipped_duplicate} trùng, ${r.failed} lỗi).` })
      await refresh()
    } catch (e) {
      setNote({ kind: 'bad', text: errMessage(e) })
    } finally { setBusy(false) }
  }

  return (
    <>
      <PageHead
        title="Kho dữ liệu"
        sub="Dữ liệu dùng để huấn luyện: video trong object store, metadata trong Redis. Thêm clip cho những từ còn thiếu mẫu. Clip mẫu của tab Từ vựng không tính vào đây."
        action={
          <div className="row gap-2">
            <button className="btn btn-ghost btn-sm" onClick={refresh} disabled={busy}>
              <RefreshCw size={13} /> Làm mới
            </button>
            <a className="btn btn-ghost btn-sm" href={mediaUrl('/api/dataset/export.csv')} download>
              <Download size={13} /> Xuất CSV
            </a>
            <button className="btn btn-ghost btn-sm" onClick={importCorpus} disabled={busy}>
              <Database size={13} /> Nạp từ corpus
            </button>
          </div>
        }
      />

      {note && <Note note={note} onClose={() => setNote(null)} />}

      <StatsStrip stats={stats} storage={storage} />

      <div className="subtabs">
        <button className="subtab" data-active={tab === 'browse'} onClick={() => setTab('browse')}>
          Duyệt clip
        </button>
        <button className="subtab" data-active={tab === 'add'} onClick={() => setTab('add')}>
          Thêm clip
        </button>
      </div>

      {tab === 'add' ? (
        <AddClips onDone={(t) => { setNote({ kind: 'ok', text: t }); refresh() }} onError={(t) => setNote({ kind: 'bad', text: t })} />
      ) : (
        <>
          <div className="card card-pad row gap-3 wrap-row">
            <input
              className="field"
              placeholder="Tìm theo từ hoặc tên file…"
              value={filters.q}
              onChange={(e) => setFilters((f) => ({ ...f, q: e.target.value }))}
              style={{ maxWidth: 280 }}
            />
            <select className="field" value={filters.split} onChange={(e) => setFilters((f) => ({ ...f, split: e.target.value }))} style={{ maxWidth: 150 }}>
              <option value="">Mọi split</option>
              {SPLITS.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
            <select className="field" value={filters.source} onChange={(e) => setFilters((f) => ({ ...f, source: e.target.value }))} style={{ maxWidth: 150 }}>
              <option value="">Mọi nguồn</option>
              {['upload', 'webcam', 'corpus', 'enroll'].map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
            <div className="grow" />
            <span className="small dim mono">{clips.length}/{total}</span>
          </div>

          {clips.length === 0 ? (
            <div className="card empty">
              <h3>Kho còn trống</h3>
              <p className="small">Thêm clip ở tab <strong>Thêm clip</strong>, hoặc bấm <em>Nạp từ corpus</em> nếu máy chủ có sẵn thư mục Dataset.</p>
            </div>
          ) : (
            <div className="grid-auto">
              {clips.map((c) => (
                <ClipCard
                  key={c.clip_id}
                  clip={c}
                  onPlay={() => setPlaying(c)}
                  onDelete={() => remove(c)}
                  onSplit={(s) => changeSplit(c, s)}
                />
              ))}
            </div>
          )}
        </>
      )}

      {playing && <PlayerModal clip={playing} onClose={() => setPlaying(null)} />}
    </>
  )
}

/* ─────────────────────────────────────────────────────────────────────── */
function Note({ note, onClose }) {
  const style =
    note.kind === 'bad'
      ? { background: 'var(--rose-soft)', color: 'var(--rose)' }
      : note.kind === 'ok'
        ? { background: 'var(--accent-soft)', color: 'var(--accent-ink)' }
        : { background: 'var(--steel-soft)', color: 'var(--steel)' }
  return (
    <div className="row gap-2 small" style={{ ...style, padding: '10px 14px', borderRadius: 'var(--r-md)' }}>
      <span className="grow">{note.text}</span>
      <button className="btn btn-quiet btn-sm" onClick={onClose} style={{ color: 'inherit' }}><X size={13} /></button>
    </div>
  )
}

function StatsStrip({ stats, storage }) {
  if (!stats) return null
  const usage = storage?.object_store?.usage
  const bytes = usage ? Object.values(usage).reduce((a, b) => a + (b.bytes || 0), 0) : stats.total_bytes

  const cells = [
    { label: 'Clip', value: stats.clips },
    { label: 'Từ', value: stats.glosses },
    { label: 'Trung bình clip/từ', value: stats.clips_per_gloss_avg },
    { label: 'Dung lượng', value: formatBytes(bytes), mono: false },
  ]

  return (
    <div className="grid-4">
      {cells.map((c) => (
        <div key={c.label} className="card card-pad">
          <div className="eyebrow">{c.label}</div>
          <div className="stat-value" style={{ marginTop: 4 }}>{c.value}</div>
        </div>
      ))}

      <div className="card card-pad" style={{ gridColumn: '1 / -1' }}>
        <div className="row gap-4 wrap-row">
          <div className="row gap-2">
            <HardDrive size={14} style={{ color: 'var(--ink-3)' }} />
            <span className="small muted">
              Redis <strong>{storage?.redis?.backend || '?'}</strong> · object store{' '}
              <strong>{storage?.object_store?.backend || '?'}</strong>
            </span>
          </div>
          <div className="grow" />
          {Object.entries(stats.by_split || {}).map(([k, v]) => (
            <span key={k} className="chip">{k} <span className="mono">{v}</span></span>
          ))}
        </div>
        {stats.needs_more?.length > 0 && (
          <p className="tiny dim" style={{ marginTop: 10 }}>
            <strong>Ưu tiên quay thêm</strong> — {stats.needs_more_total ?? stats.needs_more.length} từ có
            dưới {stats.target_clips_per_gloss ?? 4} clip. Đo trên Top-200: từ đủ{' '}
            {stats.target_clips_per_gloss ?? 4} clip đạt 97 % còn từ chỉ 2–3 clip chỉ 69 %.
            <br />
            {stats.needs_more.slice(0, 12).join(' · ')}
            {stats.needs_more.length > 12 && ' …'}
          </p>
        )}
      </div>
    </div>
  )
}

function ClipCard({ clip, onPlay, onDelete, onSplit }) {
  return (
    <article className="card">
      <button onClick={onPlay} style={{ display: 'block', width: '100%', position: 'relative' }} title="Phát clip">
        {clip.thumb_url ? (
          <img className="thumb" src={mediaUrl(clip.thumb_url)} alt={clip.gloss} loading="lazy" />
        ) : (
          <div className="thumb" style={{ display: 'grid', placeItems: 'center' }}>
            <Play size={22} style={{ color: 'var(--ink-3)' }} />
          </div>
        )}
        <span
          className="chip"
          style={{ position: 'absolute', bottom: 8, right: 8, background: 'rgba(13,20,36,.8)', color: '#fff', borderColor: 'transparent' }}
        >
          {formatBytes(clip.size_bytes)}
        </span>
      </button>

      <div className="card-pad stack gap-2" style={{ padding: '12px 14px' }}>
        <div className="row gap-2">
          <span style={{ fontWeight: 600 }} className="truncate grow">{clip.gloss}</span>
          <button className="btn btn-quiet btn-sm" onClick={onDelete} title="Xoá clip"><Trash2 size={13} /></button>
        </div>
        <div className="row gap-2 wrap-row">
          <select
            className="field"
            value={clip.split}
            onChange={(e) => onSplit(e.target.value)}
            style={{ padding: '3px 7px', fontSize: '.72rem', width: 'auto' }}
          >
            {SPLITS.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
          <span className="chip tiny">{clip.source}</span>
          <div className="grow" />
          <span className="tiny dim">{timeAgo(clip.created_at)}</span>
        </div>
      </div>
    </article>
  )
}

function PlayerModal({ clip, onClose }) {
  useEffect(() => {
    const onKey = (e) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, zIndex: 200,
        background: 'rgba(13,20,36,.55)',
        backdropFilter: 'blur(3px)',
        display: 'grid', placeItems: 'center', padding: 24,
      }}
    >
      <div className="card rise" style={{ maxWidth: 640, width: '100%', boxShadow: 'var(--shadow-lg)' }} onClick={(e) => e.stopPropagation()}>
        <div className="card-head">
          <h3 className="truncate">{clip.gloss}</h3>
          <div className="grow" />
          <button className="btn btn-quiet btn-sm" onClick={onClose}><X size={14} /></button>
        </div>
        <div className="stage" style={{ borderRadius: 0 }}>
          <video
            src={mediaUrl(clip.stream_url)}
            controls
            autoPlay
            style={{ width: '100%', height: '100%', objectFit: 'contain' }}
          />
        </div>
        <div className="card-pad row gap-2 wrap-row small muted" style={{ padding: '12px 18px' }}>
          <span className="chip">{clip.split}</span>
          <span className="chip">{clip.source}</span>
          <span className="chip mono">{formatBytes(clip.size_bytes)}</span>
          <div className="grow" />
          <span className="tiny dim truncate">{clip.object_key}</span>
        </div>
      </div>
    </div>
  )
}

function AddClips({ onDone, onError }) {
  const [gloss, setGloss] = useState('')
  const [split, setSplit] = useState('train')
  const [busy, setBusy] = useState(false)
  const cam = useWebcam({ maxFrames: 40, intervalMs: 90, quality: 0.85 })

  const onDrop = useCallback(async (files) => {
    if (!gloss.trim()) { onError('Nhập từ (gloss) trước khi tải clip lên.'); return }
    setBusy(true)
    let n = 0
    for (const f of files) {
      try { await datasetUpload(f, gloss.trim(), split); n += 1 }
      catch (e) { onError(errMessage(e)) }
    }
    setBusy(false)
    if (n) onDone(`Đã thêm ${n} clip cho “${gloss.trim()}”.`)
  }, [gloss, split, onDone, onError])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'video/*': ['.mp4', '.webm', '.mov', '.avi', '.mkv'] },
  })

  const saveWebcam = async () => {
    if (!gloss.trim()) { onError('Nhập từ (gloss) trước khi lưu.'); return }
    const frames = cam.frames.current
    if (frames.length < 8) { onError('Đoạn ghi quá ngắn.'); return }
    setBusy(true)
    try {
      await datasetWebcamClip(frames, gloss.trim(), split, 11)
      onDone(`Đã lưu clip webcam cho “${gloss.trim()}”.`)
      cam.reset()
    } catch (e) { onError(errMessage(e)) } finally { setBusy(false) }
  }

  return (
    <div className="grid-2" style={{ alignItems: 'start' }}>
      <section className="card">
        <div className="card-head"><h3>Thông tin clip</h3></div>
        <div className="card-body stack gap-3">
          <div>
            <label className="label">Từ (gloss)</label>
            <input className="field" value={gloss} onChange={(e) => setGloss(e.target.value)} placeholder="ví dụ: chào bạn" />
          </div>
          <div>
            <label className="label">Split</label>
            <select className="field" value={split} onChange={(e) => setSplit(e.target.value)}>
              {SPLITS.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>

          <div {...getRootProps()} className="dropzone" data-active={isDragActive} style={{ padding: '30px 20px' }}>
            <input {...getInputProps()} />
            <Upload size={22} strokeWidth={1.4} style={{ color: 'var(--ink-3)', marginBottom: 8 }} />
            <div style={{ fontWeight: 500, marginBottom: 3 }}>
              {busy ? 'Đang tải lên…' : 'Kéo thả một hoặc nhiều video'}
            </div>
            <div className="tiny dim">Tất cả sẽ được gán cùng một từ và split ở trên</div>
          </div>
        </div>
      </section>

      <WebcamPanel
        cam={cam}
        title="Ghi mẫu mới bằng webcam"
        maxFrames={40}
        minFrames={8}
        hint="Ghi 2–3 mẫu cho mỗi từ trong cùng điều kiện ánh sáng bạn sẽ dùng khi dịch."
        actions={
          <button className="btn btn-primary grow" onClick={saveWebcam} disabled={busy}>
            {busy ? <span className="spinner" /> : <Upload size={14} />} Lưu vào kho
          </button>
        }
      />
    </div>
  )
}
