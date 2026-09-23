import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import {
  ArrowLeft, BookOpen, Check, ChevronRight, Film, RotateCcw, Search,
  Sparkles, Target, Trash2, Upload, Video, X,
} from 'lucide-react'
import {
  errMessage, formatBytes, libraryDeleteClip, libraryOverview, libraryPractice,
  libraryResetProgress, libraryUpload, libraryWord, libraryWords, mediaUrl,
} from '../lib/api'
import { useWebcam } from '../hooks/useWebcam'
import WebcamPanel from './WebcamPanel'
import PageHead from './PageHead'

const FILTERS = [
  { id: 'has_video', label: 'Có video mẫu' },
  { id: 'todo', label: 'Chưa thuộc' },
  { id: 'learned', label: 'Đã thuộc' },
  { id: 'all', label: 'Tất cả' },
  { id: 'no_video', label: 'Thiếu video' },
]

/**
 * Kho video từ vựng: xem clip mẫu của từng từ rồi làm lại trước webcam để
 * LT-SignDiff chấm. Từ được tính là "thuộc" khi model xếp nó ở hạng 1.
 */
export default function LibraryView() {
  const [overview, setOverview] = useState(null)
  const [words, setWords] = useState([])
  const [total, setTotal] = useState(0)
  const [q, setQ] = useState('')
  const [filter, setFilter] = useState('has_video')
  const [loading, setLoading] = useState(true)
  const [active, setActive] = useState(null)      // gloss đang mở ở chế độ học
  const [note, setNote] = useState(null)

  const loadOverview = useCallback(() => {
    libraryOverview().then(setOverview).catch(() => setOverview(null))
  }, [])

  const loadWords = useCallback(() => {
    setLoading(true)
    libraryWords({ q, filter, limit: 300 })
      .then((d) => { setWords(d.words || []); setTotal(d.total || 0) })
      .catch((e) => setNote({ bad: true, text: errMessage(e) }))
      .finally(() => setLoading(false))
  }, [q, filter])

  useEffect(() => { loadOverview() }, [loadOverview])

  // Gõ tìm kiếm thì chờ một nhịp để không bắn request mỗi ký tự.
  useEffect(() => {
    const t = setTimeout(loadWords, q ? 280 : 0)
    return () => clearTimeout(t)
  }, [loadWords, q])

  const refreshAll = useCallback(() => { loadWords(); loadOverview() }, [loadWords, loadOverview])

  if (active) {
    return (
      <StudyRoom
        gloss={active}
        onBack={() => { setActive(null); refreshAll() }}
        onProgress={loadOverview}
      />
    )
  }

  return (
    <>
      <PageHead
        title="Kho từ vựng"
        sub="Mỗi từ có clip mẫu để xem lại nhiều lần. Bấm vào một từ để vào phòng luyện: xem mẫu, tự làm trước webcam và để model chấm ngay."
        action={
          overview?.attempts > 0 && (
            <button
              className="btn btn-ghost btn-sm"
              onClick={async () => {
                if (!window.confirm('Xoá toàn bộ tiến độ học của bạn?')) return
                await libraryResetProgress().catch(() => {})
                refreshAll()
              }}
            >
              <RotateCcw size={13} /> Đặt lại tiến độ
            </button>
          )
        }
      />

      {note && (
        <Banner tone={note.bad ? 'bad' : 'ok'} onClose={() => setNote(null)}>{note.text}</Banner>
      )}

      {overview && <ProgressStrip ov={overview} />}

      <div className="card card-pad stack gap-3">
        <div className="row gap-3 wrap-row">
          <div className="row grow" style={{ position: 'relative', maxWidth: 340 }}>
            <Search
              size={15}
              style={{ position: 'absolute', left: 12, color: 'var(--ink-3)', pointerEvents: 'none' }}
            />
            <input
              className="field"
              placeholder="Tìm từ trong bộ từ vựng…"
              value={q}
              onChange={(e) => setQ(e.target.value)}
              style={{ paddingLeft: 34 }}
            />
          </div>
          <div className="subtabs">
            {FILTERS.map((f) => (
              <button
                key={f.id}
                className="subtab"
                data-active={filter === f.id}
                onClick={() => setFilter(f.id)}
              >
                {f.label}
              </button>
            ))}
          </div>
          <div className="grow" />
          <span className="small dim mono">{words.length}/{total}</span>
        </div>
      </div>

      {loading ? (
        <div className="card empty"><span className="spinner spinner-lg" style={{ color: 'var(--accent)' }} /></div>
      ) : words.length === 0 ? (
        <div className="card empty">
          <h3>Không có từ nào khớp</h3>
          <p className="small">
            {filter === 'has_video'
              ? 'Kho chưa có clip mẫu nào. Thêm video cho từng từ ở phòng luyện của từ đó.'
              : 'Thử đổi bộ lọc hoặc xoá từ khoá tìm kiếm.'}
          </p>
        </div>
      ) : (
        <div className="grid-auto">
          {words.map((w) => <WordCard key={w.gloss} word={w} onOpen={() => setActive(w.gloss)} />)}
        </div>
      )}
    </>
  )
}

/* ─────────────────────────────────────────────────────────────────────── */
function ProgressStrip({ ov }) {
  const cells = [
    { label: 'Từ trong bộ', value: ov.words_total, icon: BookOpen },
    { label: 'Từ có video mẫu', value: ov.words_with_video, hint: `${ov.coverage_pct}% phủ · ${ov.clips} clip`, icon: Video },
    { label: 'Đã thuộc', value: ov.learned, hint: ov.learning ? `${ov.learning} từ đang luyện` : undefined, icon: Check },
    {
      label: 'Lượt luyện đúng',
      value: ov.attempts ? `${ov.passes}/${ov.attempts}` : '—',
      hint: ov.pass_rate_pct == null ? 'Chưa luyện lần nào' : `${ov.pass_rate_pct}% đạt`,
      icon: Target,
    },
  ]
  const pct = ov.words_total ? Math.round((ov.learned / ov.words_total) * 100) : 0

  return (
    <div className="stack gap-3">
      <div className="grid-4">
        {cells.map((c) => (
          <div key={c.label} className="card card-pad">
            <div className="row gap-2">
              <c.icon size={13} style={{ color: 'var(--ink-3)' }} />
              <span className="eyebrow">{c.label}</span>
            </div>
            <div className="stat-value" style={{ marginTop: 4 }}>{c.value}</div>
            {c.hint && <div className="tiny dim">{c.hint}</div>}
          </div>
        ))}
      </div>
      <div className="card card-pad">
        <div className="row gap-3">
          <span className="small muted grow">
            Tiến độ học: <strong>{ov.learned}</strong> / {ov.words_total} từ
          </span>
          <span className="mono small dim">{pct}%</span>
        </div>
        <div className="meter meter-green" style={{ marginTop: 8 }}><i style={{ width: `${pct}%` }} /></div>
      </div>
    </div>
  )
}

function WordCard({ word, onOpen }) {
  const noVideo = word.clips === 0
  return (
    <article className="card" style={{ transition: 'box-shadow .15s ease, border-color .15s ease' }}>
      <button
        onClick={onOpen}
        style={{ display: 'block', width: '100%', textAlign: 'left' }}
        title={`Luyện từ “${word.gloss}”`}
      >
        <div style={{ position: 'relative' }}>
          {word.thumb_url ? (
            <img className="thumb" src={mediaUrl(word.thumb_url)} alt={word.gloss} loading="lazy" />
          ) : (
            <div className="thumb" style={{ display: 'grid', placeItems: 'center' }}>
              <Film size={20} style={{ color: 'var(--ink-3)' }} />
            </div>
          )}
          {word.passed && (
            <span
              className="chip chip-green"
              style={{ position: 'absolute', top: 8, left: 8, background: 'var(--green)', color: '#fff', borderColor: 'transparent' }}
            >
              <Check size={11} /> Đã thuộc
            </span>
          )}
          {noVideo && (
            <span className="chip" style={{ position: 'absolute', top: 8, left: 8 }}>Chưa có mẫu</span>
          )}
          {word.clips > 1 && (
            <span
              className="chip"
              style={{ position: 'absolute', bottom: 8, right: 8, background: 'rgba(13,20,36,.8)', color: '#fff', borderColor: 'transparent' }}
            >
              <span className="mono">{word.clips}</span> clip
            </span>
          )}
        </div>

        <div className="stack gap-2" style={{ padding: '12px 14px' }}>
          <span style={{ fontWeight: 600, fontFamily: 'var(--font-display)' }} className="clamp-1">
            {word.gloss}
          </span>
          <div className="row gap-2">
            {word.attempts ? (
              <span className="tiny dim">
                {word.attempts} lượt · tốt nhất{' '}
                <span className="mono">{word.best_pct.toFixed(0)}%</span>
                {word.best_rank > 1 && ` (hạng ${word.best_rank})`}
              </span>
            ) : (
              <span className="tiny dim">Chưa luyện</span>
            )}
            <div className="grow" />
            <ChevronRight size={14} style={{ color: 'var(--ink-3)' }} />
          </div>
        </div>
      </button>
    </article>
  )
}

/* ─── phòng luyện một từ ────────────────────────────────────────────────── */
function StudyRoom({ gloss, onBack, onProgress }) {
  const [data, setData] = useState(null)
  const [clipIdx, setClipIdx] = useState(0)
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [note, setNote] = useState(null)
  const videoRef = useRef(null)

  const cam = useWebcam({ maxFrames: 48, intervalMs: 90, quality: 0.9 })

  const load = useCallback(() => {
    libraryWord(gloss)
      .then((d) => { setData(d); setClipIdx(0) })
      .catch((e) => setNote({ bad: true, text: errMessage(e) }))
  }, [gloss])

  useEffect(() => { load() }, [load])

  const clips = data?.clips || []
  const clip = clips[clipIdx]

  const grade = async () => {
    const frames = cam.frames.current
    if (frames.length < 20) {
      setNote({ bad: true, text: `Cần ít nhất 20 frame, mới ghi được ${frames.length}. Ghi lại lâu hơn một chút.` })
      return
    }
    setBusy(true)
    setNote(null)
    try {
      const r = await libraryPractice(frames, gloss)
      setResult(r)
      setData((d) => (d ? { ...d, progress: r.progress } : d))
      onProgress?.()
    } catch (e) {
      setNote({ bad: true, text: errMessage(e) })
    } finally {
      setBusy(false)
    }
  }

  const replay = () => {
    const v = videoRef.current
    if (!v) return
    v.currentTime = 0
    v.play().catch(() => {})
  }

  const prog = data?.progress

  return (
    <>
      <PageHead
        title={gloss}
        sub="Xem clip mẫu vài lần, chú ý hình tay và hướng chuyển động, rồi ghi lại động tác của bạn để model chấm."
        action={
          <button className="btn btn-ghost btn-sm" onClick={onBack}>
            <ArrowLeft size={13} /> Về danh sách
          </button>
        }
      />

      {note && <Banner tone={note.bad ? 'bad' : 'ok'} onClose={() => setNote(null)}>{note.text}</Banner>}

      {prog && (
        <div className="row gap-2 wrap-row">
          {prog.passes ? (
            <span className="chip chip-green"><Check size={11} /> Đã thuộc từ này</span>
          ) : (
            <span className="chip chip-amber">Đang luyện</span>
          )}
          <span className="chip">{prog.attempts} lượt thử</span>
          <span className="chip mono">tốt nhất {Number(prog.best_pct).toFixed(1)}%</span>
          {prog.best_rank && <span className="chip">hạng tốt nhất {prog.best_rank}</span>}
        </div>
      )}

      <div className="grid-2" style={{ alignItems: 'start' }}>
        <div className="stack gap-4">
          <section className="card">
            <div className="card-head">
              <Video size={15} style={{ color: 'var(--ink-3)' }} />
              <h3>Clip mẫu</h3>
              <div className="grow" />
              {clips.length > 1 && (
                <span className="chip mono">{clipIdx + 1}/{clips.length}</span>
              )}
            </div>

            {clip ? (
              <>
                <div className="stage">
                  <video
                    ref={videoRef}
                    key={clip.clip_id}
                    src={mediaUrl(clip.stream_url)}
                    controls
                    autoPlay
                    loop
                    muted
                    playsInline
                    style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                  />
                </div>
                <div className="card-pad row gap-2 wrap-row" style={{ padding: '12px 16px' }}>
                  <button className="btn btn-soft btn-sm" onClick={replay}>
                    <RotateCcw size={13} /> Xem lại
                  </button>
                  {clips.length > 1 && (
                    <div className="row gap-2">
                      {clips.map((c, i) => (
                        <button
                          key={c.clip_id}
                          className="btn btn-sm"
                          onClick={() => setClipIdx(i)}
                          style={{
                            border: `1px solid ${i === clipIdx ? 'var(--accent)' : 'var(--line)'}`,
                            background: i === clipIdx ? 'var(--accent-soft)' : 'var(--surface)',
                            color: i === clipIdx ? 'var(--accent-ink)' : 'var(--ink-2)',
                          }}
                        >
                          Mẫu {i + 1}
                        </button>
                      ))}
                    </div>
                  )}
                  <div className="grow" />
                  <span className="tiny dim mono">{formatBytes(clip.size_bytes)}</span>
                  <button
                    className="btn btn-quiet btn-sm"
                    title="Xoá clip mẫu này"
                    onClick={async () => {
                      if (!window.confirm('Xoá clip mẫu này khỏi kho từ vựng?')) return
                      await libraryDeleteClip(clip.clip_id).catch(() => {})
                      load()
                    }}
                  >
                    <Trash2 size={13} />
                  </button>
                </div>
              </>
            ) : (
              <div className="empty stack gap-2" style={{ alignItems: 'center' }}>
                <Film size={26} strokeWidth={1.4} />
                <h3>Từ này chưa có clip mẫu</h3>
                <p className="small">Tải lên một video thực hiện ký hiệu để dùng làm mẫu học.</p>
              </div>
            )}
          </section>

          <AddReferenceClip gloss={gloss} onDone={load} />
        </div>

        <div className="stack gap-4">
          <WebcamPanel
            cam={cam}
            title="Bạn thử lại"
            maxFrames={48}
            minFrames={20}
            hint="Thực hiện ký hiệu trong khoảng 4 giây, giữ hai tay và khuôn mặt trong khung hình."
            actions={
              <button className="btn btn-primary grow" onClick={grade} disabled={busy}>
                {busy ? <><span className="spinner" /> Đang chấm…</> : <><Sparkles size={14} /> Chấm điểm</>}
              </button>
            }
          />
          {result && <Verdict result={result} onRetry={() => { setResult(null); cam.reset() }} />}
        </div>
      </div>
    </>
  )
}

function Verdict({ result, onRetry }) {
  const tone =
    result.verdict === 'pass' ? 'green' : result.verdict === 'near' ? 'amber' : 'rose'
  const title =
    result.verdict === 'pass' ? 'Đúng rồi' : result.verdict === 'near' ? 'Gần đúng' : 'Chưa đúng'

  return (
    <section className="card rise">
      <div className="card-head">
        <h3>Kết quả chấm</h3>
        <span className={`chip chip-${tone}`}>{title}</span>
        <div className="grow" />
        <span className="chip mono">{result.elapsed_ms} ms</span>
      </div>
      <div className="card-body stack gap-3">
        <p className="small muted">{result.message}</p>

        <div>
          <div className="row gap-2">
            <span className="small grow">
              Độ khớp với <strong>{result.gloss}</strong>
              {result.rank && <span className="dim"> · hạng {result.rank}</span>}
            </span>
            <span className="mono small">{result.confidence_pct.toFixed(1)}%</span>
          </div>
          <div className={`meter meter-${tone === 'rose' ? 'amber' : tone}`} style={{ marginTop: 6 }}>
            <i style={{ width: `${Math.min(100, result.confidence_pct)}%` }} />
          </div>
        </div>

        <div>
          <div className="eyebrow" style={{ marginBottom: 8 }}>Model đọc ra</div>
          <div className="stack gap-2">
            {result.predictions.map((p) => {
              const isTarget = p.gloss === result.gloss
              return (
                <div key={p.rank} className="row gap-3">
                  <span className="mono dim tiny" style={{ width: 14 }}>{p.rank}</span>
                  <span
                    className="small truncate"
                    style={{ minWidth: 120, fontWeight: isTarget ? 700 : 400, color: isTarget ? 'var(--accent-ink)' : undefined }}
                  >
                    {p.gloss}
                  </span>
                  <div className="grow">
                    <div className={`meter ${isTarget ? '' : 'meter-quiet'}`} style={{ height: 4 }}>
                      <i style={{ width: `${Math.min(100, p.confidence_pct)}%` }} />
                    </div>
                  </div>
                  <span className="mono tiny dim" style={{ width: 42, textAlign: 'right' }}>
                    {p.confidence_pct.toFixed(1)}%
                  </span>
                </div>
              )
            })}
          </div>
        </div>

        <button className="btn btn-ghost btn-block" onClick={onRetry}>
          <RotateCcw size={14} /> Thử lại
        </button>
      </div>
    </section>
  )
}

function AddReferenceClip({ gloss, onDone }) {
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState(null)
  const [open, setOpen] = useState(false)

  const onDrop = useCallback(async (files) => {
    if (!files.length) return
    setBusy(true)
    setMsg(null)
    let n = 0
    for (const f of files) {
      try { await libraryUpload(f, gloss); n += 1 }
      catch (e) { setMsg({ bad: true, text: errMessage(e) }) }
    }
    setBusy(false)
    if (n) { setMsg({ bad: false, text: `Đã thêm ${n} clip mẫu.` }); onDone?.() }
  }, [gloss, onDone])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'video/*': ['.mp4', '.webm', '.mov', '.avi', '.mkv'] },
  })

  if (!open) {
    return (
      <button className="btn btn-ghost btn-block" onClick={() => setOpen(true)}>
        <Upload size={14} /> Thêm clip mẫu cho “{gloss}”
      </button>
    )
  }

  return (
    <section className="card">
      <div className="card-head">
        <h3>Thêm clip mẫu</h3>
        <div className="grow" />
        <button className="btn btn-quiet btn-sm" onClick={() => setOpen(false)}><X size={14} /></button>
      </div>
      <div className="card-body stack gap-3">
        {msg && <Banner tone={msg.bad ? 'bad' : 'ok'}>{msg.text}</Banner>}
        <div {...getRootProps()} className="dropzone" data-active={isDragActive} style={{ padding: '26px 20px' }}>
          <input {...getInputProps()} />
          <Upload size={20} strokeWidth={1.4} style={{ color: 'var(--ink-3)', marginBottom: 8 }} />
          <div style={{ fontWeight: 500, marginBottom: 3 }}>
            {busy ? 'Đang tải lên…' : 'Kéo thả video hoặc bấm để chọn'}
          </div>
          <div className="tiny dim">Mọi clip đều được gán cho từ “{gloss}”</div>
        </div>
      </div>
    </section>
  )
}

export function Banner({ tone = 'ok', children, onClose }) {
  const map = {
    ok: { background: 'var(--green-soft)', color: 'var(--green)' },
    bad: { background: 'var(--rose-soft)', color: 'var(--rose)' },
    warn: { background: 'var(--amber-soft)', color: 'var(--amber)' },
    info: { background: 'var(--steel-soft)', color: 'var(--steel)' },
  }
  return (
    <div className="row gap-2 small" style={{ ...map[tone], padding: '10px 14px', borderRadius: 'var(--r-md)' }}>
      <span className="grow">{children}</span>
      {onClose && (
        <button className="btn btn-quiet btn-sm" onClick={onClose} style={{ color: 'inherit' }}><X size={13} /></button>
      )}
    </div>
  )
}
