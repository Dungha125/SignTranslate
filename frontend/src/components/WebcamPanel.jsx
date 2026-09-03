import { Camera, Circle, RotateCcw, Square, X } from 'lucide-react'

/**
 * Khung xem webcam dùng chung (điều khiển đến từ hook `useWebcam`).
 * Nút hành động chính do component cha truyền vào qua `actions`.
 */
export default function WebcamPanel({
  cam,
  maxFrames,
  minFrames = 0,
  hint,
  actions = null,
  title = 'Webcam',
}) {
  const { videoRef, state, count, countdown, error } = cam
  const recording = state === 'recording'
  const done = state === 'done'
  const enough = count >= minFrames

  return (
    <section className="card">
      <div className="card-head">
        <h3>{title}</h3>
        {recording && (
          <span className="chip chip-rose" style={{ marginLeft: 'auto' }}>
            <span className="dot dot-bad dot-live" />
            <span className="mono">{count}/{maxFrames}</span>
          </span>
        )}
        {state !== 'idle' && !recording && (
          <button className="btn btn-quiet btn-sm" style={{ marginLeft: 'auto' }} onClick={cam.stop}>
            <X size={14} /> Tắt camera
          </button>
        )}
      </div>

      <div className="card-body stack gap-3">
        {hint && state === 'preview' && (
          <p className="small muted" style={{ background: 'var(--surface-2)', padding: '9px 12px', borderRadius: 'var(--r-md)' }}>
            {hint}
          </p>
        )}
        {error && (
          <p className="small" style={{ color: 'var(--rose)', background: 'var(--rose-soft)', padding: '9px 12px', borderRadius: 'var(--r-md)' }}>
            {error}
          </p>
        )}

        <div
          style={{
            position: 'relative',
            aspectRatio: '4 / 3',
            borderRadius: 'var(--r-md)',
            overflow: 'hidden',
            background: '#101211',
            display: 'grid',
            placeItems: 'center',
          }}
        >
          {state === 'idle' && (
            <div className="stack gap-2" style={{ alignItems: 'center', color: '#8e918a' }}>
              <Camera size={30} strokeWidth={1.4} />
              <span className="small">Camera chưa bật</span>
            </div>
          )}
          <video
            ref={videoRef}
            muted
            playsInline
            style={{
              width: '100%',
              height: '100%',
              objectFit: 'cover',
              transform: 'scaleX(-1)',
              display: state === 'idle' ? 'none' : 'block',
            }}
          />

          {countdown !== null && (
            <div
              style={{
                position: 'absolute',
                inset: 0,
                display: 'grid',
                placeItems: 'center',
                background: 'rgba(16,18,17,.55)',
              }}
            >
              <span style={{ fontFamily: 'var(--font-display)', fontSize: 84, color: '#fff', fontWeight: 500 }}>
                {countdown}
              </span>
            </div>
          )}

          {done && (
            <span
              className="chip"
              style={{
                position: 'absolute',
                top: 10,
                right: 10,
                background: enough ? 'rgba(31,95,78,.9)' : 'rgba(154,91,18,.9)',
                color: '#fff',
                borderColor: 'transparent',
              }}
            >
              <span className="mono">{count}</span> frame
            </span>
          )}

          {recording && (
            <div
              style={{
                position: 'absolute',
                left: 0,
                right: 0,
                bottom: 0,
                height: 3,
                background: 'rgba(255,255,255,.2)',
              }}
            >
              <div
                style={{
                  height: '100%',
                  width: `${(count / maxFrames) * 100}%`,
                  background: '#e0655a',
                  transition: 'width .1s linear',
                }}
              />
            </div>
          )}
        </div>

        <div className="row gap-2">
          {state === 'idle' && (
            <button className="btn btn-primary btn-block" onClick={cam.start}>
              <Camera size={15} /> Bật camera
            </button>
          )}
          {state === 'preview' && (
            <button className="btn btn-primary btn-block" onClick={() => cam.startWithCountdown(3)}>
              <Circle size={13} fill="currentColor" /> Ghi {maxFrames} frame
            </button>
          )}
          {recording && (
            <button className="btn btn-ghost btn-block" onClick={cam.stopRecording}>
              <Square size={13} fill="currentColor" /> Dừng
            </button>
          )}
          {done && (
            <>
              <button className="btn btn-ghost" onClick={cam.reset}>
                <RotateCcw size={14} /> Ghi lại
              </button>
              {actions}
            </>
          )}
        </div>
      </div>
    </section>
  )
}
