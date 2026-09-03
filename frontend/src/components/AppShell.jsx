import { useEffect, useState } from 'react'
import { Moon, Sun, Monitor, LogOut, Wifi, WifiOff } from 'lucide-react'
import { useTheme } from '../contexts/ThemeContext'
import { getHealth } from '../lib/api'

const TABS = [
  { id: 'translate', label: 'Dịch' },
  { id: 'dataset', label: 'Kho dữ liệu' },
  { id: 'insights', label: 'Thống kê' },
  { id: 'history', label: 'Lịch sử' },
  { id: 'enroll', label: 'Enroll' },
  { id: 'learn', label: 'Học từ mới' },
]

export default function AppShell({ tab, onTab, user, onLogout, children }) {
  const { mode, cycle } = useTheme()
  const [health, setHealth] = useState(null)

  useEffect(() => {
    let alive = true
    const poll = () =>
      getHealth()
        .then((d) => alive && setHealth(d))
        .catch(() => alive && setHealth({ status: 'error' }))
    poll()
    const t = setInterval(poll, 30_000)
    return () => {
      alive = false
      clearInterval(t)
    }
  }, [])

  // Alt+1..6 chuyển tab nhanh
  useEffect(() => {
    const onKey = (e) => {
      if (!e.altKey || e.ctrlKey || e.metaKey) return
      const i = parseInt(e.key, 10) - 1
      if (i >= 0 && i < TABS.length) {
        e.preventDefault()
        onTab(TABS[i].id)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onTab])

  const online = health?.status === 'ok'
  const modelOk = online && health?.model_loaded
  const ThemeIcon = mode === 'dark' ? Moon : mode === 'light' ? Sun : Monitor

  return (
    <div className="shell">
      <header
        style={{
          borderBottom: '1px solid var(--line)',
          background: 'var(--surface)',
          position: 'sticky',
          top: 0,
          zIndex: 50,
        }}
      >
        <div className="wrap row gap-4" style={{ height: 60 }}>
          <div className="row gap-3">
            <div className="brandmark">S</div>
            <div>
              <div style={{ fontFamily: 'var(--font-display)', fontSize: '1.08rem', fontWeight: 600, lineHeight: 1.15 }}>
                SignTranslate
              </div>
              <div className="eyebrow" style={{ fontSize: '.62rem' }}>
                Ngôn ngữ ký hiệu Việt
              </div>
            </div>
          </div>

          <div className="grow" />

          <StatusPill health={health} online={online} modelOk={modelOk} />

          <button className="btn btn-quiet" onClick={cycle} title={`Giao diện: ${mode}`} aria-label="Đổi giao diện">
            <ThemeIcon size={16} />
          </button>

          {user && (
            <div className="row gap-2">
              <span className="small muted" style={{ fontWeight: 500 }}>{user.username}</span>
              <button className="btn btn-quiet" onClick={onLogout} title="Đăng xuất" aria-label="Đăng xuất">
                <LogOut size={15} />
              </button>
            </div>
          )}
        </div>

        <div className="wrap">
          <nav className="tabs" style={{ borderTop: '1px solid var(--line)' }}>
            {TABS.map((t, i) => (
              <button
                key={t.id}
                className="tab"
                data-active={tab === t.id}
                onClick={() => onTab(t.id)}
                title={`Alt+${i + 1}`}
              >
                {t.label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main className="wrap stack gap-4" style={{ flex: 1, padding: '24px 28px 72px' }}>
        {children}
      </main>

      <footer style={{ borderTop: '1px solid var(--line)', padding: '16px 0', background: 'var(--surface)' }}>
        <div className="wrap row gap-4 wrap-row">
          <span className="tiny dim">
            LT-SignDiff · MediaPipe Holistic · Redis + MinIO
          </span>
          <div className="grow" />
          <span className="tiny dim">
            Chuyển tab bằng <kbd>Alt</kbd>+<kbd>1</kbd>…<kbd>6</kbd>
          </span>
        </div>
      </footer>
    </div>
  )
}

function StatusPill({ health, online, modelOk }) {
  const store = health?.storage
  // `health === null` là "chưa kiểm tra xong", không phải "backend chết" —
  // báo đỏ ngay lúc mới tải trang là báo sai.
  const checking = health == null
  const tone = checking ? '' : !online ? 'chip-rose' : !modelOk ? 'chip-amber' : 'chip-accent'
  const dot = checking ? '' : !online ? 'dot-bad' : !modelOk ? 'dot-warn' : 'dot-ok dot-live'
  const label = checking
    ? 'Đang kiểm tra…'
    : !online
      ? 'Backend chưa phản hồi'
      : !modelOk
        ? 'Model chưa nạp'
        : `${health.display_name} · ${health.num_classes} từ`

  return (
    <div className="row gap-2">
      <span className={`chip ${tone}`} title={health?.ckpt_path || ''}>
        <span className={`dot ${dot || 'dot-warn'}`} style={checking ? { opacity: 0.4 } : undefined} />
        {label}
      </span>
      {store && (
        <span
          className="chip"
          title={`Redis: ${store.redis} · Object store: ${store.object_store}`}
        >
          {store.redis_ok && store.object_store_ok ? <Wifi size={12} /> : <WifiOff size={12} />}
          {store.redis_ok ? 'redis' : 'ram'} · {store.object_store_ok ? 'minio' : 'local'}
        </span>
      )}
    </div>
  )
}
