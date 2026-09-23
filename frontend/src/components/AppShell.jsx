import { useEffect, useState } from 'react'
import {
  BarChart3, BookOpen, Database, History, LogOut, Sparkles, UserPlus, Wand2,
} from 'lucide-react'
import { getHealth } from '../lib/api'

export const TABS = [
  { id: 'translate', label: 'Dịch', icon: Wand2 },
  { id: 'library', label: 'Từ vựng', icon: BookOpen },
  { id: 'enroll', label: 'Cá nhân hoá', icon: UserPlus },
  { id: 'dataset', label: 'Kho dữ liệu', icon: Database },
  { id: 'history', label: 'Lịch sử', icon: History },
  { id: 'insights', label: 'Thống kê', icon: BarChart3 },
]

export default function AppShell({ tab, onTab, user, onLogout, children }) {
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

  return (
    <div className="shell">
      <header
        style={{
          borderBottom: '1px solid var(--line)',
          background: 'rgba(255,255,255,.86)',
          backdropFilter: 'blur(10px)',
          position: 'sticky',
          top: 0,
          zIndex: 50,
        }}
      >
        <div className="wrap row gap-4" style={{ height: 62 }}>
          <div className="row gap-3">
            <div className="brandmark"><Sparkles size={17} strokeWidth={2.2} /></div>
            <div>
              <div
                style={{
                  fontFamily: 'var(--font-display)',
                  fontSize: '1.05rem',
                  fontWeight: 700,
                  letterSpacing: '-.02em',
                  lineHeight: 1.15,
                  whiteSpace: 'nowrap',
                }}
              >
                SignTranslate
              </div>
              <div className="eyebrow brand-sub" style={{ fontSize: '.6rem', whiteSpace: 'nowrap' }}>
                Ngôn ngữ ký hiệu Việt
              </div>
            </div>
          </div>

          <div className="grow" />

          <StatusPill health={health} />

          {user && (
            <div className="row gap-2">
              <span className="small muted hide-narrow" style={{ fontWeight: 500 }}>{user.username}</span>
              <button className="btn btn-quiet" onClick={onLogout} title="Đăng xuất" aria-label="Đăng xuất">
                <LogOut size={15} />
              </button>
            </div>
          )}
        </div>

        <div className="wrap" style={{ paddingBottom: 12 }}>
          <nav className="tabs">
            {TABS.map((t, i) => (
              <button
                key={t.id}
                className="tab"
                data-active={tab === t.id}
                onClick={() => onTab(t.id)}
                title={`Alt+${i + 1}`}
              >
                <t.icon size={14} strokeWidth={2} />
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
          <span className="tiny dim">LT-SignDiff v2 · MediaPipe Holistic · FastAPI + Redis</span>
          <div className="grow" />
          <span className="tiny dim">
            Chuyển tab bằng <kbd>Alt</kbd>+<kbd>1</kbd>…<kbd>6</kbd>
          </span>
        </div>
      </footer>
    </div>
  )
}

function StatusPill({ health }) {
  // `health === null` là "chưa kiểm tra xong", không phải "backend chết" —
  // báo đỏ ngay lúc mới tải trang là báo sai.
  const checking = health == null
  const online = health?.status === 'ok'
  const modelOk = online && health?.model_loaded

  const tone = checking ? '' : !online ? 'chip-rose' : !modelOk ? 'chip-amber' : 'chip-accent'
  const dot = checking ? 'dot-warn' : !online ? 'dot-bad' : !modelOk ? 'dot-warn' : 'dot-ok dot-live'
  const label = checking
    ? 'Đang kiểm tra…'
    : !online
      ? 'Backend chưa phản hồi'
      : !modelOk
        ? 'Model chưa nạp'
        : `${health.num_classes} từ`

  return (
    <span className={`chip ${tone}`} title={health?.ckpt_path || ''}>
      <span className={`dot ${dot}`} style={checking ? { opacity: 0.4 } : undefined} />
      {label}
    </span>
  )
}
