import { useState } from 'react'
import { Sparkles } from 'lucide-react'
import { useAuth } from '../contexts/AuthContext'
import { errMessage } from '../lib/api'

export default function LoginPage() {
  const { login } = useAuth()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const submit = async (e) => {
    e.preventDefault()
    if (!username || !password) return
    setLoading(true)
    setError('')
    try {
      await login(username, password)
    } catch (err) {
      setError(errMessage(err))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div
      style={{
        minHeight: '100vh',
        display: 'grid',
        placeItems: 'center',
        padding: 24,
        background:
          'radial-gradient(900px 480px at 50% -8%, #e3ecff 0%, transparent 62%), var(--paper)',
      }}
    >
      <div className="stack gap-6 rise" style={{ width: '100%', maxWidth: 380 }}>
        <div className="stack gap-3" style={{ alignItems: 'center', textAlign: 'center' }}>
          <div className="brandmark" style={{ width: 46, height: 46 }}>
            <Sparkles size={22} strokeWidth={2.2} />
          </div>
          <div>
            <h1 style={{ fontSize: '1.7rem' }}>SignTranslate</h1>
            <p className="small muted" style={{ marginTop: 2 }}>
              Dịch và học ngôn ngữ ký hiệu Việt bằng LT-SignDiff
            </p>
          </div>
        </div>

        <form className="card card-body stack gap-4" onSubmit={submit}>
          <div>
            <label className="label" htmlFor="u">Tài khoản</label>
            <input
              id="u"
              className="field"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              autoComplete="username"
              placeholder="admin"
              autoFocus
            />
          </div>
          <div>
            <label className="label" htmlFor="p">Mật khẩu</label>
            <input
              id="p"
              className="field"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete="current-password"
              placeholder="••••••••"
            />
          </div>

          {error && (
            <p className="small" style={{ background: 'var(--rose-soft)', color: 'var(--rose)', padding: '9px 12px', borderRadius: 'var(--r-md)' }}>
              {error}
            </p>
          )}

          <button className="btn btn-primary btn-lg btn-block" type="submit" disabled={loading || !username || !password}>
            {loading ? <><span className="spinner" /> Đang đăng nhập…</> : 'Đăng nhập'}
          </button>
        </form>

        <p className="tiny dim" style={{ textAlign: 'center' }}>
          LT-SignDiff v2 · 200 từ vựng · MediaPipe Holistic
        </p>
      </div>
    </div>
  )
}
