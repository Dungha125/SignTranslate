import { useState } from 'react'
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
    <div style={{ minHeight: '100vh', display: 'grid', gridTemplateColumns: '1fr', placeItems: 'center', padding: 24 }}>
      <div className="stack gap-6 rise" style={{ width: '100%', maxWidth: 380 }}>
        <div className="stack gap-3" style={{ alignItems: 'center', textAlign: 'center' }}>
          <div className="brandmark" style={{ width: 44, height: 44, fontSize: '1.15rem' }}>S</div>
          <div>
            <h1 style={{ fontSize: '1.6rem' }}>SignTranslate</h1>
            <p className="small muted" style={{ marginTop: 2 }}>
              Dịch ngôn ngữ ký hiệu Việt bằng LT-SignDiff
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
          Tài khoản demo: <span className="mono">admin / admin123</span>
        </p>
      </div>
    </div>
  )
}
