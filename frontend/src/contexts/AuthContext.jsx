import { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react'
import axios from 'axios'

const AuthContext = createContext(null)
const STORAGE_KEY = 'signtranslate_user'
const LEGACY_KEY = 'currivsl_user'

function readStored() {
  const raw = localStorage.getItem(STORAGE_KEY) || localStorage.getItem(LEGACY_KEY)
  if (!raw) return null
  try { return JSON.parse(raw) } catch { return null }
}

/**
 * Header Authorization được gán **ngay lúc đọc/ghi user**, không đợi useEffect.
 * Nếu để trong effect thì component con có thể gọi API trước khi header kịp đặt,
 * nhận 401 và bị interceptor đá ra màn hình đăng nhập.
 */
function applyToken(token) {
  if (token) axios.defaults.headers.common.Authorization = `Bearer ${token}`
  else delete axios.defaults.headers.common.Authorization
}

const initial = readStored()
applyToken(initial?.token)

export function AuthProvider({ children }) {
  const [user, setUser] = useState(initial)
  const interceptorRef = useRef(null)

  const login = useCallback(async (username, password) => {
    const { data } = await axios.post('/api/auth/login', { username, password })
    const info = { username: data.username, token: data.token }
    applyToken(info.token)
    localStorage.setItem(STORAGE_KEY, JSON.stringify(info))
    setUser(info)
    return info
  }, [])

  const clear = useCallback(() => {
    applyToken(null)
    localStorage.removeItem(STORAGE_KEY)
    localStorage.removeItem(LEGACY_KEY)
    setUser(null)
  }, [])

  const logout = useCallback(async () => {
    try { await axios.post('/api/auth/logout', {}) } catch { /* phiên có thể đã hết hạn */ }
    clear()
  }, [clear])

  // 401 ở endpoint khác login ⇒ token hết hạn hoặc backend khởi động lại.
  useEffect(() => {
    if (interceptorRef.current != null) return undefined
    interceptorRef.current = axios.interceptors.response.use(
      (r) => r,
      (err) => {
        const url = err?.config?.url || ''
        if (err?.response?.status === 401 && !url.includes('/api/auth/login')) clear()
        return Promise.reject(err)
      },
    )
    return () => {
      if (interceptorRef.current != null) {
        axios.interceptors.response.eject(interceptorRef.current)
        interceptorRef.current = null
      }
    }
  }, [clear])

  return (
    <AuthContext.Provider value={{ user, login, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export const useAuth = () => useContext(AuthContext)
