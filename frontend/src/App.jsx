import { useEffect, useState } from 'react'
import { AuthProvider, useAuth } from './contexts/AuthContext'
import LoginPage from './pages/LoginPage'
import AppShell, { TABS } from './components/AppShell'
import TranslateView from './components/TranslateView'
import LibraryView from './components/LibraryView'
import DatasetView from './components/DatasetView'
import InsightsView from './components/InsightsView'
import HistoryView from './components/HistoryView'
import EnrollTab from './components/EnrollTab'
import { getHealth } from './lib/api'

const DEFAULT_MODEL = 'lt_signdiff_v2_top200'

function Workspace() {
  const { user, logout } = useAuth()
  const [tab, setTab] = useState(() => {
    const saved = localStorage.getItem('signtranslate_tab')
    return TABS.some((t) => t.id === saved) ? saved : 'translate'
  })
  // Hệ thống chỉ chạy LT-SignDiff nên không còn bộ chọn model; id lấy từ server
  // để lỡ có đổi checkpoint thì frontend vẫn trỏ đúng.
  const [modelId, setModelId] = useState(DEFAULT_MODEL)

  useEffect(() => { localStorage.setItem('signtranslate_tab', tab) }, [tab])

  useEffect(() => {
    getHealth()
      .then((h) => h?.default_model && setModelId(h.default_model))
      .catch(() => {})
  }, [])

  return (
    <AppShell tab={tab} onTab={setTab} user={user} onLogout={logout}>
      {tab === 'translate' && <TranslateView modelId={modelId} />}
      {tab === 'library' && <LibraryView />}
      {tab === 'enroll' && <EnrollTab modelId={modelId} />}
      {tab === 'dataset' && <DatasetView />}
      {tab === 'history' && <HistoryView />}
      {tab === 'insights' && <InsightsView />}
    </AppShell>
  )
}

function Gate() {
  const { user } = useAuth()
  return user ? <Workspace /> : <LoginPage />
}

export default function App() {
  return (
    <AuthProvider>
      <Gate />
    </AuthProvider>
  )
}
