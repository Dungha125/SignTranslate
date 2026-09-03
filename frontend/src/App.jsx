import { useEffect, useState } from 'react'
import { AuthProvider, useAuth } from './contexts/AuthContext'
import { ThemeProvider } from './contexts/ThemeContext'
import LoginPage from './pages/LoginPage'
import AppShell from './components/AppShell'
import TranslateView from './components/TranslateView'
import DatasetView from './components/DatasetView'
import InsightsView from './components/InsightsView'
import HistoryView from './components/HistoryView'
import EnrollTab from './components/EnrollTab'
import LearnTab from './components/LearnTab'
import { getHealth } from './lib/api'

function Workspace() {
  const { user, logout } = useAuth()
  const [tab, setTab] = useState(() => localStorage.getItem('signtranslate_tab') || 'translate')
  const [modelId, setModelId] = useState(
    () => localStorage.getItem('signtranslate_model') || 'lt_signdiff_top200',
  )

  useEffect(() => { localStorage.setItem('signtranslate_tab', tab) }, [tab])
  useEffect(() => { localStorage.setItem('signtranslate_model', modelId) }, [modelId])

  // Nếu model đã lưu không còn được nạp, chuyển sang model mặc định của server.
  useEffect(() => {
    getHealth()
      .then((h) => {
        const loaded = h?.models?.[modelId]?.loaded
        if (!loaded && h?.default_model) setModelId(h.default_model)
      })
      .catch(() => {})
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <AppShell tab={tab} onTab={setTab} user={user} onLogout={logout}>
      {tab === 'translate' && <TranslateView modelId={modelId} onModelId={setModelId} />}
      {tab === 'dataset' && <DatasetView />}
      {tab === 'insights' && <InsightsView />}
      {tab === 'history' && <HistoryView />}
      {tab === 'enroll' && <EnrollTab modelId={modelId} />}
      {tab === 'learn' && <LearnTab modelId={modelId} />}
    </AppShell>
  )
}

function Gate() {
  const { user } = useAuth()
  return user ? <Workspace /> : <LoginPage />
}

export default function App() {
  return (
    <ThemeProvider>
      <AuthProvider>
        <Gate />
      </AuthProvider>
    </ThemeProvider>
  )
}
