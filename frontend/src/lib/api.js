import axios from 'axios'

/** Gom mọi endpoint về một chỗ để component không phải nhớ đường dẫn. */

/**
 * Khi frontend và backend nằm khác origin (Vercel ↔ VPS), mọi đường dẫn `/api/...`
 * phải được ghép thêm tiền tố. Để trống thì dùng đường dẫn tương đối như bản dev,
 * lúc đó Vite proxy lo phần chuyển tiếp.
 */
export const API_BASE = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '')

/** Dùng cho URL đặt thẳng vào `src`/`href` — axios không can thiệp được. */
export const mediaUrl = (path) => (path ? `${API_BASE}${path}` : path)

const unwrap = (p) => p.then((r) => r.data)

export const errMessage = (e) =>
  e?.response?.data?.detail || e?.message || 'Lỗi không xác định'

// ── hệ thống ────────────────────────────────────────────────────────────
export const getHealth = () => unwrap(axios.get('/api/health'))
export const getModels = () => unwrap(axios.get('/api/models'))
export const getStorageHealth = () => unwrap(axios.get('/api/storage/health'))
export const reconnectStorage = () => unwrap(axios.post('/api/storage/reconnect'))

// ── từ vựng & gallery ───────────────────────────────────────────────────
export const getVocab = (modelId) => unwrap(axios.get(`/api/vocab/${modelId}`))
export const getGallery = (modelId) => unwrap(axios.get(`/api/gallery/${modelId}`))
export const rebuildGallery = (modelId) =>
  unwrap(axios.post(`/api/gallery/rebuild/${modelId}`, null, { timeout: 900_000 }))
export const enrollFrames = (payload) =>
  unwrap(axios.post('/api/enroll/frames', payload, { timeout: 180_000 }))

// ── dịch ────────────────────────────────────────────────────────────────
export function translateVideo(file, modelId, { saveToDataset = false, gloss = '' } = {}) {
  const fd = new FormData()
  fd.append('file', file)
  fd.append('model_id', modelId)
  fd.append('save_to_dataset', String(saveToDataset))
  if (gloss) fd.append('gloss', gloss)
  return unwrap(
    axios.post('/api/translate/video', fd, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 180_000,
    }),
  )
}

export const translateFrames = (framesB64, modelId) =>
  unwrap(axios.post('/api/translate/frames', { frames_b64: framesB64, model_id: modelId }, { timeout: 180_000 }))

export const compareModels = (framesB64, modelIds) =>
  unwrap(axios.post('/api/translate/compare', { frames_b64: framesB64, model_ids: modelIds }, { timeout: 300_000 }))

// ── kho dữ liệu ─────────────────────────────────────────────────────────
export const datasetStats = () => unwrap(axios.get('/api/dataset/stats'))
export const datasetGlosses = () => unwrap(axios.get('/api/dataset/glosses'))
export const datasetClips = (params = {}) => unwrap(axios.get('/api/dataset/clips', { params }))
export const datasetPatch = (clipId, body) => unwrap(axios.patch(`/api/dataset/clips/${clipId}`, body))
export const datasetDelete = (clipId) => unwrap(axios.delete(`/api/dataset/clips/${clipId}`))
export const datasetImport = (body) =>
  unwrap(axios.post('/api/dataset/import-corpus', body, { timeout: 900_000 }))

export function datasetUpload(file, gloss, split = 'unassigned') {
  const fd = new FormData()
  fd.append('file', file)
  fd.append('gloss', gloss)
  fd.append('split', split)
  return unwrap(
    axios.post('/api/dataset/clips', fd, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 300_000,
    }),
  )
}

export const datasetWebcamClip = (framesB64, gloss, split = 'unassigned', fps = 12) =>
  unwrap(axios.post('/api/dataset/clips/webcam', { frames_b64: framesB64, gloss, split, fps }, { timeout: 180_000 }))

// ── lịch sử & thống kê ──────────────────────────────────────────────────
export const getHistory = (params = {}) => unwrap(axios.get('/api/history', { params }))
export const clearHistory = () => unwrap(axios.delete('/api/history'))
export const sendFeedback = (entryId, correct, trueGloss) =>
  unwrap(axios.post(`/api/history/${entryId}/feedback`, { correct, true_gloss: trueGloss || null }))
export const getOverview = () => unwrap(axios.get('/api/insights/overview'))
export const getHardCases = () => unwrap(axios.get('/api/insights/hard-cases'))

// ── câu ─────────────────────────────────────────────────────────────────
export const getSentence = () => unwrap(axios.get('/api/sentence'))
export const appendSentence = (gloss, score = 0) =>
  unwrap(axios.post('/api/sentence/append', { gloss, score }))
export const popSentence = () => unwrap(axios.post('/api/sentence/pop'))
export const resetSentence = () => unwrap(axios.post('/api/sentence/reset'))

// ── tiện ích hiển thị ───────────────────────────────────────────────────
export function formatBytes(n) {
  if (!n) return '0 B'
  const u = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.min(u.length - 1, Math.floor(Math.log(n) / Math.log(1024)))
  return `${(n / 1024 ** i).toFixed(i === 0 ? 0 : 1)} ${u[i]}`
}

export function timeAgo(ts) {
  if (!ts) return ''
  const s = Math.max(0, Date.now() / 1000 - ts)
  if (s < 60) return 'vừa xong'
  if (s < 3600) return `${Math.floor(s / 60)} phút trước`
  if (s < 86400) return `${Math.floor(s / 3600)} giờ trước`
  return `${Math.floor(s / 86400)} ngày trước`
}
