import { useCallback, useEffect, useRef, useState } from 'react'

/**
 * Camera + ghi frame JPEG base64.
 *
 * Tách khỏi giao diện để cả tab Dịch, Enroll và Kho dữ liệu dùng chung một
 * đường ghi — trước đây mỗi tab tự viết lại logic này.
 */
export function useWebcam({ maxFrames = 32, intervalMs = 100, quality = 0.85 } = {}) {
  const videoRef = useRef(null)
  const streamRef = useRef(null)
  const timerRef = useRef(null)
  const canvasRef = useRef(null)
  const framesRef = useRef([])

  const [state, setState] = useState('idle')      // idle | preview | recording | done
  const [count, setCount] = useState(0)
  const [countdown, setCountdown] = useState(null)
  const [error, setError] = useState(null)

  if (!canvasRef.current && typeof document !== 'undefined') {
    canvasRef.current = document.createElement('canvas')
  }

  const stop = useCallback(() => {
    clearInterval(timerRef.current)
    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
    framesRef.current = []
    setCount(0)
    setCountdown(null)
    setState('idle')
  }, [])

  useEffect(() => () => stop(), [stop])

  const start = useCallback(async () => {
    setError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 15 } },
        audio: false,
      })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play().catch(() => {})
      }
      setState('preview')
    } catch (e) {
      setError('Không truy cập được webcam: ' + (e?.message || e))
    }
  }, [])

  const stopRecording = useCallback(() => {
    clearInterval(timerRef.current)
    setState('done')
  }, [])

  const record = useCallback(() => {
    framesRef.current = []
    setCount(0)
    setState('recording')
    timerRef.current = setInterval(() => {
      const video = videoRef.current
      const canvas = canvasRef.current
      if (!video || !canvas || !video.videoWidth) return
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      canvas.getContext('2d').drawImage(video, 0, 0)
      framesRef.current.push(canvas.toDataURL('image/jpeg', quality).split(',')[1])
      setCount(framesRef.current.length)
      if (framesRef.current.length >= maxFrames) {
        clearInterval(timerRef.current)
        setState('done')
      }
    }, intervalMs)
  }, [maxFrames, intervalMs, quality])

  const startWithCountdown = useCallback((from = 3) => {
    let c = from
    setCountdown(c)
    const t = setInterval(() => {
      c -= 1
      if (c <= 0) {
        clearInterval(t)
        setCountdown(null)
        record()
      } else {
        setCountdown(c)
      }
    }, 1000)
  }, [record])

  const reset = useCallback(() => {
    framesRef.current = []
    setCount(0)
    setState(streamRef.current ? 'preview' : 'idle')
  }, [])

  return {
    videoRef,
    state,
    count,
    countdown,
    error,
    frames: framesRef,
    start,
    stop,
    record,
    startWithCountdown,
    stopRecording,
    reset,
    setError,
  }
}
