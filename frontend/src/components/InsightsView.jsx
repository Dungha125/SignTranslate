import { useCallback, useEffect, useState } from 'react'
import { Activity, RefreshCw, Server, Timer } from 'lucide-react'
import { formatBytes, getHardCases, getOverview, getStorageHealth, reconnectStorage } from '../lib/api'
import PageHead from './PageHead'

export default function InsightsView() {
  const [ov, setOv] = useState(null)
  const [hard, setHard] = useState(null)
  const [storage, setStorage] = useState(null)
  const [busy, setBusy] = useState(false)

  const refresh = useCallback(() => {
    getOverview().then(setOv).catch(() => {})
    getHardCases().then(setHard).catch(() => {})
    getStorageHealth().then(setStorage).catch(() => {})
  }, [])

  useEffect(() => {
    refresh()
    const t = setInterval(refresh, 20_000)
    return () => clearInterval(t)
  }, [refresh])

  const reconnect = async () => {
    setBusy(true)
    try { setStorage((await reconnectStorage()).health) } catch { /* giữ nguyên trạng thái cũ */ }
    setBusy(false)
    refresh()
  }

  if (!ov) {
    return (
      <>
        <PageHead title="Thống kê" sub="Số liệu vận hành lấy từ Redis." />
        <div className="card empty"><span className="spinner spinner-lg" /></div>
      </>
    )
  }

  const lat = ov.latency
  const acc = ov.feedback.accuracy_pct

  return (
    <>
      <PageHead
        title="Thống kê"
        sub="Mọi lượt dịch được ghi vào Redis: độ trễ, phân bố độ tin cậy, và độ chính xác theo phản hồi người dùng."
        action={
          <button className="btn btn-ghost btn-sm" onClick={refresh}>
            <RefreshCw size={13} /> Làm mới
          </button>
        }
      />

      <div className="grid-4">
        <Stat label="Lượt dịch" value={ov.total_translations} />
        <Stat label="Độ trễ trung vị" value={`${lat.p50_ms}`} unit="ms" />
        <Stat label="Độ trễ p90" value={`${lat.p90_ms}`} unit="ms" />
        <Stat
          label="Chính xác (phản hồi)"
          value={acc == null ? '—' : `${acc}`}
          unit={acc == null ? '' : '%'}
          hint={acc == null ? 'Chưa có phản hồi' : `${ov.feedback.correct}/${ov.feedback.n} lượt`}
        />
      </div>

      <div className="grid-2">
        <section className="card">
          <div className="card-head">
            <Timer size={15} style={{ color: 'var(--ink-3)' }} />
            <h3>Độ trễ gần đây</h3>
            <div className="grow" />
            <span className="mono tiny dim">min {lat.min_ms} · max {lat.max_ms} ms</span>
          </div>
          <div className="card-body">
            {lat.recent.length ? (
              <>
                <Bars values={[...lat.recent].reverse()} />
                <p className="tiny dim" style={{ marginTop: 8 }}>
                  {lat.recent.length} lượt gần nhất · trung bình <span className="mono">{lat.avg_ms} ms</span>
                </p>
              </>
            ) : (
              <p className="small dim">Chưa có dữ liệu.</p>
            )}
          </div>
        </section>

        <section className="card">
          <div className="card-head">
            <Activity size={15} style={{ color: 'var(--ink-3)' }} />
            <h3>Phân bố độ tin cậy</h3>
          </div>
          <div className="card-body">
            <Bars values={ov.confidence_hist} />
            <div className="row" style={{ marginTop: 6, justifyContent: 'space-between' }}>
              <span className="tiny dim mono">0%</span>
              <span className="tiny dim mono">50%</span>
              <span className="tiny dim mono">100%</span>
            </div>
            <p className="tiny dim" style={{ marginTop: 8 }}>
              Nhiều lượt dồn về bên trái nghĩa là model đang thiếu tự tin — thêm mẫu enroll cho các từ hay gặp.
            </p>
          </div>
        </section>
      </div>

      <div className="grid-2">
        <section className="card">
          <div className="card-head"><h3>Từ được dịch nhiều nhất</h3></div>
          <div className="card-body">
            {ov.top_glosses.length ? (
              <div className="stack gap-2">
                {ov.top_glosses.map(([g, n]) => {
                  const max = ov.top_glosses[0][1] || 1
                  return (
                    <div key={g} className="row gap-3">
                      <span className="small" style={{ minWidth: 110 }}>{g}</span>
                      <div className="grow"><div className="meter"><i style={{ width: `${(n / max) * 100}%` }} /></div></div>
                      <span className="mono tiny dim" style={{ width: 24, textAlign: 'right' }}>{n}</span>
                    </div>
                  )
                })}
              </div>
            ) : <p className="small dim">Chưa có lượt dịch nào.</p>}
          </div>
        </section>

        <section className="card">
          <div className="card-head"><h3>Nhầm lẫn thường gặp</h3></div>
          <div className="card-body">
            {hard?.pairs?.length ? (
              <table className="table">
                <thead>
                  <tr><th>Model đoán</th><th>Thực tế</th><th style={{ textAlign: 'right' }}>Lần</th></tr>
                </thead>
                <tbody>
                  {hard.pairs.map((p, i) => (
                    <tr key={i}>
                      <td>{p.predicted}</td>
                      <td style={{ fontWeight: 600 }}>{p.true}</td>
                      <td className="mono" style={{ textAlign: 'right' }}>{p.count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p className="small dim">
                Chưa có dữ liệu. Bấm <em>Sai</em> ở kết quả và nhập từ đúng — các cặp nhầm lẫn sẽ xuất hiện ở đây và
                gợi ý những từ nên enroll thêm.
              </p>
            )}
          </div>
        </section>
      </div>

      <div className="grid-2">
        <section className="card">
          <div className="card-head"><h3>Model đã dùng</h3></div>
          <div className="card-body">
            {ov.models.length ? (
              <table className="table">
                <thead><tr><th>Model</th><th style={{ textAlign: 'right' }}>Lượt</th><th style={{ textAlign: 'right' }}>Trung bình</th></tr></thead>
                <tbody>
                  {ov.models.map((m) => (
                    <tr key={m.model_id}>
                      <td>{m.display_name}</td>
                      <td className="mono" style={{ textAlign: 'right' }}>{m.count}</td>
                      <td className="mono" style={{ textAlign: 'right' }}>{m.avg_ms} ms</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : <p className="small dim">Chưa có dữ liệu.</p>}
          </div>
        </section>

        <section className="card">
          <div className="card-head">
            <Server size={15} style={{ color: 'var(--ink-3)' }} />
            <h3>Hạ tầng lưu trữ</h3>
            <div className="grow" />
            <button className="btn btn-quiet btn-sm" onClick={reconnect} disabled={busy}>
              {busy ? <span className="spinner" /> : <RefreshCw size={13} />} Kết nối lại
            </button>
          </div>
          <div className="card-body stack gap-3">
            <InfraRow
              name="Redis"
              ok={storage?.redis?.available}
              detail={
                storage?.redis?.available
                  ? `v${storage.redis.version} · ${storage.redis.keys} key · ${storage.redis.used_memory_human}`
                  : storage?.redis?.error || 'Đang dùng bộ nhớ tiến trình'
              }
            />
            <InfraRow
              name="MinIO"
              ok={storage?.object_store?.available}
              detail={
                storage?.object_store?.available
                  ? Object.entries(storage.object_store.usage || {})
                      .map(([b, u]) => `${b}: ${u.objects} obj / ${formatBytes(u.bytes)}`)
                      .join(' · ')
                  : storage?.object_store?.error || 'Đang lưu xuống đĩa cục bộ'
              }
            />
            {!storage?.redis?.available && (
              <p className="tiny dim">
                Bật hạ tầng: <code className="mono">docker compose -f sign_translate/docker-compose.yml up -d</code>
              </p>
            )}
          </div>
        </section>
      </div>
    </>
  )
}

function Stat({ label, value, unit, hint }) {
  return (
    <div className="card card-pad">
      <div className="eyebrow">{label}</div>
      <div className="row gap-2" style={{ alignItems: 'baseline', marginTop: 4 }}>
        <span className="stat-value">{value}</span>
        {unit && <span className="small dim">{unit}</span>}
      </div>
      {hint && <div className="tiny dim" style={{ marginTop: 2 }}>{hint}</div>}
    </div>
  )
}

function Bars({ values }) {
  const max = Math.max(...values, 1)
  return (
    <div className="bars">
      {values.map((v, i) => (
        <i key={i} style={{ height: `${Math.max(2, (v / max) * 100)}%` }} title={String(v)} />
      ))}
    </div>
  )
}

function InfraRow({ name, ok, detail }) {
  return (
    <div className="row gap-3">
      <span className={`dot ${ok ? 'dot-ok' : 'dot-warn'}`} />
      <span style={{ fontWeight: 600, minWidth: 60 }} className="small">{name}</span>
      <span className="tiny dim grow" style={{ wordBreak: 'break-word' }}>{detail}</span>
    </div>
  )
}
