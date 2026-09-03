/** Tiêu đề trang dùng chung cho mọi tab. */
export default function PageHead({ title, sub, action }) {
  return (
    <div className="row gap-4 wrap-row" style={{ alignItems: 'flex-end' }}>
      <div>
        <h1>{title}</h1>
        {sub && <p className="muted small" style={{ maxWidth: 620, marginTop: 4 }}>{sub}</p>}
      </div>
      <div className="grow" />
      {action}
    </div>
  )
}
