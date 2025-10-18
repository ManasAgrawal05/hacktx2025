import React from 'react'

interface Props {
  original: string | null
  modified: string | null
  statistic: number | null
  onReset: () => void
}

export default function DisplayView({ original, modified, statistic, onReset }: Props): JSX.Element {
  return (
    <div className="display-wrap center-card">
      <h2>Result</h2>

      <div className="images-row">
        <figure>
          <figcaption>Original</figcaption>
          {original ? <img src={original} alt="original" /> : <div className="img-placeholder">No image</div>}
        </figure>

        <figure>
          <figcaption>Modified</figcaption>
          {modified ? <img src={modified} alt="modified" /> : <div className="img-placeholder">No image</div>}
        </figure>
      </div>

      <div className="stat-card">
        <div className="stat-label">AI Confusion</div>
        <div className="stat-value">{statistic != null ? `${statistic}%` : '—'}</div>
        <p className="muted">(Higher percentage means the model was more confused — placeholder metric.)</p>
      </div>

      <div style={{ marginTop: 14 }}>
        <button className="primary-btn" onClick={onReset}>Start New Upload</button>
      </div>
    </div>
  )
}