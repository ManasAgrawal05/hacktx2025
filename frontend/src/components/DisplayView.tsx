import React from 'react'

interface Props {
  mask: string | null
  original: string | null
  modified: string | null
  statistic: number | null
  onReset: () => void
}

export default function DisplayView({
  mask,
  original,
  modified,
  statistic,
  onReset,
}: Props): JSX.Element {
  const handleDownload = (url: string | null, filename: string) => {
    if (!url) return
    const link = document.createElement('a')
    link.href = url
    link.download = filename
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  return (
    <div className="display-wrap center-card">
      <h2>Result</h2>

      {/* Mask below */}
      <div style={{ marginTop: 28, textAlign: 'center' }}>
        <figcaption>Anchor</figcaption>
        {mask ? (
          <img
            src={mask}
            alt="mask"
            style={{
              width: '100%',
              maxWidth: 580,
              borderRadius: 8,
              marginTop: 6,
            }}
          />
        ) : (
          <div
            className="img-placeholder"
            style={{ margin: '0 auto', width: 280 }}
          >
            No Anchor
          </div>
        )}
      </div>
      
      {/* Original & Modified side by side */}
      <div
        className="images-row"
        style={{
          display: 'flex',
          justifyContent: 'center',
          gap: 24,
          flexWrap: 'wrap',
          marginTop: 12,
        }}
      >
        {/* Original */}
        <figure style={{ textAlign: 'center' }}>
          <figcaption>Original</figcaption>
          {original ? (
            <>
              <img
                src={original}
                alt="original"
                style={{ maxWidth: '100%', height: 'auto' }}
              />
              <div style={{ marginTop: 6 }}>
                <button
                  className="primary-btn"
                  onClick={() => handleDownload(original, 'original.jpg')}
                >
                  Download Original
                </button>
              </div>
            </>
          ) : (
            <div className="img-placeholder">No image</div>
          )}
        </figure>

        {/* Modified */}
        <figure style={{ textAlign: 'center' }}>
          <figcaption>Modified</figcaption>
          {modified ? (
            <>
              <img
                src={modified}
                alt="modified"
                style={{ maxWidth: '100%', height: 'auto' }}
              />
              <div style={{ marginTop: 6 }}>
                <button
                  className="primary-btn"
                  onClick={() => handleDownload(modified, 'modified.jpg')}
                >
                  Download Modified
                </button>
              </div>
            </>
          ) : (
            <div className="img-placeholder">No image</div>
          )}
        </figure>
      </div>

      

      {/* Statistic below */}
      <div style={{ marginTop: 28, textAlign: 'center' }}>
        <div className="stat-value" style={{ fontSize: 28 }}>
          {statistic != null ? `${statistic}%` : '—'}
        </div>
        <p className="muted" style={{ marginTop: 8 }}>
          Higher percentages mean the AI was more confused when identifying the
          subject — indicating better anonymization.
        </p>
      </div>

      <div style={{ marginTop: 20 }}>
        <button className="primary-btn" onClick={onReset}>
          Start New Upload
        </button>
      </div>
    </div>
  )
}
