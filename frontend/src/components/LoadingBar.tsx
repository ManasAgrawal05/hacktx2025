import React from 'react'

interface Props { progress?: number; message?: string }

export default function LoadingBar({ progress = 0, message = '' }: Props): JSX.Element {
  return (
    <div className="loading-wrap">
      <h3>Working on it...</h3>
      <div className="loading-bar" aria-hidden>
        <div className="loading-fill" style={{ width: `${progress}%` }} />
      </div>
      <p className="muted">{message}</p>
      <div className="progress-number" aria-live="polite">{progress}%</div>
    </div>
  )
}