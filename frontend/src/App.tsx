import React, { useState, useCallback } from 'react'
import UploadArea from './components/UploadArea.tsx'
import LoadingBar from './components/LoadingBar.tsx'
import DisplayView from './components/DisplayView.tsx'
import Starfield from './components/StarField.js'
import StarfieldParticles from './components/StarFieldParticles.js'
import StarfieldWithConstellations from './components/StarFieldWithConstellations.js'

type AppState = 'ready' | 'loading' | 'display'

export default function App(): JSX.Element {
  const [state, setState] = useState<AppState>('ready')
  const [fileURL, setFileURL] = useState<string | null>(null)
  const [modifiedURL, setModifiedURL] = useState<string | null>(null)
  const [progress, setProgress] = useState<number>(0)
  const [statusText, setStatusText] = useState<string>('')
  const [statistic, setStatistic] = useState<number | null>(null)

  const numStarsNorm = 100;
  const clustersNorm = 10;
  const numStarsConstellations = 40;
  const clustersConstellations = 4;

  const resetAll = useCallback(() => {
    setState('ready')
    setProgress(0)
    setStatusText('')
    setStatistic(null)
    if (fileURL) {
      URL.revokeObjectURL(fileURL)
      setFileURL(null)
    }
    if (modifiedURL) {
      URL.revokeObjectURL(modifiedURL)
      setModifiedURL(null)
    }
  }, [fileURL, modifiedURL])

  const handleUpload = useCallback((file: File) => {
    if (!file) return
    const url = URL.createObjectURL(file)
    setFileURL(url)
    setState('loading')
    setProgress(0)
    setStatusText('Initializing transformation...')

    let p = 0
    const interval = window.setInterval(() => {
      p += Math.floor(Math.random() * 12) + 5
      if (p >= 100) p = 100
      setProgress(p)
      setStatusText(p < 100 ? `Transforming — ${p}%` : 'Finalizing results...')
      if (p === 100) {
        clearInterval(interval)
        simulateModification(file).then(({ url: modUrl, stat }) => {
          setModifiedURL(modUrl)
          setStatistic(stat)
          setState('display')
        })
      }
    }, 450)

  }, [])

  return (
    <div className="app-root">
      
      <header className="app-header">
        <h1 className="logo">BlackHole</h1>
        <p className="tag">Your digital Invisibilty Cloak</p>
      </header>

      <main className="app-main">
        {state === 'ready' && (
          <UploadArea onUpload={handleUpload} />
        )}

        {state === 'loading' && (
          <div className="center-card">
            <LoadingBar progress={progress} message={statusText} />
            <div style={{ marginTop: 18 }}>
              <button className="ghost-btn" onClick={resetAll}>Cancel & Reset</button>
            </div>
          </div>
        )}

        {state === 'display' && (
          <DisplayView
            original={fileURL}
            modified={modifiedURL}
            statistic={statistic}
            onReset={resetAll}
          />
        )}
      </main>

      <footer className="app-footer">
        <small>Space themed prototype — demo only</small>
      </footer>
      <Starfield numStars={numStarsNorm} clusters={clustersNorm} />
      <StarfieldWithConstellations numStars={numStarsConstellations} clusters={clustersConstellations} />
    </div>
  )
}

// ---------- Helper: simulateModification ----------

async function simulateModification(file: File): Promise<{ url: string | null; stat: number | null }> {
  return new Promise((resolve) => {
    const img = new Image()
    img.onload = () => {
      const w = img.naturalWidth
      const h = img.naturalHeight
      const canvas = document.createElement('canvas')
      canvas.width = w
      canvas.height = h
      const ctx = canvas.getContext('2d')!
      ctx.drawImage(img, 0, 0)

      const size = Math.max(6, Math.floor(Math.min(w, h) / 40))
      for (let y = 0; y < h; y += size) {
        for (let x = 0; x < w; x += size) {
          const data = ctx.getImageData(x, y, size, size)
          let r = 0, g = 0, b = 0, a = 0
          const len = data.data.length / 4
          for (let i = 0; i < data.data.length; i += 4) { r += data.data[i]; g += data.data[i + 1]; b += data.data[i + 2]; a += data.data[i + 3] }
          r = Math.round(r / len); g = Math.round(g / len); b = Math.round(b / len); a = Math.round(a / len);
          ctx.fillStyle = `rgba(${r},${g},${b},${a / 255})`
          ctx.fillRect(x, y, size, size)
        }
      }

      ctx.globalCompositeOperation = 'screen'
      ctx.fillStyle = 'rgba(255,255,255,0.02)'
      for (let i = 0; i < 200; i++) {
        const rx = Math.random() * w
        const ry = Math.random() * h
        const r = Math.random() * 1.4
        ctx.beginPath(); ctx.arc(rx, ry, r, 0, Math.PI * 2); ctx.fill();
      }

      canvas.toBlob((blob) => {
        if (!blob) { resolve({ url: null, stat: null }); return }
        const url = URL.createObjectURL(blob)
        const stat = Math.round(70 + Math.random() * 25)
        resolve({ url, stat })
      }, 'image/jpeg', 0.85)
    }
    img.onerror = () => {
      resolve({ url: null, stat: null })
    }
    img.src = URL.createObjectURL(file)
  })
}