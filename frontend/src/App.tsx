import React, { useState, useCallback } from 'react'
import UploadArea from './components/UploadArea.tsx'
import LoadingBar from './components/LoadingBar.tsx'
import DisplayView from './components/DisplayView.tsx'
import Starfield from './components/StarField.js'
import StarfieldWithConstellations from './components/StarFieldWithConstellations.js'

type AppState = 'ready' | 'loading' | 'display'

export default function App(): JSX.Element {
  const [state, setState] = useState<AppState>('ready')
  const [fileURL, setFileURL] = useState<string | null>(null)
  const [modifiedURL, setModifiedURL] = useState<string | null>(null)
  const [maskURL, setMaskURL] = useState<string | null>(null)
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
    if (maskURL) {
      URL.revokeObjectURL(maskURL)
      setMaskURL(null)
    }
  }, [fileURL, modifiedURL, maskURL])

  const createPlaceholderPhoto = async (): Promise<File> => {
    const w = 1280
    const h = 720
    const canvas = document.createElement('canvas')
    canvas.width = w
    canvas.height = h
    const ctx = canvas.getContext('2d')!

    // background
    ctx.fillStyle = '#000'
    ctx.fillRect(0, 0, w, h)

    // subtle radial glow
    const grad = ctx.createRadialGradient(w * 0.5, h * 0.4, 10, w * 0.5, h * 0.4, Math.max(w, h) * 0.8)
    grad.addColorStop(0, 'rgba(255,255,255,0.06)')
    grad.addColorStop(1, 'rgba(255,255,255,0)')
    ctx.fillStyle = grad
    ctx.fillRect(0, 0, w, h)

    // placeholder subject
    ctx.fillStyle = '#1a1a1a'
    ctx.fillRect(w * 0.15, h * 0.2, w * 0.7, h * 0.6)
    ctx.fillStyle = '#222'
    ctx.beginPath()
    ctx.arc(w * 0.5, h * 0.45, Math.min(w, h) * 0.12, 0, Math.PI * 2)
    ctx.fill()

    // label text
    ctx.fillStyle = '#ddd'
    ctx.font = '36px sans-serif'
    ctx.textAlign = 'center'
    ctx.fillText('BlackHole Capture', w * 0.5, h * 0.85)

    const blob: Blob | null = await new Promise((resolve) => canvas.toBlob((b) => resolve(b), 'image/jpeg', 0.9))
    if (!blob) return new File([], 'placeholder.jpg', { type: 'image/jpeg' })
    return new File([blob], 'capture.jpg', { type: 'image/jpeg' })
  }


  const handleTakePhoto = useCallback(async () => {
    // create placeholder capture
    const file = await createPlaceholderPhoto()
    const previewUrl = URL.createObjectURL(file)
    setFileURL(previewUrl)

    // start simulated progress + transformation pipeline
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
        // run the modification that also yields a mask
        simulateModification(file).then(({ url: modUrl, stat, mask }) => {
          setModifiedURL(modUrl)
          setMaskURL(mask)
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
          <UploadArea onTakePhoto={handleTakePhoto}/>
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
          <DisplayView mask={maskURL} original={fileURL} modified={modifiedURL} statistic={statistic} onReset={resetAll} />
        )}
      </main>

      <footer className="app-footer">
        <small>Space themed prototype — demo only</small>
      </footer>
      (
        <div>
          <Starfield numStars={numStarsNorm} clusters={clustersNorm} state={state}/>
          <StarfieldWithConstellations numStars={numStarsConstellations} clusters={clustersConstellations} state={state}/>
        </div>
      )
    </div>
  )
}


async function simulateModification(file: File): Promise<{ url: string | null; stat: number | null; mask: string | null }> {
    return new Promise((resolve) => {
      const img = new Image()
      img.onload = () => {
        const w = img.naturalWidth
        const h = img.naturalHeight

        // canvas for modified image
        const canvas = document.createElement('canvas')
        canvas.width = w
        canvas.height = h
        const ctx = canvas.getContext('2d')!
        ctx.drawImage(img, 0, 0)

        // canvas for mask (black background, white where obfuscated)
        const maskCanvas = document.createElement('canvas')
        maskCanvas.width = w
        maskCanvas.height = h
        const maskCtx = maskCanvas.getContext('2d')!
        maskCtx.fillStyle = '#000'
        maskCtx.fillRect(0, 0, w, h)

        // apply pixelation and at the same time mark mask blocks
        const size = Math.max(6, Math.floor(Math.min(w, h) / 40))
        for (let y = 0; y < h; y += size) {
          for (let x = 0; x < w; x += size) {
            const data = ctx.getImageData(x, y, size, size)
            let r = 0,
              g = 0,
              b = 0,
              a = 0
            const len = data.data.length / 4
            for (let i = 0; i < data.data.length; i += 4) {
              r += data.data[i]
              g += data.data[i + 1]
              b += data.data[i + 2]
              a += data.data[i + 3]
            }
            r = Math.round(r / len)
            g = Math.round(g / len)
            b = Math.round(b / len)
            a = Math.round(a / len)
            // fill the modified image block
            ctx.fillStyle = `rgba(${r},${g},${b},${a / 255})`
            ctx.fillRect(x, y, size, size)
            // mark mask block (white)
            maskCtx.fillStyle = 'rgba(255,255,255,1)'
            maskCtx.fillRect(x, y, size, size)
          }
        }

        // subtle star overlay on modified (keeps theme)
        ctx.globalCompositeOperation = 'screen'
        ctx.fillStyle = 'rgba(255,255,255,0.02)'
        for (let i = 0; i < 200; i++) {
          const rx = Math.random() * w
          const ry = Math.random() * h
          const r = Math.random() * 1.4
          ctx.beginPath()
          ctx.arc(rx, ry, r, 0, Math.PI * 2)
          ctx.fill()
        }

        // produce blobs for modified and mask
        canvas.toBlob(
          (blob) => {
            if (!blob) {
              resolve({ url: null, stat: null, mask: null })
              return
            }
            const modifiedUrl = URL.createObjectURL(blob)

            maskCanvas.toBlob((maskBlob) => {
              if (!maskBlob) {
                resolve({ url: modifiedUrl, stat: null, mask: null })
                return
              }
              const maskUrl = URL.createObjectURL(maskBlob)
              const stat = Math.round(70 + Math.random() * 25)
              resolve({ url: modifiedUrl, stat, mask: maskUrl })
            }, 'image/png')
          },
          'image/jpeg',
          0.85,
        )
      }

      img.onerror = () => {
        resolve({ url: null, stat: null, mask: null })
      }

      img.src = URL.createObjectURL(file)
    })
  }
