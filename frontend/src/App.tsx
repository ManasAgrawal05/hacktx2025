import React, { useState, useCallback } from 'react'
import UploadArea from './components/UploadArea.tsx'
import LoadingBar from './components/LoadingBar.tsx'
import DisplayView from './components/DisplayView.tsx'
import Starfield from './components/StarField.js'
import StarfieldWithConstellations from './components/StarFieldWithConstellations.js'

import originalImg from './uploads/original_full_b.jpg'
import modifiedImg from './uploads/full_modified.jpg'
import maskImg from './uploads/original_full.jpg'


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


  const handleTakePhoto = useCallback(async () => {
  // Start loading simulation
  setState('loading')
  setProgress(0)
  setStatusText('Initializing transformation...')

  let p = 0
  const interval = window.setInterval(() => {
    // Simulate progress
    p += Math.floor(Math.random() * 12) + 5
    if (p >= 100) p = 100
    setProgress(p)
    setStatusText(p < 100 ? `Transforming — ${p}%` : 'Finalizing results...')

    if (p === 100) {
      clearInterval(interval)

      // Set the images using the symlink imports
      setFileURL(originalImg)
      setModifiedURL(modifiedImg)
      setMaskURL(maskImg)
      setStatistic(78) // placeholder metric
      setState('display')
    }
  }, 450)
}, [])


  const waitForFiles = async (files: string[], intervalMs = 1000) => {
    const checkFile = async (url: string) => {
      try {
        const res = await fetch(url, { method: 'HEAD' }) // only check if exists
        return res.ok
      } catch {
        return false
      }
    }

    let allExist = false
    while (!allExist) {
      const statuses = await Promise.all(files.map(f => checkFile(f)))
      allExist = statuses.every(s => s)
      if (!allExist) await new Promise(res => setTimeout(res, intervalMs))
    }
  }




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

