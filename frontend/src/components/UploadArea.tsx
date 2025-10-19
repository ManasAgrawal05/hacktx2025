import React, { useRef } from 'react'

interface Props { onTakePhoto: () => void; expanded?: boolean }

export default function UploadArea({ onTakePhoto, expanded=false }: Props): JSX.Element {
  return (
    <div className="upload-area-circle">
      
      <div className="upload-content">
        <h2>Photo Button Title</h2>
        <p className="muted">Point your camera, or take a photo to test the obfuscation</p>

        <button className="primary-btn" onClick={onTakePhoto}>Take a Photo</button>
      </div>
    </div>
  )
}

