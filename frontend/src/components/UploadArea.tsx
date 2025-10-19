import React, { useRef } from 'react'

interface Props { onUpload: (file: File) => void }

export default function UploadArea({ onUpload }: Props): JSX.Element {
  const inputRef = useRef<HTMLInputElement | null>(null)

  const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    if (f) onUpload(f)
  }

  const handleClick = () => inputRef.current?.click()

  return (
    <div className="upload-area-circle">
      
      <div className="upload-content">
        <h2>Ready for upload</h2>
        <p className="muted">Point your camera, or upload an image to test the obfuscation.</p>

        <button className="primary-btn" onClick={handleClick}>Upload Image</button>
        <input ref={inputRef} type="file" accept="image/*" onChange={handleFile} style={{ display: 'none' }} />
      </div>
    </div>
  )
}
