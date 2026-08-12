"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import "./css/SelfTesting.css"
import Navbar from "./Navbar.jsx"

const SELF_TEST_API = process.env.REACT_APP_SELFTEST_API_URL || "http://localhost:5001"

// How often we poll the backend for a fresh prediction.
const POLL_MS = 400
// How long a sign has to be held (with confidence above the threshold)
// before it's committed to the sentence. This is what turns "the model's
// raw per-frame guess" into something you can actually spell words with --
// without it, a held letter would spam itself into the sentence dozens of
// times per second.
const HOLD_MS = 900
const CONFIDENCE_THRESHOLD = 0.65

// The training set's label set (see server/labels.txt) includes three
// non-alphabetic classes alongside A-Z:
const BLANK_SIGN = "blank"   // resting hand / no gesture -- ignored
const DELETE_SIGN = "delete" // backspace
const SPACE_SIGN = "space"   // word separator

const speak = (text) => {
  if (!text || !("speechSynthesis" in window)) return
  window.speechSynthesis.cancel()
  const utterance = new SpeechSynthesisUtterance(text)
  utterance.rate = 0.9
  window.speechSynthesis.speak(utterance)
}

const ASLDetector = () => {
  const [prediction, setPrediction] = useState("")
  const [confidence, setConfidence] = useState(0)
  const [holdProgress, setHoldProgress] = useState(0)
  const [sentence, setSentence] = useState("")
  const [autoSpeak, setAutoSpeak] = useState(false)
  const [speechSupported] = useState(() => typeof window !== "undefined" && "speechSynthesis" in window)

  // Read inside the polling loop without re-subscribing the effect on every
  // toggle/keystroke.
  const autoSpeakRef = useRef(autoSpeak)
  useEffect(() => {
    autoSpeakRef.current = autoSpeak
  }, [autoSpeak])

  const candidateRef = useRef({ sign: null, since: null })
  const lastCommittedRef = useRef(null)

  const commitSign = useCallback((sign) => {
    if (sign === DELETE_SIGN) {
      setSentence((prev) => prev.slice(0, -1))
      return
    }
    const char = sign === SPACE_SIGN ? " " : sign
    setSentence((prev) => prev + char)
    if (autoSpeakRef.current) {
      speak(sign === SPACE_SIGN ? "space" : sign)
    }
  }, [])

  useEffect(() => {
    let cancelled = false

    const interval = setInterval(async () => {
      try {
        const response = await fetch(`${SELF_TEST_API}/prediction`)
        const data = await response.json()
        if (cancelled) return

        setPrediction(data.class || "")
        setConfidence(data.confidence || 0)

        const isTrackable =
          data.class && data.class !== BLANK_SIGN && data.confidence >= CONFIDENCE_THRESHOLD

        if (!isTrackable) {
          // No hand, low confidence, or the resting "blank" class: reset the
          // hold timer and, importantly, the repeat-guard -- so showing the
          // same letter twice in a row (e.g. the two Ls in "HELLO") works as
          // long as the hand passes through a rest/blank state in between.
          candidateRef.current = { sign: null, since: null }
          lastCommittedRef.current = null
          setHoldProgress(0)
          return
        }

        const now = Date.now()
        if (data.class !== candidateRef.current.sign) {
          candidateRef.current = { sign: data.class, since: now }
        }

        const heldFor = now - candidateRef.current.since
        setHoldProgress(Math.min(1, heldFor / HOLD_MS))

        if (heldFor >= HOLD_MS && candidateRef.current.sign !== lastCommittedRef.current) {
          commitSign(candidateRef.current.sign)
          lastCommittedRef.current = candidateRef.current.sign
        }
      } catch (error) {
        console.error("Error fetching prediction:", error)
      }
    }, POLL_MS)

    return () => {
      cancelled = true
      clearInterval(interval)
    }
  }, [commitSign])

  const handleSpace = () => setSentence((prev) => prev + " ")
  const handleBackspace = () => setSentence((prev) => prev.slice(0, -1))
  const handleClear = () => setSentence("")
  const handleSpeak = () => speak(sentence.trim())

  return (
    <div className="asl-detector">
      <Navbar />

      {/* Main Content */}
      <div className="asl-content">
        <h1 className="title">Real-Time ASL Detection</h1>
        <div className="video-wrapper">
          <img src={`${SELF_TEST_API}/video_feed`} alt="Live ASL Stream" className="video-feed" />
        </div>

        <div className="output-box">
          <h2>Detected Sign</h2>
          <div className={`detected-sign ${prediction ? "" : "no-hand"}`}>
            {prediction || "Waiting for sign..."}
          </div>
          {prediction && (
            <div className="confidence-row">
              <span className="confidence-label">
                {(confidence * 100).toFixed(0)}% confidence
              </span>
              <div className="hold-track" title="Hold the sign steady to add it to the sentence">
                <div className="hold-fill" style={{ width: `${holdProgress * 100}%` }} />
              </div>
            </div>
          )}
          <p className="hint-text">
            Hold a sign steady for under a second to spell it out below. Show
            "delete" to backspace and "space" to add a space.
          </p>
        </div>

        <div className="sentence-box">
          <h2>Spelled Sentence</h2>
          <div className="sentence-text">
            {sentence || <span className="sentence-placeholder">Your spelled letters will appear here...</span>}
            <span className="sentence-cursor" aria-hidden="true">|</span>
          </div>
          <div className="sentence-controls">
            <button type="button" onClick={handleSpace} className="sentence-btn">Space</button>
            <button type="button" onClick={handleBackspace} className="sentence-btn">Backspace</button>
            <button type="button" onClick={handleClear} className="sentence-btn danger">Clear</button>
            <button
              type="button"
              onClick={handleSpeak}
              className="sentence-btn primary"
              disabled={!speechSupported || !sentence.trim()}
              title={speechSupported ? "Speak the sentence aloud" : "Speech synthesis isn't supported in this browser"}
            >
              Speak
            </button>
            <label className="autospeak-toggle">
              <input
                type="checkbox"
                checked={autoSpeak}
                onChange={(e) => setAutoSpeak(e.target.checked)}
                disabled={!speechSupported}
              />
              Speak each letter as I sign it
            </label>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ASLDetector
