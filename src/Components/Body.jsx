import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './Body.css'; // Assuming your CSS file is correctly linked

const Body = () => {
    const [input, setInput] = useState('');
    const [response, setResponse] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [parsedResult, setParsedResult] = useState(null);

    // --- Voice Recognition States and Refs ---
    const [isListeningVoice, setIsListeningVoice] = useState(false);
    const [voiceTranscript, setVoiceTranscript] = useState('');
    const recognitionRef = useRef(null); // To store the SpeechRecognition object
    // --- End Voice Recognition States and Refs ---

    useEffect(() => {
        // Check for Web Speech API compatibility
        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
            console.warn('Web Speech API is not fully supported by this browser. Voice recognition will not be available.');
            // Optionally disable the mic button or show a message
            return;
        }

        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        const recognition = new SpeechRecognition();

        recognition.continuous = false; // Set to false to get one result and stop
        recognition.interimResults = false; // Only get final results
        recognition.lang = 'en-IN'; // Set the language, e.g., English (India)

        recognition.onstart = () => {
            console.log('Voice recognition started.');
            setIsListeningVoice(true);
            setVoiceTranscript(''); // Clear previous transcript on start
            setError(null); // Clear any previous errors
        };

        recognition.onresult = (event) => {
            const finalTranscript = Array.from(event.results)
                .map(result => result[0].transcript)
                .join('');
            setVoiceTranscript(finalTranscript);
            setInput(finalTranscript); // Automatically set the recognized text to your input field
            console.log('Final Transcript:', finalTranscript);
        };

        recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            setIsListeningVoice(false);
            if (event.error === 'no-speech') {
                setError('No speech detected. Please speak clearly.');
            } else if (event.error === 'not-allowed') {
                setError('Microphone access denied. Please enable microphone in browser settings.');
            } else if (event.error === 'network') {
                setError('Network error during voice recognition.');
            } else {
                setError(`Voice recognition error: ${event.error}`);
            }
        };

        recognition.onend = () => {
            console.log('Voice recognition ended.');
            setIsListeningVoice(false);
        };

        recognitionRef.current = recognition;

        // Cleanup on component unmount
        return () => {
            if (recognitionRef.current) {
                recognitionRef.current.stop();
            }
        };
    }, []); // Empty dependency array means this effect runs once on mount

    const handleSubmit = async () => {
        if (!input.trim()) {
            setError('Please enter a crime narration or speak into the microphone.');
            return;
        }

        setLoading(true);
        setError(null);
        try {
            const result = await axios.post('http://localhost:8000/analyze', {
                narration: input
            });
            
            // Set the response with latency
            setResponse({
                response: result.data.response,
                latency: `${result.data.latency_seconds}s`
            });

            // Display the raw response for now
            setParsedResult({
                crime: result.data.response,
                bns: '',
                judgment: ''
            });
        } catch (err) {
            setError('Error analyzing the narration. Please try again.');
            console.error('Error:', err);
        } finally {
            setLoading(false);
        }
    };

    // --- Voice Recognition Handler ---
    const toggleVoiceRecognition = () => {
        if (recognitionRef.current) {
            if (isListeningVoice) {
                recognitionRef.current.stop();
            } else {
                try {
                    recognitionRef.current.start();
                } catch (e) {
                    // Catch error if recognition is already running (e.g., multiple rapid clicks)
                    console.warn("Attempted to start recognition, but it might already be active:", e);
                }
            }
        } else {
            setError('Voice recognition is not supported in your browser.');
        }
    };
    // --- End Voice Recognition Handler ---

    return (
        <div>
            <section className="hero">
                <div id="home" className="hero-text">
                    <h1>Navigate Law with <span className="glow">AI Precision</span></h1>
                    <p>Your intelligent legal companion providing clarity, document analysis, and case visualization powered by artificial intelligence.</p>
                    <div className="search-container">
                        <div className="search-bar">
                            <input
                                type="text"
                                placeholder="Enter crime narration for analysis..."
                                value={input}
                                onChange={(e) => setInput(e.target.value)}
                            />
                            <button
                                id="micBtn"
                                aria-label="Voice search"
                                onClick={toggleVoiceRecognition}
                                disabled={loading || (!recognitionRef.current && !('webkitSpeechRecognition' in window))}
                            >
                                <i className={`fas fa-microphone ${isListeningVoice ? 'listening-active' : ''}`}></i>
                                <span className="tooltip">{isListeningVoice ? 'Stop Speaking' : 'Voice Search'}</span>
                            </button>
                        </div>
                        <button className="cta" onClick={handleSubmit} disabled={loading}>
                            <i className="fas fa-rocket"></i>
                            {loading ? 'Analyzing...' : 'Analyze Narration'}
                        </button>
                    </div>

                    {error && (
                        <div className="error-message">
                            {error}
                        </div>
                    )}

                    {parsedResult && (
                        <div className="response-container">
                            <h2>Analysis Result:</h2>
                            <div className="response-content">
                                {parsedResult.crime}
                            </div>
                            {response?.latency && (
                                <p className="latency">Response time: {response.latency}</p>
                            )}
                        </div>
                    )}
                </div>
                <div className="hero-image">
                    <div className="screen-mockup floating">
                        <img src="/hero.png" alt="Legal Analysis" />
                    </div>
                </div>
            </section>
        </div>
    );
};

export default Body;