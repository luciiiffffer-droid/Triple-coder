/**
 * AI Voice Chatbot — Core JavaScript
 *
 * Provides:  API (auth + fetch wrapper), VoiceChat (MediaRecorder + WebSocket), Toast
 */

// ═══════════════════════════════════════════════════════════
//  CONFIG
// ═══════════════════════════════════════════════════════════
const CONFIG = {
  API_BASE: window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? 'http://localhost:8000'
    : '',
  WS_BASE: window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? 'ws://localhost:8000'
    : `ws://${window.location.host}`,
};

// ═══════════════════════════════════════════════════════════
//  API — Auth + fetch wrapper
// ═══════════════════════════════════════════════════════════
const API = {
  getToken() {
    return localStorage.getItem('token');
  },

  setAuth(data) {
    localStorage.setItem('token', data.access_token);
    localStorage.setItem('user_id', data.user_id);
    localStorage.setItem('is_admin', data.is_admin);
  },

  logout() {
    localStorage.clear();
    window.location.href = 'index.html';
  },

  requireAuth() {
    if (!this.getToken()) {
      window.location.href = 'index.html';
    }
  },

  async request(path, options = {}) {
    const url = `${CONFIG.API_BASE}${path}`;
    const headers = {
      'Content-Type': 'application/json',
      ...(options.headers || {}),
    };

    const token = this.getToken();
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }

    const res = await fetch(url, { ...options, headers });

    if (res.status === 401) {
      this.logout();
      throw new Error('Session expired');
    }

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Request failed' }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    return res.json();
  },

  async login(username, password) {
    const data = await this.request('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify({ username, password }),
    });
    this.setAuth(data);
    return data;
  },

  async register(username, email, password) {
    const data = await this.request('/api/auth/register', {
      method: 'POST',
      body: JSON.stringify({ username, email, password }),
    });
    this.setAuth(data);
    return data;
  },
};

// ═══════════════════════════════════════════════════════════
//  TOAST — notification system
// ═══════════════════════════════════════════════════════════
const Toast = {
  _container: null,

  _getContainer() {
    if (!this._container) {
      this._container = document.getElementById('toast-container');
      if (!this._container) {
        this._container = document.createElement('div');
        this._container.className = 'toast-container';
        this._container.id = 'toast-container';
        document.body.appendChild(this._container);
      }
    }
    return this._container;
  },

  show(message, type = 'info', duration = 4000) {
    const container = this._getContainer();
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateX(100px)';
      toast.style.transition = 'all 0.3s ease';
      setTimeout(() => toast.remove(), 300);
    }, duration);
  },

  success(msg) { this.show(msg, 'success'); },
  error(msg)   { this.show(msg, 'error'); },
  info(msg)    { this.show(msg, 'info'); },
};

// ═══════════════════════════════════════════════════════════
//  VOICE CHAT — MediaRecorder + WebSocket
// ═══════════════════════════════════════════════════════════
class VoiceChat {
  constructor(callbacks = {}) {
    this.callbacks = callbacks;
    this.mediaRecorder = null;
    this.audioChunks = [];
    this.isRecording = false;
    this.ws = null;
    this.sessionId = this._generateId();
    this.stream = null;
  }

  _generateId() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
      const r = Math.random() * 16 | 0;
      return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
    });
  }

  async _ensureWebSocket() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) return;

    return new Promise((resolve, reject) => {
      const url = `${CONFIG.WS_BASE}/ws/voice/${this.sessionId}`;
      this.ws = new WebSocket(url);

      this.ws.onopen = () => {
        this.callbacks.onStatusChange?.('Connected');
        resolve();
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'response') {
            this.callbacks.onTranscript?.(data.transcript);
            this.callbacks.onResponse?.(data);
          } else if (data.type === 'error') {
            this.callbacks.onError?.(data.message);
          }
        } catch (e) {
          console.error('WS message parse error:', e);
        }
      };

      this.ws.onerror = (err) => {
        this.callbacks.onError?.('WebSocket connection error');
        reject(err);
      };

      this.ws.onclose = () => {
        this.callbacks.onStatusChange?.('Disconnected');
      };
    });
  }

  async toggleRecording() {
    if (this.isRecording) {
      this.stopRecording();
    } else {
      await this.startRecording();
    }
  }

  async startRecording() {
    try {
      await this._ensureWebSocket();

      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        }
      });

      this.audioChunks = [];
      this.mediaRecorder = new MediaRecorder(this.stream, {
        mimeType: MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
          ? 'audio/webm;codecs=opus'
          : 'audio/webm'
      });

      this.mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) this.audioChunks.push(e.data);
      };

      this.mediaRecorder.onstop = async () => {
        const blob = new Blob(this.audioChunks, { type: 'audio/webm' });
        const buffer = await blob.arrayBuffer();

        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
          this.ws.send(buffer);
          this.callbacks.onStatusChange?.('Processing…');
        } else {
          this.callbacks.onError?.('WebSocket not connected');
        }

        // Cleanup stream
        if (this.stream) {
          this.stream.getTracks().forEach(t => t.stop());
          this.stream = null;
        }
      };

      this.mediaRecorder.start();
      this.isRecording = true;
      this.callbacks.onRecordingStart?.();
      this.callbacks.onStatusChange?.('Recording…');

    } catch (err) {
      console.error('Recording error:', err);
      if (err.name === 'NotAllowedError') {
        this.callbacks.onError?.('Microphone access denied. Please allow microphone access.');
      } else {
        this.callbacks.onError?.('Failed to start recording: ' + err.message);
      }
    }
  }

  stopRecording() {
    if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
      this.mediaRecorder.stop();
      this.isRecording = false;
      this.callbacks.onRecordingStop?.();
    }
  }

  static playAudioBase64(base64) {
    if (!base64) return;
    try {
      const audio = new Audio(`data:audio/mpeg;base64,${base64}`);
      audio.play().catch(e => console.warn('Audio playback error:', e));
    } catch (e) {
      console.error('Audio playback error:', e);
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    if (this.stream) {
      this.stream.getTracks().forEach(t => t.stop());
    }
  }
}
