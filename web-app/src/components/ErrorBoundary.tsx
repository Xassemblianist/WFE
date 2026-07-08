import { Component, ReactNode } from 'react'

interface State {
  error: Error | null
}

export default class ErrorBoundary extends Component<{ children: ReactNode }, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  render() {
    if (this.state.error) {
      return (
        <div style={{ padding: 24, color: 'var(--fg)', fontFamily: 'monospace', overflow: 'auto' }}>
          <h2 style={{ color: 'var(--danger)' }}>Arayüz hatası</h2>
          <pre id="wfe-error" style={{ whiteSpace: 'pre-wrap', fontSize: 12 }}>
            {this.state.error.message}
            {'\n\n'}
            {this.state.error.stack}
          </pre>
        </div>
      )
    }
    return this.props.children
  }
}
