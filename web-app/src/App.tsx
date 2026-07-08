import { lazy, Suspense } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import Header from './components/Header'
import ErrorBoundary from './components/ErrorBoundary'
import HomePage from './pages/HomePage'

// Harita (maplibre) ve Hakkında ilk boyamayı yavaşlatmasın diye tembel yüklenir.
const MapPage = lazy(() => import('./pages/MapPage'))
const AboutPage = lazy(() => import('./pages/AboutPage'))

function RouteFallback() {
  return (
    <div style={{ flex: 1, display: 'grid', placeItems: 'center' }}>
      <div className="spinner" />
    </div>
  )
}

export default function App() {
  return (
    <>
      <Header />
      <ErrorBoundary>
        <Suspense fallback={<RouteFallback />}>
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/harita" element={<MapPage />} />
            <Route path="/hakkinda" element={<AboutPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </Suspense>
      </ErrorBoundary>
    </>
  )
}
