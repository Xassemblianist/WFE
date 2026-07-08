import { NavLink, Link } from 'react-router-dom'
import { useTheme } from '../theme'
import { IcSun, IcMoon } from './icons'

// İzotah/akım-çizgisi işareti: üç kademeli akış hattı — teknik, tek renkli.
function Logo() {
  return (
    <svg className="logo" viewBox="0 0 32 32" aria-hidden>
      <rect width="32" height="32" rx="8" fill="#0b1826" />
      <g stroke="#2fb5ee" strokeWidth="2.6" strokeLinecap="round" fill="none">
        <path d="M6 10.5h13c4 0 4-5-.5-4.4" />
        <path d="M6 16h20" opacity="0.55" />
        <path d="M6 21.5h10c4.5 0 4.5 5 0 4.6" opacity="0.85" />
      </g>
    </svg>
  )
}

export default function Header() {
  const { theme, toggle } = useTheme()
  return (
    <header className="app-header">
      <Link to="/" className="brand">
        <Logo />
        <span>
          WFE
          <span className="sub">Weather Forecast Engine</span>
        </span>
      </Link>
      <nav className="nav">
        <NavLink to="/" end>
          <span>Ana Sayfa</span>
        </NavLink>
        <NavLink to="/harita">
          <span>Harita</span>
        </NavLink>
        <NavLink to="/hakkinda">
          <span>Hakkında</span>
        </NavLink>
      </nav>
      <div className="header-spacer" />
      <button className="icon-btn" onClick={toggle} title="Tema değiştir" aria-label="Tema değiştir">
        {theme === 'dark' ? <IcSun /> : <IcMoon />}
      </button>
    </header>
  )
}
