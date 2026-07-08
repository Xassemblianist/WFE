import type { FieldKey } from '../api'
import { FIELD_DEFS, FIELD_ORDER } from '../lib/fields'
import { IcThermo, IcWind, IcRain, IcCloud, IcGauge } from './icons'

const ICONS: Record<FieldKey, (p: { size?: number }) => JSX.Element> = {
  t2m: (p) => <IcThermo {...p} />,
  wind: (p) => <IcWind {...p} />,
  precip: (p) => <IcRain {...p} />,
  cloud: (p) => <IcCloud {...p} />,
  mslp: (p) => <IcGauge {...p} />,
}

interface Props {
  fields: FieldKey[]
  value: FieldKey
  onChange: (f: FieldKey) => void
}

export default function LayerRail({ fields, value, onChange }: Props) {
  const list = FIELD_ORDER.filter((f) => fields.includes(f))
  return (
    <div className="layer-rail glass" role="radiogroup" aria-label="Katman seç">
      {list.map((f) => {
        const d = FIELD_DEFS[f]
        const Ico = ICONS[f]
        return (
          <button
            key={f}
            className={`rail-item ${value === f ? 'active' : ''}`}
            onClick={() => onChange(f)}
            role="radio"
            aria-checked={value === f}
            aria-label={d.label}
          >
            <Ico size={21} />
            <span className="tip">{d.label}</span>
          </button>
        )
      })}
    </div>
  )
}
