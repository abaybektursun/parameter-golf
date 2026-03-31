interface StatsCardProps {
  label: string;
  value: string;
  unit?: string;
}

export default function StatsCard({ label, value, unit }: StatsCardProps) {
  return (
    <div className="rounded-lg border border-[var(--border)] bg-[var(--card)] p-6">
      <p className="text-sm text-[var(--muted)] mb-1">{label}</p>
      <p className="text-2xl font-semibold">
        {value}
        {unit && (
          <span className="text-sm font-normal text-[var(--muted)] ml-1">
            {unit}
          </span>
        )}
      </p>
    </div>
  );
}
