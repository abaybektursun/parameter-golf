interface Props {
  children: React.ReactNode;
}

export default function InsightCallout({ children }: Props) {
  return (
    <div className="rounded-lg border border-[var(--border)] bg-[var(--card)] px-6 py-5 mt-8">
      <p className="text-sm font-semibold text-[var(--foreground)] mb-1">Key Insight</p>
      <div className="text-[var(--muted)] text-[15px] leading-relaxed">
        {children}
      </div>
    </div>
  );
}
