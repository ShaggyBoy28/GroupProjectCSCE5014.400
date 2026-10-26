// Shared UI pieces used across the pages.
import type { ComponentPropsWithoutRef, ReactNode } from 'react';
import { Link, type LinkProps } from 'react-router-dom';

import { UNCERTAIN, type ActivityLabel } from './api';
import logoUrl from './assets/icons/logo.svg';
import { ACTIVITIES, getActivity, IDLE_COLOR } from './lib';

export function AppHeader() {
  return (
    <header className="sticky top-0 z-10 border-b border-black/8 bg-white/80 backdrop-blur-[10px]">
      <nav className="mx-auto flex h-[52px] max-w-[1280px] items-center justify-between px-4 md:px-16">
        <Link to="/" className="flex items-center gap-2.5">
          <img src={logoUrl} width={22} height={22} alt="" />
          <span className="text-[14.6px] font-semibold tracking-[-0.15px]">Activity Detection</span>
        </Link>
        <Link to="/" className="text-[13px] hover:text-accent">
          New analysis
        </Link>
      </nav>
    </header>
  );
}

/** Centered page column below the header. */
export function Page({ children, className = '' }: { children: ReactNode; className?: string }) {
  return (
    <main className={`mx-auto flex max-w-[1280px] flex-col px-4 pt-7 pb-12 md:px-16 ${className}`}>
      {children}
    </main>
  );
}

export function Breadcrumb({ current }: { current: string }) {
  return (
    <nav aria-label="Breadcrumb" className="min-h-[26px] truncate py-[5px] text-[12.4px] leading-4">
      <Link to="/" className="text-accent hover:underline">
        Home
      </Link>
      <span className="text-muted">{' › '}</span>
      <span className="text-muted" aria-current="page">
        {current}
      </span>
    </nav>
  );
}

/** White rounded panel used throughout the design. */
export function Card({ className = '', ...props }: ComponentPropsWithoutRef<'section'>) {
  return <section className={`rounded-card bg-white ${className}`} {...props} />;
}

export function CardHeader({ title, note }: { title: string; note?: string }) {
  return (
    <div className="flex items-baseline justify-between gap-4">
      <h2 className="text-[21px] font-semibold tracking-[-0.21px]">{title}</h2>
      {note && <p className="text-right text-[12.5px] text-muted">{note}</p>}
    </div>
  );
}

// ---------- Pill buttons ----------

type Variant = 'primary' | 'secondary';

export function pillClassName(variant: Variant = 'primary', className = '') {
  const colors =
    variant === 'primary' ? 'bg-accent text-white hover:bg-[#0058b0]' : 'bg-fill text-ink hover:bg-[#dcdce1]';
  return `inline-flex h-11 shrink-0 items-center justify-center rounded-full px-[22px] text-[14px] font-medium whitespace-nowrap transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:cursor-not-allowed disabled:opacity-50 ${colors} ${className}`;
}

export function PillButton({
  variant,
  className,
  ...props
}: ComponentPropsWithoutRef<'button'> & { variant?: Variant }) {
  return <button type="button" className={pillClassName(variant, className)} {...props} />;
}

export function PillLink({ variant, className, ...props }: LinkProps & { variant?: Variant }) {
  return <Link className={pillClassName(variant, className)} {...props} />;
}

// ---------- Activities ----------

export function ActivityDot({ label, size = 8 }: { label: ActivityLabel | null; size?: number }) {
  return (
    <span
      aria-hidden
      className="inline-block shrink-0 rounded-full"
      style={{ width: size, height: size, backgroundColor: label ? getActivity(label).color : IDLE_COLOR }}
    />
  );
}

/** Colored dot followed by the activity's display name. */
export function ActivityName({ label }: { label: ActivityLabel }) {
  return (
    <span className="inline-flex items-center gap-2">
      <ActivityDot label={label} />
      {getActivity(label).name}
    </span>
  );
}

/**
 * Row of activity colors. `labels` defaults to the current 6 classes; `idleLabel` adds the grey
 * "First 32 frames" entry and `showUncertain` the "Uncertain" entry (extension 1).
 */
export function ActivityLegend({
  labels = ACTIVITIES.map((a) => a.label),
  idleLabel,
  showUncertain = false,
  className = '',
}: {
  labels?: ActivityLabel[];
  idleLabel?: string;
  showUncertain?: boolean;
  className?: string;
}) {
  return (
    <ul className={`flex flex-wrap gap-x-[22px] gap-y-2 text-[13px] ${className}`} aria-label="Activity colors">
      {(showUncertain ? [...labels, UNCERTAIN] : labels).map((label) => (
        <li key={label} className="flex items-center gap-[7px]">
          <ActivityDot label={label} />
          {getActivity(label).name}
        </li>
      ))}
      {idleLabel && (
        <li className="flex items-center gap-[7px]">
          <ActivityDot label={null} />
          {idleLabel}
        </li>
      )}
    </ul>
  );
}

// ---------- Stats and status ----------

/** Small grey label above a large value (use inside a <dl>). */
export function Stat({
  label,
  value,
  note,
  valueClassName = 'text-[28px]',
}: {
  label: string;
  value: ReactNode;
  note?: string;
  valueClassName?: string;
}) {
  return (
    <div className="flex min-w-0 flex-col gap-1">
      <dt className="text-[12px] text-muted">{label}</dt>
      <dd className={`font-semibold tracking-[-0.56px] ${valueClassName}`}>{value}</dd>
      {note && <dd className="text-[12px] text-muted">{note}</dd>}
    </div>
  );
}

/** Centered loading / error / empty state. */
export function StatusMessage({
  title,
  message,
  showHomeLink = false,
}: {
  title: string;
  message?: string;
  showHomeLink?: boolean;
}) {
  return (
    <div role="status" className="flex flex-col items-center gap-3 py-24 text-center">
      <h1 className="text-[24px] font-semibold tracking-[-0.24px]">{title}</h1>
      {message && <p className="max-w-[520px] text-[15px] text-muted">{message}</p>}
      {showHomeLink && (
        <PillLink to="/" variant="secondary" className="mt-2">
          Back to home
        </PillLink>
      )}
    </div>
  );
}
