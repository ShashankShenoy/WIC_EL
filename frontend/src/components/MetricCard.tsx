import React from 'react'
import { cn } from '@/lib/utils'

interface MetricCardProps {
  title: string
  value: string | number
  subtitle?: string
  delta?: {
    value: number
    isPositive: boolean
  }
  icon?: React.ReactNode
  className?: string
}

export function MetricCard({ title, value, subtitle, delta, icon, className }: MetricCardProps) {
  return (
    <div className={cn(
      "group rounded-lg border border-zinc-800 bg-zinc-950 p-5 transition-all duration-200",
      className
    )}>
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <p className="text-sm font-medium text-zinc-500 mb-1">{title}</p>
          <div className="flex items-baseline gap-2">
            <p className="text-3xl font-semibold text-white">
              {value}
            </p>
            {delta && (
              <span className={cn(
                "flex items-center gap-0.5 text-xs font-medium px-2 py-0.5 rounded bg-zinc-800 text-zinc-400"
              )}>
                <span className="text-sm">{delta.isPositive ? '↑' : '↓'}</span>
                {Math.abs(delta.value).toFixed(1)}%
              </span>
            )}
          </div>
          {subtitle && (
            <p className="mt-1 text-xs text-zinc-500">{subtitle}</p>
          )}
        </div>
        {icon && (
          <div className="flex-shrink-0 p-3 bg-zinc-900 border border-zinc-800 rounded-lg text-zinc-500 transition-transform">
            {icon}
          </div>
        )}
      </div>
    </div>
  )
}

