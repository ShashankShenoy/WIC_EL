import React from 'react'
import { cn } from '@/lib/utils'

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'default' | 'destructive' | 'outline' | 'secondary' | 'ghost'
  size?: 'default' | 'sm' | 'lg' | 'icon'
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = 'default', size = 'default', ...props }, ref) => {
    return (
      <button
        className={cn(
          "inline-flex items-center justify-center rounded-lg text-sm font-medium transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-500 focus-visible:ring-offset-2 focus-visible:ring-offset-black disabled:pointer-events-none disabled:opacity-50 active:scale-95",
          {
            'bg-zinc-700 text-white hover:bg-zinc-600 border border-zinc-600': variant === 'default',
            'bg-red-500 text-white hover:bg-red-600': variant === 'destructive',
            'border border-zinc-700 bg-zinc-900 hover:bg-zinc-800 text-zinc-300': variant === 'outline',
            'bg-zinc-800 text-zinc-300 hover:bg-zinc-700': variant === 'secondary',
            'hover:bg-zinc-900 text-zinc-400': variant === 'ghost',
          },
          {
            'h-10 px-5 py-2': size === 'default',
            'h-8 rounded px-3 text-xs': size === 'sm',
            'h-12 rounded-lg px-8': size === 'lg',
            'h-9 w-9': size === 'icon',
          },
          className
        )}
        ref={ref}
        {...props}
      />
    )
  }
)
Button.displayName = 'Button'

export { Button }

