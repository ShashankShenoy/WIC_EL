import { Loader2 } from 'lucide-react'

interface LoadingSpinnerProps {
  text?: string
  size?: 'sm' | 'md' | 'lg'
}

export function LoadingSpinner({ text = 'Loading...', size = 'md' }: LoadingSpinnerProps) {
  const sizeClasses = {
    sm: 'h-4 w-4',
    md: 'h-8 w-8',
    lg: 'h-12 w-12'
  }

  return (
    <div className="flex flex-col items-center justify-center gap-3 p-8">
      <Loader2 className={`${sizeClasses[size]} animate-spin text-zinc-400`} />
      {text && (
        <p className="text-sm font-medium text-zinc-500 animate-pulse">{text}</p>
      )}
    </div>
  )
}

export function LoadingOverlay({ text }: { text?: string }) {
  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50">
      <div className="bg-zinc-950 border border-zinc-800 rounded-xl shadow-2xl p-8">
        <LoadingSpinner text={text} size="lg" />
      </div>
    </div>
  )
}
