import React from 'react'
import { cn } from '../../lib/utils'

interface TabsProps {
  children: React.ReactNode
  defaultValue: string
  value?: string
  onValueChange?: (value: string) => void
  className?: string
}

interface TabsListProps {
  children: React.ReactNode
  className?: string
}

interface TabsTriggerProps {
  children: React.ReactNode
  value: string
  className?: string
}

interface TabsContentProps {
  children: React.ReactNode
  value: string
  className?: string
}

const TabsContext = React.createContext<{
  value: string
  onValueChange: (value: string) => void
}>({ value: '', onValueChange: () => {} })

export function Tabs({ children, defaultValue, value: controlledValue, onValueChange, className }: TabsProps) {
  const [internalValue, setInternalValue] = React.useState(defaultValue)
  const value = controlledValue !== undefined ? controlledValue : internalValue
  const handleValueChange = onValueChange || setInternalValue

  return (
    <TabsContext.Provider value={{ value, onValueChange: handleValueChange }}>
      <div className={cn('w-full', className)}>{children}</div>
    </TabsContext.Provider>
  )
}

export function TabsList({ children, className }: TabsListProps) {
  return (
    <div className={cn('inline-flex h-10 items-center justify-start rounded-lg bg-zinc-900 p-1 text-zinc-400', className)}>
      {children}
    </div>
  )
}

export function TabsTrigger({ children, value, className }: TabsTriggerProps) {
  const { value: selectedValue, onValueChange } = React.useContext(TabsContext)
  const isActive = value === selectedValue

  return (
    <button
      onClick={() => onValueChange(value)}
      className={cn(
        'inline-flex items-center justify-center whitespace-nowrap rounded-md px-3 py-1.5 text-sm font-medium transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-500 disabled:pointer-events-none disabled:opacity-50',
        isActive
          ? 'bg-zinc-800 text-white shadow-sm'
          : 'text-zinc-400 hover:bg-zinc-800/50 hover:text-zinc-300',
        className
      )}
    >
      {children}
    </button>
  )
}

export function TabsContent({ children, value, className }: TabsContentProps) {
  const { value: selectedValue } = React.useContext(TabsContext)
  
  if (value !== selectedValue) return null

  return (
    <div className={cn('mt-6 focus-visible:outline-none', className)}>
      {children}
    </div>
  )
}
