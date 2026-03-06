import React from "react"

export function Card({ children, className }: { children: React.ReactNode, className?: string }) {
    return <div className={`bg-white rounded-2xl border border-neutral-200 overflow-hidden ${className}`}>{children}</div>
}

export function CardHeader({ children, className }: { children: React.ReactNode, className?: string }) {
    return <div className={`p-6 border-b border-neutral-100 ${className}`}>{children}</div>
}

export function CardTitle({ children, className }: { children: React.ReactNode, className?: string }) {
    return <h3 className={`text-xl font-bold text-neutral-900 ${className}`}>{children}</h3>
}

export function CardDescription({ children, className }: { children: React.ReactNode, className?: string }) {
    return <p className={`text-sm text-neutral-500 mt-1 ${className}`}>{children}</p>
}

export function CardContent({ children, className }: { children: React.ReactNode, className?: string }) {
    return <div className={`p-6 ${className}`}>{children}</div>
}
