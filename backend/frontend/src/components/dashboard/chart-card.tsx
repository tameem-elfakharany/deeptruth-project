"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

export function ChartCard({
  title,
  children,
  right
}: {
  title: string
  right?: React.ReactNode
  children: React.ReactNode
}) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0">
        <CardTitle>{title}</CardTitle>
        {right}
      </CardHeader>
      <CardContent className="h-[300px]">{children}</CardContent>
    </Card>
  )
}

