"use client"

import { PolarAngleAxis, PolarGrid, Radar, RadarChart as RechartsRadarChart, ResponsiveContainer } from "recharts"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"

interface RadarChartProps {
    data: {
        subject: string
        A: number
        fullMark: number
    }[]
    title?: string
    description?: string
    color?: string
}

export function RadarChart({
    data,
    title = "Evaluation Dimensions",
    description = "Performance across 6 key metrics",
    color = "hsl(var(--primary))"
}: RadarChartProps) {
    return (
        <Card className="h-full border-none shadow-none bg-transparent">
            <CardHeader className="pb-4 items-center">
                <CardTitle className="text-sm font-medium">{title}</CardTitle>
                <CardDescription className="text-xs">{description}</CardDescription>
            </CardHeader>
            <CardContent className="pb-0">
                <div className="h-[250px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                        <RechartsRadarChart cx="50%" cy="50%" outerRadius="70%" data={data}>
                            <PolarGrid stroke="hsl(var(--muted-foreground))" strokeOpacity={0.2} />
                            <PolarAngleAxis
                                dataKey="subject"
                                tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 10 }}
                            />
                            <Radar
                                name="Score"
                                dataKey="A"
                                stroke={color}
                                fill={color}
                                fillOpacity={0.3}
                            />
                        </RechartsRadarChart>
                    </ResponsiveContainer>
                </div>
            </CardContent>
        </Card>
    )
}
