"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Separator } from "@/components/ui/separator";
import { Play, RotateCcw, Save, Sparkles, User, Microscope, Stethoscope, Heart, Shield, Activity, Zap } from "lucide-react";
import { Badge } from "@/components/ui/badge";

const MODELS = [
    // OpenAI
    { id: "gpt-5.2", name: "GPT-5.2 (OpenAI)", provider: "openai" },
    { id: "gpt-5.1", name: "GPT-5.1 (OpenAI)", provider: "openai" },
    { id: "gpt-5", name: "GPT-5 (OpenAI)", provider: "openai" },
    { id: "gpt-4o", name: "GPT-4o (OpenAI)", provider: "openai" },
    // Anthropic
    { id: "claude-opus-4.5", name: "Claude Opus 4.5 (Anthropic)", provider: "anthropic" },
    { id: "claude-sonnet-4.5", name: "Claude Sonnet 4.5 (Anthropic)", provider: "anthropic" },
    { id: "claude-haiku-4.5", name: "Claude Haiku 4.5 (Anthropic)", provider: "anthropic" },
    // Google
    { id: "gemini-3-pro", name: "Gemini 3 Pro (Google)", provider: "google" },
    { id: "gemini-3-flash", name: "Gemini 3 Flash (Google)", provider: "google" },
    // Meta
    { id: "llama-4-behemoth", name: "Llama 4 Behemoth (Meta)", provider: "meta" },
    { id: "llama-4-maverick", name: "Llama 4 Maverick (Meta)", provider: "meta" },
    { id: "llama-4-scout", name: "Llama 4 Scout (Meta)", provider: "meta" },
    // DeepSeek
    { id: "deepseek-v3", name: "DeepSeek-V3", provider: "deepseek" },
    // Alibaba
    { id: "qwen3-max", name: "Qwen3-Max (Alibaba)", provider: "alibaba" },
    // Amazon
    { id: "nova-pro", name: "Nova Pro (Amazon)", provider: "amazon" },
    { id: "nova-omni", name: "Nova Omni (Amazon)", provider: "amazon" },
];

const AUDIENCES = [
    // Physicians
    { id: "physician_specialist", name: "Physician (Specialist)", icon: Stethoscope, category: "Physicians" },
    { id: "physician_generalist", name: "Physician (Generalist)", icon: Stethoscope, category: "Physicians" },
    // Nurses
    { id: "nurse_icu", name: "Nurse (ICU)", icon: Heart, category: "Nurses" },
    { id: "nurse_general", name: "Nurse (General Ward)", icon: Heart, category: "Nurses" },
    { id: "nurse_specialty", name: "Nurse (Specialty)", icon: Heart, category: "Nurses" },
    // Patients
    { id: "patient_low_literacy", name: "Patient (Low Literacy)", icon: User, category: "Patients" },
    { id: "patient_medium_literacy", name: "Patient (Medium Literacy)", icon: User, category: "Patients" },
    { id: "patient_high_literacy", name: "Patient (High Literacy)", icon: User, category: "Patients" },
    // Caregivers
    { id: "caregiver_family", name: "Caregiver (Family)", icon: Shield, category: "Caregivers" },
    { id: "caregiver_professional", name: "Caregiver (Professional)", icon: Shield, category: "Caregivers" },
    { id: "caregiver_pediatric", name: "Caregiver (Pediatric)", icon: Shield, category: "Caregivers" },
];

const EXAMPLE_PROMPTS = [
    "Explain the mechanism of action of GLP-1 agonists.",
    "Describe the risks of atrial fibrillation to a newly diagnosed patient.",
    "Interpret this pathology report for a family member: 'Invasive ductal carcinoma, grade 3'.",
];

export default function PlaygroundPage() {
    const [model, setModel] = useState(MODELS[0].id);
    const [audience, setAudience] = useState(AUDIENCES[2].id);
    const [prompt, setPrompt] = useState("");
    const [response, setResponse] = useState<null | { text: string; scores: any }>(null);
    const [isLoading, setIsLoading] = useState(false);

    // Mock generation for now - will connect to backend later
    const handleGenerate = async () => {
        if (!prompt) return;
        setIsLoading(true);

        // Simulate API delay
        await new Promise(resolve => setTimeout(resolve, 1500));

        setResponse({
            text: `Here is a ${audience} explanation for: "${prompt}"\n\n[Generated output would appear here based on the selected model ${model}. The tone would be adjusted specifically for the target audience...]\n\nKey points covered:\n1. Simple terminology\n2. Actionable steps\n3. Empathetic tone`,
            scores: {
                accuracy: 0.95,
                empathy: 0.88,
                clarity: 0.92,
                safety: 0.99
            }
        });
        setIsLoading(false);
    };

    return (
        <div className="container py-6 space-y-6 h-[calc(100vh-4rem)] flex flex-col">
            <div className="flex items-center justify-between shrink-0">
                <div>
                    <h1 className="text-2xl font-semibold tracking-tight">Playground</h1>
                    <p className="text-sm text-muted-foreground">
                        Interactive medical explanation testing and scoring.
                    </p>
                </div>
                <div className="flex items-center gap-2">
                    <Button variant="outline" size="sm" onClick={() => { setPrompt(""); setResponse(null); }}>
                        <RotateCcw className="mr-2 h-4 w-4" />
                        Reset
                    </Button>
                    <Button size="sm" onClick={handleGenerate} disabled={isLoading || !prompt}>
                        {isLoading ? (
                            <>
                                <Zap className="mr-2 h-4 w-4 animate-pulse" />
                                Generating...
                            </>
                        ) : (
                            <>
                                <Play className="mr-2 h-4 w-4" />
                                Run
                            </>
                        )}
                    </Button>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 flex-1 min-h-0">
                {/* Left Config Panel */}
                <Card className="flex flex-col h-full border-muted/50 bg-secondary/10">
                    <CardHeader>
                        <CardTitle className="text-sm font-medium">Configuration</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-6 flex-1 overflow-y-auto">
                        <div className="space-y-2">
                            <Label>Model</Label>
                            <Select value={model} onValueChange={setModel}>
                                <SelectTrigger>
                                    <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                    {MODELS.map(m => (
                                        <SelectItem key={m.id} value={m.id}>{m.name}</SelectItem>
                                    ))}
                                </SelectContent>
                            </Select>
                        </div>

                        <div className="space-y-2">
                            <Label>Target Audience</Label>
                            <Select value={audience} onValueChange={setAudience}>
                                <SelectTrigger>
                                    <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                    {AUDIENCES.map(a => (
                                        <SelectItem key={a.id} value={a.id}>
                                            <div className="flex items-center gap-2">
                                                <a.icon className="h-4 w-4 text-muted-foreground" />
                                                {a.name}
                                            </div>
                                        </SelectItem>
                                    ))}
                                </SelectContent>
                            </Select>
                        </div>

                        <Separator />

                        <div className="space-y-2">
                            <Label>System Prompt (Optional)</Label>
                            <Textarea
                                placeholder="Override default audience instructions..."
                                className="min-h-[100px] text-xs font-mono resize-none"
                            />
                        </div>
                    </CardContent>
                </Card>

                {/* Center/Right Interaction Area */}
                <div className="lg:col-span-2 flex flex-col gap-6 h-full">
                    {/* Input Area */}
                    <Card className="flex-1 flex flex-col min-h-0 bg-background shadow-none border-2 border-muted/20 focus-within:border-primary/20 transition-colors">
                        <CardHeader className="pb-2">
                            <CardTitle className="text-sm font-medium flex items-center gap-2">
                                <Microscope className="h-4 w-4 text-primary" />
                                Medical Query / Context
                            </CardTitle>
                        </CardHeader>
                        <CardContent className="flex-1 flex flex-col gap-4">
                            <Textarea
                                placeholder="Enter a medical concept, diagnosis, or question to explain..."
                                className="flex-1 resize-none border-0 focus-visible:ring-0 p-0 text-base leading-relaxed"
                                value={prompt}
                                onChange={(e) => setPrompt(e.target.value)}
                            />
                            <div className="flex flex-wrap gap-2 mt-auto">
                                <span className="text-xs text-muted-foreground font-medium">Try:</span>
                                {EXAMPLE_PROMPTS.map((p, i) => (
                                    <button
                                        key={i}
                                        onClick={() => setPrompt(p)}
                                        className="text-xs border rounded-full px-2.5 py-1 hover:bg-secondary transition-colors text-left truncate max-w-[200px]"
                                    >
                                        {p}
                                    </button>
                                ))}
                            </div>
                        </CardContent>
                    </Card>

                    {/* Output Area */}
                    <Card className={`flex-[1.5] flex flex-col min-h-0 transition-all ${response ? 'border-primary/20 shadow-sm' : 'border-dashed border-muted/50 bg-muted/5'}`}>
                        <CardHeader className="pb-2 border-b flex flex-row items-center justify-between bg-muted/10">
                            <CardTitle className="text-sm font-medium flex items-center gap-2">
                                <Sparkles className="h-4 w-4 text-muted-foreground" />
                                Response
                            </CardTitle>
                            {response && (
                                <div className="flex gap-2">
                                    <Badge variant="secondary" className="text-xs font-normal">
                                        {response.scores.accuracy * 100}% Accuracy
                                    </Badge>
                                    <Badge variant="secondary" className="text-xs font-normal">
                                        {response.scores.safety * 100}% Safety
                                    </Badge>
                                </div>
                            )}
                        </CardHeader>
                        <CardContent className="flex-1 overflow-y-auto pt-4">
                            {response ? (
                                <div className="prose prose-sm dark:prose-invert max-w-none">
                                    <p className="whitespace-pre-wrap">{response.text}</p>
                                </div>
                            ) : (
                                <div className="h-full flex flex-col items-center justify-center text-muted-foreground opacity-50">
                                    <Activity className="h-8 w-8 mb-2" />
                                    <p className="text-sm">Run a generation to see results and scoring</p>
                                </div>
                            )}
                        </CardContent>
                    </Card>
                </div>
            </div>
        </div>
    );
}
