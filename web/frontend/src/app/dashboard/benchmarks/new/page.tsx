"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { ArrowLeft, Play, X, Check, Cpu, Users, BarChart3 } from "lucide-react";
import Link from "next/link";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { benchmarkAPI } from "@/lib/api";

// API-based models from README
const MODELS = [
  // OpenAI
  { id: "gpt-5.2", name: "GPT-5.2", provider: "OpenAI" },
  { id: "gpt-5.1", name: "GPT-5.1", provider: "OpenAI" },
  { id: "gpt-5", name: "GPT-5", provider: "OpenAI" },
  { id: "gpt-4o", name: "GPT-4o", provider: "OpenAI" },
  // Anthropic
  { id: "claude-opus-4.5", name: "Claude Opus 4.5", provider: "Anthropic" },
  { id: "claude-sonnet-4.5", name: "Claude Sonnet 4.5", provider: "Anthropic" },
  { id: "claude-haiku-4.5", name: "Claude Haiku 4.5", provider: "Anthropic" },
  // Google
  { id: "gemini-3-pro", name: "Gemini 3 Pro", provider: "Google" },
  { id: "gemini-3-flash", name: "Gemini 3 Flash", provider: "Google" },
  // Meta
  { id: "llama-4-behemoth", name: "Llama 4 Behemoth", provider: "Meta" },
  { id: "llama-4-maverick", name: "Llama 4 Maverick", provider: "Meta" },
  { id: "llama-4-scout", name: "Llama 4 Scout", provider: "Meta" },
  // DeepSeek
  { id: "deepseek-v3", name: "DeepSeek-V3", provider: "DeepSeek" },
  // Alibaba
  { id: "qwen3-max", name: "Qwen3-Max", provider: "Alibaba" },
  // Amazon
  { id: "nova-pro", name: "Nova Pro", provider: "Amazon" },
  { id: "nova-omni", name: "Nova Omni", provider: "Amazon" },
];

// Target audiences from README
const AUDIENCES = [
  // Physicians
  { id: "physician_specialist", name: "Physician (Specialist)", category: "Physicians" },
  { id: "physician_generalist", name: "Physician (Generalist)", category: "Physicians" },
  // Nurses
  { id: "nurse_icu", name: "Nurse (ICU)", category: "Nurses" },
  { id: "nurse_general", name: "Nurse (General Ward)", category: "Nurses" },
  { id: "nurse_specialty", name: "Nurse (Specialty)", category: "Nurses" },
  // Patients
  { id: "patient_low_literacy", name: "Patient (Low Literacy)", category: "Patients" },
  { id: "patient_medium_literacy", name: "Patient (Medium Literacy)", category: "Patients" },
  { id: "patient_high_literacy", name: "Patient (High Literacy)", category: "Patients" },
  // Caregivers
  { id: "caregiver_family", name: "Caregiver (Family)", category: "Caregivers" },
  { id: "caregiver_professional", name: "Caregiver (Professional)", category: "Caregivers" },
  { id: "caregiver_pediatric", name: "Caregiver (Pediatric)", category: "Caregivers" },
];

// Medical evaluation dimensions from README
const DIMENSIONS = [
  { id: "factual_accuracy", name: "Factual Accuracy", weight: "25%", description: "Clinical correctness and evidence alignment" },
  { id: "terminological_appropriateness", name: "Terminological Appropriateness", weight: "15%", description: "Language complexity matching audience needs" },
  { id: "explanatory_completeness", name: "Explanatory Completeness", weight: "20%", description: "Comprehensive yet accessible coverage" },
  { id: "actionability", name: "Actionability", weight: "15%", description: "Clear, practical guidance" },
  { id: "safety", name: "Safety", weight: "15%", description: "Appropriate warnings and harm avoidance" },
  { id: "empathy_tone", name: "Empathy & Tone", weight: "10%", description: "Audience-appropriate communication style" },
];

export default function NewBenchmarkPage() {
  const router = useRouter();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [selectedAudiences, setSelectedAudiences] = useState<string[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const toggleModel = (modelId: string) => {
    setSelectedModels((prev) =>
      prev.includes(modelId)
        ? prev.filter((id) => id !== modelId)
        : [...prev, modelId]
    );
  };

  const toggleAudience = (audienceId: string) => {
    setSelectedAudiences((prev) =>
      prev.includes(audienceId)
        ? prev.filter((id) => id !== audienceId)
        : [...prev, audienceId]
    );
  };

  const handleSubmit = async () => {
    if (selectedModels.length === 0 || selectedAudiences.length === 0) {
      setError("Please select at least one model and one audience");
      return;
    }

    setIsSubmitting(true);
    setError(null);

    try {
      const result = await benchmarkAPI.create({
        name: name || `Medical Evaluation ${new Date().toLocaleDateString()}`,
        description,
        models: selectedModels,
        tasks: selectedAudiences, // audiences are the "tasks" for medical eval
      });

      router.push(`/dashboard/benchmarks/${result.id}`);
    } catch (error) {
      console.error("Failed to create evaluation:", error);
      setError(
        error instanceof Error ? error.message : "Failed to create evaluation"
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="mx-auto max-w-4xl space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <Link href="/dashboard/benchmarks">
          <Button variant="ghost" size="icon" className="h-8 w-8">
            <ArrowLeft className="h-4 w-4" />
          </Button>
        </Link>
        <div>
          <h1 className="text-xl font-semibold">New Medical Evaluation</h1>
          <p className="text-sm text-muted-foreground">
            Evaluate medical explanation quality across audiences
          </p>
        </div>
      </div>

      {/* Form */}
      <div className="space-y-6">
        {/* Basic Info */}
        <Card>
          <CardHeader className="pb-4">
            <CardTitle className="text-base">Evaluation Details</CardTitle>
            <CardDescription className="text-xs">
              Name and describe this evaluation run
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <label className="text-sm font-medium">Name</label>
              <Input
                placeholder="Diabetes Explanation Evaluation"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="mt-1.5"
              />
            </div>
            <div>
              <label className="text-sm font-medium">Description</label>
              <Input
                placeholder="Comparing model explanations of Type 2 diabetes across patient literacy levels..."
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                className="mt-1.5"
              />
            </div>
          </CardContent>
        </Card>

        {/* Model Selection */}
        <Card>
          <CardHeader className="pb-4">
            <div className="flex items-center gap-2">
              <Cpu className="h-4 w-4" />
              <CardTitle className="text-base">Select Models</CardTitle>
            </div>
            <CardDescription className="text-xs">
              {selectedModels.length} model{selectedModels.length !== 1 ? "s" : ""} selected
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
              {MODELS.map((model) => (
                <button
                  key={model.id}
                  onClick={() => toggleModel(model.id)}
                  className={`relative flex flex-col items-start rounded-md border p-3 text-left transition-colors ${selectedModels.includes(model.id)
                      ? "border-foreground bg-accent"
                      : "hover:bg-accent/50"
                    }`}
                >
                  {selectedModels.includes(model.id) && (
                    <div className="absolute right-2 top-2 rounded-full bg-foreground p-0.5">
                      <Check className="h-2.5 w-2.5 text-background" />
                    </div>
                  )}
                  <span className="font-medium text-sm">{model.name}</span>
                  <span className="mt-0.5 text-xs text-muted-foreground">
                    {model.provider}
                  </span>
                </button>
              ))}
            </div>

            {selectedModels.length > 0 && (
              <div className="mt-4 flex flex-wrap gap-1.5">
                {selectedModels.map((modelId) => (
                  <Badge key={modelId} variant="secondary" className="gap-1 text-xs">
                    {MODELS.find((m) => m.id === modelId)?.name || modelId}
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        toggleModel(modelId);
                      }}
                      className="ml-0.5 hover:text-destructive"
                    >
                      <X className="h-3 w-3" />
                    </button>
                  </Badge>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Audience Selection */}
        <Card>
          <CardHeader className="pb-4">
            <div className="flex items-center gap-2">
              <Users className="h-4 w-4" />
              <CardTitle className="text-base">Select Target Audiences</CardTitle>
            </div>
            <CardDescription className="text-xs">
              {selectedAudiences.length} audience{selectedAudiences.length !== 1 ? "s" : ""} selected
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
              {AUDIENCES.map((audience) => (
                <button
                  key={audience.id}
                  onClick={() => toggleAudience(audience.id)}
                  className={`relative flex flex-col items-start rounded-md border p-3 text-left transition-colors ${selectedAudiences.includes(audience.id)
                      ? "border-foreground bg-accent"
                      : "hover:bg-accent/50"
                    }`}
                >
                  {selectedAudiences.includes(audience.id) && (
                    <div className="absolute right-2 top-2 rounded-full bg-foreground p-0.5">
                      <Check className="h-2.5 w-2.5 text-background" />
                    </div>
                  )}
                  <span className="font-medium text-sm">{audience.name}</span>
                  <span className="mt-0.5 text-xs text-muted-foreground">
                    {audience.category}
                  </span>
                </button>
              ))}
            </div>

            {selectedAudiences.length > 0 && (
              <div className="mt-4 flex flex-wrap gap-1.5">
                {selectedAudiences.map((audienceId) => (
                  <Badge key={audienceId} variant="secondary" className="gap-1 text-xs">
                    {AUDIENCES.find((a) => a.id === audienceId)?.name || audienceId}
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        toggleAudience(audienceId);
                      }}
                      className="ml-0.5 hover:text-destructive"
                    >
                      <X className="h-3 w-3" />
                    </button>
                  </Badge>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Evaluation Dimensions Info */}
        <Card>
          <CardHeader className="pb-4">
            <div className="flex items-center gap-2">
              <BarChart3 className="h-4 w-4" />
              <CardTitle className="text-base">Evaluation Dimensions</CardTitle>
            </div>
            <CardDescription className="text-xs">
              All explanations are scored across 6 dimensions
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
              {DIMENSIONS.map((dim) => (
                <div
                  key={dim.id}
                  className="rounded-md border p-3"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-medium text-sm">{dim.name}</span>
                    <Badge variant="outline" className="text-xs">{dim.weight}</Badge>
                  </div>
                  <p className="mt-1 text-xs text-muted-foreground">
                    {dim.description}
                  </p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Error */}
        {error && (
          <div className="rounded-md border border-destructive/50 bg-destructive/10 p-3 text-sm text-destructive">
            {error}
          </div>
        )}

        {/* Submit */}
        <div className="flex justify-end gap-3">
          <Link href="/dashboard/benchmarks">
            <Button variant="outline" size="sm">Cancel</Button>
          </Link>
          <Button
            onClick={handleSubmit}
            disabled={
              isSubmitting ||
              selectedModels.length === 0 ||
              selectedAudiences.length === 0
            }
            size="sm"
            className="gap-2"
          >
            {isSubmitting ? (
              <>
                <div className="h-3 w-3 animate-spin rounded-full border-2 border-current border-t-transparent" />
                Creating...
              </>
            ) : (
              <>
                <Play className="h-3 w-3" />
                Start Evaluation
              </>
            )}
          </Button>
        </div>
      </div>
    </div>
  );
}
