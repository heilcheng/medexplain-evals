"use client";

import {
  Stethoscope,
  Heart,
  User,
  Shield,
  BookOpen,
} from "lucide-react";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

// Target audiences from README with full details
const AUDIENCE_GROUPS = [
  {
    category: "Physicians",
    icon: Stethoscope,
    healthLiteracy: "Expert",
    audiences: [
      { id: "physician_specialist", name: "Specialist", description: "Domain experts requiring technical precision and clinical terminology" },
      { id: "physician_generalist", name: "Generalist", description: "Primary care physicians needing broad medical explanations" },
    ],
  },
  {
    category: "Nurses",
    icon: Heart,
    healthLiteracy: "Professional",
    audiences: [
      { id: "nurse_icu", name: "ICU", description: "Critical care nurses needing detailed physiological information" },
      { id: "nurse_general", name: "General Ward", description: "Floor nurses requiring practical care instructions" },
      { id: "nurse_specialty", name: "Specialty", description: "Specialized nurses (oncology, pediatric, etc.)" },
    ],
  },
  {
    category: "Patients",
    icon: User,
    healthLiteracy: "Variable",
    audiences: [
      { id: "patient_low_literacy", name: "Low Literacy", description: "Grade 5-6 reading level, minimal medical knowledge" },
      { id: "patient_medium_literacy", name: "Medium Literacy", description: "Grade 8-10 reading level, basic health understanding" },
      { id: "patient_high_literacy", name: "High Literacy", description: "College level, comfortable with medical terminology" },
    ],
  },
  {
    category: "Caregivers",
    icon: Shield,
    healthLiteracy: "Variable",
    audiences: [
      { id: "caregiver_family", name: "Family", description: "Family members caring for loved ones at home" },
      { id: "caregiver_professional", name: "Professional", description: "Home health aides and professional caregivers" },
      { id: "caregiver_pediatric", name: "Pediatric", description: "Parents/guardians caring for children" },
    ],
  },
];

// Evaluation dimensions from README
const DIMENSIONS = [
  { name: "Factual Accuracy", weight: "25%", description: "Clinical correctness and evidence alignment" },
  { name: "Terminological Appropriateness", weight: "15%", description: "Language complexity matching audience needs" },
  { name: "Explanatory Completeness", weight: "20%", description: "Comprehensive yet accessible coverage" },
  { name: "Actionability", weight: "15%", description: "Clear, practical guidance" },
  { name: "Safety", weight: "15%", description: "Appropriate warnings and harm avoidance" },
  { name: "Empathy & Tone", weight: "10%", description: "Audience-appropriate communication style" },
];

export default function AudiencesPage() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Target Audiences</h1>
        <p className="text-sm text-muted-foreground mt-1">
          Medical explanation personas for audience-adaptive evaluation
        </p>
      </div>

      {/* Audience Groups */}
      <div className="grid gap-6 lg:grid-cols-2">
        {AUDIENCE_GROUPS.map((group) => (
          <Card key={group.category}>
            <CardHeader className="pb-3">
              <div className="flex items-start gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-md border bg-background">
                  <group.icon className="h-5 w-5" />
                </div>
                <div className="flex-1 min-w-0">
                  <CardTitle className="text-base">{group.category}</CardTitle>
                  <div className="flex items-center gap-2 mt-1">
                    <Badge variant="outline" className="text-xs">
                      {group.healthLiteracy} Literacy
                    </Badge>
                  </div>
                </div>
              </div>
            </CardHeader>
            <CardContent className="pt-0">
              <div className="space-y-3">
                {group.audiences.map((audience) => (
                  <div
                    key={audience.id}
                    className="rounded-md border p-3"
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-medium text-sm">{audience.name}</span>
                      <code className="text-xs bg-muted px-1.5 py-0.5 rounded">
                        {audience.id}
                      </code>
                    </div>
                    <p className="mt-1 text-xs text-muted-foreground">
                      {audience.description}
                    </p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Evaluation Framework */}
      <Card>
        <CardHeader className="pb-4">
          <CardTitle className="text-base">Evaluation Dimensions</CardTitle>
          <CardDescription className="text-xs">
            Each explanation is scored across 6 weighted dimensions
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {DIMENSIONS.map((dim) => (
              <div
                key={dim.name}
                className="rounded-md border p-3"
              >
                <div className="flex items-center justify-between">
                  <span className="font-medium text-sm">{dim.name}</span>
                  <Badge variant="secondary" className="text-xs">{dim.weight}</Badge>
                </div>
                <p className="mt-1 text-xs text-muted-foreground">
                  {dim.description}
                </p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Info Section */}
      <Card>
        <CardContent className="flex items-start gap-4 p-5">
          <div className="rounded-md border bg-background p-2">
            <BookOpen className="h-5 w-5" />
          </div>
          <div className="flex-1">
            <h3 className="font-medium text-sm">About Audience-Adaptive Evaluation</h3>
            <p className="text-sm text-muted-foreground mt-1">
              MedExplain-Evals measures how well LLMs adapt their medical explanations
              for different audiences. Each audience persona represents a distinct
              health literacy level and communication need.
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
