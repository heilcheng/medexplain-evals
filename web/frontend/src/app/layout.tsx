import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "MedExplain-Evals - Medical Explanation Quality Benchmark",
  description: "Benchmark for evaluating audience-adaptive medical explanation quality in LLMs",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body className="min-h-screen bg-background antialiased">
        {children}
      </body>
    </html>
  );
}
