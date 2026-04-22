"use client"

import { Topbar } from "@/components/layout/topbar"
import { ComingSoon } from "@/components/coming-soon/coming-soon"

export default function FusionPage() {
  return (
    <div>
      <Topbar title="Multimodal Fusion" subtitle="Milestone 4 placeholder for unified decision support." />
      <div className="p-6">
        <ComingSoon
          title="Multimodal Fusion Console"
          milestone="Milestone 4"
          description="Planned fusion layer that combines image, audio, and video signals into a unified, explainable decision view."
          planned={[
            "Unified decision score combining modalities",
            "Modality-weight explainability and confidence calibration",
            "Case timeline with synchronized evidence",
            "Dashboard-level risk scoring for content collections"
          ]}
        />
      </div>
    </div>
  )
}

