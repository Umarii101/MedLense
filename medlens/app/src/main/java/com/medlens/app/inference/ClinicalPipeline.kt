package com.medlens.app.inference

import android.net.Uri
import android.util.Log

/**
 * Combined BiomedCLIP → MedGemma clinical pipeline.
 *
 * Flow:
 *   1. Image → BiomedCLIP → 512-dim embedding
 *   2. Embedding → MedicalClassifier → top conditions
 *   3. Conditions → auto-prompt → MedGemma
 *   4. MedGemma → structured clinical report (streaming)
 */
class ClinicalPipeline(
    private val biomedClip: BiomedClipInference,
    private val medGemma: MedGemmaInference,
    private val classifier: MedicalClassifier
) {
    companion object {
        private const val TAG = "ClinicalPipeline"
    }

    data class PipelineState(
        val stage: Stage,
        val clipTimeMs: Float = 0f,
        val classifications: List<MedicalClassifier.ClassificationResult> = emptyList(),
        val classificationSummary: String = "",
        val report: String = "",
        val error: String? = null
    )

    enum class Stage {
        IDLE,
        ANALYZING_IMAGE,
        CLASSIFYING,
        GENERATING_REPORT,
        COMPLETE,
        ERROR
    }

    /**
     * Run the full pipeline. Returns the final structured report.
     *
     * @param imageUri URI of the medical image to analyze
     * @param onStateUpdate Callback for each pipeline stage change
     * @return Final report string
     */
    fun runPipeline(
        imageUri: Uri,
        maxTokens: Int = 256,
        onStateUpdate: (PipelineState) -> Unit = {}
    ): String {
        try {
            // Stage 1: BiomedCLIP embedding
            onStateUpdate(PipelineState(Stage.ANALYZING_IMAGE))
            Log.i(TAG, "Stage 1: Running BiomedCLIP...")

            val clipResult = biomedClip.getEmbedding(imageUri)
            Log.i(TAG, "BiomedCLIP done: ${clipResult.inferenceTimeMs}ms")

            // Stage 2: Classification
            onStateUpdate(PipelineState(
                Stage.CLASSIFYING,
                clipTimeMs = clipResult.inferenceTimeMs
            ))
            Log.i(TAG, "Stage 2: Classifying...")

            val (classifications, summary) = classifier.classifyWithThreshold(
                clipResult.embedding
            )
            Log.i(TAG, "Classification: $summary")

            // Stage 3: Generate MedGemma prompt and run
            val prompt = buildMedGemmaPrompt(classifications, summary)
            Log.i(TAG, "Stage 3: Generating report...")

            onStateUpdate(PipelineState(
                Stage.GENERATING_REPORT,
                clipTimeMs = clipResult.inferenceTimeMs,
                classifications = classifications,
                classificationSummary = summary
            ))

            val report = medGemma.generate(prompt, maxTokens)

            // Stage 4: Complete
            val finalState = PipelineState(
                Stage.COMPLETE,
                clipTimeMs = clipResult.inferenceTimeMs,
                classifications = classifications,
                classificationSummary = summary,
                report = report
            )
            onStateUpdate(finalState)

            Log.i(TAG, "Pipeline complete")
            return report

        } catch (e: Exception) {
            Log.e(TAG, "Pipeline error: ${e.message}", e)
            val errorState = PipelineState(
                Stage.ERROR,
                error = e.message ?: "Unknown error"
            )
            onStateUpdate(errorState)
            return "[Error: ${e.message}]"
        }
    }

    /**
     * Build a structured prompt for MedGemma based on BiomedCLIP classification.
     */
    private fun buildMedGemmaPrompt(
        classifications: List<MedicalClassifier.ClassificationResult>,
        summary: String
    ): String {
        val topFindings = classifications.take(3).joinToString("\n") { result ->
            "- ${result.condition} (${result.category}): " +
            "${"%.1f".format(result.confidence * 100)}% confidence"
        }

        return """You are a medical AI assistant analyzing a clinical image. An image analysis model has detected the following findings:

Image Analysis Summary: $summary

Top findings:
$topFindings

Based on these automated findings, provide a brief structured clinical report covering:
1. **Primary Finding**: Most likely condition and confidence assessment
2. **Clinical Significance**: Why this finding matters
3. **Recommended Actions**: Suggested next steps
4. **Important Limitations**: Remind that this is AI-assisted and requires clinical verification

Keep the report concise and clinically relevant. Use professional medical terminology."""
    }
}
