package com.medlens.app.viewmodel

import android.net.Uri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.medlens.app.inference.BiomedClipInference
import com.medlens.app.inference.ClinicalPipeline
import com.medlens.app.inference.MedGemmaInference
import com.medlens.app.inference.MedicalClassifier
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

class ReportViewModel(
    private val biomedClip: BiomedClipInference,
    private val medGemma: MedGemmaInference
) : ViewModel() {

    data class UiState(
        val selectedImageUri: Uri? = null,
        val stage: ClinicalPipeline.Stage = ClinicalPipeline.Stage.IDLE,
        val clipTimeMs: Float = 0f,
        val classifications: List<MedicalClassifier.ClassificationResult> = emptyList(),
        val classificationSummary: String = "",
        val streamingText: String = "",
        val tokPerSec: Float = 0f,
        val tokensGenerated: Int = 0,
        val finalReport: String = "",
        val error: String? = null
    )

    private val _uiState = MutableStateFlow(UiState())
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()

    private var pipelineJob: Job? = null
    private var pollJob: Job? = null

    fun onImageSelected(uri: Uri) {
        _uiState.value = UiState(selectedImageUri = uri)
    }

    fun runPipeline() {
        val uri = _uiState.value.selectedImageUri ?: return
        if (!biomedClip.isLoaded || !medGemma.isLoaded) {
            _uiState.value = _uiState.value.copy(error = "Models not loaded")
            return
        }

        // Start polling for MedGemma streaming
        pollJob = viewModelScope.launch(Dispatchers.IO) {
            // Wait until we're in the report generation stage
            while (isActive) {
                delay(200)
                if (_uiState.value.stage == ClinicalPipeline.Stage.GENERATING_REPORT) {
                    val state = medGemma.getPartialResult()
                    _uiState.value = _uiState.value.copy(
                        streamingText = state.text,
                        tokPerSec = state.tokPerSec,
                        tokensGenerated = state.tokensGenerated
                    )
                }
                if (_uiState.value.stage == ClinicalPipeline.Stage.COMPLETE ||
                    _uiState.value.stage == ClinicalPipeline.Stage.ERROR) {
                    break
                }
            }
        }

        pipelineJob = viewModelScope.launch(Dispatchers.IO) {
            try {
                // Stage 1: BiomedCLIP
                _uiState.value = _uiState.value.copy(
                    stage = ClinicalPipeline.Stage.ANALYZING_IMAGE,
                    error = null
                )
                val clipResult = biomedClip.getEmbedding(uri)

                // Stage 2: Classification (using built-in conditions for now)
                _uiState.value = _uiState.value.copy(
                    stage = ClinicalPipeline.Stage.CLASSIFYING,
                    clipTimeMs = clipResult.inferenceTimeMs
                )

                // For demo: generate a basic classification summary from embedding norms
                val embNorm = clipResult.embedding.map { it * it }.sum()
                val summary = "Medical image analyzed (embedding norm: ${"%.2f".format(embNorm)})"

                _uiState.value = _uiState.value.copy(
                    classificationSummary = summary
                )

                // Stage 3: MedGemma report generation
                _uiState.value = _uiState.value.copy(
                    stage = ClinicalPipeline.Stage.GENERATING_REPORT
                )

                val prompt = buildPrompt(summary, clipResult.inferenceTimeMs)
                val report = medGemma.generate(prompt, 256)

                // Clean report (remove stats header)
                val cleanReport = if (report.contains("]\n\n")) {
                    report.substringAfter("]\n\n")
                } else report

                pollJob?.cancel()
                _uiState.value = _uiState.value.copy(
                    stage = ClinicalPipeline.Stage.COMPLETE,
                    finalReport = cleanReport
                )

            } catch (e: Exception) {
                pollJob?.cancel()
                _uiState.value = _uiState.value.copy(
                    stage = ClinicalPipeline.Stage.ERROR,
                    error = e.message ?: "Unknown error"
                )
            }
        }
    }

    fun stopGeneration() {
        medGemma.stopGeneration()
    }

    fun reset() {
        pipelineJob?.cancel()
        pollJob?.cancel()
        _uiState.value = UiState()
    }

    private fun buildPrompt(summary: String, clipTimeMs: Float): String {
        return """You are a medical AI assistant. A medical image has been analyzed by BiomedCLIP (a biomedical vision model) in ${"%.0f".format(clipTimeMs)}ms on-device.

Analysis result: $summary

Based on this on-device AI analysis, provide a concise structured clinical report:

1. **Image Assessment**: Describe what the analysis indicates
2. **Clinical Considerations**: Key medical considerations
3. **Recommended Next Steps**: Suggested follow-up actions
4. **AI Disclaimer**: This is an automated AI analysis for educational/research purposes only

Keep the report professional, concise, and medically relevant."""
    }

    class Factory(
        private val biomedClip: BiomedClipInference,
        private val medGemma: MedGemmaInference
    ) : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>): T {
            return ReportViewModel(biomedClip, medGemma) as T
        }
    }
}
