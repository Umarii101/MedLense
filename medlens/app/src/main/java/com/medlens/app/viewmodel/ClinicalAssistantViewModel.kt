package com.medlens.app.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.medlens.app.inference.MedGemmaInference
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

class ClinicalAssistantViewModel(
    private val medGemma: MedGemmaInference
) : ViewModel() {

    data class Message(
        val text: String,
        val isUser: Boolean,
        val tokPerSec: Float = 0f,
        val tokensGenerated: Int = 0
    )

    data class UiState(
        val messages: List<Message> = emptyList(),
        val currentInput: String = "",
        val isGenerating: Boolean = false,
        val streamingText: String = "",
        val tokPerSec: Float = 0f,
        val tokensGenerated: Int = 0,
        val error: String? = null
    )

    private val _uiState = MutableStateFlow(UiState())
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()

    private var generateJob: Job? = null
    private var pollJob: Job? = null

    val suggestedPrompts = listOf(
        "Describe common signs of pneumonia on a chest X-ray.",
        "What are the differential diagnoses for bilateral pleural effusions?",
        "Explain the clinical significance of cardiomegaly.",
        "What features distinguish benign from malignant skin lesions?",
        "Describe the stages of diabetic retinopathy."
    )

    fun onInputChanged(text: String) {
        _uiState.value = _uiState.value.copy(currentInput = text)
    }

    fun sendMessage(text: String? = null) {
        val prompt = text ?: _uiState.value.currentInput.trim()
        if (prompt.isBlank() || !medGemma.isLoaded) return

        val userMessage = Message(prompt, isUser = true)
        _uiState.value = _uiState.value.copy(
            messages = _uiState.value.messages + userMessage,
            currentInput = "",
            isGenerating = true,
            streamingText = "",
            error = null
        )

        // Start polling for streaming updates
        pollJob = viewModelScope.launch(Dispatchers.IO) {
            while (isActive) {
                delay(200)
                val state = medGemma.getPartialResult()
                _uiState.value = _uiState.value.copy(
                    streamingText = state.text,
                    tokPerSec = state.tokPerSec,
                    tokensGenerated = state.tokensGenerated
                )
                if (!state.isGenerating && state.tokensGenerated > 0) break
            }
        }

        // Run generation on IO thread
        generateJob = viewModelScope.launch(Dispatchers.IO) {
            try {
                val result = medGemma.generate(prompt, 256)
                pollJob?.cancel()

                // Extract the actual text (after the stats header)
                val cleanResult = if (result.contains("]\n\n")) {
                    result.substringAfter("]\n\n")
                } else result

                val lastState = medGemma.getPartialResult()
                val assistantMessage = Message(
                    text = cleanResult,
                    isUser = false,
                    tokPerSec = lastState.tokPerSec,
                    tokensGenerated = lastState.tokensGenerated
                )

                _uiState.value = _uiState.value.copy(
                    messages = _uiState.value.messages + assistantMessage,
                    isGenerating = false,
                    streamingText = ""
                )
            } catch (e: Exception) {
                pollJob?.cancel()
                _uiState.value = _uiState.value.copy(
                    isGenerating = false,
                    error = "Generation failed: ${e.message}"
                )
            }
        }
    }

    fun stopGeneration() {
        medGemma.stopGeneration()
    }

    fun clearChat() {
        _uiState.value = UiState()
    }

    class Factory(private val medGemma: MedGemmaInference) : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>): T {
            return ClinicalAssistantViewModel(medGemma) as T
        }
    }
}
