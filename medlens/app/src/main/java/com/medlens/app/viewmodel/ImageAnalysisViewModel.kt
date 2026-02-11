package com.medlens.app.viewmodel

import android.net.Uri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.medlens.app.inference.BiomedClipInference
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

class ImageAnalysisViewModel(
    private val biomedClip: BiomedClipInference
) : ViewModel() {

    data class UiState(
        val selectedImageUri: Uri? = null,
        val isAnalyzing: Boolean = false,
        val embedding: FloatArray? = null,
        val inferenceTimeMs: Float = 0f,
        val embeddingPreview: String = "",
        val error: String? = null
    )

    private val _uiState = MutableStateFlow(UiState())
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()

    fun onImageSelected(uri: Uri) {
        _uiState.value = _uiState.value.copy(
            selectedImageUri = uri,
            embedding = null,
            error = null,
            embeddingPreview = ""
        )
    }

    fun analyzeImage() {
        val uri = _uiState.value.selectedImageUri ?: return
        if (!biomedClip.isLoaded) {
            _uiState.value = _uiState.value.copy(error = "BiomedCLIP model not loaded")
            return
        }

        viewModelScope.launch(Dispatchers.IO) {
            _uiState.value = _uiState.value.copy(isAnalyzing = true, error = null)
            try {
                val result = biomedClip.getEmbedding(uri)
                val preview = result.embedding.take(8)
                    .joinToString(", ") { "%.4f".format(it) } + "..."

                _uiState.value = _uiState.value.copy(
                    isAnalyzing = false,
                    embedding = result.embedding,
                    inferenceTimeMs = result.inferenceTimeMs,
                    embeddingPreview = "[${preview}]\n(${result.embedding.size} dimensions)"
                )
            } catch (e: Exception) {
                _uiState.value = _uiState.value.copy(
                    isAnalyzing = false,
                    error = "Analysis failed: ${e.message}"
                )
            }
        }
    }

    class Factory(private val biomedClip: BiomedClipInference) : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>): T {
            return ImageAnalysisViewModel(biomedClip) as T
        }
    }
}
