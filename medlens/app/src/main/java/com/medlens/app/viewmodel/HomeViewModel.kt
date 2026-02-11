package com.medlens.app.viewmodel

import android.app.Application
import android.os.Environment
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.medlens.app.inference.BiomedClipInference
import com.medlens.app.inference.MedGemmaInference
import com.medlens.app.inference.MedicalClassifier
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

class HomeViewModel(application: Application) : AndroidViewModel(application) {

    companion object {
        private const val TAG = "HomeVM"
    }

    // ── Inference engines (shared across screens) ───────────────────────────
    val biomedClip = BiomedClipInference(application)
    val medGemma = MedGemmaInference(application)
    val classifier = MedicalClassifier(application)

    // ── UI state ────────────────────────────────────────────────────────────
    data class UiState(
        val clipStatus: ModelStatus = ModelStatus.NOT_FOUND,
        val gemmaStatus: ModelStatus = ModelStatus.NOT_FOUND,
        val clipPath: String? = null,
        val gemmaPath: String? = null,
        val clipSizeMb: Float = 0f,
        val gemmaSizeMb: Float = 0f,
        val clipLoadTimeMs: Long = 0,
        val gemmaLoadTimeMs: Long = 0,
        val statusMessage: String = "Searching for models...",
        val hasStoragePermission: Boolean = false
    )

    enum class ModelStatus {
        NOT_FOUND, FOUND, LOADING, READY, ERROR
    }

    private val _uiState = MutableStateFlow(UiState())
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()

    fun onStoragePermissionGranted() {
        _uiState.value = _uiState.value.copy(hasStoragePermission = true)
        searchAndLoadModels()
    }

    fun searchAndLoadModels() {
        viewModelScope.launch(Dispatchers.IO) {
            val ctx = getApplication<Application>()

            // Search for BiomedCLIP
            val clipPath = BiomedClipInference.findModelPath(ctx)
            val gemmaPath = MedGemmaInference.findModelPath(ctx)

            _uiState.value = _uiState.value.copy(
                clipStatus = if (clipPath != null) ModelStatus.FOUND else ModelStatus.NOT_FOUND,
                gemmaStatus = if (gemmaPath != null) ModelStatus.FOUND else ModelStatus.NOT_FOUND,
                clipPath = clipPath,
                gemmaPath = gemmaPath,
                statusMessage = buildSearchMessage(clipPath, gemmaPath)
            )

            // Load BiomedCLIP (fast — ~1s)
            if (clipPath != null) {
                try {
                    _uiState.value = _uiState.value.copy(
                        clipStatus = ModelStatus.LOADING,
                        statusMessage = "Loading BiomedCLIP..."
                    )
                    val loadTime = biomedClip.loadModel(clipPath)
                    _uiState.value = _uiState.value.copy(
                        clipStatus = ModelStatus.READY,
                        clipSizeMb = biomedClip.modelSizeMb,
                        clipLoadTimeMs = loadTime,
                        statusMessage = "BiomedCLIP ready. Loading MedGemma..."
                    )
                } catch (e: Exception) {
                    Log.e(TAG, "BiomedCLIP load failed", e)
                    _uiState.value = _uiState.value.copy(
                        clipStatus = ModelStatus.ERROR,
                        statusMessage = "BiomedCLIP error: ${e.message}"
                    )
                }
            }

            // Load reference embeddings
            classifier.loadReferenceEmbeddings()

            // Load MedGemma (slow — ~25-30s for 2.2GB)
            if (gemmaPath != null) {
                try {
                    _uiState.value = _uiState.value.copy(
                        gemmaStatus = ModelStatus.LOADING,
                        statusMessage = "Loading MedGemma (2.2 GB)... This takes ~30s"
                    )
                    val loadTime = medGemma.loadModel(gemmaPath)
                    _uiState.value = _uiState.value.copy(
                        gemmaStatus = ModelStatus.READY,
                        gemmaSizeMb = medGemma.modelSizeMb,
                        gemmaLoadTimeMs = loadTime,
                        statusMessage = "All models ready!"
                    )
                } catch (e: Exception) {
                    Log.e(TAG, "MedGemma load failed", e)
                    _uiState.value = _uiState.value.copy(
                        gemmaStatus = ModelStatus.ERROR,
                        statusMessage = "MedGemma error: ${e.message}"
                    )
                }
            }
        }
    }

    private fun buildSearchMessage(clipPath: String?, gemmaPath: String?): String {
        val missing = mutableListOf<String>()
        if (clipPath == null) missing.add("BiomedCLIP (${BiomedClipInference.MODEL_FILENAME})")
        if (gemmaPath == null) missing.add("MedGemma (${MedGemmaInference.MODEL_FILENAME})")

        return if (missing.isEmpty()) {
            "Models found! Loading..."
        } else {
            "Missing: ${missing.joinToString(", ")}\n" +
            "Place model files in /sdcard/MedGemmaEdge/"
        }
    }

    override fun onCleared() {
        super.onCleared()
        biomedClip.release()
        medGemma.release()
    }
}
