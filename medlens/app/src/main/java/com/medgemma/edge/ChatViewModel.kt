package com.medgemma.edge

import android.app.Application
import android.net.Uri
import android.util.Log
import androidx.compose.runtime.mutableStateListOf
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.medgemma.edge.inference.BiomedClipInference
import com.medgemma.edge.inference.MedGemmaInference
import com.medgemma.edge.inference.MedicalClassifier
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Central ViewModel for the MedLens chat interface.
 *
 * Manages:
 * - Model loading (BiomedCLIP, MedGemma, classifier)
 * - Chat message history
 * - Image analysis pipeline: BiomedCLIP → Classifier → MedGemma
 * - Streaming text generation with polling
 */
class ChatViewModel(application: Application) : AndroidViewModel(application) {

    companion object {
        private const val TAG = "ChatVM"
        private const val STREAM_POLL_MS = 100L
        private const val MAX_TOKENS = 512
    }

    // ── Chat message model ──────────────────────────────────────────────
    enum class MessageRole { USER, ASSISTANT, SYSTEM }

    data class ChatMessage(
        val role: MessageRole,
        val text: String,
        val imageUri: Uri? = null,
        val classificationSummary: String? = null,
        val isStreaming: Boolean = false
    )

    // ── Exposed state ───────────────────────────────────────────────────

    /** All chat messages in order. */
    val messages = mutableStateListOf<ChatMessage>()

    /** Model loading status. */
    data class ModelStatus(
        val biomedClipLoaded: Boolean = false,
        val medGemmaLoaded: Boolean = false,
        val classifierLoaded: Boolean = false,
        val loadingMessage: String = "Initializing...",
        val isLoading: Boolean = true,
        val errorMessage: String? = null
    )

    private val _modelStatus = MutableStateFlow(ModelStatus())
    val modelStatus: StateFlow<ModelStatus> = _modelStatus.asStateFlow()

    private val _isGenerating = MutableStateFlow(false)
    val isGenerating: StateFlow<Boolean> = _isGenerating.asStateFlow()

    private val _streamingStats = MutableStateFlow("")
    val streamingStats: StateFlow<String> = _streamingStats.asStateFlow()

    // ── Inference engines ───────────────────────────────────────────────
    private var biomedClip: BiomedClipInference? = null
    private var medGemma: MedGemmaInference? = null
    private var classifier: MedicalClassifier? = null
    private var streamJob: Job? = null

    // ── Initialization ──────────────────────────────────────────────────

    private var modelsLoadStarted = false

    /**
     * Load all models asynchronously on IO dispatcher.
     * Called by the UI layer once storage permission is confirmed.
     */
    fun loadModels() {
        if (modelsLoadStarted) return
        modelsLoadStarted = true
        viewModelScope.launch(Dispatchers.IO) {
            try {
                // 1. Load classifier (text embeddings from assets — fast)
                _modelStatus.value = _modelStatus.value.copy(
                    loadingMessage = "Loading classifier..."
                )
                val ctx = getApplication<Application>()
                classifier = MedicalClassifier(ctx).also { it.loadEmbeddings() }
                _modelStatus.value = _modelStatus.value.copy(classifierLoaded = true)
                Log.i(TAG, "Classifier loaded")

                // 2. Load BiomedCLIP ONNX
                _modelStatus.value = _modelStatus.value.copy(
                    loadingMessage = "Loading BiomedCLIP..."
                )
                val clipPath = BiomedClipInference.findModelPath(ctx)
                if (clipPath != null) {
                    biomedClip = BiomedClipInference(ctx).also { it.loadModel(clipPath) }
                    _modelStatus.value = _modelStatus.value.copy(biomedClipLoaded = true)
                    Log.i(TAG, "BiomedCLIP loaded from $clipPath")
                } else {
                    Log.w(TAG, "BiomedCLIP model not found on device")
                }

                // 3. Load MedGemma GGUF (heavy — ~5s)
                _modelStatus.value = _modelStatus.value.copy(
                    loadingMessage = "Loading MedGemma (2.2 GB)..."
                )
                val gemmaPath = MedGemmaInference.findModelPath(ctx)
                if (gemmaPath != null) {
                    medGemma = MedGemmaInference(ctx).also { it.loadModel(gemmaPath) }
                    _modelStatus.value = _modelStatus.value.copy(medGemmaLoaded = true)
                    Log.i(TAG, "MedGemma loaded from $gemmaPath")
                } else {
                    Log.w(TAG, "MedGemma model not found on device")
                }

                _modelStatus.value = _modelStatus.value.copy(
                    isLoading = false,
                    loadingMessage = "Ready"
                )
            } catch (e: Exception) {
                Log.e(TAG, "Model loading failed", e)
                _modelStatus.value = _modelStatus.value.copy(
                    isLoading = false,
                    errorMessage = "Failed: ${e.message}"
                )
            }
        }
    }

    // ── User actions ────────────────────────────────────────────────────

    /**
     * Send a text-only message. MedGemma generates a response.
     */
    fun sendTextMessage(text: String) {
        if (text.isBlank() || _isGenerating.value) return
        messages.add(ChatMessage(role = MessageRole.USER, text = text))
        generateResponse(text, classificationContext = null)
    }

    /**
     * Analyze a medical image then generate a MedGemma response.
     *
     * Pipeline:
     * 1. BiomedCLIP encodes image → 512-dim embedding
     * 2. Classifier matches embedding → top conditions
     * 3. MedGemma gets classification context + user query → text response
     */
    fun analyzeImage(imageUri: Uri, userMessage: String = "Analyze this medical image.") {
        if (_isGenerating.value) return

        messages.add(ChatMessage(
            role = MessageRole.USER,
            text = userMessage,
            imageUri = imageUri
        ))

        viewModelScope.launch(Dispatchers.IO) {
            _isGenerating.value = true
            try {
                // Step 1: BiomedCLIP image embedding
                val clip = biomedClip
                val cls = classifier
                var classificationText: String? = null
                var shortLabel: String? = null

                if (clip != null && cls != null && clip.isLoaded && cls.isLoaded) {
                    _streamingStats.value = "Analyzing image with BiomedCLIP..."
                    val inferenceResult = clip.runInference(imageUri)
                    val clipMs = inferenceResult.inferenceTimeMs

                    // Step 2: Zero-shot classification
                    _streamingStats.value = "Classifying..."
                    val classResult = cls.classify(inferenceResult.embedding)
                    classificationText = cls.formatForLLM(classResult)
                    shortLabel = cls.shortSummary(classResult)

                    _streamingStats.value = "BiomedCLIP: ${clipMs.toInt()}ms | ${classResult.topResults.first().label}"
                    Log.i(TAG, "Classification done: $shortLabel (${clipMs}ms)")
                } else {
                    classificationText = "Note: BiomedCLIP or classifier not available."
                    _streamingStats.value = "BiomedCLIP unavailable, using text-only..."
                }

                // Step 3: MedGemma generates response with classification context
                val contextPrompt = buildString {
                    appendLine("The BiomedCLIP vision model has analyzed a medical image and produced the following findings:")
                    appendLine()
                    if (classificationText != null) {
                        appendLine(classificationText)
                    } else {
                        appendLine("No classification data available.")
                    }
                    appendLine()
                    appendLine("The user asks: $userMessage")
                    appendLine()
                    appendLine("Using the classification findings above, provide a clinical assessment. " +
                            "Explain what the top findings suggest, list differential diagnoses, " +
                            "and recommend appropriate next steps. " +
                            "This is for educational purposes only.")
                }

                // Add assistant placeholder for streaming
                val assistantIdx = messages.size
                withContext(Dispatchers.Main) {
                    messages.add(ChatMessage(
                        role = MessageRole.ASSISTANT,
                        text = "",
                        classificationSummary = shortLabel,
                        isStreaming = true
                    ))
                }

                streamGeneration(contextPrompt, assistantIdx)
            } catch (e: Exception) {
                Log.e(TAG, "Image analysis failed", e)
                withContext(Dispatchers.Main) {
                    messages.add(ChatMessage(
                        role = MessageRole.ASSISTANT,
                        text = "Error analyzing image: ${e.message}"
                    ))
                }
                _isGenerating.value = false
                _streamingStats.value = ""
            }
        }
    }

    /**
     * Generate a text response from MedGemma with optional classification context.
     */
    private fun generateResponse(userText: String, classificationContext: String?) {
        viewModelScope.launch(Dispatchers.IO) {
            _isGenerating.value = true

            val prompt = if (classificationContext != null) {
                "$classificationContext\n\nUser: $userText"
            } else {
                userText
            }

            // Add empty assistant message for streaming
            val assistantIdx = messages.size
            withContext(Dispatchers.Main) {
                messages.add(ChatMessage(
                    role = MessageRole.ASSISTANT,
                    text = "",
                    isStreaming = true
                ))
            }

            streamGeneration(prompt, assistantIdx)
        }
    }

    /**
     * Stream MedGemma generation with polling.
     * Runs generate() on IO thread, polls getPartialResult() for updates.
     */
    private suspend fun streamGeneration(prompt: String, assistantMessageIdx: Int) {
        val gemma = medGemma
        if (gemma == null || !gemma.isLoaded) {
            withContext(Dispatchers.Main) {
                if (assistantMessageIdx < messages.size) {
                    messages[assistantMessageIdx] = messages[assistantMessageIdx].copy(
                        text = "MedGemma model not loaded. Please ensure the model file is on device.",
                        isStreaming = false
                    )
                }
            }
            _isGenerating.value = false
            _streamingStats.value = ""
            return
        }

        try {
            // Always use the provided prompt (which contains classification context
            // for image analysis) wrapped with the chat template.
            val formattedPrompt = MedGemmaInference.formatSimplePrompt(prompt)

            // Launch generation in background
            val genJob = viewModelScope.launch(Dispatchers.IO) {
                gemma.generate(formattedPrompt, MAX_TOKENS)
            }

            // Poll for streaming updates
            streamJob = viewModelScope.launch(Dispatchers.IO) {
                while (isActive && genJob.isActive) {
                    delay(STREAM_POLL_MS)
                    val state = gemma.getPartialResult()
                    if (state.text.isNotEmpty()) {
                        withContext(Dispatchers.Main) {
                            if (assistantMessageIdx < messages.size) {
                                messages[assistantMessageIdx] = messages[assistantMessageIdx].copy(
                                    text = state.text,
                                    isStreaming = state.isGenerating
                                )
                            }
                        }
                        if (state.tokPerSec > 0) {
                            _streamingStats.value = "${state.tokensGenerated} tokens · ${"%.1f".format(state.tokPerSec)} tok/s"
                        }
                    }
                }
            }

            genJob.join()
            streamJob?.cancel()

            // Final update
            val finalState = gemma.getPartialResult()
            withContext(Dispatchers.Main) {
                if (assistantMessageIdx < messages.size) {
                    messages[assistantMessageIdx] = messages[assistantMessageIdx].copy(
                        text = finalState.text.ifEmpty { "No response generated." },
                        isStreaming = false
                    )
                }
            }
            _streamingStats.value = if (finalState.tokPerSec > 0) {
                "${finalState.tokensGenerated} tokens · ${"%.1f".format(finalState.tokPerSec)} tok/s"
            } else ""

        } catch (e: Exception) {
            Log.e(TAG, "Generation failed", e)
            withContext(Dispatchers.Main) {
                if (assistantMessageIdx < messages.size) {
                    messages[assistantMessageIdx] = messages[assistantMessageIdx].copy(
                        text = "Generation error: ${e.message}",
                        isStreaming = false
                    )
                }
            }
        } finally {
            _isGenerating.value = false
        }
    }

    /**
     * Build conversation history from messages for multi-turn prompt.
     * Only includes text messages (images are represented by their classification).
     */
    private fun buildConversationHistory(upToIdx: Int): List<MedGemmaInference.ChatMessage> {
        val history = mutableListOf<MedGemmaInference.ChatMessage>()
        // Include up to last 6 messages for context (3 turns)
        val start = maxOf(0, upToIdx - 6)
        for (i in start until upToIdx) {
            val msg = messages[i]
            val role = when (msg.role) {
                MessageRole.USER -> MedGemmaInference.Role.USER
                MessageRole.ASSISTANT -> MedGemmaInference.Role.MODEL
                MessageRole.SYSTEM -> MedGemmaInference.Role.USER
            }
            // For image messages, include the classification context
            val content = if (msg.imageUri != null && msg.classificationSummary != null) {
                "${msg.text}\n[Image classification: ${msg.classificationSummary}]"
            } else {
                msg.text
            }
            if (content.isNotBlank()) {
                history.add(MedGemmaInference.ChatMessage(role, content))
            }
        }
        return history
    }

    /**
     * Stop the current generation.
     */
    fun stopGeneration() {
        medGemma?.stopGeneration()
        streamJob?.cancel()
        _isGenerating.value = false
    }

    /**
     * Clear chat history.
     */
    fun clearChat() {
        if (!_isGenerating.value) {
            messages.clear()
            _streamingStats.value = ""
        }
    }

    override fun onCleared() {
        super.onCleared()
        streamJob?.cancel()
        biomedClip?.release()
        medGemma?.release()
        classifier?.release()
    }
}
