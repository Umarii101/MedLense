package com.medgemma.edge.inference

import android.content.Context
import android.util.Log
import java.io.File

/**
 * Kotlin wrapper for llama.cpp-based MedGemma GGUF inference.
 *
 * Loads the native JNI library (libmedgemma-jni.so) which includes
 * llama.cpp compiled for Android. Provides model loading, text generation,
 * and benchmarking.
 */
class MedGemmaInference(private val context: Context) {

    /** Role for chat messages. */
    enum class Role { USER, MODEL, SYSTEM }

    /** A single message in a conversation. */
    data class ChatMessage(val role: Role, val content: String)

    companion object {
        private const val TAG = "MedGemma"
        private const val MODEL_FILENAME = "medgemma-4b-q4_k_s-final.gguf"

        /** Default medical system prompt. */
        const val MEDICAL_SYSTEM_PROMPT =
            "You are a medical AI assistant that is part of an on-device clinical pipeline. " +
            "A separate vision model (BiomedCLIP) has already analyzed the medical image and " +
            "provided classification results. You will receive these pre-computed findings as text. " +
            "Your role is to interpret the classification results, explain the likely condition, " +
            "provide differential diagnoses, and suggest next steps. " +
            "Provide clear, evidence-based information. " +
            "Always recommend consulting a healthcare professional for diagnosis and treatment."

        /**
         * Format a conversation into Gemma 3 chat template.
         *
         * Template: <start_of_turn>user\n{text}<end_of_turn>\n<start_of_turn>model\n
         * System messages are prepended to the first user turn.
         * BOS token is added by the tokenizer (add_special=true in C++).
         */
        fun formatChatPrompt(
            messages: List<ChatMessage>,
            systemPrompt: String? = MEDICAL_SYSTEM_PROMPT
        ): String {
            val sb = StringBuilder()
            var firstUserPrefix = ""
            val loopMessages: List<ChatMessage>

            // Handle system prompt: prepend to first user message
            if (systemPrompt != null && systemPrompt.isNotBlank()) {
                firstUserPrefix = systemPrompt + "\n\n"
                loopMessages = messages
            } else if (messages.isNotEmpty() && messages[0].role == Role.SYSTEM) {
                firstUserPrefix = messages[0].content + "\n\n"
                loopMessages = messages.drop(1)
            } else {
                loopMessages = messages
            }

            var isFirst = true
            for (msg in loopMessages) {
                val role = when (msg.role) {
                    Role.USER -> "user"
                    Role.MODEL -> "model"
                    Role.SYSTEM -> "user"
                }
                sb.append("<start_of_turn>")
                sb.append(role)
                sb.append("\n")
                if (isFirst && firstUserPrefix.isNotEmpty()) {
                    sb.append(firstUserPrefix)
                    isFirst = false
                }
                sb.append(msg.content.trim())
                sb.append("<end_of_turn>\n")
            }
            // Add generation prompt
            sb.append("<start_of_turn>model\n")
            return sb.toString()
        }

        /**
         * Format a simple single-turn prompt with system context.
         */
        fun formatSimplePrompt(
            userMessage: String,
            systemPrompt: String? = MEDICAL_SYSTEM_PROMPT
        ): String {
            return formatChatPrompt(
                listOf(ChatMessage(Role.USER, userMessage)),
                systemPrompt
            )
        }

        @Volatile
        private var libraryLoaded = false

        /**
         * Search common locations on device storage for the GGUF model.
         */
        fun findModelPath(context: Context): String? {
            val searchPaths = listOf(
                File(context.getExternalFilesDir(null), "models/$MODEL_FILENAME"),
                File("/storage/emulated/0/MedGemmaEdge/$MODEL_FILENAME"),
                File("/storage/emulated/0/Download/$MODEL_FILENAME"),
            )
            return searchPaths.firstOrNull { it.exists() }?.absolutePath
        }
    }

    // ── Public state ────────────────────────────────────────────────────────
    var isLoaded: Boolean = false
        private set
    var modelSizeMb: Float = 0f
        private set

    // ── JNI declarations ────────────────────────────────────────────────────
    private external fun nativeInit(nativeLibDir: String)
    private external fun nativeLoadModel(modelPath: String): Int
    private external fun nativeGenerate(prompt: String, maxTokens: Int): String
    private external fun nativeGetPartialResult(): String
    private external fun nativeStopGeneration()
    private external fun nativeBench(pp: Int, tg: Int, reps: Int): String
    private external fun nativeSystemInfo(): String
    private external fun nativeUnload()

    // ── Public API ──────────────────────────────────────────────────────────

    /**
     * Initialize the native library. Called automatically by loadModel().
     * Safe to call multiple times.
     */
    fun initialize() {
        if (!libraryLoaded) {
            try {
                System.loadLibrary("medgemma-jni")
                nativeInit(context.applicationInfo.nativeLibraryDir)
                libraryLoaded = true
                Log.i(TAG, "Native library loaded. System info:\n${nativeSystemInfo()}")
            } catch (e: UnsatisfiedLinkError) {
                throw RuntimeException(
                    "Failed to load medgemma-jni native library. " +
                    "Make sure you cloned llama.cpp and rebuilt the project.", e
                )
            }
        }
    }

    /**
     * Load a GGUF model from the given path.
     * @return load time in milliseconds
     */
    fun loadModel(modelPath: String): Long {
        val startTime = System.currentTimeMillis()

        val file = File(modelPath)
        if (!file.exists()) {
            throw RuntimeException("Model file not found: $modelPath")
        }
        modelSizeMb = file.length() / (1024f * 1024f)

        initialize()

        val result = nativeLoadModel(modelPath)
        if (result != 0) {
            throw RuntimeException("Failed to load model (error code: $result)")
        }

        isLoaded = true
        val loadTime = System.currentTimeMillis() - startTime
        Log.i(TAG, "Model loaded in ${loadTime}ms, size: ${"%.1f".format(modelSizeMb)}MB")
        return loadTime
    }

    /**
     * Generate text from a prompt (blocking, writes to streaming buffer).
     * Poll getPartialResult() from another coroutine for streaming updates.
     * @param prompt The input prompt text
     * @param maxTokens Maximum tokens to generate
     * @return Final generated text with stats header
     */
    fun generate(prompt: String, maxTokens: Int = 128): String {
        check(isLoaded) { "Model not loaded. Call loadModel() first." }
        return nativeGenerate(prompt, maxTokens)
    }

    /**
     * Generate with Gemma 3 chat template applied.
     * Wraps the user message with proper <start_of_turn>/<end_of_turn> tokens
     * and a system prompt so MedGemma responds as an instruction-following model.
     */
    fun generateWithTemplate(
        userMessage: String,
        maxTokens: Int = 256,
        systemPrompt: String? = MEDICAL_SYSTEM_PROMPT
    ): String {
        val formatted = formatSimplePrompt(userMessage, systemPrompt)
        Log.d(TAG, "Formatted prompt (${formatted.length} chars)")
        return generate(formatted, maxTokens)
    }

    /**
     * Generate from a multi-turn conversation with chat template.
     */
    fun generateFromConversation(
        messages: List<ChatMessage>,
        maxTokens: Int = 256,
        systemPrompt: String? = MEDICAL_SYSTEM_PROMPT
    ): String {
        val formatted = formatChatPrompt(messages, systemPrompt)
        Log.d(TAG, "Formatted conversation (${messages.size} turns, ${formatted.length} chars)")
        return generate(formatted, maxTokens)
    }

    /**
     * Get current streaming generation state.
     * @return StreamingState with tokens generated, speed, generating flag, and partial text
     */
    fun getPartialResult(): StreamingState {
        val raw = nativeGetPartialResult()
        // Format from C++: "tokens|speed|is_generating|text"
        val parts = raw.split("|", limit = 4)
        return if (parts.size >= 4) {
            StreamingState(
                tokensGenerated = parts[0].toIntOrNull() ?: 0,
                tokPerSec = parts[1].toFloatOrNull() ?: 0f,
                isGenerating = parts[2] == "1",
                text = parts[3]
            )
        } else {
            StreamingState(0, 0f, false, raw)
        }
    }

    /**
     * Signal the native generation loop to stop early.
     */
    fun stopGeneration() {
        nativeStopGeneration()
    }

    data class StreamingState(
        val tokensGenerated: Int,
        val tokPerSec: Float,
        val isGenerating: Boolean,
        val text: String
    )

    /**
     * Run prompt-processing and token-generation benchmarks.
     * @param pp Number of prompt tokens to benchmark
     * @param tg Number of generation tokens to benchmark
     * @param reps Number of repetitions
     * @return Formatted benchmark results string
     */
    fun benchmark(pp: Int = 512, tg: Int = 128, reps: Int = 3): String {
        check(isLoaded) { "Model not loaded. Call loadModel() first." }
        return nativeBench(pp, tg, reps)
    }

    /**
     * Get llama.cpp system info (CPU features, etc.)
     */
    fun systemInfo(): String {
        initialize()
        return nativeSystemInfo()
    }

    /**
     * Unload the model and free native resources.
     */
    fun release() {
        if (isLoaded) {
            nativeUnload()
            isLoaded = false
            Log.i(TAG, "Model released")
        }
    }
}
