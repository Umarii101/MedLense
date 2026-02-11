package com.medlens.app.inference

import android.content.Context
import android.util.Log
import java.io.File

/**
 * Kotlin wrapper for llama.cpp-based MedGemma GGUF inference.
 *
 * Loads the native JNI library (libmedgemma-jni.so) and provides
 * model loading, streaming text generation, and benchmarking.
 */
class MedGemmaInference(private val context: Context) {

    companion object {
        private const val TAG = "MedGemma"
        const val MODEL_FILENAME = "medgemma-4b-q4_k_s-final.gguf"

        @Volatile
        private var libraryLoaded = false

        fun findModelPath(context: Context): String? {
            val searchPaths = listOf(
                File(context.getExternalFilesDir(null), "models/$MODEL_FILENAME"),
                File("/storage/emulated/0/MedGemmaEdge/$MODEL_FILENAME"),
                File("/storage/emulated/0/Download/$MODEL_FILENAME"),
            )
            return searchPaths.firstOrNull { it.exists() }?.absolutePath
        }
    }

    var isLoaded: Boolean = false; private set
    var modelSizeMb: Float = 0f; private set

    // ── JNI declarations ────────────────────────────────────────────────────
    private external fun nativeInit(nativeLibDir: String)
    private external fun nativeLoadModel(modelPath: String): Int
    private external fun nativeGenerate(prompt: String, maxTokens: Int): String
    private external fun nativeGetPartialResult(): String
    private external fun nativeStopGeneration()
    private external fun nativeBench(pp: Int, tg: Int, reps: Int): String
    private external fun nativeSystemInfo(): String
    private external fun nativeUnload()

    fun initialize() {
        if (!libraryLoaded) {
            try {
                System.loadLibrary("medgemma-jni")
                nativeInit(context.applicationInfo.nativeLibraryDir)
                libraryLoaded = true
                Log.i(TAG, "Native library loaded")
            } catch (e: UnsatisfiedLinkError) {
                throw RuntimeException("Failed to load medgemma-jni native library", e)
            }
        }
    }

    fun loadModel(modelPath: String): Long {
        val startTime = System.currentTimeMillis()
        val file = File(modelPath)
        if (!file.exists()) throw RuntimeException("Model file not found: $modelPath")
        modelSizeMb = file.length() / (1024f * 1024f)

        initialize()
        val result = nativeLoadModel(modelPath)
        if (result != 0) throw RuntimeException("Failed to load model (error: $result)")

        isLoaded = true
        val loadTime = System.currentTimeMillis() - startTime
        Log.i(TAG, "Model loaded in ${loadTime}ms (${"%.1f".format(modelSizeMb)}MB)")
        return loadTime
    }

    fun generate(prompt: String, maxTokens: Int = 256): String {
        check(isLoaded) { "Model not loaded" }
        return nativeGenerate(prompt, maxTokens)
    }

    fun getPartialResult(): StreamingState {
        val raw = nativeGetPartialResult()
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

    fun stopGeneration() {
        nativeStopGeneration()
    }

    fun benchmark(pp: Int = 512, tg: Int = 128, reps: Int = 3): String {
        check(isLoaded) { "Model not loaded" }
        return nativeBench(pp, tg, reps)
    }

    fun systemInfo(): String {
        initialize()
        return nativeSystemInfo()
    }

    fun release() {
        if (isLoaded) {
            nativeUnload()
            isLoaded = false
            Log.i(TAG, "Model released")
        }
    }

    data class StreamingState(
        val tokensGenerated: Int,
        val tokPerSec: Float,
        val isGenerating: Boolean,
        val text: String
    )
}
