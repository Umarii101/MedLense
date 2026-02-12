package com.medgemma.edge.inference

import android.content.Context
import android.util.Log
import org.json.JSONObject
import kotlin.math.sqrt

/**
 * Zero-shot medical image classifier using BiomedCLIP embeddings.
 *
 * Loads pre-computed text embeddings from assets (generated offline by
 * BiomedCLIP's text encoder) and compares them to image embeddings
 * produced by BiomedClipInference via cosine similarity.
 *
 * Returns ranked conditions with confidence scores and category metadata.
 */
class MedicalClassifier(private val context: Context) {

    companion object {
        private const val TAG = "MedClassifier"
        private const val EMBEDDINGS_ASSET = "text_embeddings.json"
        private const val CATEGORIES_ASSET = "label_categories.json"
        private const val EMBEDDING_DIM = 512
    }

    /** A single classification result. */
    data class ClassificationResult(
        val label: String,
        val confidence: Float,
        val category: String
    )

    /** Full output from classifying an image. */
    data class ClassificationOutput(
        val topResults: List<ClassificationResult>,
        val allResults: List<ClassificationResult>,
        val classificationTimeMs: Float,
        val dominantCategory: String
    )

    // Pre-loaded reference embeddings: label → 512-dim float array
    private var referenceEmbeddings: Map<String, FloatArray> = emptyMap()
    // Category map: label → category name
    private var labelCategories: Map<String, String> = emptyMap()

    var isLoaded: Boolean = false
        private set

    /**
     * Load text embeddings and category metadata from assets.
     */
    fun loadEmbeddings() {
        try {
            // Load text embeddings
            val embJson = context.assets.open(EMBEDDINGS_ASSET).bufferedReader().use { it.readText() }
            val embObj = JSONObject(embJson)
            val embeddings = mutableMapOf<String, FloatArray>()

            for (key in embObj.keys()) {
                val arr = embObj.getJSONArray(key)
                val floats = FloatArray(arr.length()) { arr.getDouble(it).toFloat() }
                embeddings[key] = floats
            }
            referenceEmbeddings = embeddings

            // Load category mapping
            val catJson = context.assets.open(CATEGORIES_ASSET).bufferedReader().use { it.readText() }
            val catObj = JSONObject(catJson)
            val categories = mutableMapOf<String, String>()
            for (key in catObj.keys()) {
                categories[key] = catObj.getString(key)
            }
            labelCategories = categories

            isLoaded = true
            Log.d(TAG, "Loaded ${referenceEmbeddings.size} reference embeddings across ${labelCategories.values.toSet().size} categories")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load embeddings: ${e.message}")
            throw e
        }
    }

    /**
     * Classify an image embedding against all reference conditions.
     *
     * @param imageEmbedding 512-dim embedding from BiomedClipInference
     * @param topK Number of top results to return
     * @return ClassificationOutput with ranked results
     */
    fun classify(imageEmbedding: FloatArray, topK: Int = 5): ClassificationOutput {
        check(isLoaded) { "Embeddings not loaded. Call loadEmbeddings() first." }
        require(imageEmbedding.size == EMBEDDING_DIM) {
            "Expected ${EMBEDDING_DIM}-dim embedding, got ${imageEmbedding.size}"
        }

        val startTime = System.nanoTime()

        // Normalize image embedding
        val normImage = normalize(imageEmbedding)

        // Compute cosine similarity with all reference embeddings
        val similarities = referenceEmbeddings.map { (label, refEmb) ->
            val similarity = dotProduct(normImage, normalize(refEmb))
            ClassificationResult(
                label = label,
                confidence = similarity,
                category = labelCategories[label] ?: "unknown"
            )
        }.sortedByDescending { it.confidence }

        val classificationTimeMs = (System.nanoTime() - startTime) / 1_000_000f

        // Determine dominant category from top results
        val dominantCategory = similarities.take(topK)
            .groupBy { it.category }
            .maxByOrNull { it.value.sumOf { r -> r.confidence.toDouble() } }
            ?.key ?: "unknown"

        val output = ClassificationOutput(
            topResults = similarities.take(topK),
            allResults = similarities,
            classificationTimeMs = classificationTimeMs,
            dominantCategory = dominantCategory
        )

        Log.d(TAG, "Classification: ${classificationTimeMs}ms, top: ${output.topResults.first().label} (${output.topResults.first().confidence})")
        return output
    }

    /**
     * Format classification results as a human-readable summary
     * suitable for feeding into MedGemma as context.
     */
    fun formatForLLM(output: ClassificationOutput): String {
        val sb = StringBuilder()
        sb.appendLine("Image Classification Results (BiomedCLIP zero-shot):")
        sb.appendLine("Dominant category: ${output.dominantCategory}")
        sb.appendLine("Top findings:")
        output.topResults.forEachIndexed { i, result ->
            val pct = (result.confidence * 100).toInt()
            sb.appendLine("  ${i + 1}. ${result.label} (${pct}% confidence, ${result.category})")
        }
        return sb.toString().trim()
    }

    /**
     * Get a short single-line summary (e.g., for display chips).
     */
    fun shortSummary(output: ClassificationOutput): String {
        val top = output.topResults.first()
        val pct = (top.confidence * 100).toInt()
        return "${top.label} ($pct%)"
    }

    // ── Vector math utilities ──────────────────────────────────────────

    private fun normalize(vec: FloatArray): FloatArray {
        val norm = sqrt(vec.sumOf { (it * it).toDouble() }).toFloat()
        return if (norm > 0f) FloatArray(vec.size) { vec[it] / norm } else vec.copyOf()
    }

    private fun dotProduct(a: FloatArray, b: FloatArray): Float {
        var sum = 0f
        for (i in a.indices) {
            sum += a[i] * b[i]
        }
        return sum
    }

    fun release() {
        referenceEmbeddings = emptyMap()
        labelCategories = emptyMap()
        isLoaded = false
    }
}
