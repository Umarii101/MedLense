package com.medlens.app.inference

import android.content.Context
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.sqrt

/**
 * Medical image classifier using BiomedCLIP embeddings.
 *
 * Compares image embeddings against pre-computed reference embeddings
 * via cosine similarity to classify medical images into conditions.
 *
 * Reference embeddings are stored in assets/reference_embeddings.json.
 * They can be pre-computed offline using the BiomedCLIP model on
 * representative images for each condition.
 */
class MedicalClassifier(private val context: Context) {

    companion object {
        private const val TAG = "MedClassifier"
        private const val EMBEDDINGS_FILE = "reference_embeddings.json"
    }

    data class ClassificationResult(
        val condition: String,
        val category: String,
        val confidence: Float,
        val description: String
    )

    data class ReferenceCondition(
        val name: String,
        val category: String,
        val description: String,
        val embeddings: List<FloatArray>  // Multiple reference embeddings per condition
    )

    private var referenceConditions: List<ReferenceCondition> = emptyList()
    var isLoaded: Boolean = false; private set

    /**
     * Load reference embeddings from assets.
     */
    fun loadReferenceEmbeddings() {
        try {
            val json = context.assets.open(EMBEDDINGS_FILE).bufferedReader().readText()
            val root = JSONObject(json)
            val conditions = root.getJSONArray("conditions")

            referenceConditions = (0 until conditions.length()).map { i ->
                val cond = conditions.getJSONObject(i)
                val embArray = cond.getJSONArray("embeddings")
                val embeddings = (0 until embArray.length()).map { j ->
                    val emb = embArray.getJSONArray(j)
                    FloatArray(emb.length()) { k -> emb.getDouble(k).toFloat() }
                }
                ReferenceCondition(
                    name = cond.getString("name"),
                    category = cond.getString("category"),
                    description = cond.getString("description"),
                    embeddings = embeddings
                )
            }
            isLoaded = true
            Log.i(TAG, "Loaded ${referenceConditions.size} reference conditions")
        } catch (e: Exception) {
            Log.w(TAG, "No reference embeddings found, using built-in defaults")
            referenceConditions = getBuiltInConditions()
            isLoaded = true
        }
    }

    /**
     * Classify an image embedding against reference conditions.
     * Returns sorted list of matches with confidence scores.
     */
    fun classify(
        embedding: FloatArray,
        topK: Int = 5
    ): List<ClassificationResult> {
        if (referenceConditions.isEmpty()) return emptyList()

        return referenceConditions.map { condition ->
            // Average cosine similarity across all reference embeddings
            val avgSimilarity = if (condition.embeddings.isNotEmpty()) {
                condition.embeddings.map { ref -> cosineSimilarity(embedding, ref) }.average().toFloat()
            } else {
                0f
            }

            ClassificationResult(
                condition = condition.name,
                category = condition.category,
                confidence = avgSimilarity.coerceIn(0f, 1f),
                description = condition.description
            )
        }
            .sortedByDescending { it.confidence }
            .take(topK)
    }

    /**
     * Classify with a simple threshold-based categorization.
     */
    fun classifyWithThreshold(
        embedding: FloatArray,
        threshold: Float = 0.3f
    ): Pair<List<ClassificationResult>, String> {
        val results = classify(embedding)
        val topResult = results.firstOrNull()

        val summary = when {
            topResult == null -> "Unable to classify"
            topResult.confidence >= 0.7f -> "High confidence: ${topResult.condition}"
            topResult.confidence >= threshold -> "Possible: ${topResult.condition}"
            else -> "No strong match found — further review needed"
        }

        return Pair(results, summary)
    }

    private fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        if (a.size != b.size) return 0f
        var dot = 0f; var normA = 0f; var normB = 0f
        for (i in a.indices) {
            dot += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        val denom = sqrt(normA) * sqrt(normB)
        return if (denom > 0f) dot / denom else 0f
    }

    /**
     * Built-in condition definitions (without embeddings — for UI structure).
     * Real classification requires pre-computed reference embeddings.
     */
    private fun getBuiltInConditions(): List<ReferenceCondition> {
        return listOf(
            ReferenceCondition(
                "Normal Chest X-Ray", "Radiology",
                "No significant abnormalities detected", emptyList()
            ),
            ReferenceCondition(
                "Pneumonia", "Pulmonology",
                "Lung infection causing inflammation of air sacs", emptyList()
            ),
            ReferenceCondition(
                "Pleural Effusion", "Pulmonology",
                "Abnormal fluid accumulation between lungs and chest wall", emptyList()
            ),
            ReferenceCondition(
                "Cardiomegaly", "Cardiology",
                "Enlarged heart visible on chest imaging", emptyList()
            ),
            ReferenceCondition(
                "Atelectasis", "Pulmonology",
                "Partial or complete collapse of lung tissue", emptyList()
            ),
            ReferenceCondition(
                "Consolidation", "Pulmonology",
                "Region of lung tissue filled with fluid instead of air", emptyList()
            ),
            ReferenceCondition(
                "Pneumothorax", "Emergency Medicine",
                "Air in pleural space causing lung collapse", emptyList()
            ),
            ReferenceCondition(
                "Skin Lesion — Benign", "Dermatology",
                "Non-cancerous skin growth or mark", emptyList()
            ),
            ReferenceCondition(
                "Skin Lesion — Suspicious", "Dermatology",
                "Skin lesion requiring further evaluation", emptyList()
            ),
            ReferenceCondition(
                "Retinal — Normal", "Ophthalmology",
                "Normal retinal appearance, no pathology", emptyList()
            ),
            ReferenceCondition(
                "Diabetic Retinopathy", "Ophthalmology",
                "Retinal damage from diabetes", emptyList()
            ),
        )
    }
}
