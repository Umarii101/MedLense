package com.medlens.app.inference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.util.Log
import java.io.File
import java.nio.FloatBuffer

/**
 * BiomedCLIP ONNX INT8 inference engine.
 *
 * Loads biomedclip_vision_int8.onnx from device storage and runs
 * image → 512-dim embedding inference.
 */
class BiomedClipInference(private val context: Context) {

    companion object {
        private const val TAG = "BiomedCLIP"
        const val MODEL_FILENAME = "biomedclip_vision_int8.onnx"
        const val IMAGE_SIZE = 224
        const val EMBEDDING_DIM = 512

        // ImageNet normalization (used by BiomedCLIP / OpenCLIP)
        private val MEAN = floatArrayOf(0.48145466f, 0.4578275f, 0.40821073f)
        private val STD = floatArrayOf(0.26862954f, 0.26130258f, 0.27577711f)

        fun findModelPath(context: Context): String? {
            val searchPaths = listOf(
                File(context.getExternalFilesDir(null), "models/$MODEL_FILENAME"),
                File("/storage/emulated/0/MedGemmaEdge/$MODEL_FILENAME"),
                File("/storage/emulated/0/Download/$MODEL_FILENAME"),
            )
            return searchPaths.firstOrNull { it.exists() }?.absolutePath
        }
    }

    private var ortEnv: OrtEnvironment? = null
    private var session: OrtSession? = null

    var modelSizeMb: Float = 0f; private set
    var loadTimeMs: Long = 0; private set
    var isLoaded: Boolean = false; private set

    fun loadModel(modelPath: String): Long {
        Log.d(TAG, "Loading model from: $modelPath")
        val startTime = System.currentTimeMillis()
        val modelFile = File(modelPath)
        modelSizeMb = modelFile.length() / (1024f * 1024f)

        ortEnv = OrtEnvironment.getEnvironment()
        val sessionOptions = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(4)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            try { addNnapi(); Log.d(TAG, "NNAPI enabled") }
            catch (e: Exception) { Log.d(TAG, "NNAPI not available: ${e.message}") }
        }
        session = ortEnv!!.createSession(modelPath, sessionOptions)

        loadTimeMs = System.currentTimeMillis() - startTime
        isLoaded = true
        Log.d(TAG, "Model loaded: ${modelSizeMb}MB in ${loadTimeMs}ms")
        return loadTimeMs
    }

    /**
     * Run inference on an image URI → 512-dim embedding.
     */
    fun getEmbedding(imageUri: Uri): InferenceResult {
        check(isLoaded) { "Model not loaded" }

        val bitmap = decodeBitmap(imageUri)
        val inputTensor = preprocessImage(bitmap)

        val startTime = System.nanoTime()
        val inputName = session!!.inputInfo.keys.first()
        val shape = longArrayOf(1, 3, IMAGE_SIZE.toLong(), IMAGE_SIZE.toLong())
        val onnxTensor = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(inputTensor), shape)

        val results = session!!.run(mapOf(inputName to onnxTensor))
        val output = results[0].value as Array<FloatArray>
        val embedding = output[0]
        val inferenceTimeMs = (System.nanoTime() - startTime) / 1_000_000f

        onnxTensor.close()
        results.close()

        Log.d(TAG, "Inference: ${inferenceTimeMs}ms, dims: ${embedding.size}")
        return InferenceResult(embedding, inferenceTimeMs)
    }

    /**
     * Run inference on a Bitmap directly → 512-dim embedding.
     */
    fun getEmbedding(bitmap: Bitmap): InferenceResult {
        check(isLoaded) { "Model not loaded" }

        val resized = Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true)
        val inputTensor = preprocessImage(resized)

        val startTime = System.nanoTime()
        val inputName = session!!.inputInfo.keys.first()
        val shape = longArrayOf(1, 3, IMAGE_SIZE.toLong(), IMAGE_SIZE.toLong())
        val onnxTensor = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(inputTensor), shape)

        val results = session!!.run(mapOf(inputName to onnxTensor))
        val output = results[0].value as Array<FloatArray>
        val embedding = output[0]
        val inferenceTimeMs = (System.nanoTime() - startTime) / 1_000_000f

        onnxTensor.close()
        results.close()

        return InferenceResult(embedding, inferenceTimeMs)
    }

    private fun decodeBitmap(uri: Uri): Bitmap {
        val inputStream = context.contentResolver.openInputStream(uri)
        val bitmap = BitmapFactory.decodeStream(inputStream)
        inputStream?.close()
        return Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true)
    }

    private fun preprocessImage(bitmap: Bitmap): FloatArray {
        val pixels = IntArray(IMAGE_SIZE * IMAGE_SIZE)
        bitmap.getPixels(pixels, 0, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE)

        val floatArray = FloatArray(3 * IMAGE_SIZE * IMAGE_SIZE)
        for (i in pixels.indices) {
            val pixel = pixels[i]
            floatArray[i] = ((pixel shr 16 and 0xFF) / 255f - MEAN[0]) / STD[0]
            floatArray[IMAGE_SIZE * IMAGE_SIZE + i] = ((pixel shr 8 and 0xFF) / 255f - MEAN[1]) / STD[1]
            floatArray[2 * IMAGE_SIZE * IMAGE_SIZE + i] = ((pixel and 0xFF) / 255f - MEAN[2]) / STD[2]
        }
        return floatArray
    }

    fun release() {
        session?.close()
        ortEnv?.close()
        session = null
        ortEnv = null
        isLoaded = false
    }

    data class InferenceResult(
        val embedding: FloatArray,
        val inferenceTimeMs: Float
    )
}
