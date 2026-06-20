package com.medgemma.edge

import android.graphics.Bitmap
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.util.concurrent.TimeUnit

data class ChatCard(
    val chatId: Int,
    val lastMessage: String,
    val lastTimestamp: String
)

data class ChatResponse(
    val status: String,
    val chatId: Int,
    val userMessage: String,
    val response: String
)

data class ChatHistoryMessage(
    val senderType: String,
    val message: String,
    val messageType: String,
    val fileUrl: String?,
    val timestampAt: String
)

class ChatApiClient(private val onUnauthorized: () -> Unit) {
    private val client = OkHttpClient.Builder()
        .connectTimeout(300, TimeUnit.SECONDS)
        .readTimeout(300, TimeUnit.SECONDS)
        .writeTimeout(300, TimeUnit.SECONDS)
        .addInterceptor { chain ->
            val requestBuilder = chain.request().newBuilder()
            val token = AppPreferences.accessToken
            if (token != null) {
                requestBuilder.addHeader("Authorization", "Bearer $token")
            }
            val response = chain.proceed(requestBuilder.build())
            if (response.code == 401) {
                AppPreferences.accessToken = null
                onUnauthorized()
            }
            response
        }
        .build()

    fun getChats(): Result<List<ChatCard>> {
        val url = ApiConstants.getBaseUrl + ApiConstants.ENDPOINT_CHATS
        val request = Request.Builder()
            .url(url)
            .get()
            .build()

        return try {
            client.newCall(request).execute().use { response ->
                if (response.isSuccessful) {
                    val bodyString = response.body?.string() ?: "{}"
                    val jsonResponse = JSONObject(bodyString)
                    val status = jsonResponse.optString("status")
                    if (status == "success") {
                        val chatsArray = jsonResponse.optJSONArray("chats")
                        val chats = mutableListOf<ChatCard>()
                        if (chatsArray != null) {
                            for (i in 0 until chatsArray.length()) {
                                val chatObj = chatsArray.getJSONObject(i)
                                chats.add(
                                    ChatCard(
                                        chatId = chatObj.optInt("chat_id"),
                                        lastMessage = chatObj.optString("last_message"),
                                        lastTimestamp = chatObj.optString("last_timestamp")
                                    )
                                )
                            }
                        }
                        Result.success(chats)
                    } else {
                        Result.failure(Exception("Failed to fetch chats"))
                    }
                } else {
                    val errorBody = response.body?.string() ?: ""
                    val errorMessage = try {
                        val json = JSONObject(errorBody)
                        json.optString("detail", json.optString("error", "Get chats failed"))
                    } catch (e: Exception) {
                        "Server returned error code ${response.code}"
                    }
                    Result.failure(Exception(errorMessage))
                }
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    fun sendMessage(message: String, chatId: Int?, bitmap: Bitmap?): Result<ChatResponse> {
        val url = ApiConstants.getBaseUrl + ApiConstants.ENDPOINT_CHAT
        
        val builder = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("message", message)
            
        if (chatId != null && chatId != 0) {
            builder.addFormDataPart("chat_id", chatId.toString())
        }
        
        if (bitmap != null) {
            val stream = ByteArrayOutputStream()
            val maxWidth = 512
            val maxHeight = 512
            val width = bitmap.width
            val height = bitmap.height
            val scale = minOf(maxWidth.toFloat() / width, maxHeight.toFloat() / height)
            
            val scaledBitmap = if (scale < 1.0f) {
                Bitmap.createScaledBitmap(bitmap, (width * scale).toInt(), (height * scale).toInt(), true)
            } else {
                bitmap
            }
            
            scaledBitmap.compress(Bitmap.CompressFormat.JPEG, 85, stream)
            val byteArray = stream.toByteArray()
            
            builder.addFormDataPart(
                "file", "image.jpg",
                byteArray.toRequestBody("image/jpeg".toMediaTypeOrNull(), 0, byteArray.size)
            )
        }
        
        val requestBody = builder.build()
        val request = Request.Builder()
            .url(url)
            .post(requestBody)
            .build()

        return try {
            client.newCall(request).execute().use { response ->
                if (response.isSuccessful) {
                    val bodyString = response.body?.string() ?: "{}"
                    val jsonResponse = JSONObject(bodyString)
                    val status = jsonResponse.optString("status")
                    if (status == "success") {
                        val responseObj = ChatResponse(
                            status = status,
                            chatId = jsonResponse.optInt("chat_id"),
                            userMessage = jsonResponse.optString("user_message"),
                            response = jsonResponse.optString("response")
                        )
                        Result.success(responseObj)
                    } else {
                        Result.failure(Exception("Failed to send message"))
                    }
                } else {
                    val errorBody = response.body?.string() ?: ""
                    val errorMessage = try {
                        val json = JSONObject(errorBody)
                        json.optString("detail", json.optString("error", "Send message failed"))
                    } catch (e: Exception) {
                        "Server returned error code ${response.code}"
                    }
                    Result.failure(Exception(errorMessage))
                }
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    fun fetchChatHistory(chatId: Int): Result<List<ChatHistoryMessage>> {
        val url = "${ApiConstants.getBaseUrl}chat/$chatId/"
        val request = Request.Builder()
            .url(url)
            .get()
            .build()

        return try {
            client.newCall(request).execute().use { response ->
                if (response.isSuccessful) {
                    val bodyString = response.body?.string() ?: "{}"
                    val jsonResponse = JSONObject(bodyString)
                    if (jsonResponse.optString("status") == "success") {
                        val msgsArray = jsonResponse.optJSONArray("messages")
                        val history = mutableListOf<ChatHistoryMessage>()
                        if (msgsArray != null) {
                            for (i in 0 until msgsArray.length()) {
                                val msgObj = msgsArray.getJSONObject(i)
                                history.add(
                                    ChatHistoryMessage(
                                        senderType = msgObj.optString("sender_type"),
                                        message = msgObj.optString("message"),
                                        messageType = msgObj.optString("message_type"),
                                        fileUrl = msgObj.optString("file_url").takeIf { it != "null" && it.isNotEmpty() },
                                        timestampAt = msgObj.optString("timestamp_at")
                                    )
                                )
                            }
                        }
                        Result.success(history)
                    } else {
                        Result.failure(Exception("Failed to fetch history"))
                    }
                } else {
                    val errorBody = response.body?.string() ?: ""
                    val errorMessage = try {
                        val json = JSONObject(errorBody)
                        json.optString("detail", json.optString("error", "Unknown error"))
                    } catch (e: Exception) {
                        "Server returned error code ${response.code}"
                    }
                    Result.failure(Exception(errorMessage))
                }
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
}
