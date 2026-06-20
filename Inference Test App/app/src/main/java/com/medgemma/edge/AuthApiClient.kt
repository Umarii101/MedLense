package com.medgemma.edge

import android.content.Context
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.IOException

class AuthApiClient(private val context: Context) {
    private val client = OkHttpClient()
    private val JSON = "application/json; charset=utf-8".toMediaType()

    private val prefs = context.getSharedPreferences("com.medgemma.edge.auth_prefs", Context.MODE_PRIVATE)

    fun register(username: String, password: String): Result<Unit> {
        val url = ApiConstants.getBaseUrl + ApiConstants.ENDPOINT_REGISTER
        val json = JSONObject().apply {
            put("username", username)
            put("password", password)
        }
        val body = json.toString().toRequestBody(JSON)
        val request = Request.Builder()
            .url(url)
            .post(body)
            .build()

        return try {
            client.newCall(request).execute().use { response ->
                if (response.isSuccessful) {
                    Result.success(Unit)
                } else {
                    val errorBody = response.body?.string() ?: ""
                    val errorMessage = try {
                        JSONObject(errorBody).optString("error", "Registration failed")
                    } catch (e: Exception) {
                        "Registration failed: Server returned ${response.code}"
                    }
                    Result.failure(Exception(errorMessage))
                }
            }
        } catch (e: IOException) {
            Result.failure(e)
        }
    }

    fun login(username: String, password: String): Result<String> {
        val url = ApiConstants.getBaseUrl + ApiConstants.ENDPOINT_LOGIN
        val json = JSONObject().apply {
            put("username", username)
            put("password", password)
        }
        val body = json.toString().toRequestBody(JSON)
        val request = Request.Builder()
            .url(url)
            .post(body)
            .build()

        return try {
            client.newCall(request).execute().use { response ->
                if (response.isSuccessful) {
                    val bodyString = response.body?.string() ?: ""
                    val jsonResponse = JSONObject(bodyString)
                    val access = jsonResponse.optString("access", "")
                    if (access.isNotEmpty()) {
                        saveToken(access)
                        Result.success(access)
                    } else {
                        Result.failure(Exception("No access token found in response"))
                    }
                } else {
                    val errorBody = response.body?.string() ?: ""
                    val errorMessage = try {
                        val json = JSONObject(errorBody)
                        json.optString("detail", json.optString("error", "Login failed"))
                    } catch (e: Exception) {
                        "Login failed: Server returned ${response.code}"
                    }
                    Result.failure(Exception(errorMessage))
                }
            }
        } catch (e: IOException) {
            Result.failure(e)
        }
    }

    fun logout() {
        prefs.edit().remove("access_token").apply()
        AppPreferences.accessToken = null
    }

    fun getToken(): String? {
        return prefs.getString("access_token", null)
    }

    private fun saveToken(token: String) {
        prefs.edit().putString("access_token", token).apply()
        AppPreferences.accessToken = token
    }
}
