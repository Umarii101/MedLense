package com.medgemma.edge.ui

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.medgemma.edge.AppPreferences
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import okhttp3.OkHttpClient
import okhttp3.Request

class SettingsViewModel : ViewModel() {
    private val _urlState = MutableStateFlow(AppPreferences.serverBaseUrl)
    val urlState: StateFlow<String> = _urlState

    private val _testStatus = MutableStateFlow<String?>(null)
    val testStatus: StateFlow<String?> = _testStatus

    fun onUrlChange(newUrl: String) {
        _urlState.value = newUrl
    }

    fun saveUrl() {
        AppPreferences.serverBaseUrl = _urlState.value
        _testStatus.value = "Saved."
    }

    fun testConnection() {
        _testStatus.value = "Testing..."
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val client = OkHttpClient()
                val request = Request.Builder().url(_urlState.value).get().build()
                client.newCall(request).execute().use { response ->
                    if (response.isSuccessful || response.code == 404) {
                        _testStatus.value = "✅ Connected"
                    } else {
                        _testStatus.value = "❌ Unreachable (${response.code})"
                    }
                }
            } catch (e: Exception) {
                _testStatus.value = "❌ Unreachable"
            }
        }
    }
}
