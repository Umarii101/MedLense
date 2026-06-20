package com.medgemma.edge.ui

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.medgemma.edge.AppPreferences
import com.medgemma.edge.AuthApiClient
import com.medgemma.edge.ChatApiClient
import com.medgemma.edge.ChatCard
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

data class ChatListUiState(
    val isLoading: Boolean = false,
    val chats: List<ChatCard> = emptyList(),
    val error: String? = null
)

class ChatListViewModel(application: Application) : AndroidViewModel(application) {

    private val authApiClient = AuthApiClient(application)

    // ── Events fired to the UI ──────────────────────────────────────────────────────────
    private val _navigateToLogin = MutableSharedFlow<Unit>(replay = 0)
    val navigateToLogin: SharedFlow<Unit> = _navigateToLogin

    private val chatApiClient = ChatApiClient(
        onUnauthorized = {
            // Called on main thread by the interceptor when refresh also fails
            viewModelScope.launch {
                _navigateToLogin.emit(Unit)
            }
        }
    )

    private val _uiState = MutableStateFlow(ChatListUiState())
    val uiState: StateFlow<ChatListUiState> = _uiState

    init {
        loadChats()
    }

    fun loadChats() {
        _uiState.value = _uiState.value.copy(isLoading = true, error = null)
        viewModelScope.launch(Dispatchers.IO) {
            val result = chatApiClient.getChats()
            if (result.isSuccess) {
                val chats = result.getOrDefault(emptyList())
                _uiState.value = _uiState.value.copy(
                    isLoading = false,
                    chats = chats
                )
            } else {
                val errorMsg = result.exceptionOrNull()?.message ?: "Failed to load conversations"
                if (errorMsg.contains("Failed to fetch chats") || errorMsg.contains("Get chats failed", ignoreCase = true)) {
                    _uiState.value = _uiState.value.copy(isLoading = false, error = null, chats = emptyList())
                } else {
                    _uiState.value = _uiState.value.copy(isLoading = false, error = errorMsg)
                }
            }
        }
    }

    fun logout() {
        authApiClient.logout()
        viewModelScope.launch {
            _navigateToLogin.emit(Unit)
        }
    }
}
