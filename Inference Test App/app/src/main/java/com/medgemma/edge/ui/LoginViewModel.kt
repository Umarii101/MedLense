package com.medgemma.edge.ui

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.medgemma.edge.AuthApiClient
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

data class LoginUiState(
    val isLoginMode: Boolean = true,
    val username: String = "",
    val password: String = "",
    val isLoading: Boolean = false,
    val isSuccess: Boolean = false,
    val error: String? = null
)

class LoginViewModel(application: Application) : AndroidViewModel(application) {

    private val authApiClient = AuthApiClient(application)

    private val _uiState = MutableStateFlow(LoginUiState())
    val uiState: StateFlow<LoginUiState> = _uiState

    fun onUsernameChange(value: String) {
        _uiState.value = _uiState.value.copy(username = value, error = null)
    }

    fun onPasswordChange(value: String) {
        _uiState.value = _uiState.value.copy(password = value, error = null)
    }

    fun toggleMode() {
        _uiState.value = _uiState.value.copy(
            isLoginMode = !_uiState.value.isLoginMode,
            error = null,
            password = ""           // clear password when switching tabs
        )
    }

    fun resetState() {
        _uiState.value = LoginUiState()
    }

    fun submit() {
        val state = _uiState.value
        val username = state.username.trim()
        val password = state.password

        if (username.isBlank()) {
            _uiState.value = state.copy(error = "Username cannot be empty")
            return
        }
        if (password.isBlank()) {
            _uiState.value = state.copy(error = "Password cannot be empty")
            return
        }

        _uiState.value = state.copy(isLoading = true, error = null)

        viewModelScope.launch(Dispatchers.IO) {
            if (state.isLoginMode) {
                val result = authApiClient.login(username, password)
                _uiState.value = if (result.isSuccess) {
                    _uiState.value.copy(isLoading = false, isSuccess = true)
                } else {
                    _uiState.value.copy(
                        isLoading = false,
                        password = "",          // clear password on failed login
                        error = result.exceptionOrNull()?.message ?: "Login failed"
                    )
                }
            } else {
                // Register then auto-login
                val regResult = authApiClient.register(username, password)
                if (regResult.isFailure) {
                    _uiState.value = _uiState.value.copy(
                        isLoading = false,
                        error = regResult.exceptionOrNull()?.message ?: "Registration failed"
                    )
                    return@launch
                }
                val loginResult = authApiClient.login(username, password)
                _uiState.value = if (loginResult.isSuccess) {
                    _uiState.value.copy(isLoading = false, isSuccess = true)
                } else {
                    _uiState.value.copy(
                        isLoading = false,
                        error = "Registered. Please log in.",
                        isLoginMode = true      // switch to login tab
                    )
                }
            }
        }
    }
}
