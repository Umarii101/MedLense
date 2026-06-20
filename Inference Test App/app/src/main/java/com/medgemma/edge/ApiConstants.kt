package com.medgemma.edge

object ApiConstants {
    val getBaseUrl: String
        get() = AppPreferences.serverBaseUrl.removeSuffix("/") + "/api/"

    const val ENDPOINT_REGISTER = "register/"
    const val ENDPOINT_LOGIN = "login/"
    const val ENDPOINT_CHAT = "chat/"
    const val ENDPOINT_CHATS = "chats/"
}
