package com.medgemma.edge

import android.content.Context
import android.content.SharedPreferences

object AppPreferences {
    private const val PREF_NAME = "com.medgemma.edge.prefs"
    private lateinit var prefs: SharedPreferences

    fun init(context: Context) {
        prefs = context.applicationContext.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)
    }

    var serverBaseUrl: String
        get() = prefs.getString("server_base_url", "https://077b538r-8000.asse.devtunnels.ms") ?: "https://077b538r-8000.asse.devtunnels.ms"
        set(value) = prefs.edit().putString("server_base_url", value).apply()

    var accessToken: String?
        get() = prefs.getString("access_token", null)
        set(value) {
            if (value == null) {
                prefs.edit().remove("access_token").apply()
            } else {
                prefs.edit().putString("access_token", value).apply()
            }
        }
}
