package com.medlens.app.ui.theme

import android.app.Activity
import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext

private val DarkColorScheme = darkColorScheme(
    primary = MedTeal80,
    secondary = MedBlue80,
    tertiary = MedGreen80,
    background = Color(0xFF0F1A1C),
    surface = Color(0xFF1A2628),
    surfaceVariant = CardBackgroundDark
)

private val LightColorScheme = lightColorScheme(
    primary = MedTeal40,
    secondary = MedBlue40,
    tertiary = MedGreen40,
    background = Color(0xFFF8FBFC),
    surface = Color.White,
    surfaceVariant = CardBackground
)

@Composable
fun MedLensTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    dynamicColor: Boolean = false, // Disabled: use our medical palette
    content: @Composable () -> Unit
) {
    val colorScheme = when {
        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {
            val context = LocalContext.current
            if (darkTheme) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
        }
        darkTheme -> DarkColorScheme
        else -> LightColorScheme
    }

    MaterialTheme(
        colorScheme = colorScheme,
        typography = Typography,
        content = content
    )
}