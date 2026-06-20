package com.medgemma.edge.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.ui.graphics.Color
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable

/**
 * MedLens always runs in dark mode.
 * Dynamic color is disabled â€?we enforce the teal medical palette.
 */
private val MedLensColorScheme = darkColorScheme(
    // Primary â€?teal
    primary                = TealPrimary,
    onPrimary              = WhiteText,
    primaryContainer       = TealDark,
    onPrimaryContainer     = MintAccent,

    // Secondary â€?muted teal
    secondary              = TealLight,
    onSecondary            = NavyBackground,
    secondaryContainer     = CardSurfaceAlt,
    onSecondaryContainer   = SilverText,

    // Tertiary â€?amber (Fast mode accent)
    tertiary               = AmberAccent,
    onTertiary             = NavyBackground,
    tertiaryContainer      = Color(0xFF2D1F00),
    onTertiaryContainer    = AmberAccent,

    // Error
    error                  = ErrorRed,
    onError                = WhiteText,
    errorContainer         = ErrorContainer,
    onErrorContainer       = ErrorRed,

    // Backgrounds
    background             = NavyBackground,
    onBackground           = WhiteText,

    // Surface hierarchy
    surface                = CardSurface,
    onSurface              = SilverText,
    surfaceVariant         = DarkSurface,
    onSurfaceVariant       = MutedText,
    surfaceContainerHigh   = CardSurfaceAlt,
    surfaceContainerHighest= DarkSurface,

    // Borders
    outline                = MutedText,
    outlineVariant         = DarkSurface,
)

@Composable
fun MedGemmaEdgeTheme(
    content: @Composable () -> Unit
) {
    MaterialTheme(
        colorScheme = MedLensColorScheme,
        typography = Typography,
        content = content
    )
}
