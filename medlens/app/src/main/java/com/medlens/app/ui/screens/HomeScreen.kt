package com.medlens.app.ui.screens

import androidx.compose.animation.animateColorAsState
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Analytics
import androidx.compose.material.icons.filled.Chat
import androidx.compose.material.icons.filled.Description
import androidx.compose.material.icons.filled.LocalHospital
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.medlens.app.ui.theme.StatusError
import com.medlens.app.ui.theme.StatusIdle
import com.medlens.app.ui.theme.StatusLoading
import com.medlens.app.ui.theme.StatusReady
import com.medlens.app.viewmodel.HomeViewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HomeScreen(
    viewModel: HomeViewModel,
    onNavigateToImageAnalysis: () -> Unit,
    onNavigateToClinicalAssistant: () -> Unit,
    onNavigateToReport: () -> Unit
) {
    val state by viewModel.uiState.collectAsState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Spacer(modifier = Modifier.height(16.dp))

        // ── App Header ──────────────────────────────────────────────────────
        Icon(
            Icons.Default.LocalHospital,
            contentDescription = null,
            modifier = Modifier.size(64.dp),
            tint = MaterialTheme.colorScheme.primary
        )
        Spacer(modifier = Modifier.height(8.dp))
        Text(
            "MedLens",
            style = MaterialTheme.typography.headlineLarge,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.primary
        )
        Text(
            "On-Device Medical AI",
            style = MaterialTheme.typography.titleMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )

        Spacer(modifier = Modifier.height(24.dp))

        // ── Model Status Cards ──────────────────────────────────────────────
        ModelStatusCard(
            name = "BiomedCLIP",
            description = "Vision Encoder • INT8 ONNX",
            status = state.clipStatus,
            sizeMb = state.clipSizeMb,
            loadTimeMs = state.clipLoadTimeMs
        )
        Spacer(modifier = Modifier.height(8.dp))
        ModelStatusCard(
            name = "MedGemma 4B",
            description = "Language Model • Q4_K_S GGUF",
            status = state.gemmaStatus,
            sizeMb = state.gemmaSizeMb,
            loadTimeMs = state.gemmaLoadTimeMs
        )

        Spacer(modifier = Modifier.height(8.dp))

        // Status message
        Text(
            state.statusMessage,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center,
            modifier = Modifier.fillMaxWidth()
        )

        Spacer(modifier = Modifier.height(24.dp))

        // ── Action Cards ────────────────────────────────────────────────────
        ActionCard(
            icon = Icons.Default.Analytics,
            title = "Analyze Image",
            description = "Run BiomedCLIP on a medical image to get embeddings",
            enabled = state.clipStatus == HomeViewModel.ModelStatus.READY,
            onClick = onNavigateToImageAnalysis
        )
        Spacer(modifier = Modifier.height(12.dp))

        ActionCard(
            icon = Icons.Default.Chat,
            title = "Clinical Assistant",
            description = "Chat with MedGemma about medical topics",
            enabled = state.gemmaStatus == HomeViewModel.ModelStatus.READY,
            onClick = onNavigateToClinicalAssistant
        )
        Spacer(modifier = Modifier.height(12.dp))

        ActionCard(
            icon = Icons.Default.Description,
            title = "Full Report",
            description = "Image → BiomedCLIP → MedGemma → Clinical Report",
            enabled = state.clipStatus == HomeViewModel.ModelStatus.READY &&
                      state.gemmaStatus == HomeViewModel.ModelStatus.READY,
            onClick = onNavigateToReport
        )

        Spacer(modifier = Modifier.height(24.dp))

        // ── Disclaimer ──────────────────────────────────────────────────────
        Card(
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.errorContainer.copy(alpha = 0.3f)
            ),
            modifier = Modifier.fillMaxWidth()
        ) {
            Text(
                "⚠ For research and educational purposes only. " +
                "Not intended for clinical diagnosis. Always consult healthcare professionals.",
                style = MaterialTheme.typography.bodySmall,
                modifier = Modifier.padding(12.dp),
                color = MaterialTheme.colorScheme.onErrorContainer,
                textAlign = TextAlign.Center
            )
        }

        Spacer(modifier = Modifier.height(16.dp))
    }
}

@Composable
private fun ModelStatusCard(
    name: String,
    description: String,
    status: HomeViewModel.ModelStatus,
    sizeMb: Float,
    loadTimeMs: Long
) {
    val statusColor by animateColorAsState(
        targetValue = when (status) {
            HomeViewModel.ModelStatus.READY -> StatusReady
            HomeViewModel.ModelStatus.LOADING -> StatusLoading
            HomeViewModel.ModelStatus.ERROR -> StatusError
            HomeViewModel.ModelStatus.FOUND -> StatusLoading
            HomeViewModel.ModelStatus.NOT_FOUND -> StatusIdle
        },
        label = "statusColor"
    )

    val statusText = when (status) {
        HomeViewModel.ModelStatus.READY -> "Ready"
        HomeViewModel.ModelStatus.LOADING -> "Loading..."
        HomeViewModel.ModelStatus.ERROR -> "Error"
        HomeViewModel.ModelStatus.FOUND -> "Found"
        HomeViewModel.ModelStatus.NOT_FOUND -> "Not Found"
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant
        )
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // Status indicator
            Surface(
                modifier = Modifier.size(12.dp),
                shape = MaterialTheme.shapes.small,
                color = statusColor
            ) {}

            Spacer(modifier = Modifier.width(12.dp))

            Column(modifier = Modifier.weight(1f)) {
                Text(name, style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold)
                Text(description, style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            }

            Column(horizontalAlignment = Alignment.End) {
                Text(statusText, style = MaterialTheme.typography.labelMedium, color = statusColor)
                if (status == HomeViewModel.ModelStatus.READY && sizeMb > 0) {
                    Text(
                        "${"%.0f".format(sizeMb)} MB • ${loadTimeMs}ms",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
                if (status == HomeViewModel.ModelStatus.LOADING) {
                    Spacer(modifier = Modifier.height(4.dp))
                    LinearProgressIndicator(modifier = Modifier.width(60.dp))
                }
            }
        }
    }
}

@Composable
private fun ActionCard(
    icon: ImageVector,
    title: String,
    description: String,
    enabled: Boolean,
    onClick: () -> Unit
) {
    Card(
        onClick = onClick,
        enabled = enabled,
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = if (enabled)
                MaterialTheme.colorScheme.primaryContainer
            else
                MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f)
        )
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(
                icon,
                contentDescription = null,
                modifier = Modifier.size(40.dp),
                tint = if (enabled)
                    MaterialTheme.colorScheme.onPrimaryContainer
                else
                    MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.5f)
            )
            Spacer(modifier = Modifier.width(16.dp))
            Column {
                Text(
                    title,
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold,
                    color = if (enabled)
                        MaterialTheme.colorScheme.onPrimaryContainer
                    else
                        MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.5f)
                )
                Text(
                    description,
                    style = MaterialTheme.typography.bodySmall,
                    color = if (enabled)
                        MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.7f)
                    else
                        MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.4f)
                )
            }
        }
    }
}
