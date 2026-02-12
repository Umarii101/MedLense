package com.medgemma.edge.ui

import android.net.Uri
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.DeleteOutline
import androidx.compose.material.icons.filled.Image
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil.compose.AsyncImage
import com.medgemma.edge.ChatViewModel

/**
 * Main chat screen composable.
 * Displays message history, image previews, streaming responses,
 * and input controls (text, camera, gallery).
 */
@OptIn(ExperimentalMaterial3Api::class, ExperimentalLayoutApi::class)
@Composable
fun ChatScreen(
    viewModel: ChatViewModel,
    pendingImageUri: Uri? = null,
    onPendingImageClear: () -> Unit = {},
    onPickImage: () -> Unit,
    onOpenCamera: () -> Unit
) {
    val modelStatus by viewModel.modelStatus.collectAsState()
    val isGenerating by viewModel.isGenerating.collectAsState()
    val streamingStats by viewModel.streamingStats.collectAsState()
    var inputText by remember { mutableStateOf("") }
    // Local pending image state — can be set by gallery picker or camera capture
    var localPendingImage by remember { mutableStateOf<Uri?>(null) }
    val listState = rememberLazyListState()

    // Accept pending image from parent (camera/gallery)
    LaunchedEffect(pendingImageUri) {
        if (pendingImageUri != null) {
            localPendingImage = pendingImageUri
        }
    }

    // Auto-scroll to bottom when messages change
    LaunchedEffect(viewModel.messages.size) {
        if (viewModel.messages.isNotEmpty()) {
            listState.animateScrollToItem(viewModel.messages.size - 1)
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text(
                            "MedLens",
                            style = MaterialTheme.typography.titleLarge,
                            fontWeight = FontWeight.Bold
                        )
                        val subtitle = when {
                            modelStatus.isLoading -> modelStatus.loadingMessage
                            modelStatus.errorMessage != null -> modelStatus.errorMessage!!
                            streamingStats.isNotEmpty() && isGenerating -> streamingStats
                            else -> buildString {
                                if (modelStatus.biomedClipLoaded) append("Vision ✓  ")
                                if (modelStatus.medGemmaLoaded) append("LLM ✓  ")
                                if (modelStatus.classifierLoaded) append("Classifier ✓")
                            }
                        }
                        Text(
                            subtitle,
                            style = MaterialTheme.typography.bodySmall,
                            color = if (modelStatus.errorMessage != null)
                                MaterialTheme.colorScheme.error
                            else
                                MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                },
                actions = {
                    if (viewModel.messages.isNotEmpty() && !isGenerating) {
                        IconButton(onClick = { viewModel.clearChat() }) {
                            Icon(
                                Icons.Default.DeleteOutline,
                                contentDescription = "Clear chat"
                            )
                        }
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.surface
                )
            )
        }
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .imePadding()
        ) {
            // Loading indicator
            AnimatedVisibility(visible = modelStatus.isLoading) {
                LinearProgressIndicator(
                    modifier = Modifier.fillMaxWidth()
                )
            }

            // Messages list
            LazyColumn(
                state = listState,
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth(),
                contentPadding = PaddingValues(horizontal = 12.dp, vertical = 8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                // Welcome message when empty
                if (viewModel.messages.isEmpty()) {
                    item {
                        WelcomeCard()
                    }
                }

                items(viewModel.messages) { message ->
                    MessageBubble(message)
                }
            }

            // Suggestion chips (when no messages)
            if (viewModel.messages.isEmpty() && !modelStatus.isLoading) {
                FlowRow(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 12.dp, vertical = 4.dp),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    SuggestionChip("What is pneumonia?") { inputText = it }
                    SuggestionChip("Explain chest X-ray findings") { inputText = it }
                    SuggestionChip("Skin lesion types") { inputText = it }
                    SuggestionChip("Diabetic retinopathy signs") { inputText = it }
                }
            }

            // Pending image preview
            localPendingImage?.let { uri ->
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 12.dp, vertical = 4.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    AsyncImage(
                        model = uri,
                        contentDescription = "Selected image",
                        modifier = Modifier
                            .size(56.dp)
                            .clip(RoundedCornerShape(8.dp)),
                        contentScale = ContentScale.Crop
                    )
                    Spacer(Modifier.width(8.dp))
                    Text(
                        "Add a message or tap send to analyze",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    Spacer(Modifier.weight(1f))
                    IconButton(onClick = {
                        localPendingImage = null
                        onPendingImageClear()
                    }) {
                        Icon(Icons.Default.Close, "Remove image", modifier = Modifier.size(18.dp))
                    }
                }
            }

            // Input bar
            InputBar(
                text = inputText,
                onTextChange = { inputText = it },
                isGenerating = isGenerating,
                hasImage = localPendingImage != null,
                onSend = {
                    val uri = localPendingImage
                    if (uri != null) {
                        viewModel.analyzeImage(uri, inputText.ifBlank { "Analyze this medical image." })
                        localPendingImage = null
                        onPendingImageClear()
                    } else {
                        viewModel.sendTextMessage(inputText)
                    }
                    inputText = ""
                },
                onStop = { viewModel.stopGeneration() },
                onPickImage = onPickImage,
                onOpenCamera = onOpenCamera
            )

            // Medical disclaimer
            Text(
                "⚕ AI-assisted analysis for educational purposes only. Not a medical diagnosis.",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp, vertical = 4.dp)
            )
        }
    }
}

// ── Sub-composables ─────────────────────────────────────────────────────

@Composable
private fun WelcomeCard() {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.3f)
        )
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                "Welcome to MedLens",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )
            Spacer(Modifier.height(4.dp))
            Text(
                "AI-powered medical image analysis running entirely on your device. " +
                "No data leaves your phone.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Spacer(Modifier.height(8.dp))
            Text(
                "• Take a photo or pick a medical image\n" +
                "• BiomedCLIP analyzes the image locally\n" +
                "• MedGemma provides clinical insights\n" +
                "• Ask follow-up questions naturally",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                lineHeight = 20.sp
            )
        }
    }
}

@Composable
private fun SuggestionChip(text: String, onClick: (String) -> Unit) {
    AssistChip(
        onClick = { onClick(text) },
        label = {
            Text(text, maxLines = 1, overflow = TextOverflow.Ellipsis)
        }
    )
}

@Composable
private fun MessageBubble(message: ChatViewModel.ChatMessage) {
    val isUser = message.role == ChatViewModel.MessageRole.USER
    val screenWidth = LocalConfiguration.current.screenWidthDp.dp

    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = if (isUser) Arrangement.End else Arrangement.Start
    ) {
        Surface(
            shape = RoundedCornerShape(
                topStart = 16.dp,
                topEnd = 16.dp,
                bottomStart = if (isUser) 16.dp else 4.dp,
                bottomEnd = if (isUser) 4.dp else 16.dp
            ),
            color = if (isUser)
                MaterialTheme.colorScheme.primaryContainer
            else
                MaterialTheme.colorScheme.surfaceContainerHigh,
            modifier = Modifier.widthIn(max = screenWidth * 0.85f)
        ) {
            Column(modifier = Modifier.padding(12.dp)) {
                // Image preview (for user messages with images)
                message.imageUri?.let { uri ->
                    AsyncImage(
                        model = uri,
                        contentDescription = "Medical image",
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(180.dp)
                            .clip(RoundedCornerShape(8.dp)),
                        contentScale = ContentScale.Crop
                    )
                    Spacer(Modifier.height(8.dp))
                }

                // Classification badge
                message.classificationSummary?.let { summary ->
                    Surface(
                        shape = RoundedCornerShape(12.dp),
                        color = MaterialTheme.colorScheme.tertiaryContainer,
                        modifier = Modifier.padding(bottom = 6.dp)
                    ) {
                        Text(
                            text = "🔬 $summary",
                            style = MaterialTheme.typography.labelSmall,
                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                            color = MaterialTheme.colorScheme.onTertiaryContainer
                        )
                    }
                }

                // Message text
                if (message.text.isNotEmpty()) {
                    Text(
                        text = message.text,
                        style = MaterialTheme.typography.bodyMedium,
                        color = if (isUser)
                            MaterialTheme.colorScheme.onPrimaryContainer
                        else
                            MaterialTheme.colorScheme.onSurface
                    )
                }

                // Streaming indicator
                if (message.isStreaming) {
                    Spacer(Modifier.height(4.dp))
                    StreamingDots()
                }
            }
        }
    }
}

@Composable
private fun StreamingDots() {
    val infiniteTransition = rememberInfiniteTransition(label = "dots")
    val alpha by infiniteTransition.animateFloat(
        initialValue = 0.3f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(600, easing = LinearEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "alpha"
    )
    Text(
        "●  ●  ●",
        style = MaterialTheme.typography.bodySmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
        modifier = Modifier.graphicsLayer { this.alpha = alpha }
    )
}

@Composable
private fun InputBar(
    text: String,
    onTextChange: (String) -> Unit,
    isGenerating: Boolean,
    hasImage: Boolean,
    onSend: () -> Unit,
    onStop: () -> Unit,
    onPickImage: () -> Unit,
    onOpenCamera: () -> Unit
) {
    Surface(
        tonalElevation = 3.dp,
        modifier = Modifier.fillMaxWidth()
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 8.dp, vertical = 6.dp),
            verticalAlignment = Alignment.Bottom
        ) {
            // Camera button
            IconButton(onClick = onOpenCamera, enabled = !isGenerating) {
                Icon(
                    Icons.Default.CameraAlt,
                    contentDescription = "Take photo",
                    tint = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }

            // Gallery button
            IconButton(onClick = onPickImage, enabled = !isGenerating) {
                Icon(
                    Icons.Default.Image,
                    contentDescription = "Pick image",
                    tint = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }

            // Text input
            OutlinedTextField(
                value = text,
                onValueChange = onTextChange,
                modifier = Modifier.weight(1f),
                placeholder = {
                    Text(
                        if (hasImage) "Describe what to analyze..."
                        else "Ask a medical question...",
                        style = MaterialTheme.typography.bodyMedium
                    )
                },
                maxLines = 4,
                shape = RoundedCornerShape(24.dp),
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = MaterialTheme.colorScheme.primary,
                    unfocusedBorderColor = MaterialTheme.colorScheme.outlineVariant
                ),
                enabled = !isGenerating
            )

            Spacer(Modifier.width(4.dp))

            // Send / Stop button
            if (isGenerating) {
                IconButton(
                    onClick = onStop,
                    modifier = Modifier
                        .size(48.dp)
                        .background(
                            MaterialTheme.colorScheme.errorContainer,
                            CircleShape
                        )
                ) {
                    Icon(
                        Icons.Default.Stop,
                        contentDescription = "Stop",
                        tint = MaterialTheme.colorScheme.onErrorContainer
                    )
                }
            } else {
                IconButton(
                    onClick = onSend,
                    enabled = text.isNotBlank() || hasImage,
                    modifier = Modifier
                        .size(48.dp)
                        .background(
                            if (text.isNotBlank() || hasImage)
                                MaterialTheme.colorScheme.primary
                            else
                                MaterialTheme.colorScheme.surfaceContainerHigh,
                            CircleShape
                        )
                ) {
                    Icon(
                        Icons.AutoMirrored.Filled.Send,
                        contentDescription = "Send",
                        tint = if (text.isNotBlank() || hasImage)
                            MaterialTheme.colorScheme.onPrimary
                        else
                            MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
        }
    }
}
