package com.medgemma.edge

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.Settings
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.lifecycle.viewmodel.compose.viewModel
import com.medgemma.edge.ui.CameraCapture
import com.medgemma.edge.ui.ChatScreen
import com.medgemma.edge.ui.theme.MedGemmaEdgeTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            MedGemmaEdgeTheme {
                MedLensApp()
            }
        }
    }
}

@Composable
fun MedLensApp() {
    val context = LocalContext.current
    val viewModel: ChatViewModel = viewModel()

    // ── Permission state ────────────────────────────────────────────────
    var hasStoragePermission by remember {
        mutableStateOf(
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                Environment.isExternalStorageManager()
            } else true
        )
    }

    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) ==
                    PackageManager.PERMISSION_GRANTED
        )
    }

    // Re-check permissions on resume
    val lifecycleOwner = androidx.lifecycle.compose.LocalLifecycleOwner.current
    DisposableEffect(lifecycleOwner) {
        val observer = androidx.lifecycle.LifecycleEventObserver { _, event ->
            if (event == androidx.lifecycle.Lifecycle.Event.ON_RESUME) {
                hasStoragePermission = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                    Environment.isExternalStorageManager()
                } else true
                hasCameraPermission = ContextCompat.checkSelfPermission(
                    context, Manifest.permission.CAMERA
                ) == PackageManager.PERMISSION_GRANTED
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose { lifecycleOwner.lifecycle.removeObserver(observer) }
    }

    // ── Activity result launchers ───────────────────────────────────────

    // Camera permission launcher
    val cameraPermissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        hasCameraPermission = granted
        if (!granted) {
            Toast.makeText(context, "Camera permission required", Toast.LENGTH_SHORT).show()
        }
    }

    // Image picker launcher
    var pendingImageUri by remember { mutableStateOf<Uri?>(null) }
    val imagePickerLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri ->
        if (uri != null) {
            pendingImageUri = uri
        }
    }

    // ── Camera state ────────────────────────────────────────────────────
    var showCamera by remember { mutableStateOf(false) }

    // State for image captured from camera — shown as pending in chat
    var cameraCapturedUri by remember { mutableStateOf<Uri?>(null) }

    // Handle gallery-picked image → set as pending in ChatScreen
    LaunchedEffect(pendingImageUri) {
        pendingImageUri?.let { uri ->
            cameraCapturedUri = uri
            pendingImageUri = null
        }
    }

    // ── UI ──────────────────────────────────────────────────────────────

    // Trigger model loading once storage permission is available
    LaunchedEffect(hasStoragePermission) {
        if (hasStoragePermission) {
            viewModel.loadModels()
        }
    }

    if (!hasStoragePermission) {
        // Storage permission gate
        StoragePermissionScreen(context)
    } else if (showCamera) {
        // Full-screen camera — captured image goes to pending preview
        CameraCapture(
            onImageCaptured = { uri ->
                showCamera = false
                cameraCapturedUri = uri
            },
            onClose = { showCamera = false }
        )
    } else {
        // Main chat interface
        ChatScreen(
            viewModel = viewModel,
            pendingImageUri = cameraCapturedUri,
            onPendingImageClear = { cameraCapturedUri = null },
            onPickImage = { imagePickerLauncher.launch("image/*") },
            onOpenCamera = {
                if (hasCameraPermission) {
                    showCamera = true
                } else {
                    cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
                }
            }
        )
    }
}

@Composable
private fun StoragePermissionScreen(context: android.content.Context) {
    Scaffold { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(24.dp),
            verticalArrangement = Arrangement.Center
        ) {
            Text(
                "Storage Permission Required",
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Bold
            )
            Spacer(Modifier.height(12.dp))
            Text(
                "MedLens needs \"All Files Access\" to read model files (.onnx, .gguf) " +
                        "from /sdcard/MedGemmaEdge/.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Spacer(Modifier.height(16.dp))
            Button(
                onClick = {
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                        val intent = Intent(
                            Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION,
                            Uri.parse("package:${context.packageName}")
                        )
                        context.startActivity(intent)
                    }
                }
            ) {
                Text("Grant Storage Permission")
            }
        }
    }
}