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
import androidx.core.content.ContextCompat
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navArgument
import com.medgemma.edge.ui.CameraCapture
import com.medgemma.edge.ui.ChatListScreen
import com.medgemma.edge.ui.ChatListViewModel
import com.medgemma.edge.ui.ChatScreen
import com.medgemma.edge.ui.LoginScreen
import com.medgemma.edge.ui.LoginViewModel
import com.medgemma.edge.ui.SettingsScreen
import com.medgemma.edge.ui.SettingsViewModel
import com.medgemma.edge.ui.theme.MedGemmaEdgeTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        AppPreferences.init(applicationContext)
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
    
    // ViewModels
    val chatViewModel: ChatViewModel = viewModel()
    val chatListViewModel: ChatListViewModel = viewModel()
    val loginViewModel: LoginViewModel = viewModel()
    val settingsViewModel: SettingsViewModel = viewModel()

    //  Permission state 
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

    // Trigger model loading once storage permission is available
    LaunchedEffect(hasStoragePermission) {
        if (hasStoragePermission) {
            chatViewModel.loadModels()
        }
    }

    //  Activity result launchers 
    val showCamera = remember { mutableStateOf(false) }

    val cameraPermissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        hasCameraPermission = granted
        if (granted) {
            showCamera.value = true
        } else {
            Toast.makeText(context, "Camera permission required", Toast.LENGTH_SHORT).show()
        }
    }

    val pendingImageUri = remember { mutableStateOf<Uri?>(null) }
    val cameraCapturedUri = remember { mutableStateOf<Uri?>(null) }

    val imagePickerLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri ->
        if (uri != null) {
            pendingImageUri.value = uri
        }
    }

    LaunchedEffect(pendingImageUri.value) {
        pendingImageUri.value?.let { uri ->
            cameraCapturedUri.value = uri
            pendingImageUri.value = null
        }
    }

    val navController = rememberNavController()
    val startDestination = if (AppPreferences.accessToken != null) "chat_list" else "login"
    
    //  UI 
    if (!hasStoragePermission) {
        StoragePermissionScreen(context)
    } else {
        NavHost(navController = navController, startDestination = startDestination) {
            composable("login") {
                LoginScreen(
                    viewModel = loginViewModel,
                    onLoginSuccess = {
                        navController.navigate("chat_list") {
                            popUpTo("login") { inclusive = true }
                        }
                    },
                    onNavigateToSettings = {
                        navController.navigate("settings")
                    }
                )
            }
            
            composable("settings") {
                SettingsScreen(
                    viewModel = settingsViewModel,
                    onNavigateBack = {
                        navController.popBackStack()
                    }
                )
            }
            
            composable("chat_list") {
                ChatListScreen(
                    viewModel = chatListViewModel,
                    onNavigateToChat = { chatId ->
                        if (chatId != null) {
                            navController.navigate("chat?chatId=$chatId")
                        } else {
                            navController.navigate("chat")
                        }
                    },
                    onNavigateToSettings = {
                        navController.navigate("settings")
                    },
                    onLogout = {
                        chatListViewModel.logout()
                        loginViewModel.resetState()
                        navController.navigate("login") {
                            popUpTo("chat_list") { inclusive = true }
                        }
                    }
                )
            }
            
            composable(
                route = "chat?chatId={chatId}",
                arguments = listOf(navArgument("chatId") {
                    type = NavType.StringType
                    nullable = true
                })
            ) { backStackEntry ->
                val chatIdStr = backStackEntry.arguments?.getString("chatId")
                val chatId = chatIdStr?.toIntOrNull()
                
                if (showCamera.value) {
                    CameraCapture(
                        onImageCaptured = { uri ->
                            showCamera.value = false
                            cameraCapturedUri.value = uri
                        },
                        onClose = { showCamera.value = false }
                    )
                } else {
                    ChatScreen(
                        viewModel = chatViewModel,
                        chatId = chatId,
                        pendingImageUri = cameraCapturedUri.value,
                        onPendingImageClear = { cameraCapturedUri.value = null },
                        onPickImage = { imagePickerLauncher.launch("image/*") },
                        onOpenCamera = {
                            if (hasCameraPermission) {
                                showCamera.value = true
                            } else {
                                cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
                            }
                        },
                        onNavigateBack = {
                            // If returning to chat array, reset states
                            chatViewModel.clearChat()
                            navController.popBackStack()
                        }
                    )
                }
            }
        }
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
            //noinspection SpellCheckingInspection
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
