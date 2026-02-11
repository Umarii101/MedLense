package com.medlens.app

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.Settings
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.compose.rememberNavController
import com.medlens.app.navigation.MedLensNavGraph
import com.medlens.app.ui.theme.MedLensTheme
import com.medlens.app.viewmodel.HomeViewModel

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            MedLensTheme {
                MedLensApp()
            }
        }
    }
}

@Composable
fun MedLensApp() {
    val navController = rememberNavController()
    val homeViewModel: HomeViewModel = viewModel()
    val state by homeViewModel.uiState.collectAsState()

    var hasPermission by remember { mutableStateOf(checkStoragePermission()) }

    // Permission launcher for MANAGE_EXTERNAL_STORAGE (Android 11+)
    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.StartActivityForResult()
    ) {
        hasPermission = checkStoragePermission()
        if (hasPermission) {
            homeViewModel.onStoragePermissionGranted()
        }
    }

    // Legacy permission launcher (Android < 11)
    val legacyPermissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { granted ->
        hasPermission = granted
        if (granted) {
            homeViewModel.onStoragePermissionGranted()
        }
    }

    // Request permission on first load
    LaunchedEffect(Unit) {
        if (hasPermission) {
            homeViewModel.onStoragePermissionGranted()
        }
    }

    Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
        if (!hasPermission) {
            // Permission request screen
            PermissionScreen(
                modifier = Modifier.padding(innerPadding),
                onRequestPermission = {
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                        val intent = Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION).apply {
                            data = Uri.parse("package:com.medlens.app")
                        }
                        permissionLauncher.launch(intent)
                    } else {
                        legacyPermissionLauncher.launch(Manifest.permission.READ_EXTERNAL_STORAGE)
                    }
                }
            )
        } else {
            MedLensNavGraph(
                navController = navController,
                homeViewModel = homeViewModel,
                modifier = Modifier.padding(innerPadding)
            )
        }
    }
}

@Composable
private fun PermissionScreen(
    modifier: Modifier = Modifier,
    onRequestPermission: () -> Unit
) {
    Box(
        modifier = modifier.fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        Card(modifier = Modifier.padding(32.dp)) {
            Column(
                modifier = Modifier.padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(
                    "Storage Permission Required",
                    style = MaterialTheme.typography.titleLarge
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    "MedLens needs access to read model files from device storage " +
                    "(/sdcard/MedGemmaEdge/).",
                    style = MaterialTheme.typography.bodyMedium
                )
                Spacer(modifier = Modifier.height(16.dp))
                Button(onClick = onRequestPermission) {
                    Text("Grant Permission")
                }
            }
        }
    }
}

private fun checkStoragePermission(): Boolean {
    return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
        Environment.isExternalStorageManager()
    } else {
        true // Handled by runtime permission
    }
}