package com.medlens.app.navigation

import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import com.medlens.app.ui.screens.ClinicalAssistantScreen
import com.medlens.app.ui.screens.HomeScreen
import com.medlens.app.ui.screens.ImageAnalysisScreen
import com.medlens.app.ui.screens.ReportScreen
import com.medlens.app.viewmodel.ClinicalAssistantViewModel
import com.medlens.app.viewmodel.HomeViewModel
import com.medlens.app.viewmodel.ImageAnalysisViewModel
import com.medlens.app.viewmodel.ReportViewModel

@Composable
fun MedLensNavGraph(
    navController: NavHostController,
    homeViewModel: HomeViewModel,
    modifier: Modifier = Modifier
) {
    NavHost(
        navController = navController,
        startDestination = Routes.HOME,
        modifier = modifier
    ) {
        composable(Routes.HOME) {
            HomeScreen(
                viewModel = homeViewModel,
                onNavigateToImageAnalysis = { navController.navigate(Routes.IMAGE_ANALYSIS) },
                onNavigateToClinicalAssistant = { navController.navigate(Routes.CLINICAL_ASSISTANT) },
                onNavigateToReport = { navController.navigate(Routes.REPORT) }
            )
        }

        composable(Routes.IMAGE_ANALYSIS) {
            val viewModel: ImageAnalysisViewModel = viewModel(
                factory = ImageAnalysisViewModel.Factory(homeViewModel.biomedClip)
            )
            ImageAnalysisScreen(
                viewModel = viewModel,
                onBack = { navController.popBackStack() }
            )
        }

        composable(Routes.CLINICAL_ASSISTANT) {
            val viewModel: ClinicalAssistantViewModel = viewModel(
                factory = ClinicalAssistantViewModel.Factory(homeViewModel.medGemma)
            )
            ClinicalAssistantScreen(
                viewModel = viewModel,
                onBack = { navController.popBackStack() }
            )
        }

        composable(Routes.REPORT) {
            val viewModel: ReportViewModel = viewModel(
                factory = ReportViewModel.Factory(
                    homeViewModel.biomedClip,
                    homeViewModel.medGemma
                )
            )
            ReportScreen(
                viewModel = viewModel,
                onBack = { navController.popBackStack() }
            )
        }
    }
}
