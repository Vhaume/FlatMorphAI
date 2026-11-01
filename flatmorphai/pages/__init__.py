"""Stacked-page widgets for each step and pipeline."""

from .base_page import BasePage, create_worker_thread
from .step1_page import Step1Page
from .step2_page import Step2Page
from .step3_page import Step3Page
from .step4_page import Step4Page
from .step5_page import Step5Page
from .step6_page import Step6Page
from .pipeline_a_page import PipelineAPage
from .pipeline_b_page import PipelineBPage
from .pipeline_c_page import PipelineCPage
from .stereotype_classification_page import StereotypeClassificationPage
from .pipeline_d_page import PipelineDPage
from .pipeline_e_page import PipelineEPage
from .regeneration_page import RegenerationPage
from .eyespots_counter_page import EyespotsCounterPage
from .regeneration_editor_page import RegenerationEditorPage
from .settings_page import SettingsPage
from .about_page import AboutPage

__all__ = [
    "BasePage",
    "create_worker_thread",
    "Step1Page",
    "Step2Page",
    "Step3Page",
    "Step4Page",
    "Step5Page",
    "Step6Page",
    "PipelineAPage",
    "PipelineBPage",
    "PipelineCPage",
    "StereotypeClassificationPage",
    "PipelineDPage",
    "PipelineEPage",
    "RegenerationPage",
    "EyespotsCounterPage",
    "RegenerationEditorPage",
    "SettingsPage",
    "AboutPage",
]
