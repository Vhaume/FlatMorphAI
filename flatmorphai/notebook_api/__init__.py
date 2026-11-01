"""
Thin wrappers around the Jupyter-notebook functions listed in the v2
specification (sections 3 and 6).

The spec forbids copy-pasting inference code into worker classes. These
modules import the already-existing Python files from the working
directory (``custom_dataset``, ``model_wrapper``,
``datasets``, ``polyloss``, ``improved_mask2former_wrapper``,
``loss_initialization``, ``thin_worm_patch``, ``refine_1x_patch``,
``train_tfms_320``, ``evaluate_test_images_v2``) and expose the six
public notebook entry points expected by the dashboard:

``process_video``, ``process_all_videos_in_directory``
``load_model_for_segmentation``, ``forecast_mask_with_a_mask2former``
``get_mask_folders``, ``generate_stereotypie_image_for_classification``
``process_videos``,
``forecast_classes_in_image_dataloader``
``modify_column_of_row_in_csv``
``fuse_videos_into_periods``, ``compute_grouped_statistics``

References for the planarian-stereotype pipeline that these wrappers
implement:
  - Zhang et al. (2011), "Automated image-based tracking and its
    application in ecology", Trends Ecol. Evol. 26(8), 420-428,
    DOI 10.1016/j.tree.2011.04.004
  - Cheng et al. (2022), "Masked-attention Mask Transformer for Universal
    Image Segmentation" (Mask2Former), arXiv:2112.01527
  - Hatamizadeh et al. (2023), "FasterViT: Fast Vision Transformers with
    Hierarchical Attention", arXiv:2306.06189
"""
