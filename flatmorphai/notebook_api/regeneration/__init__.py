"""
Regeneration-state forecasting (page "Forecast regeneration states").

The package wraps the self-contained ``Export_head_detection`` folder
(segmentation, head cascade, eye-cup heat maps, blastema
segmenter, darkness index, state classifier) and the per-image
descriptor functions of its ``metrics/`` scripts, and orchestrates
them over every ``images/`` folder found under a processing root.

Modules
-------
naming           image-name convention ``PLATE_POSITION_DATE[_comment].ext``
export_bridge    lazy import of the export, model caches, parameter files
metrics_adapter  the 97 state descriptors, computed image by image
pipeline         masks -> eye cups -> blastema -> head -> thumbnails -> state
"""
