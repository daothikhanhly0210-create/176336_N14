"""Source package for the BRSET diabetic retinopathy project.

Modules
-------
``src.data``          Dataset, dataloaders and patient level splitting.
``src.preprocessing`` ROI cropping, CLAHE and circular masking.
``src.models``        Backbones, losses and the training loop.
``src.evaluation``    Metrics and error analysis.
``src.xai``           Grad-CAM explanations.
``src.utils``         Config loading and logging.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
