# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- Support for GTSRB dataset's _samples attribute
- Enhanced plot visualization with clean baseline
- Improved bar plotting with proper grouping
- Added proper JSON to CSV conversion for experiment results
- Added detailed experiment metrics in consolidated CSV output

### Changed

- Standardized checkpoint file extensions to .pt
- Improved checkpoint handling with better validation
- Enhanced model saving/loading with device support
- Removed seaborn dependency for plotting
- Simplified training loop checkpoint handling
- Updated progress bars to show only poisoned samples
- Improved experiment results consolidation workflow

### Fixed

- Correct dataset reference in poison_dataset
- Handle both ImageFolder and GTSRB dataset types
- Resolve data alignment issues in plots
- Rename all instances of 'Genetic Algorithm (GA)' to 'Gradient Ascent (GA)' for accuracy
- Fixed progress bar totals in all poisoning attacks
- Fixed results consolidation to properly handle JSON to CSV conversion
