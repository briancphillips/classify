# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- Support for GTSRB dataset's \_samples attribute
- Enhanced plot visualization with clean baseline
- Improved bar plotting with proper grouping

### Changed

- Standardized checkpoint file extensions to .pt
- Improved checkpoint handling with better validation
- Enhanced model saving/loading with device support
- Removed seaborn dependency for plotting
- Simplified training loop checkpoint handling

### Fixed

- Correct dataset reference in poison_dataset
- Handle both ImageFolder and GTSRB dataset types
- Resolve data alignment issues in plots
- Rename all instances of 'Genetic Algorithm (GA)' to 'Gradient Ascent (GA)' for accuracy
