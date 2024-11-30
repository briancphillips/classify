# TODO List

## Remove Hardcoded Values

### poison.py
- [ ] Move DataLoader parameters to config:
  ```python
  poisoned_loader = DataLoader(
      poisoned_dataset,
      batch_size=128,  # Should come from config
      shuffle=True,
      num_workers=4,  # Should come from config
      pin_memory=True
  )
  ```

### run_experiments.py
- [ ] Remove hardcoded default poison_ratio:
  ```python
  "--poison-ratio", str(experiment.get("poison_ratio", 0.1))  # Remove hardcoded 0.1
  ```
  - Should use value from POISON_DEFAULTS instead

### traditional_classifiers.py
- [ ] Move classifier hyperparameters to config:
  ```python
  "KNN": KNeighborsClassifier(n_neighbors=3),  # Move to config
  "logistic_regression": LogisticRegression(max_iter=1000),  # Move to config
  "random_forest": RandomForestClassifier(n_estimators=100),  # Move to config
  ```

### Add New Config Sections in defaults.py
- [ ] Add DATALOADER_DEFAULTS:
  ```python
  DATALOADER_DEFAULTS = {
      "batch_size": 128,
      "num_workers": 4,
      "pin_memory": True,
      "shuffle": True
  }
  ```

- [ ] Add TRADITIONAL_CLASSIFIER_DEFAULTS:
  ```python
  TRADITIONAL_CLASSIFIER_DEFAULTS = {
      "knn": {
          "n_neighbors": 3
      },
      "logistic_regression": {
          "max_iter": 1000
      },
      "random_forest": {
          "n_estimators": 100
      }
  }
  ```

## Performance Optimizations

### Gradient Ascent Attack
- [ ] Evaluate and potentially reduce `ga_iterations` (currently 100) as it may be more than necessary
- [ ] Consider adding early stopping if attack succeeds before max iterations
- [ ] Experiment with different batch sizes (currently 32) to find optimal GPU utilization
