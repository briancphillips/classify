#!/bin/bash

echo "Fetching and pruning remote references..."
git fetch --all --prune

echo "Updating remote HEAD reference..."
git remote set-head origin -a

echo "Removing old feature branch reference..."
git branch -rd origin/feature/new-classifier-temp

echo "Updating develop branch..."
git checkout develop
git pull origin develop

echo "Done! Current branch status:"
git branch -a
