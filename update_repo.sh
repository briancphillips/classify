#!/bin/bash

echo "Fetching all changes..."
git fetch --all

echo "Updating main branch..."
git checkout main
git pull origin main

echo "Setting up develop branch..."
git checkout -b develop origin/develop 2>/dev/null || git checkout develop && git pull origin develop

echo "Cleaning up old feature branches..."
git branch -D feature/new-classifier feature/new-classifier-temp 2>/dev/null || true

echo "Done! You are now on the develop branch."
