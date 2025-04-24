import os
import json

# Target kernel metadata
new_kernelspec = {
    "name": "fall_risk_env",
    "display_name": "Python (fall_risk_env)",
    "language": "python"
}

# Minimal valid notebook template
minimal_notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": new_kernelspec,
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

for root, _, files in os.walk("."):
    for file in files:
        if file.endswith(".ipynb"):
            path = os.path.join(root, file)
            print(f"🔍 Processing: {path}")

            try:
                if os.stat(path).st_size == 0:
                    print(f"⚠️  Empty notebook — initializing: {path}")
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(minimal_notebook, f, indent=1)
                else:
                    with open(path, "r", encoding="utf-8") as f:
                        nb = json.load(f)

                    nb.setdefault("metadata", {})
                    nb["metadata"]["kernelspec"] = new_kernelspec

                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(nb, f, indent=1)
                    print(f"✅ Updated: {path}")

            except Exception as e:
                print(f"❌ Error with {path}: {e}")
