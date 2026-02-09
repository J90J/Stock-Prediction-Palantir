#!/bin/bash
# Remove large/unnecessary files
rm "Book2.xlsx"
rm "Final Project_Jens Jung_UPLOAD.pdf"
rm "Final_Project_Jens_Jung_UPLOAD.ipynb"
rm "Project proposal EE_WIP.pptx"
rm "test_news.py"
rm -rf "Project EE_DL"
rm -rf "results" # User can regenerate these
rm -rf "data"    # User can download fresh data
rm -rf "checkpoints" # User can retrain
# Keep critical source files
echo "Cleanup complete."
