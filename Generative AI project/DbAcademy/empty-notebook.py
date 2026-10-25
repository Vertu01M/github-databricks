# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Cell 1
# DAG Workflow:
# Task 1: Bronze_to_Silver_transformation
# Task 2: Silver_to_Gold_transformation (depends on Task 1)

print("This notebook is for creating a Databricks Job to orchestrate the transformation pipeline.")
print("Use the Jobs UI to create a multi-task workflow with:")
print("  - Task 1: /Users/vertubouka@gmail.com/Bronze_to_Silver_transformation")
print("  - Task 2: /Users/vertubouka@gmail.com/Silver_to_Gold_transformation (depends on Task 1)")