
# -----------------------------------------
# EXPLORATORY DATA ANALYSIS (EDA)
# Student Performance Dataset
# -----------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import os

# Set the project folders
file_path = r"D:\internship\week2\StudentPerfomance_Project\dataset\StudentPerfomance.csv"

output_folder = r"D:\internship\week2\StudentPerfomance_Project\visualizations"

# Create visualizations folder if needed
os.makedirs(output_folder, exist_ok=True)

# Load dataset
df = pd.read_csv(file_path)

print("\n========== STUDENT DATASET ==========")
print(df)

# -----------------------------------------
# 1. BASIC DATASET INFORMATION
# -----------------------------------------

print("\n========== DATASET INFORMATION ==========")

print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

# -----------------------------------------
# 2. MISSING VALUE ANALYSIS
# -----------------------------------------

print("\n========== MISSING VALUE ANALYSIS ==========")
print(df.isnull().sum())

# -----------------------------------------
# 3. DUPLICATE VALUE CHECKING
# -----------------------------------------

print("\n========== DUPLICATE CHECKING ==========")
print("Duplicate Records:", df.duplicated().sum())

# -----------------------------------------
# 4. OUTLIER DETECTION USING IQR
# -----------------------------------------

print("\n========== OUTLIER DETECTION ==========")

numeric_columns = ["Age", "Study_Hours", "Attendance", "Marks"]

for column in numeric_columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_limit) |
        (df[column] > upper_limit)
    ]

    print(column, "Outliers:", len(outliers))

# -----------------------------------------
# 5. UNIVARIATE ANALYSIS
# -----------------------------------------

# Histogram of Marks
plt.figure(figsize=(7, 5))
plt.hist(df["Marks"], bins=5, edgecolor="black")
plt.title("Distribution of Student Marks")
plt.xlabel("Marks")
plt.ylabel("Number of Students")
plt.savefig(os.path.join(output_folder, "marks_histogram.png"))
plt.show()

# Histogram of Study Hours
plt.figure(figsize=(7, 5))
plt.hist(df["Study_Hours"], bins=5, edgecolor="black")
plt.title("Distribution of Study Hours")
plt.xlabel("Study Hours")
plt.ylabel("Number of Students")
plt.savefig(os.path.join(output_folder, "study_hours_histogram.png"))
plt.show()

# Gender Count Bar Chart
gender_counts = df["Gender"].value_counts()

plt.figure(figsize=(7, 5))
plt.bar(gender_counts.index, gender_counts.values, edgecolor="black")
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Number of Students")
plt.savefig(os.path.join(output_folder, "gender_count.png"))
plt.show()

# -----------------------------------------
# 6. BIVARIATE ANALYSIS
# -----------------------------------------

# Study Hours vs Marks
plt.figure(figsize=(7, 5))
for gender in df["Gender"].unique():
    subset = df[df["Gender"] == gender]
    plt.scatter(
        subset["Study_Hours"],
        subset["Marks"],
        label=gender,
        s=80
    )

plt.title("Study Hours vs Marks")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.legend()
plt.savefig(os.path.join(output_folder, "study_hours_vs_marks.png"))
plt.show()

# Attendance vs Marks
plt.figure(figsize=(7, 5))
for gender in df["Gender"].unique():
    subset = df[df["Gender"] == gender]
    plt.scatter(
        subset["Attendance"],
        subset["Marks"],
        label=gender,
        s=80
    )

plt.title("Attendance vs Marks")
plt.xlabel("Attendance (%)")
plt.ylabel("Marks")
plt.legend()
plt.savefig(os.path.join(output_folder, "attendance_vs_marks.png"))
plt.show()

# Marks by Gender - Box Plot
male_marks = df[df["Gender"] == "Male"]["Marks"]
female_marks = df[df["Gender"] == "Female"]["Marks"]

plt.figure(figsize=(7, 5))
plt.boxplot(
    [male_marks, female_marks],
    tick_labels=["Male", "Female"]
)
plt.title("Marks by Gender")
plt.xlabel("Gender")
plt.ylabel("Marks")
plt.savefig(os.path.join(output_folder, "marks_by_gender.png"))
plt.show()

# -----------------------------------------
# 7. MULTIVARIATE ANALYSIS
# -----------------------------------------

# Correlation Heatmap using Matplotlib
correlation = df[
    ["Age", "Study_Hours", "Attendance", "Marks"]
].corr()

plt.figure(figsize=(8, 6))
plt.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)

plt.colorbar(label="Correlation")

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=45
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

for i in range(len(correlation.columns)):
    for j in range(len(correlation.columns)):
        plt.text(
            j, i,
            round(correlation.iloc[i, j], 2),
            ha="center",
            va="center"
        )

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "correlation_heatmap.png"))
plt.show()

# -----------------------------------------
# 8. FINAL EDA SUMMARY
# -----------------------------------------

print("\n========== FINAL EDA SUMMARY ==========")

print("Total Students:", len(df))
print("Average Study Hours:", df["Study_Hours"].mean())
print("Average Attendance:", df["Attendance"].mean())
print("Average Marks:", df["Marks"].mean())
print("Highest Marks:", df["Marks"].max())
print("Lowest Marks:", df["Marks"].min())

print("\nAll EDA operations completed successfully!")

print("Graphs saved in:", output_folder)
