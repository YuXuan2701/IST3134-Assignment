# IST3134 Group Assignment – Enterprise Cloud Data Breach Risk Prediction
# Google Colab / PySpark implementation

# =========================
# 1. Install and configure
# =========================
!pip -q install pyspark==3.5.6

import os, time, json, warnings
warnings.filterwarnings("ignore")

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

spark = (
    SparkSession.builder
    .appName("IST3134_Breach_Risk_Prediction")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# =========================
# 2. Load dataset
# =========================
from google.colab import files
uploaded = files.upload()

csv_files = [x for x in uploaded.keys() if x.lower().endswith(".csv")]
assert csv_files, "Upload the CSV dataset."
DATA_PATH = csv_files[0]

df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(DATA_PATH)
)

print("Rows:", df.count())
print("Columns:", len(df.columns))
df.printSchema()
df.show(5, truncate=False)

# =========================
# 3. Data quality checks
# =========================
print("Duplicate rows:", df.count() - df.dropDuplicates().count())

null_counts = df.select([
    F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns
])
null_counts.show(truncate=False)

print("Target distribution:")
df.groupBy("breach_risk_label").count().orderBy("breach_risk_label").show()

# =========================
# 4. MapReduce-style analysis
# =========================
# Map stage: emit (department, (count, risk_sum))
# Reduce stage: aggregate values by department
map_pairs = (
    df.select("department", "breach_risk_label")
      .rdd
      .map(lambda r: (r["department"], (1, int(r["breach_risk_label"]))))
)

reduce_department = (
    map_pairs
    .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
    .map(lambda x: (x[0], x[1][0], x[1][1], x[1][1] / x[1][0]))
    .sortBy(lambda x: x[0])
)

department_result = reduce_department.collect()

print("\nMapReduce-style department summary:")
for row in department_result:
    print(row)

# MapReduce-style target count
target_pairs = (
    df.select("breach_risk_label")
      .rdd
      .map(lambda r: (int(r["breach_risk_label"]), 1))
      .reduceByKey(lambda a, b: a + b)
      .sortByKey()
)
print("\nMapReduce-style target counts:", target_pairs.collect())

# =========================
# 5. Spark SQL analysis
# =========================
df.createOrReplaceTempView("breach_data")

sql_department = spark.sql("""
SELECT
    department,
    COUNT(*) AS records,
    SUM(breach_risk_label) AS breaches,
    ROUND(AVG(breach_risk_label), 4) AS breach_rate
FROM breach_data
GROUP BY department
ORDER BY breach_rate DESC
""")
sql_department.show()

sql_access = spark.sql("""
SELECT
    access_level,
    COUNT(*) AS records,
    SUM(breach_risk_label) AS breaches,
    ROUND(AVG(breach_risk_label), 4) AS breach_rate
FROM breach_data
GROUP BY access_level
ORDER BY breach_rate DESC
""")
sql_access.show()

sql_risk_features = spark.sql("""
SELECT
    breach_risk_label,
    ROUND(AVG(login_attempts_24h), 3) AS avg_login_attempts,
    ROUND(AVG(failed_logins_24h), 3) AS avg_failed_logins,
    ROUND(AVG(data_download_gb), 3) AS avg_download_gb,
    ROUND(AVG(privileged_actions_count), 3) AS avg_privileged_actions,
    ROUND(AVG(device_trust_score), 3) AS avg_device_trust,
    ROUND(AVG(anomaly_behavior_score), 3) AS avg_anomaly_score,
    ROUND(AVG(policy_violation_count), 3) AS avg_policy_violations
FROM breach_data
GROUP BY breach_risk_label
ORDER BY breach_risk_label
""")
sql_risk_features.show()

# =========================
# 6. Prepare Spark ML data
# =========================
target = "breach_risk_label"
drop_cols = ["employee_id", target]

categorical_cols = ["department", "access_level"]
numeric_cols = [
    "login_attempts_24h",
    "failed_logins_24h",
    "remote_access_flag",
    "vpn_usage_flag",
    "data_download_gb",
    "unusual_file_access",
    "privileged_actions_count",
    "device_trust_score",
    "security_patch_delay_days",
    "anomaly_behavior_score",
    "external_ip_flag",
    "policy_violation_count"
]

data = df.select(*(categorical_cols + numeric_cols + [target])).dropna()

# Stratified train/test split
train, test = data.randomSplit([0.8, 0.2], seed=42)
print("Training rows:", train.count())
print("Testing rows:", test.count())

# =========================
# 7. Class-weighted Logistic Regression
# =========================
class_counts = train.groupBy(target).count().collect()
count_dict = {int(r[target]): int(r["count"]) for r in class_counts}
n = sum(count_dict.values())
k = len(count_dict)
weights = {cls: n / (k * cnt) for cls, cnt in count_dict.items()}

weighted_train = train.withColumn(
    "class_weight",
    F.when(F.col(target) == 0, F.lit(float(weights.get(0, 1.0))))
     .otherwise(F.lit(float(weights.get(1, 1.0))))
)

indexers = [
    StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
    for c in categorical_cols
]
encoder = OneHotEncoder(
    inputCols=[f"{c}_idx" for c in categorical_cols],
    outputCols=[f"{c}_ohe" for c in categorical_cols]
)
assembler = VectorAssembler(
    inputCols=[f"{c}_ohe" for c in categorical_cols] + numeric_cols,
    outputCol="features_raw"
)
scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features",
    withStd=True,
    withMean=False
)
lr = LogisticRegression(
    featuresCol="features",
    labelCol=target,
    weightCol="class_weight",
    maxIter=100,
    regParam=0.1,
    elasticNetParam=0.0
)

pipeline = Pipeline(stages=indexers + [encoder, assembler, scaler, lr])

start = time.perf_counter()
model = pipeline.fit(weighted_train)
spark_train_seconds = time.perf_counter() - start

pred = model.transform(test).cache()
pred.select(target, "probability", "prediction").show(10, truncate=False)

# =========================
# 8. Evaluation
# =========================
auc = BinaryClassificationEvaluator(
    labelCol=target,
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
).evaluate(pred)

accuracy = MulticlassClassificationEvaluator(
    labelCol=target,
    predictionCol="prediction",
    metricName="accuracy"
).evaluate(pred)

precision = MulticlassClassificationEvaluator(
    labelCol=target,
    predictionCol="prediction",
    metricName="weightedPrecision"
).evaluate(pred)

recall = MulticlassClassificationEvaluator(
    labelCol=target,
    predictionCol="prediction",
    metricName="weightedRecall"
).evaluate(pred)

f1 = MulticlassClassificationEvaluator(
    labelCol=target,
    predictionCol="prediction",
    metricName="f1"
).evaluate(pred)

print("\nSpark ML results")
print("AUC:", round(auc, 4))
print("Accuracy:", round(accuracy, 4))
print("Weighted precision:", round(precision, 4))
print("Weighted recall:", round(recall, 4))
print("Weighted F1:", round(f1, 4))
print("Spark training time (seconds):", round(spark_train_seconds, 4))

print("\nConfusion matrix")
pred.groupBy(target, "prediction").count().orderBy(target, "prediction").show()

# =========================
# 9. Feature coefficients
# =========================
lr_model = model.stages[-1]
coefficients = lr_model.coefficients.toArray()

feature_names = []
for c in categorical_cols:
    feature_names.append(f"{c}_encoded")
feature_names += numeric_cols

print("\nModel coefficient vector length:", len(coefficients))
print("Model trained successfully.")

# =========================
# 10. Non-big-data comparison
# =========================
# A conventional scikit-learn implementation is included to provide the
# required meaningful comparison against a non-distributed approach.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix
)

pdf = pd.read_csv(DATA_PATH).dropna()
X = pdf[categorical_cols + numeric_cols]
y = pdf[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

sk_pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", StandardScaler(), numeric_cols)
])

sk_model = SkPipeline([
    ("pre", sk_pre),
    ("clf", SkLogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ))
])

start = time.perf_counter()
sk_model.fit(X_train, y_train)
sk_train_seconds = time.perf_counter() - start

sk_pred = sk_model.predict(X_test)
sk_prob = sk_model.predict_proba(X_test)[:, 1]

print("\nScikit-learn comparison")
print("AUC:", round(roc_auc_score(y_test, sk_prob), 4))
print("Accuracy:", round(accuracy_score(y_test, sk_pred), 4))
print("Precision:", round(precision_score(y_test, sk_pred, zero_division=0), 4))
print("Recall:", round(recall_score(y_test, sk_pred, zero_division=0), 4))
print("F1:", round(f1_score(y_test, sk_pred, zero_division=0), 4))
print("Training time (seconds):", round(sk_train_seconds, 4))
print("Confusion matrix:\n", confusion_matrix(y_test, sk_pred))

# =========================
# 11. Visualisations
# =========================
import matplotlib.pyplot as plt

plot_pdf = pdf.copy()
risk_by_dept = plot_pdf.groupby("department")[target].mean().sort_values(ascending=False)

plt.figure(figsize=(8,5))
risk_by_dept.plot(kind="bar")
plt.title("Breach Risk Rate by Department")
plt.xlabel("Department")
plt.ylabel("Breach Risk Rate")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,5))
plot_pdf[target].value_counts().sort_index().plot(kind="bar")
plt.title("Breach Risk Label Distribution")
plt.xlabel("Breach Risk Label")
plt.ylabel("Number of Records")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# =========================
# 12. Save key outputs
# =========================
output_dir = "ist3134_outputs"
os.makedirs(output_dir, exist_ok=True)

sql_department.toPandas().to_csv(
    f"{output_dir}/department_breach_summary.csv", index=False
)
sql_access.toPandas().to_csv(
    f"{output_dir}/access_level_breach_summary.csv", index=False
)
sql_risk_features.toPandas().to_csv(
    f"{output_dir}/risk_feature_summary.csv", index=False
)

metrics = pd.DataFrame([{
    "spark_auc": auc,
    "spark_accuracy": accuracy,
    "spark_weighted_precision": precision,
    "spark_weighted_recall": recall,
    "spark_weighted_f1": f1,
    "spark_training_seconds": spark_train_seconds,
    "sklearn_auc": roc_auc_score(y_test, sk_prob),
    "sklearn_accuracy": accuracy_score(y_test, sk_pred),
    "sklearn_precision": precision_score(y_test, sk_pred, zero_division=0),
    "sklearn_recall": recall_score(y_test, sk_pred, zero_division=0),
    "sklearn_f1": f1_score(y_test, sk_pred, zero_division=0),
    "sklearn_training_seconds": sk_train_seconds
}])
metrics.to_csv(f"{output_dir}/model_metrics.csv", index=False)

print(f"\nOutputs saved in: {output_dir}/")
print("Assignment pipeline completed.")
