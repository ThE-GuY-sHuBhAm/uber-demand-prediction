import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# ---------------- LOAD ARTIFACTS ----------------

root = Path(__file__).parent

df = pd.read_csv(
    root / "data/processed/test.csv",
    parse_dates=["tpep_pickup_datetime"]
).set_index("tpep_pickup_datetime")

encoder = joblib.load(root / "models/encoder.joblib")
model = joblib.load(root / "models/model.joblib")
kmeans = joblib.load(root / "models/mb_kmeans.joblib")

pipe = Pipeline([
    ("encoder", encoder),
    ("model", model)
])

# ---------------- UI ----------------

st.title("🚕 NYC Taxi Demand Recommendation System")

st.sidebar.header("Driver Options")

map_scope = st.sidebar.radio(
    "Prediction Scope",
    ["All NYC Regions", "Nearby Regions Only"]
)

K_NEIGHBORS = st.sidebar.slider(
    "Nearby regions to consider",
    min_value=3,
    max_value=15,
    value=8
)

selected_time = st.selectbox(
    "Choose prediction timestamp",
    df.index.unique().sort_values()
)

st.write("Predicting demand for next 15 minutes after:")
st.code(selected_time)

# ---------------- FEATURE PREP ----------------

current_snapshot = df.loc[selected_time].copy()
X = current_snapshot.drop(columns=["target"])

# ---------------- PREDICT ----------------

current_snapshot["predicted_demand"] = pipe.predict(X)

ranked = current_snapshot.sort_values(
    "predicted_demand",
    ascending=False
)

# ---------------- FILTER NEARBY ----------------

if map_scope == "Nearby Regions Only":

    current_region = int(ranked.iloc[0]["region"])
    center = kmeans.cluster_centers_[current_region]

    distances = kmeans.transform([center]).ravel()

    nearest_regions = sorted(
        list(enumerate(distances)),
        key=lambda x: x[1]
    )[:K_NEIGHBORS]

    region_ids = [r[0] for r in nearest_regions]

    ranked = ranked[ranked["region"].isin(region_ids)]

# ---------------- BAR CHART ----------------

st.subheader("📊 Predicted Demand by Region")

top_regions = ranked.head(10)

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(
    top_regions["region"].astype(str),
    top_regions["predicted_demand"],
    color="#3498db"
)

ax.invert_yaxis()
ax.set_xlabel("Expected Pickups (next 15 min)")
ax.set_ylabel("Region")
ax.set_title("Top Demand Regions")

st.pyplot(fig)

# ---------------- METRICS ----------------

st.subheader("📈 Recommended Regions")

for _, row in top_regions.iterrows():
    st.metric(
        label=f"Region {int(row['region'])}",
        value=f"{int(row['predicted_demand'])} pickups"
    )

# ---------------- TABLE ----------------

st.subheader("Full Ranking")

st.dataframe(
    ranked[["region", "predicted_demand"]]
    .reset_index(drop=True)
)