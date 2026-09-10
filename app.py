import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    r2_score
)

# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------

st.set_page_config(
    page_title="Tourism Experience Analytics",
    page_icon="🌍",
    layout="wide"
)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("🌍 Tourism Experience Analytics")
st.caption("Classification • Rating Prediction • Recommendation System")

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

@st.cache_data
def load_data():

    transaction = pd.read_excel("data/transaction.xlsx")
    user = pd.read_excel("data/user.xlsx")
    city = pd.read_excel("data/city.xlsx")
    item = pd.read_excel("data/item.xlsx")
    type_data = pd.read_excel("data/type.xlsx")

    return transaction, user, city, item, type_data


transaction, user, city, item, type_data = load_data()

# ---------------------------------------------------
# BASIC DATA CLEANING
# ---------------------------------------------------

transaction = transaction.drop_duplicates()
user = user.drop_duplicates()
city = city.drop_duplicates()
item = item.drop_duplicates()
type_data = type_data.drop_duplicates()

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Dashboard",
        "EDA",
        "Visit Mode Prediction",
        "Rating Prediction",
        "Recommendation"
    ]
)

# ===================================================
# DASHBOARD
# ===================================================

if page == "Dashboard":

    st.header("📊 Tourism Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Transactions",
        f"{len(transaction):,}"
    )

    col2.metric(
        "Total Users",
        f"{len(user):,}"
    )

    col3.metric(
        "Total Attractions",
        f"{len(item):,}"
    )

    col4.metric(
        "Average Rating",
        round(transaction["Rating"].mean(), 2)
    )

    st.divider()

    st.subheader("🧾 Transaction Data")

    st.dataframe(
        transaction.head(20),
        use_container_width=True
    )

    st.subheader("⭐ Rating Distribution")

    rating_counts = transaction["Rating"].value_counts().sort_index()

    st.bar_chart(rating_counts)


# ===================================================
# EDA
# ===================================================

elif page == "EDA":

    st.header("📈 Exploratory Data Analysis")

    st.subheader("⭐ Ratings Distribution")

    rating_counts = transaction["Rating"].value_counts().sort_index()

    st.bar_chart(rating_counts)

    st.subheader("📅 Visits by Year")

    year_counts = transaction["VisitYear"].value_counts().sort_index()

    st.bar_chart(year_counts)

    st.subheader("🗺️ Most Visited Attractions")

    attraction_counts = transaction["AttractionId"].value_counts().head(10)

    attraction_names = item[
        ["AttractionId", "Attraction"]
    ].drop_duplicates()

    attraction_counts = attraction_counts.reset_index()

    attraction_counts.columns = [
        "AttractionId",
        "Visits"
    ]

    attraction_counts = attraction_counts.merge(
        attraction_names,
        on="AttractionId",
        how="left"
    )

    st.dataframe(
        attraction_counts[
            ["Attraction", "Visits"]
        ],
        use_container_width=True
    )

    st.subheader("🏆 Top Attractions")

    st.bar_chart(
        attraction_counts.set_index(
            "Attraction"
        )["Visits"]
    )


# ===================================================
# VISIT MODE CLASSIFICATION
# ===================================================

elif page == "Visit Mode Prediction":

    st.header("🤖 Visit Mode Prediction")

    st.write(
        "Predict the likely tourism visit mode using historical "
        "tourist, attraction and visit information."
    )

    # ---------------------------------------------------
    # Prepare Data
    # ---------------------------------------------------

    data = transaction.copy()

    user_features = user[
        ["UserId", "ContinentId", "RegionId", "CountryId", "CityId"]
    ].drop_duplicates()

    data = data.merge(
        user_features,
        on="UserId",
        how="left"
    )

    data = data.dropna(subset=["VisitMode"])

    features = [
        "VisitYear",
        "VisitMonth",
        "AttractionId",
        "Rating",
        "ContinentId",
        "RegionId",
        "CountryId",
        "CityId"
    ]

    data = data.dropna(subset=features)

    X = data[features]
    y = data["VisitMode"].astype(str)

    # ---------------------------------------------------
    # Encode Visit Mode
    # ---------------------------------------------------

    encoder = LabelEncoder()

    y_encoded = encoder.fit_transform(y)

    # ---------------------------------------------------
    # Train/Test Split
    # ---------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.20,
        random_state=42,
        stratify=y_encoded
    )

    # ---------------------------------------------------
    # Random Forest
    # ---------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # ---------------------------------------------------
    # Evaluation Metrics
    # ---------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    # ---------------------------------------------------
    # Display Metrics
    # ---------------------------------------------------

    st.subheader("📊 Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )

    col2.metric(
        "Precision",
        f"{precision * 100:.2f}%"
    )

    col3.metric(
        "Recall",
        f"{recall * 100:.2f}%"
    )

    col4.metric(
        "F1 Score",
        f"{f1 * 100:.2f}%"
    )

    st.divider()

    # ---------------------------------------------------
    # Classification Report
    # ---------------------------------------------------

    st.subheader("📋 Classification Report")

    report = classification_report(
        y_test,
        predictions,
        target_names=encoder.classes_,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(3),
        use_container_width=True
    )

    # ---------------------------------------------------
    # Confusion Matrix
    # ---------------------------------------------------

    st.subheader("🔎 Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        predictions
    )

    cm_df = pd.DataFrame(
        cm,
        index=encoder.classes_,
        columns=encoder.classes_
    )

    st.dataframe(
        cm_df,
        use_container_width=True
    )

    st.caption(
        "Rows represent the actual visit mode and columns represent "
        "the predicted visit mode."
    )

    st.divider()

    # ---------------------------------------------------
    # Prediction Section
    # ---------------------------------------------------

    st.subheader("🎯 Predict Tourist Visit Mode")

    col1, col2 = st.columns(2)

    with col1:

        year = st.number_input(
            "Visit Year",
            min_value=int(data["VisitYear"].min()),
            max_value=int(data["VisitYear"].max()),
            value=int(data["VisitYear"].mode()[0])
        )

        month = st.number_input(
            "Visit Month",
            min_value=1,
            max_value=12,
            value=6
        )

        attraction = st.selectbox(
            "Attraction ID",
            sorted(data["AttractionId"].unique())
        )

        rating = st.slider(
            "Expected Rating",
            1.0,
            5.0,
            4.0
        )

    with col2:

        continent = st.selectbox(
            "Continent ID",
            sorted(data["ContinentId"].unique())
        )

        region = st.selectbox(
            "Region ID",
            sorted(data["RegionId"].unique())
        )

        country = st.selectbox(
            "Country ID",
            sorted(data["CountryId"].unique())
        )

        city_id = st.selectbox(
            "City ID",
            sorted(data["CityId"].unique())
        )

    # ---------------------------------------------------
    # Make Prediction
    # ---------------------------------------------------

    if st.button("🔮 Predict Visit Mode"):

        input_data = pd.DataFrame({

            "VisitYear": [year],

            "VisitMonth": [month],

            "AttractionId": [attraction],

            "Rating": [rating],

            "ContinentId": [continent],

            "RegionId": [region],

            "CountryId": [country],

            "CityId": [city_id]
        })

        result = model.predict(
            input_data
        )

        visit_mode = encoder.inverse_transform(
            result
        )[0]

        st.success(
            f"🎯 Predicted Visit Mode: **{visit_mode}**"
        )

        st.info(
            "The prediction is based on historical tourist "
            "and attraction information."
        )
# ===================================================
# RATING REGRESSION
# ===================================================

elif page == "Rating Prediction":

    st.header("⭐ Attraction Rating Prediction")

    st.write(
        "Predict the rating a tourist may give to an attraction."
    )

    data = transaction.copy()

    data = data.dropna(
        subset=["Rating"]
    )

    X = data[
        [
            "VisitYear",
            "VisitMonth",
            "AttractionId"
        ]
    ]

    y = data["Rating"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    score = r2_score(
        y_test,
        predictions
    )

    st.metric(
        "R² Score",
        f"{score:.3f}"
    )

    st.divider()

    year = st.number_input(
        "Visit Year",
        min_value=int(data["VisitYear"].min()),
        max_value=int(data["VisitYear"].max()),
        value=int(data["VisitYear"].mode()[0]),
        key="rating_year"
    )

    month = st.number_input(
        "Visit Month",
        min_value=1,
        max_value=12,
        value=6,
        key="rating_month"
    )

    attraction = st.selectbox(
        "Select Attraction",
        sorted(
            data["AttractionId"].unique()
        ),
        key="rating_attraction"
    )

    if st.button(
        "⭐ Predict Rating"
    ):

        input_data = pd.DataFrame(
            {
                "VisitYear": [year],
                "VisitMonth": [month],
                "AttractionId": [attraction]
            }
        )

        predicted_rating = model.predict(
            input_data
        )[0]

        predicted_rating = max(
            1,
            min(5, predicted_rating)
        )

        st.success(
            f"Predicted Rating: **{predicted_rating:.2f} / 5 ⭐**"
        )


# ===================================================
# RECOMMENDATION SYSTEM
# ===================================================

elif page == "Recommendation":
    st.header("🗺️ Personalized Attraction Recommendation")
    st.write(
        "Select an attraction you are interested in and discover "
        "other highly-rated attractions."
    )

    # ---------------------------------------------------
    # Prepare attraction rating information
    # ---------------------------------------------------
    attraction_stats = (
        transaction.groupby("AttractionId")["Rating"]
        .agg(["mean", "count"])
        .reset_index()
    )

    attraction_stats.columns = [
        "AttractionId",
        "AverageRating",
        "NumberOfReviews"
    ]

    # Add attraction names and addresses
    attraction_info = item[
        ["AttractionId", "Attraction", "AttractionAddress"]
    ].drop_duplicates()

    recommendations = attraction_stats.merge(
        attraction_info,
        on="AttractionId",
        how="left"
    )

    # ---------------------------------------------------
    # Attraction selection
    # ---------------------------------------------------
    st.subheader("🎯 Choose an Attraction")

    available_attractions = recommendations[
        recommendations["Attraction"].notna()
    ].sort_values("Attraction")

    selected_attraction = st.selectbox(
        "Select an attraction you are interested in:",
        available_attractions["Attraction"].tolist()
    )

    # Get selected attraction details
    selected_row = recommendations[
        recommendations["Attraction"] == selected_attraction
    ].iloc[0]

    st.divider()

    # ---------------------------------------------------
    # Selected attraction information
    # ---------------------------------------------------
    st.subheader("📌 Selected Attraction")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "⭐ Average Rating",
        f"{selected_row['AverageRating']:.2f}/5"
    )

    col2.metric(
        "👥 Reviews",
        f"{int(selected_row['NumberOfReviews']):,}"
    )

    col3.metric(
        "🆔 Attraction ID",
        int(selected_row["AttractionId"])
    )

    if pd.notna(selected_row["AttractionAddress"]):
        st.write(
            f"📍 **Address:** {selected_row['AttractionAddress']}"
        )

    # ---------------------------------------------------
    # Generate recommendations
    # ---------------------------------------------------
    other_attractions = recommendations[
        recommendations["Attraction"] != selected_attraction
    ].copy()

    # Sort by rating and popularity
    other_attractions = other_attractions.sort_values(
        ["AverageRating", "NumberOfReviews"],
        ascending=False
    )

    # Top 5 recommendations
    top_recommendations = other_attractions.head(5)

    st.divider()

    st.subheader("🏆 Recommended Attractions")

    for i, (_, row) in enumerate(
        top_recommendations.iterrows(),
        start=1
    ):

        st.markdown(
            f"""
            ### {i}. {row['Attraction']}

            ⭐ **Rating:** {row['AverageRating']:.2f}/5

            👥 **Reviews:** {int(row['NumberOfReviews']):,}

            📍 **Address:** {row['AttractionAddress']}
            """
        )

        st.divider()

    # ---------------------------------------------------
    # Explanation
    # ---------------------------------------------------
    st.info(
        "💡 Recommendations are generated using historical tourist "
        "ratings and attraction popularity. Attractions with higher "
        "average ratings and more reviews are prioritized."
    )