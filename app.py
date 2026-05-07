import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

st.set_page_config(
    page_title="Netflix Analytics Dashboard",
    page_icon="🎬",
    layout="wide"
)

st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #E50914;
    text-align: center;
}
.sub-title {
    font-size: 18px;
    text-align: center;
    color: #666;
}
.metric-card {
    background-color: #111827;
    padding: 20px;
    border-radius: 15px;
    color: white;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎬 Netflix Movies & TV Shows Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">EDA + Machine Learning Clustering Dashboard</div>', unsafe_allow_html=True)

st.sidebar.title("📌 Dashboard Menu")
uploaded_file = st.sidebar.file_uploader("Upload Netflix CSV", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.write("Built with Python, Streamlit, Plotly & Scikit-learn")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    df.columns = df.columns.str.strip()

    st.markdown("## 📂 Dataset Overview")
    st.dataframe(df.head(), use_container_width=True)

    total_titles = df.shape[0]
    total_columns = df.shape[1]
    missing_values = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Titles", total_titles)
    col2.metric("Total Columns", total_columns)
    col3.metric("Missing Values", missing_values)
    col4.metric("Duplicate Rows", duplicate_rows)

    st.markdown("---")

    st.markdown("## 🎥 Content Type Distribution")
    if "type" in df.columns:
        type_count = df["type"].value_counts().reset_index()
        type_count.columns = ["Type", "Count"]

        fig = px.pie(
            type_count,
            names="Type",
            values="Count",
            hole=0.45,
            title="Movies vs TV Shows"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("## 🌍 Country & Genre Analysis")

    col1, col2 = st.columns(2)

    with col1:
        if "country" in df.columns:
            country_data = (
                df["country"]
                .dropna()
                .str.split(", ")
                .explode()
                .value_counts()
                .head(10)
            )

            fig = px.bar(
                x=country_data.values,
                y=country_data.index,
                orientation="h",
                title="Top 10 Countries",
                labels={"x": "Number of Titles", "y": "Country"}
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "listed_in" in df.columns:
            genre_data = (
                df["listed_in"]
                .dropna()
                .str.split(", ")
                .explode()
                .value_counts()
                .head(10)
            )

            fig = px.bar(
                x=genre_data.values,
                y=genre_data.index,
                orientation="h",
                title="Top 10 Genres",
                labels={"x": "Number of Titles", "y": "Genre"}
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("## 📅 Netflix Content Trend")

    if "date_added" in df.columns:
        df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
        df["year_added"] = df["date_added"].dt.year

        year_data = df["year_added"].value_counts().sort_index()

        fig = px.line(
            x=year_data.index,
            y=year_data.values,
            markers=True,
            title="Content Added Over the Years",
            labels={"x": "Year", "y": "Number of Titles"}
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("## 🎭 Rating Analysis")

    if "rating" in df.columns:
        rating_data = df["rating"].value_counts().head(10)

        fig = px.bar(
            x=rating_data.index,
            y=rating_data.values,
            title="Top Content Ratings",
            labels={"x": "Rating", "y": "Count"}
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.markdown("## 🤖 Machine Learning: Content Clustering")

    if "description" in df.columns:
        df["description"] = df["description"].fillna("")

        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=1000
        )

        X = vectorizer.fit_transform(df["description"])

        kmeans = KMeans(
            n_clusters=5,
            random_state=42,
            n_init=10
        )

        df["Cluster"] = kmeans.fit_predict(X)

        st.success("Successfully clustered Netflix titles into 5 groups using TF-IDF and KMeans.")

        cluster_data = df["Cluster"].value_counts().sort_index()

        fig = px.bar(
            x=cluster_data.index,
            y=cluster_data.values,
            title="Netflix Content Clusters",
            labels={"x": "Cluster Number", "y": "Number of Titles"}
        )
        st.plotly_chart(fig, use_container_width=True)

        selected_cluster = st.selectbox(
            "Select a cluster to explore titles:",
            sorted(df["Cluster"].unique())
        )

        cluster_df = df[df["Cluster"] == selected_cluster]

        show_cols = [
            col for col in
            ["title", "type", "country", "release_year", "rating", "listed_in", "description"]
            if col in cluster_df.columns
        ]

        st.dataframe(cluster_df[show_cols].head(25), use_container_width=True)

    st.markdown("---")

    st.markdown("## ✅ Business Insights")

    st.info("""
    This dashboard helps analyze Netflix content patterns such as movie vs TV show distribution,
    top content-producing countries, popular genres, yearly content growth, ratings, and similarity-based
    clustering using machine learning.
    """)

    st.markdown("## 🎯 Project Conclusion")

    st.success("""
    Netflix can use this type of analysis to understand content trends, improve recommendation systems,
    identify popular genres, and make better content acquisition decisions.
    """)

else:
    st.warning("👈 Please upload your Netflix CSV file from the sidebar to start the dashboard.")