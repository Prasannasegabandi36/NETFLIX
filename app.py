import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

st.set_page_config(
    page_title="Netflix Cinematic Analytics",
    page_icon="🍿",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #050505 0%, #111111 45%, #1a0000 100%);
    color: white;
}

[data-testid="stSidebar"] {
    background: #080808;
}

.hero {
    padding: 45px;
    border-radius: 25px;
    background: linear-gradient(120deg, rgba(229,9,20,0.95), rgba(20,20,20,0.95));
    box-shadow: 0 0 35px rgba(229,9,20,0.45);
    text-align: center;
    margin-bottom: 30px;
}

.hero h1 {
    font-size: 58px;
    font-weight: 900;
    color: white;
}

.hero p {
    font-size: 20px;
    color: white;
}

.card {
    background: linear-gradient(145deg, #181818, #0b0b0b);
    padding: 24px;
    border-radius: 20px;
    border: 1px solid rgba(229,9,20,0.4);
    box-shadow: 0 0 18px rgba(229,9,20,0.18);
    text-align: center;
}

.card h3 {
    color: white;
    font-size: 16px;
}

.card h2 {
    color: #E50914;
    font-size: 36px;
    font-weight: 900;
}

.section-title {
    font-size: 30px;
    font-weight: 800;
    color: #ff4b4b;
    margin-top: 30px;
    margin-bottom: 15px;
}

.movie-card {
    background: #141414;
    border-left: 5px solid #E50914;
    padding: 18px;
    border-radius: 14px;
    margin-bottom: 12px;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>NETFLIX CONTENT UNIVERSE</h1>
    <p>Cinematic Data Analytics Dashboard powered by Python, Streamlit & Machine Learning</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("## 🎬 Netflix Dashboard")
st.sidebar.markdown("## 🎛️ Control Panel")
uploaded_file = st.sidebar.file_uploader("Upload Netflix CSV", type=["csv"])

def style_chart(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        title_font_color="white",
        legend=dict(
            font=dict(color="white", size=14),
            title_font=dict(color="white", size=15)
        ),
        xaxis=dict(
            color="white",
            title_font=dict(color="white"),
            tickfont=dict(color="white"),
            gridcolor="rgba(255,255,255,0.15)"
        ),
        yaxis=dict(
            color="white",
            title_font=dict(color="white"),
            tickfont=dict(color="white"),
            gridcolor="rgba(255,255,255,0.15)"
        )
    )
    return fig

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    st.sidebar.markdown("### 🔍 Filters")

    if "type" in df.columns:
        selected_type = st.sidebar.multiselect(
            "Select Content Type",
            df["type"].dropna().unique(),
            default=df["type"].dropna().unique()
        )
        df = df[df["type"].isin(selected_type)]

    if "country" in df.columns:
        countries = df["country"].dropna().str.split(", ").explode().unique()
        selected_country = st.sidebar.selectbox(
            "Select Country",
            ["All"] + sorted(countries.tolist())
        )
        if selected_country != "All":
            df = df[df["country"].fillna("").str.contains(selected_country, regex=False)]

    if "release_year" in df.columns:
        min_year = int(df["release_year"].min())
        max_year = int(df["release_year"].max())
        year_range = st.sidebar.slider(
            "Select Release Year Range",
            min_year,
            max_year,
            (min_year, max_year)
        )
        df = df[(df["release_year"] >= year_range[0]) & (df["release_year"] <= year_range[1])]

    total_titles = df.shape[0]
    total_movies = df[df["type"] == "Movie"].shape[0] if "type" in df.columns else 0
    total_tv = df[df["type"] == "TV Show"].shape[0] if "type" in df.columns else 0
    total_countries = df["country"].dropna().str.split(", ").explode().nunique() if "country" in df.columns else 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f'<div class="card"><h3>Total Titles</h3><h2>{total_titles}</h2></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="card"><h3>Movies</h3><h2>{total_movies}</h2></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="card"><h3>TV Shows</h3><h2>{total_tv}</h2></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="card"><h3>Countries</h3><h2>{total_countries}</h2></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">🎬 Content Type Breakdown</div>', unsafe_allow_html=True)

    if "type" in df.columns:
        type_count = df["type"].value_counts().reset_index()
        type_count.columns = ["Type", "Count"]

        fig = px.pie(
            type_count,
            names="Type",
            values="Count",
            hole=0.55,
            color_discrete_sequence=["#E50914", "#ff6b6b"],
            title="Movies vs TV Shows"
        )

        fig.update_traces(
            textfont_color="white",
            textfont_size=15,
            marker=dict(line=dict(color="black", width=2))
        )

        fig = style_chart(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">🌍 Global Netflix Presence</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        if "country" in df.columns:
            country_data = (
                df["country"].dropna()
                .str.split(", ")
                .explode()
                .value_counts()
                .head(10)
            )

            fig = px.bar(
                x=country_data.values,
                y=country_data.index,
                orientation="h",
                title="Top 10 Content Producing Countries",
                labels={"x": "Titles", "y": "Country"},
                color=country_data.values,
                color_continuous_scale="Reds"
            )

            fig = style_chart(fig)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "listed_in" in df.columns:
            genre_data = (
                df["listed_in"].dropna()
                .str.split(", ")
                .explode()
                .value_counts()
                .head(10)
            )

            fig = px.bar(
                x=genre_data.values,
                y=genre_data.index,
                orientation="h",
                title="Top 10 Netflix Genres",
                labels={"x": "Titles", "y": "Genre"},
                color=genre_data.values,
                color_continuous_scale="Reds"
            )

            fig = style_chart(fig)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">📅 Netflix Timeline</div>', unsafe_allow_html=True)

    if "date_added" in df.columns:
        df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
        df["year_added"] = df["date_added"].dt.year
        year_data = df["year_added"].value_counts().sort_index()

        fig = px.area(
            x=year_data.index,
            y=year_data.values,
            title="Content Added Over the Years",
            labels={"x": "Year", "y": "Number of Titles"}
        )

        fig.update_traces(line_color="#E50914", fillcolor="rgba(229,9,20,0.35)")
        fig = style_chart(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">🎭 Rating Universe</div>', unsafe_allow_html=True)

    if "rating" in df.columns:
        rating_data = df["rating"].value_counts().head(10)

        fig = px.bar(
            x=rating_data.index,
            y=rating_data.values,
            title="Top Netflix Ratings",
            labels={"x": "Rating", "y": "Count"},
            color=rating_data.values,
            color_continuous_scale="Reds"
        )

        fig = style_chart(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">🤖 AI Content Clustering</div>', unsafe_allow_html=True)

    if "description" in df.columns and len(df) >= 5:
        ml_df = df.copy()
        ml_df["description"] = ml_df["description"].fillna("")

        vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
        X = vectorizer.fit_transform(ml_df["description"])

        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        ml_df["Cluster"] = kmeans.fit_predict(X)

        st.success("AI successfully grouped Netflix titles into 5 similarity-based content clusters.")

        cluster_data = ml_df["Cluster"].value_counts().sort_index()

        fig = px.bar(
            x=cluster_data.index,
            y=cluster_data.values,
            title="Netflix AI Content Clusters",
            labels={"x": "Cluster", "y": "Number of Titles"},
            color=cluster_data.values,
            color_continuous_scale="Reds"
        )

        fig = style_chart(fig)
        st.plotly_chart(fig, use_container_width=True)

        selected_cluster = st.selectbox(
            "🍿 Explore Similar Netflix Titles by Cluster",
            sorted(ml_df["Cluster"].unique())
        )

        cluster_df = ml_df[ml_df["Cluster"] == selected_cluster].head(10)

        for _, row in cluster_df.iterrows():
            title = row["title"] if "title" in row else "Unknown Title"
            content_type = row["type"] if "type" in row else "N/A"
            rating = row["rating"] if "rating" in row else "N/A"
            year = row["release_year"] if "release_year" in row else "N/A"
            desc = row["description"] if "description" in row else "No description available."

            st.markdown(f"""
            <div class="movie-card">
                <h3>🎞️ {title}</h3>
                <p><b>Type:</b> {content_type} | <b>Rating:</b> {rating} | <b>Year:</b> {year}</p>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">📂 Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown('<div class="section-title">✅ Final Insight</div>', unsafe_allow_html=True)
    st.info("""
    This cinematic Netflix dashboard helps understand content trends, country contribution,
    genre popularity, yearly growth, rating distribution, and similar content groups using Machine Learning.
    """)

else:
    st.warning("👈 Upload your Netflix CSV file from the sidebar to start the cinematic dashboard.")
