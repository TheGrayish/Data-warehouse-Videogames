import streamlit as st
import pandas as pd
import numpy as np
import mariadb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Configuración de la página
st.set_page_config(page_title="Analytics de Videojuegos", layout="wide")


# Función para conectar a la base de datos
def get_connection():
    return mariadb.connect(
        user="root",
        password="Yez31446736",
        host="localhost",
        port=3307,
        database="data_w"
    )


# Función para cargar y normalizar los datos
def load_data():
    try:
        conn = get_connection()
        query = """
        SELECT 
            g.game_name, g.Team, g.Rating, g.Times_Listed, g.Number_of_Reviews,
            gen.genre_name,
            t.year, t.month,
            f.Plays, f.Playing, f.Backlogs, f.Wishlist
        FROM fact_sales f
        JOIN dim_games g ON f.game_id = g.game_id
        JOIN dim_genres gen ON f.genre_id = gen.genre_id
        JOIN dim_time t ON f.time_id = t.time_id
        """
        df = pd.read_sql(query, conn)
        conn.close()

        # Normalizar géneros (eliminar corchetes y comillas si existen)
        df['genre_name'] = df['genre_name'].apply(
            lambda x: x.strip("[]'").split(',')[0].strip() if isinstance(x, str) else x)

        return df
    except Exception as e:
        st.error(f"Error al cargar datos: {str(e)}")
        return pd.DataFrame()


# Función para preparar datos para el modelo predictivo
def prepare_data(df):
    # Codificar variables categóricas
    le_team = LabelEncoder()
    le_genre = LabelEncoder()

    df['Team_encoded'] = le_team.fit_transform(df['Team'].fillna('Unknown'))
    df['genre_encoded'] = le_genre.fit_transform(df['genre_name'])

    # Preparar features
    features = ['Team_encoded', 'Rating', 'Times_Listed', 'Number_of_Reviews',
                'genre_encoded', 'year', 'month', 'Playing', 'Backlogs', 'Wishlist']

    X = df[features].fillna(0)
    y = df['Plays'].fillna(0)

    return X, y, le_team, le_genre


# Función para entrenar el modelo
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test


def main():
    st.title("🎮 Dashboard de Análisis de Videojuegos")

    # Cargar datos
    with st.spinner('Cargando datos...'):
        df = load_data()

    if df.empty:
        st.error("No se pudieron cargar los datos. Por favor verifica la conexión a la base de datos.")
        return

    # Manejar valores nulos en Team
    df['Team'] = df['Team'].fillna('Desconocido')

    # Sidebar para filtros
    st.sidebar.header("Filtros")
    selected_year = st.sidebar.selectbox("Año", sorted(df['year'].unique()))
    unique_genres = sorted([g for g in df['genre_name'].unique() if pd.notna(g)])
    selected_genre = st.sidebar.selectbox("Género", ['Todos'] + list(unique_genres))

    # Filtrar datos
    filtered_df = df[df['year'] == selected_year]
    if selected_genre != 'Todos':
        filtered_df = filtered_df[filtered_df['genre_name'] == selected_genre]

    # Layout en columnas
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Top Juegos más Jugados")
        if not filtered_df.empty:
            fig_top_games = px.bar(
                filtered_df.nlargest(10, 'Plays'),
                x='game_name',
                y='Plays',
                color='Rating',
                title="Top Juegos por Número de Jugadores"
            )
            st.plotly_chart(fig_top_games)
        else:
            st.warning("No hay datos para mostrar con los filtros seleccionados")

    with col2:
        st.subheader("🎯 Distribución por Género")
        if not filtered_df.empty:
            genre_data = filtered_df.groupby('genre_name')['Plays'].sum().reset_index()
            fig_genres = px.pie(
                genre_data,
                values='Plays',
                names='genre_name',
                title="Distribución de Jugadores por Género"
            )
            st.plotly_chart(fig_genres)
        else:
            st.warning("No hay datos para mostrar con los filtros seleccionados")

    # Modelo Predictivo
    st.header("🤖 Predicción de Popularidad")

    try:
        # Preparar y entrenar modelo
        X, y, le_team, le_genre = prepare_data(df)
        model, X_test, y_test = train_model(X, y)

        # Interfaz para predicciones
        col3, col4, col5 = st.columns(3)

        with col3:
            # Obtener desarrolladores únicos y válidos
            unique_teams = sorted([team for team in df['Team'].unique() if pd.notna(team)])
            team = st.selectbox("Desarrollador", unique_teams)
            rating = st.slider("Rating", 0.0, 5.0, 4.0)
            times_listed = st.number_input("Times Listed", 0, 1000000, 1000)

        with col4:
            genre = st.selectbox("Género", unique_genres)
            release_year = st.number_input("Año de Lanzamiento", 1980, 2024, 2024)
            release_month = st.slider("Mes de Lanzamiento", 1, 12, 6)

        with col5:
            playing = st.number_input("Jugando Actualmente", 0, 1000000, 1000)
            backlogs = st.number_input("En Backlog", 0, 1000000, 1000)
            wishlist = st.number_input("En Wishlist", 0, 1000000, 1000)

        if st.button("Predecir Popularidad"):
            # Preparar datos para predicción
            prediction_data = pd.DataFrame({
                'Team_encoded': [le_team.transform([team])[0]],
                'Rating': [rating],
                'Times_Listed': [times_listed],
                'Number_of_Reviews': [0],  # Placeholder
                'genre_encoded': [le_genre.transform([genre])[0]],
                'year': [release_year],
                'month': [release_month],
                'Playing': [playing],
                'Backlogs': [backlogs],
                'Wishlist': [wishlist]
            })

            # Realizar predicción
            prediction = model.predict(prediction_data)[0]

            # Mostrar resultado
            st.success(f"Predicción de número de jugadores: {int(prediction):,}")

            # Mostrar importancia de características
            feature_importance = pd.DataFrame({
                'Feature': ['Team', 'Rating', 'Times Listed', 'Reviews', 'Genre',
                            'Year', 'Month', 'Playing', 'Backlogs', 'Wishlist'],
                'Importance': model.feature_importances_
            })
            feature_importance = feature_importance.sort_values('Importance', ascending=False)

            st.subheader("📈 Importancia de Características")
            fig_importance = px.bar(
                feature_importance,
                x='Feature',
                y='Importance',
                title="Importancia de Características en la Predicción"
            )
            st.plotly_chart(fig_importance)

    except Exception as e:
        st.error(f"Error en el modelo predictivo: {str(e)}")


if __name__ == "__main__":
    main()