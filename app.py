import re

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from pathlib import Path


# ============================================================
# CONFIGURAÇÃO
# ============================================================

st.set_page_config(
    page_title="Marvel Database Explorer",
    page_icon="🦸",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_FILE = Path("power_grid_geral.csv")

ATTRIBUTES = [
    "Intelligence",
    "Strength",
    "Speed",
    "Durability",
    "Energy Projection",
    "Fighting Skills",
]

PROFILE_COLUMNS = [
    "Appearances",
    "CurrentAlias",
    "Gender",
    "Height_cm",
    "Weight_kg",
    "Eyes",
    "Hair",
    "Reality",
]

GROUP_COLUMNS = ["Gender_Group", "Eye_Group", "Hair_Group", "Reality_Group"]

UNKNOWN = "Unknown"

# Campos de texto da infobox que podem conter citações como "[ 2 ]".
TEXT_PROFILE_COLUMNS = ["CurrentAlias", "Gender", "Eyes", "Hair", "Reality"]


# ============================================================
# PALETA DOS GRÁFICOS
# ============================================================

PRIMARY_RED = "#E23636"

# O vermelho é a cor principal (barras simples, histogramas, dispersão).
# As demais cores só aparecem em gráficos com várias séries, como a
# comparação de personagens e o box plot.
CHART_COLORS = [
    PRIMARY_RED,
    "#2D3047",
    "#F4A259",
    "#5B8E7D",
    "#8F2D56",
    "#7D8597",
    "#BC4B51",
    "#8CB369",
]

px.defaults.color_discrete_sequence = CHART_COLORS

# Escala contínua (ex.: Top 15 por Grid Total): rosa claro -> vermelho -> vinho.
px.defaults.color_continuous_scale = ["#FAD4D4", PRIMARY_RED, "#6E0F0F"]


def style_bar_labels(fig, size=None):
    """
    Rótulos de contagem sempre na horizontal (o Plotly os gira em barras
    estreitas) e, opcionalmente, em fonte maior. Se não couberem dentro da
    barra, o Plotly os coloca fora dela.
    """
    fig.update_traces(textangle=0, cliponaxis=False)
    if size:
        fig.update_traces(textfont_size=size)
    return fig


def histogram_with_counts(values, nbins, x_label, title):
    """
    Histograma com a contagem de personagens sobre cada barra.
    Faixas vazias não recebem rótulo, para não poluir caudas longas.
    """
    counts, edges = np.histogram(pd.Series(values).dropna(), bins=nbins)
    bins = pd.DataFrame(
        {
            "Center": (edges[:-1] + edges[1:]) / 2,
            "Characters": counts,
            "Range": [
                f"{start:,.1f} – {end:,.1f}"
                for start, end in zip(edges[:-1], edges[1:])
            ],
        }
    )
    bins["Label"] = bins["Characters"].astype(str).where(bins["Characters"] > 0, "")

    fig = px.bar(
        bins,
        x="Center",
        y="Characters",
        text="Label",
        title=title,
        labels={"Center": x_label, "Characters": "Number of characters"},
        hover_data={"Range": True, "Characters": True, "Center": False, "Label": False},
    )
    fig.update_traces(
        width=float(edges[1] - edges[0]),
        marker_line_color="white",
        marker_line_width=1,
        textposition="outside",
        cliponaxis=False,
    )
    return fig


# ============================================================
# ESTILO
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    [data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.22);
        padding: 14px;
        border-radius: 10px;
    }

    /* ---- Sidebar: layout mais compacto ---- */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.6rem;
    }
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
        align-items: center;
        gap: 0.5rem;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        margin-bottom: 0;
    }

    /* ---- Sliders: sem os limites 0/7; o valor atual fica no selo ---- */
    [data-testid="stSidebar"] [data-testid="stSlider"] {
        padding-bottom: 0;
    }
    [data-testid="stSidebar"] [data-testid="stSliderTickBar"],
    [data-testid="stSidebar"] [data-testid="stTickBar"] {
        display: none;
    }
    [data-testid="stSidebar"] [data-testid="stSliderThumbValue"] {
        font-size: 0.75rem;
    }

    /* ---- Selo com o intervalo selecionado ---- */
    .pg-badge-wrap {
        display: flex;
        justify-content: flex-end;
    }
    .pg-badge {
        font-size: 0.78rem;
        font-weight: 600;
        font-variant-numeric: tabular-nums;
        padding: 0.05rem 0.6rem;
        border-radius: 999px;
        border: 1px solid rgba(128,128,128,0.30);
        color: rgba(128,128,128,0.95);
        white-space: nowrap;
    }
    .pg-badge--active {
        background: #E23636;
        border-color: #E23636;
        color: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CATEGORIZAÇÃO DOS CAMPOS DE PERFIL
# ============================================================

def strip_citations(series):
    """Remove notas como '[ 2 ]' ou '[ 1 ]' de uma coluna de texto, preservando NaN."""
    result = series.astype(object)
    mask = result.notna()
    result.loc[mask] = (
        result.loc[mask]
        .astype(str)
        .str.replace(r"\s*\[[^\]]*\]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .replace("", np.nan)
    )
    return result


def group_gender(value):
    """Usa o primeiro valor listado (o atual) e ignora notas como [ 2 ]."""
    if pd.isna(value):
        return UNKNOWN
    text = re.sub(r"\[.*?\]", "", str(value))
    first = text.split(",")[0].strip().lower()

    if "agender" in first:
        return "Agender"
    if "genderfluid" in first or "male/female" in first or "female/male" in first:
        return "Genderfluid / Other"
    if first.startswith("female"):
        return "Female"
    if first.startswith("male"):
        return "Male"
    return "Genderfluid / Other"


def group_eyes(value):
    """Agrupa pela cor da íris atual; ignora cores '(formerly)'."""
    if pd.isna(value):
        return UNKNOWN
    low = str(value).lower()

    if "no eyes at all" in low:
        return "No eyes"

    match = re.search(r"irises:\s*(.*)", low)
    iris = match.group(1).strip() if match else low.strip()

    if not iris or iris.startswith("no visible"):
        return "No visible irises"

    first = iris.split()[0]
    if first == "variable":
        return "Variable"
    if first in {"blue", "brown", "green", "black", "red", "grey", "hazel"}:
        return first.capitalize()
    return "Other color"


def group_hair(value):
    """Agrupa pelo primeiro valor listado (o atual)."""
    if pd.isna(value):
        return UNKNOWN
    first = str(value).split(",")[0].strip().lower()

    if any(k in first for k in ("no hair", "bald", "shaved")):
        return "Bald / No hair"
    if "blond" in first:
        return "Blond"
    if "black" in first:
        return "Black"
    if "brown" in first:
        return "Brown"
    if "red" in first or "auburn" in first:
        return "Red / Auburn"
    if any(k in first for k in ("grey", "gray", "silver", "white")):
        return "Grey / White"
    if "variable" in first:
        return "Variable"
    return "Other color"


def clean_reality(value):
    """Extrai o ID da Terra: 'Earth-295 , (currently displaced...)' -> 'Earth-295'."""
    if pd.isna(value):
        return UNKNOWN
    text = str(value).strip()
    match = re.match(r"(Earth-\d+)", text)
    return match.group(1) if match else text


def group_reality(base):
    if base == UNKNOWN:
        return UNKNOWN
    if base == "Earth-616":
        return "Earth-616 (main)"
    if base == "Earth-1610":
        return "Earth-1610 (Ultimate)"
    if base.startswith("Earth-"):
        return "Other Earths"
    return "Other dimensions"


# ============================================================
# CARREGAMENTO DOS DADOS
# ============================================================

@st.cache_data
def load_data(file_path, modified_time):
    df = pd.read_csv(file_path)

    # Compatibilidade com CSVs que salvam o índice como primeira coluna.
    if "Name" not in df.columns:
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "Name"})

    df["Name"] = df["Name"].astype(str)

    # Compatibilidade com datasets antigos que ainda não possuem os novos campos.
    for column in PROFILE_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan

    for attr in ATTRIBUTES:
        if attr not in df.columns:
            df[attr] = np.nan
        df[attr] = pd.to_numeric(df[attr], errors="coerce")

    df["Appearances"] = pd.to_numeric(df["Appearances"], errors="coerce")

    for column in ["Height_cm", "Weight_kg"]:
        # O scraper pode registrar múltiplas medidas separadas por '; '.
        # Para análises numéricas, usa-se a primeira medida; a string original é preservada.
        df[f"{column}_numeric"] = pd.to_numeric(
            df[column].astype(str).str.split(";").str[0].str.strip(),
            errors="coerce",
        )

    for column in ["CurrentAlias", "Gender", "Eyes", "Hair", "Reality"]:
        df[column] = df[column].replace({"nan": np.nan, "None": np.nan, "": np.nan})

    # Remove citações/notas de rodapé: "Abdul Alhazred [ 2 ] [ 1 ]" -> "Abdul Alhazred".
    for column in TEXT_PROFILE_COLUMNS:
        df[column] = strip_citations(df[column])

    # Conforme solicitado: dados ausentes passam a ser zero.
    df[ATTRIBUTES] = df[ATTRIBUTES].fillna(0)

    # Os níveis válidos do Power Grid são de 1 a 7.
    # Zero é reservado para representar dado ausente.
    for attr in ATTRIBUTES:
        df.loc[~df[attr].between(1, 7), attr] = 0

    # Categorias gerais para filtros e gráficos.
    df["Gender_Group"] = df["Gender"].map(group_gender)
    df["Eye_Group"] = df["Eyes"].map(group_eyes)
    df["Hair_Group"] = df["Hair"].map(group_hair)
    df["Reality_Base"] = df["Reality"].map(clean_reality)
    df["Reality_Group"] = df["Reality_Base"].map(group_reality)

    # Alias já limpo (sem notas como "[ 1 ]") para busca e exibição.
    alias_clean = df["CurrentAlias"].fillna("").astype(str)
    has_alias = alias_clean.ne("") & alias_clean.str.lower().ne(df["Name"].str.lower())

    # Texto usado na busca: nome + alias, em minúsculas.
    df["Search_Text"] = (df["Name"] + " " + alias_clean).str.lower()

    # Rótulo para listas de seleção: "James Howlett (Wolverine)".
    df["Display_Name"] = np.where(
        has_alias,
        df["Name"] + " (" + alias_clean.str[:80] + ")",
        df["Name"],
    )

    return df.drop_duplicates(subset="Name", keep="first")


if not DATA_FILE.exists():
    st.error(
        f"Arquivo não encontrado: {DATA_FILE.resolve()}\n\n"
        "Coloque power_grid_geral_2.csv na mesma pasta de app.py."
    )
    st.stop()

df = load_data(str(DATA_FILE), DATA_FILE.stat().st_mtime)

# Métricas calculadas por personagem.
df["Grid Total"] = df[ATTRIBUTES].replace(0, np.nan).sum(axis=1)
df["Attributes Available"] = (df[ATTRIBUTES] > 0).sum(axis=1)


# ============================================================
# CABEÇALHO
# ============================================================

st.title("🦸 Marvel Database Explorer")

st.markdown(
    """
    Explore, compare and analyze Marvel characters using their
    six Power Grid attributes.
    """
)

st.caption(
    "Dataset based on Marvel Database. "
    "A score of 0 indicates a missing or unrecognized attribute, "
    "not an official Power Grid level."
)


# ============================================================
# BARRA LATERAL: FILTROS GLOBAIS
# ============================================================

def ordered_options(series):
    """Categorias por frequência; 'Unknown' sempre por último."""
    counts = series.value_counts()
    options = [c for c in counts.index if c != UNKNOWN]
    if UNKNOWN in counts.index:
        options.append(UNKNOWN)
    return options


st.sidebar.title("Filters")

st.sidebar.caption(
    "Filters apply to the dashboard's charts and character table."
)

search_name = st.sidebar.text_input(
    "Search character",
    placeholder="e.g. Spider-Man or Wolverine",
    help="Searches the character name and current alias.",
)

st.sidebar.subheader("Appearances")

appearance_values = df["Appearances"].dropna()

if not appearance_values.empty:
    appearances_min = int(appearance_values.min())
    appearances_max = int(appearance_values.max())

    if appearances_min < appearances_max:
        min_appearances = st.sidebar.number_input(
            "Minimum number of appearances",
            min_value=appearances_min,
            max_value=appearances_max,
            value=appearances_min,
            step=1,
            help=(
                "Shows only characters with at least this many appearances. "
                "Characters without a recorded count are excluded while the "
                "filter is above the minimum."
            ),
        )
    else:
        min_appearances = None
        st.sidebar.caption(f"Only available value: {appearances_min}")
else:
    appearances_min, appearances_max = 0, 0
    min_appearances = None
    st.sidebar.caption("No appearance counts available.")


def numeric_bounds(series):
    """Menor e maior valor inteiro que cobrem a coluna, ou None se vazia."""
    values = series.dropna()
    if values.empty:
        return None
    return int(np.floor(values.min())), int(np.ceil(values.max()))


st.sidebar.subheader("Height and weight")

# coluna numérica -> (rótulo, unidade)
MEASURE_FILTERS = {
    "Height_cm_numeric": ("Height", "cm"),
    "Weight_kg_numeric": ("Weight", "kg"),
}

measure_ranges = {}

for measure_column, (measure_label, measure_unit) in MEASURE_FILTERS.items():
    bounds = numeric_bounds(df[measure_column])

    if bounds is None:
        st.sidebar.caption(f"No {measure_label.lower()} data available.")
        continue

    lower_bound, upper_bound = bounds

    st.sidebar.markdown(
        f"**{measure_label} ({measure_unit})**",
        help=(
            "Characters without a recorded measurement are excluded while "
            "this filter is narrowed. When several values were recorded, "
            "the first one is used."
        ),
    )

    low_col, high_col = st.sidebar.columns(2)
    measure_low = low_col.number_input(
        "Min",
        min_value=lower_bound,
        max_value=upper_bound,
        value=lower_bound,
        step=1,
        key=f"filter_{measure_column}_min",
    )
    measure_high = high_col.number_input(
        "Max",
        min_value=lower_bound,
        max_value=upper_bound,
        value=upper_bound,
        step=1,
        key=f"filter_{measure_column}_max",
    )

    measure_ranges[measure_column] = (
        measure_low, measure_high, lower_bound, upper_bound,
    )

st.sidebar.subheader("Character traits")

selected_genders = st.sidebar.multiselect(
    "Gender",
    options=ordered_options(df["Gender_Group"]),
    placeholder="All genders",
    help="Leave empty to include all.",
)
selected_eyes = st.sidebar.multiselect(
    "Eyes",
    options=ordered_options(df["Eye_Group"]),
    placeholder="All eye types",
    help="Leave empty to include all.",
)
selected_hair = st.sidebar.multiselect(
    "Hair",
    options=ordered_options(df["Hair_Group"]),
    placeholder="All hair types",
    help="Leave empty to include all.",
)

st.sidebar.subheader("Reality")

selected_reality_groups = st.sidebar.multiselect(
    "Reality group",
    options=ordered_options(df["Reality_Group"]),
    placeholder="All realities",
    help="Leave empty to include all.",
)

# Detalhe opcional, restrito aos grupos escolhidos acima.
detail_pool = df if not selected_reality_groups else df[
    df["Reality_Group"].isin(selected_reality_groups)
]
selected_realities = st.sidebar.multiselect(
    "Specific reality (optional)",
    options=ordered_options(detail_pool["Reality_Base"]),
    placeholder="All in selected groups",
)

st.sidebar.subheader("Power Grid attributes")

ATTRIBUTE_FILTER_HELP = "0 means missing data; valid Power Grid levels are 1–7."


def format_level_range(low, high):
    """'3' quando os dois extremos coincidem; caso contrário '3–5'."""
    return str(low) if low == high else f"{low}–{high}"


filters = {}

for attr in ATTRIBUTES:
    # Cabeçalho compacto: nome (com ajuda) à esquerda, seleção à direita.
    # As colunas são criadas antes do slider para ficarem acima dele; o selo
    # é preenchido depois, quando o valor do slider já é conhecido.
    name_col, value_col = st.sidebar.columns([3, 2])
    name_col.markdown(f"**{attr}**", help=ATTRIBUTE_FILTER_HELP)

    filters[attr] = st.sidebar.slider(
        attr,
        min_value=0,
        max_value=7,
        value=(0, 7),
        step=1,
        key=f"filter_{attr}",
        label_visibility="collapsed",
    )

    low, high = filters[attr]
    badge_class = "pg-badge pg-badge--active" if (low, high) != (0, 7) else "pg-badge"
    value_col.markdown(
        f'<div class="pg-badge-wrap">'
        f'<span class="{badge_class}">{format_level_range(low, high)}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )

st.sidebar.divider()

ignore_missing = st.sidebar.checkbox(
    "Ignore missing values in statistical charts",
    value=True,
    help=(
        "Treat zeros as missing when calculating distributions, "
        "means and correlations. The original values remain zero "
        "in the dataset and table."
    ),
)

st.sidebar.caption(f"{len(df):,} characters loaded")


# ============================================================
# APLICAÇÃO DOS FILTROS
# ============================================================

filtered_df = df.copy()

# Busca por nome ou alias atual.
if search_name.strip():
    filtered_df = filtered_df[
        filtered_df["Search_Text"].str.contains(
            search_name.strip().lower(), regex=False
        )
    ]

# Só filtra por aparições se o mínimo foi aumentado; caso contrário,
# personagens sem contagem (NaN) seriam excluídos silenciosamente.
if min_appearances is not None and min_appearances > appearances_min:
    filtered_df = filtered_df[filtered_df["Appearances"] >= min_appearances]

# Altura e peso: mesma regra, só filtra quando o intervalo foi estreitado.
for measure_column, (
    measure_low, measure_high, lower_bound, upper_bound
) in measure_ranges.items():
    if measure_low > lower_bound or measure_high < upper_bound:
        filtered_df = filtered_df[
            filtered_df[measure_column].between(measure_low, measure_high)
        ]

for column, selected in [
    ("Gender_Group", selected_genders),
    ("Eye_Group", selected_eyes),
    ("Hair_Group", selected_hair),
    ("Reality_Group", selected_reality_groups),
    ("Reality_Base", selected_realities),
]:
    if selected:
        filtered_df = filtered_df[filtered_df[column].isin(selected)]

for attr, (minimum, maximum) in filters.items():
    filtered_df = filtered_df[filtered_df[attr].between(minimum, maximum)]

# Cópia destinada aos gráficos estatísticos.
analysis_df = filtered_df.copy()

if ignore_missing:
    analysis_df[ATTRIBUTES] = analysis_df[ATTRIBUTES].replace(0, np.nan)

analysis_df["Grid Total"] = analysis_df[ATTRIBUTES].sum(axis=1, min_count=1)

st.caption(
    f"Showing {len(filtered_df):,} of {len(df):,} characters after filtering."
)

if filtered_df.empty:
    st.warning("No characters match the current filters.")
    st.stop()


# ============================================================
# ABAS
# ============================================================

tab_overview, tab_compare, tab_distribution, tab_relationships, tab_profiles, tab_data = st.tabs(
    [
        "📊 Overview",
        "⚔️ Character Comparison",
        "📈 Distributions",
        "🔬 Attribute Relationships",
        "🧬 Character Profiles",
        "📋 Data Explorer",
    ]
)


# ============================================================
# 1. OVERVIEW
# ============================================================

with tab_overview:
    st.header("Dataset Overview")

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric(
        "Characters",
        f"{len(filtered_df):,}",
    )

    metric2.metric(
        "Attributes",
        len(ATTRIBUTES),
    )

    valid_pct = (
        (filtered_df[ATTRIBUTES] > 0).sum().sum()
        / (len(filtered_df) * len(ATTRIBUTES))
        * 100
        if len(filtered_df) > 0
        else 0
    )

    metric3.metric(
        "Data completeness",
        f"{valid_pct:.1f}%",
    )

    metric4.metric(
        "Missing attribute values",
        f"{(filtered_df[ATTRIBUTES] == 0).sum().sum():,}",
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Attribute distributions")

        distribution_attr = st.selectbox(
            "Choose an attribute",
            ATTRIBUTES,
            key="overview_distribution_attr",
        )

        values = filtered_df[distribution_attr]

        if ignore_missing:
            values = values[values > 0]

        counts = (
            values.value_counts()
            .reindex(range(1, 8), fill_value=0)
            .rename_axis("Level")
            .reset_index(name="Characters")
        )

        fig = px.bar(
            counts,
            x="Level",
            y="Characters",
            text="Characters",
            title=f"{distribution_attr}: level frequency",
            labels={
                "Level": "Power Grid level",
                "Characters": "Number of characters",
            },
        )

        fig.update_xaxes(dtick=1)
        fig.update_layout(showlegend=False)
        style_bar_labels(fig)
        st.plotly_chart(fig, use_container_width=True, key="overview_level_frequency")

    with right:
        st.subheader("Highest Grid Totals")

        ranking = filtered_df[
            ["Name", "Grid Total", "Attributes Available"]
        ].copy()

        ranking = ranking.sort_values(
            "Grid Total",
            ascending=False,
        ).head(15)

        fig = px.bar(
            ranking.sort_values("Grid Total"),
            x="Grid Total",
            y="Name",
            orientation="h",
            color="Grid Total",
            text="Grid Total",
            hover_data=["Attributes Available"],
            title="Top 15 by sum of known attributes",
            labels={
                "Grid Total": "Sum of known levels",
                "Name": "Character",
                "Attributes Available": "Available attributes",
            },
        )

        fig.update_traces(texttemplate="%{x:.0f}")
        style_bar_labels(fig, size=15)

        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Grid Total is a descriptive sum, not an official Marvel "
            "rating. Characters with missing attributes may have "
            "underestimated totals."
        )

    st.subheader("Average level by attribute")

    averages = analysis_df[ATTRIBUTES].mean().rename_axis(
        "Attribute"
    ).reset_index(name="Mean Level")

    fig = px.bar(
        averages.sort_values("Mean Level"),
        x="Mean Level",
        y="Attribute",
        orientation="h",
        text="Mean Level",
        labels={"Mean Level": "Mean Power Grid level"},
        title="Mean attribute levels among filtered characters",
    )

    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_xaxes(range=[0, 7])
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 2. CHARACTER COMPARISON
# ============================================================

with tab_compare:
    st.header("Character Comparison")

    st.markdown(
        "Select characters to compare their Power Grid profiles. "
        "You can search by name or current alias."
    )

    all_names = sorted(filtered_df["Name"].unique())
    display_names = dict(zip(filtered_df["Name"], filtered_df["Display_Name"]))

    # Mantém a seleção anterior, removendo apenas quem saiu dos filtros.
    st.session_state["comparison_characters"] = [
        name
        for name in st.session_state.get("comparison_characters", [])
        if name in all_names
    ]

    selected_characters = st.multiselect(
        "Characters",
        options=all_names,
        format_func=lambda name: display_names.get(name, name),
        placeholder="Search by name or alias...",
        key="comparison_characters",
    )

    if not selected_characters:
        st.info("Select at least one character to display comparisons.")

    else:
        comparison = filtered_df[
            filtered_df["Name"].isin(selected_characters)
        ].copy()

        # Os gráficos usam o rótulo com alias na legenda.
        comparison_long = comparison.melt(
            id_vars=["Display_Name"],
            value_vars=ATTRIBUTES,
            var_name="Attribute",
            value_name="Level",
        )

        comparison_long["Level"] = comparison_long["Level"].astype(float)

        st.subheader("Radar chart")

        radar = px.line_polar(
            comparison_long,
            r="Level",
            theta="Attribute",
            color="Display_Name",
            line_close=True,
            markers=True,
            range_r=[0, 7],
            title="Power Grid profiles",
        )

        radar.update_traces(fill="toself")
        radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 7],
                    dtick=1,
                )
            ),
            legend_title_text="Character",
        )

        st.plotly_chart(radar, use_container_width=True)

        st.subheader("Attribute-by-attribute comparison")

        bar_fig = px.bar(
            comparison_long,
            x="Attribute",
            y="Level",
            color="Display_Name",
            barmode="group",
            category_orders={"Attribute": ATTRIBUTES},
            title="Power Grid levels",
            labels={
                "Level": "Power Grid level",
                "Display_Name": "Character",
            },
        )

        bar_fig.update_yaxes(range=[0, 7], dtick=1)
        st.plotly_chart(bar_fig, use_container_width=True)

        st.subheader("Comparison table")

        comparison_table = comparison.set_index("Display_Name")[
            ATTRIBUTES + ["Grid Total", "Attributes Available"]
        ].T

        st.dataframe(
            comparison_table,
            use_container_width=True,
        )

        csv_comparison = comparison.to_csv(
            index=False
        ).encode("utf-8-sig")

        st.download_button(
            "Download comparison as CSV",
            data=csv_comparison,
            file_name="marvel_power_grid_comparison.csv",
            mime="text/csv",
        )


# ============================================================
# 3. DISTRIBUTIONS
# ============================================================

with tab_distribution:
    st.header("Attribute Distributions")

    distribution_attr = st.selectbox(
        "Select attribute",
        ATTRIBUTES,
        key="distribution_attribute",
    )

    raw_values = filtered_df[distribution_attr]

    # Estatísticas calculadas sem considerar zeros como níveis válidos,
    # quando a opção de ignorar dados ausentes está ativada.
    values = (
        raw_values[raw_values > 0]
        if ignore_missing
        else raw_values
    )

    # --------------------------------------------------------
    # SUMMARY STATISTICS
    # --------------------------------------------------------

    st.subheader(f"Summary Statistics — {distribution_attr}")

    valid_values = raw_values[raw_values > 0]
    missing_count = int((raw_values == 0).sum())

    if not values.empty:
        mean_value = values.mean()
        median_value = values.median()

        modes = values.mode()
        mode_value = (
            ", ".join(
                str(int(v)) if float(v).is_integer() else str(v)
                for v in modes.tolist()
            )
            if not modes.empty
            else "N/A"
        )
    else:
        mean_value = np.nan
        median_value = np.nan
        mode_value = "N/A"

    metric1, metric2, metric3, metric4, metric5 = st.columns(5)

    metric1.metric(
        "Mean",
        f"{mean_value:.2f}" if pd.notna(mean_value) else "N/A",
    )

    metric2.metric(
        "Median",
        f"{median_value:.1f}" if pd.notna(median_value) else "N/A",
    )

    metric3.metric(
        "Mode",
        mode_value,
    )

    metric4.metric(
        "Valid values",
        f"{len(valid_values):,}",
    )

    metric5.metric(
        "Missing values",
        f"{missing_count:,}",
    )

    st.caption(
        "Statistics exclude missing values when the corresponding "
        "sidebar option is enabled. Mean and median are descriptive "
        "summaries of ordinal Power Grid levels."
    )

    st.divider()

    # --------------------------------------------------------
    # FREQUENCY DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("Frequency of Each Level")

    levels = range(1, 8) if ignore_missing else range(0, 8)

    frequency = (
        values.value_counts()
        .reindex(levels, fill_value=0)
        .rename_axis("Level")
        .reset_index(name="Characters")
    )

    frequency["Level"] = frequency["Level"].astype(int)

    fig = px.bar(
        frequency,
        x="Level",
        y="Characters",
        text="Characters",
        title=f"{distribution_attr}: level frequency",
        labels={
            "Level": "Power Grid level",
            "Characters": "Number of characters",
        },
    )

    fig.update_xaxes(dtick=1)
    fig.update_layout(showlegend=False)
    style_bar_labels(fig)

    st.plotly_chart(fig, use_container_width=True, key="distribution_level_frequency")

    st.divider()

    # --------------------------------------------------------
    # SCATTER PLOT: TWO SELECTABLE VARIABLES
    # --------------------------------------------------------

    st.subheader("Compare Two Variables")

    scatter_options = ATTRIBUTES + ["Grid Total"]

    col_x, col_y = st.columns(2)

    with col_x:
        x_attr = st.selectbox(
            "X-axis",
            scatter_options,
            index=scatter_options.index(distribution_attr),
            key="distribution_scatter_x",
        )

    with col_y:
        default_y = "Grid Total" if x_attr != "Grid Total" else "Strength"

        y_attr = st.selectbox(
            "Y-axis",
            scatter_options,
            index=scatter_options.index(default_y),
            key="distribution_scatter_y",
        )

    scatter_df = filtered_df[["Name"] + ATTRIBUTES + ["Grid Total"]].copy()

    # Zero representa dado ausente, não um nível real.
    if ignore_missing:
        for attr in [x_attr, y_attr]:
            if attr in ATTRIBUTES:
                scatter_df.loc[scatter_df[attr] == 0, attr] = np.nan

    scatter_df = scatter_df.dropna(subset=[x_attr, y_attr])

    if x_attr == y_attr:
        st.info("Choose different variables for the X and Y axes.")

    elif scatter_df.empty:
        st.warning("No valid observations available for this combination.")

    else:
        # Keep original values for hover and apply jitter only for display.
        rng = np.random.default_rng(42)

        scatter_df["X_plot"] = scatter_df[x_attr]
        scatter_df["Y_plot"] = scatter_df[y_attr]

        jitter_amount = 0.15

        if x_attr in ATTRIBUTES:
            scatter_df["X_plot"] = (
                scatter_df["X_plot"]
                + rng.uniform(
                    -jitter_amount,
                    jitter_amount,
                    size=len(scatter_df),
                )
            )

        if y_attr in ATTRIBUTES:
            scatter_df["Y_plot"] = (
                scatter_df["Y_plot"]
                + rng.uniform(
                    -jitter_amount,
                    jitter_amount,
                    size=len(scatter_df),
                )
            )

        fig = px.scatter(
            scatter_df,
            x="X_plot",
            y="Y_plot",
            hover_name="Name",
            hover_data={
                **{attr: True for attr in ATTRIBUTES},
                "Grid Total": True,
                "X_plot": False,
                "Y_plot": False,
            },
            opacity=0.45,
            title=f"{y_attr} vs. {x_attr}",
            labels={
                "X_plot": x_attr,
                "Y_plot": y_attr,
            },
        )

        if x_attr in ATTRIBUTES:
            fig.update_xaxes(dtick=1, range=[0.5, 7.5])

        if y_attr in ATTRIBUTES:
            fig.update_yaxes(dtick=1, range=[0.5, 7.5])

        fig.update_layout(
            xaxis_title=x_attr,
            yaxis_title=y_attr,
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            f"{len(scatter_df):,} characters shown. "
            "Jitter is applied to discrete attributes for visibility; "
            "hover values remain unchanged. Grid Total is the sum of "
            "known attribute levels, so missing attributes may lead "
            "to underestimated totals."
        )

    st.divider()

    # --------------------------------------------------------
    # BOX PLOT: ALL SIX ATTRIBUTES
    # --------------------------------------------------------

    st.subheader("Distribution Comparison Across All Attributes")

    long_data = analysis_df.melt(
        id_vars=["Name"],
        value_vars=ATTRIBUTES,
        var_name="Attribute",
        value_name="Level",
    ).dropna(subset=["Level"])

    if not long_data.empty:
        fig = px.box(
            long_data,
            x="Attribute",
            y="Level",
            color="Attribute",
            points=False,
            category_orders={"Attribute": ATTRIBUTES},
            title="Power Grid level distributions",
            labels={
                "Attribute": "Attribute",
                "Level": "Power Grid level",
            },
        )

        fig.update_yaxes(
            range=[0.5, 7.5] if ignore_missing else [-0.5, 7.5],
            dtick=1,
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No valid data available for the box plot.")


# ============================================================
# 4. ATTRIBUTE RELATIONSHIPS — CORRELATION MATRIX
# ============================================================

with tab_relationships:
    st.header("Attribute Relationships")

    correlation_columns = ATTRIBUTES + ["Grid Total"]

    correlation_data = analysis_df[correlation_columns]

    corr = correlation_data.corr(
        method="spearman",
        min_periods=2,
    )

    # Hide the upper triangle, keeping the diagonal and lower triangle.
    mask = np.triu(
        np.ones(corr.shape, dtype=bool),
        k=1,
    )

    fig = px.imshow(
        corr,
        x=correlation_columns,
        y=correlation_columns,
        color_continuous_scale="RdBu_r",  # correlações positivas em vermelho
        zmin=-1,
        zmax=1,
        aspect="equal",
        text_auto=".2f",
        title="Spearman Correlation Matrix",
        labels={
            "x": "Variable",
            "y": "Variable",
            "color": "Correlation",
        },
    )

    # Mask the upper triangle.
    z = corr.to_numpy().astype(float)
    z[mask] = np.nan

    fig.update_traces(
        z=z,
        text=np.where(mask, "", np.vectorize(
            lambda v: f"{v:.2f}" if pd.notna(v) else ""
        )(corr.to_numpy())),
        texttemplate="%{text}",
    )

    fig.update_xaxes(side="bottom")
    fig.update_yaxes(autorange="reversed")

    fig.update_layout(
        width=800,
        height=800,
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Spearman correlations measure monotonic associations. "
        "Grid Total is calculated from the available attribute levels, "
        "so its correlations with individual attributes are partly "
        "driven by its definition. Missing values are excluded pairwise."
    )


# ============================================================
# 5. CHARACTER PROFILES: PROFILE DATA, APPEARANCES, REALITY
# ============================================================

with tab_profiles:
    st.header("Character Profiles")
    st.markdown(
        "Explore profile characteristics, appearance counts, and "
        "realities recorded in the infoboxes."
    )

    profile_df = filtered_df.copy()
    total_profiles = len(profile_df)

    profile_metric_cols = st.columns(4)
    profile_metric_cols[0].metric("Characters", f"{total_profiles:,}")
    profile_metric_cols[1].metric("With appearance count", f"{profile_df['Appearances'].notna().sum():,}")
    profile_metric_cols[2].metric("With height", f"{profile_df['Height_cm'].notna().sum():,}")
    profile_metric_cols[3].metric("With weight", f"{profile_df['Weight_kg'].notna().sum():,}")

    st.divider()
    left, right = st.columns(2)

    with left:
        st.subheader("Appearances")
        appearances_data = profile_df.dropna(subset=["Appearances"]).sort_values("Appearances", ascending=False)
        if not appearances_data.empty:
            fig = histogram_with_counts(
                appearances_data["Appearances"], 30,
                "Number of appearances", "Distribution of appearance counts",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f"Median: {appearances_data['Appearances'].median():,.0f} appearances · "
                f"Maximum: {appearances_data['Appearances'].max():,.0f}."
            )
        else:
            st.info("No appearance counts available for the current filters.")

    with right:
        st.subheader("Most appearances")
        top_appearances = profile_df.dropna(subset=["Appearances"]).nlargest(15, "Appearances")
        if not top_appearances.empty:
            fig = px.bar(
                top_appearances.sort_values("Appearances"), x="Appearances", y="Name",
                orientation="h", title="Top 15 by appearance count",
                labels={"Appearances": "Number of appearances", "Name": "Character"},
                hover_data=["Reality", "Gender"],
                text="Appearances",
            )
            fig.update_traces(texttemplate="%{x:,.0f}")
            style_bar_labels(fig, size=15)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No appearance counts available for the current filters.")

    st.divider()
    st.subheader("Profile characteristics")
    profile_left, profile_right = st.columns(2)

    with profile_left:
        gender_counts = (
            profile_df["Gender_Group"].dropna().value_counts()
            .rename_axis("Gender").reset_index(name="Characters")
        )
        if not gender_counts.empty:
            fig = px.bar(gender_counts, x="Gender", y="Characters", title="Gender distribution", text="Characters")
            style_bar_labels(fig)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No gender data available for the current filters.")

    with profile_right:
        profile_measure = st.selectbox(
            "Measurement", ["Height_cm", "Weight_kg"],
            format_func=lambda value: "Height (cm)" if value == "Height_cm" else "Weight (kg)",
            key="profile_measurement",
        )
        numeric_column = f"{profile_measure}_numeric"
        measure_data = profile_df.dropna(subset=[numeric_column])
        if not measure_data.empty:
            unit = "cm" if profile_measure == "Height_cm" else "kg"
            fig = histogram_with_counts(
                measure_data[numeric_column], 30,
                f"{profile_measure.replace('_', ' ')} ({unit})",
                f"Distribution of {profile_measure.replace('_', ' ').lower()}",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "When multiple measurements are present, the histogram uses the first value; "
                "the table below preserves all extracted values."
            )
        else:
            st.info("No measurements available for the current filters.")

    # Olhos e cabelo, agora com categorias gerais.
    eyes_col, hair_col = st.columns(2)

    with eyes_col:
        eye_counts = (
            profile_df["Eye_Group"].value_counts()
            .rename_axis("Eyes").reset_index(name="Characters")
        )
        fig = px.bar(
            eye_counts.sort_values("Characters"), x="Characters", y="Eyes",
            orientation="h", title="Eye color distribution", text="Characters",
        )
        style_bar_labels(fig, size=14)
        st.plotly_chart(fig, use_container_width=True)

    with hair_col:
        hair_counts = (
            profile_df["Hair_Group"].value_counts()
            .rename_axis("Hair").reset_index(name="Characters")
        )
        fig = px.bar(
            hair_counts.sort_values("Characters"), x="Characters", y="Hair",
            orientation="h", title="Hair color distribution", text="Characters",
        )
        style_bar_labels(fig, size=14)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Reality distribution")
    reality_counts = (
        profile_df["Reality_Group"].value_counts()
        .rename_axis("Reality").reset_index(name="Characters")
    )
    if not reality_counts.empty:
        fig = px.bar(
            reality_counts.sort_values("Characters"), x="Characters", y="Reality",
            orientation="h", title="Characters by reality group", text="Characters",
            labels={"Characters": "Characters", "Reality": "Reality"},
        )
        style_bar_labels(fig, size=14)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Profile table")
    profile_table_columns = ["Name"] + PROFILE_COLUMNS
    st.dataframe(
        profile_df[profile_table_columns].sort_values("Name"),
        use_container_width=True, hide_index=True, height=450,
    )
    profile_csv = profile_df[profile_table_columns].to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download character profiles as CSV", profile_csv,
        "marvel_character_profiles.csv", "text/csv",
    )


# ============================================================
# 6. DATA EXPLORER
# ============================================================

with tab_data:
    st.header("Data Explorer")

    st.subheader("Sort and inspect")

    sort_options = ["Name"] + ATTRIBUTES + [
        "Grid Total", "Attributes Available", "Appearances",
        "Height_cm_numeric", "Weight_kg_numeric",
    ]

    left, right = st.columns([2, 1])

    with left:
        sort_by = st.selectbox(
            "Sort by",
            sort_options,
            index=sort_options.index("Grid Total"),
        )

    with right:
        ascending = st.selectbox(
            "Order",
            ["Descending", "Ascending"],
            index=0,
        ) == "Ascending"

    sorted_df = filtered_df.sort_values(
        by=sort_by,
        ascending=ascending,
    )

    display_columns = (
        ["Name"]
        + ATTRIBUTES
        + ["Grid Total", "Attributes Available"]
        + PROFILE_COLUMNS
        + GROUP_COLUMNS
    )

    st.dataframe(
        sorted_df[display_columns],
        use_container_width=True,
        height=500,
        hide_index=True,
    )

    st.caption(
        f"{len(sorted_df):,} characters displayed."
    )

    csv_data = sorted_df[display_columns].to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        "Download filtered data as CSV",
        data=csv_data,
        file_name="marvel_power_grid_filtered.csv",
        mime="text/csv",
    )

    st.subheader("Data quality")

    quality = pd.DataFrame(
        {
            "Attribute": ATTRIBUTES,
            "Available": [
                int((filtered_df[attr] > 0).sum())
                for attr in ATTRIBUTES
            ],
            "Missing": [
                int((filtered_df[attr] == 0).sum())
                for attr in ATTRIBUTES
            ],
        }
    )

    quality["Completeness (%)"] = (
        quality["Available"] / len(filtered_df) * 100
    ).round(2)

    st.dataframe(
        quality,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Profile field completeness")
    metadata_quality = pd.DataFrame({
        "Field": PROFILE_COLUMNS,
        "Available": [int(filtered_df[field].notna().sum()) for field in PROFILE_COLUMNS],
        "Missing": [int(filtered_df[field].isna().sum()) for field in PROFILE_COLUMNS],
    })
    metadata_quality["Completeness (%)"] = (
        metadata_quality["Available"] / len(filtered_df) * 100
    ).round(2)
    st.dataframe(metadata_quality, use_container_width=True, hide_index=True)