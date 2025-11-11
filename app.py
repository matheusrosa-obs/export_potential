from __future__ import annotations

import re
from typing import List, Tuple
import streamlit as st
import polars as pl
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import tracemalloc
import psutil
import os

from warnings import filterwarnings
filterwarnings("ignore")

# Add at the very top after imports
if 'memory_started' not in st.session_state:
    tracemalloc.start()
    st.session_state.memory_started = True
    st.session_state.snapshots = []

def show_memory_usage(label=""):
    current, peak = tracemalloc.get_traced_memory()
    process = psutil.Process(os.getpid())
    
    snapshot = {
        'label': label,
        'current_mb': current / 1024 / 1024,
        'peak_mb': peak / 1024 / 1024,
        'rss_mb': process.memory_info().rss / 1024 / 1024
    }
    
    st.session_state.snapshots.append(snapshot)
    
    st.sidebar.markdown(f"**{label}**")
    st.sidebar.text(f"Current: {snapshot['current_mb']:.2f} MB")
    st.sidebar.text(f"Peak: {snapshot['peak_mb']:.2f} MB")
    st.sidebar.text(f"Process: {snapshot['rss_mb']:.2f} MB")
    st.sidebar.markdown("---")
if 'last_memory' not in st.session_state:
    st.session_state.last_memory = None

def show_memory_delta(label=""):
    process = psutil.Process(os.getpid())
    current_mb = process.memory_info().rss / 1024 / 1024
    
    if st.session_state.last_memory is None:
        delta_mb = 0
        st.session_state.last_memory = current_mb
    else:
        delta_mb = current_mb - st.session_state.last_memory
        st.session_state.last_memory = current_mb
    
    delta_color = "🔴" if abs(delta_mb) > 50 else "🟡" if abs(delta_mb) > 10 else "🟢"
    
    st.sidebar.markdown(f"**{label}**")
    st.sidebar.text(f"Memory: {current_mb:.2f} MB ({delta_color} {delta_mb:+.2f} MB)")
    st.sidebar.markdown("---")

######## Setting the directories ########
# Diretórios relativos ao arquivo atual
try:
    app = Path(__file__).parent
except NameError:
    app = Path.cwd()
project_root = app.parent
references = app / 'references'

######## Setting the style ########
with open(app / "style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(
    page_title="Potencial de exportações - Observatório FIESC",
    page_icon=app / "logo_dark_mini.png",
    layout="wide"
)

### FUNÇÔES

def format_contabil(value):
    if value >= 1e9:
        return f"{value/1e9:,.1f} bi".replace(",", "X").replace(".", ",").replace("X", ".")
    elif value >= 1e6:
        return f"{value/1e6:,.1f} mi".replace(",", "X").replace(".", ",").replace("X", ".")
    elif value >= 1e3:
        return f"{value/1e3:,.1f} mil".replace(",", "X").replace(".", ",").replace("X", ".")
    else:
        return f"{value:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".")

def format_decimal(value, decimals=1):
    return f"{value:.{decimals}f}".replace(".", ",")


######## Loading the data ########
### Munic and VP list ###
df_munic_vp = pl.read_excel(references / 'munic_vp.xlsx')

vp = df_munic_vp['vp'].unique().to_list()
munic = df_munic_vp['munic'].unique().to_list()

### EPI scores SH6 ###
@st.cache_resource(ttl=1800, show_spinner=False)
def load_epi_scores_sh6():
    return pl.read_parquet(app / 'data' / 'epi_scores_sh6.parquet')
df_epi_sh6 = load_epi_scores_sh6()

df_epi_sh6 = df_epi_sh6.with_columns(
    pl.col("epi_score_normalized").round(3)
)
@st.cache_resource(ttl=1800, show_spinner=False)
def load_epi_countries():
    return pl.read_parquet(app / 'data' / 'epi_scores_countries.parquet')
df_epi_countries = load_epi_countries()

df_epi_countries = df_epi_countries.with_columns(
    pl.col("epi_score_normalized").round(3)
)
### EPI scores ###

@st.cache_resource(ttl=1800, show_spinner=False)
def load_epi_scores():
    return pl.read_parquet(app / 'data' / 'epi_scores_processed.parquet')
df_epi = load_epi_scores()

### EPI scores SC Competitiva ###
@st.cache_resource(ttl=1800, show_spinner=False)
def load_epi_scores_sc_comp():
    return pl.read_parquet(app / 'data' / 'epi_scores_sc_comp.parquet')
df_epi_sc_comp = load_epi_scores_sc_comp()

df_epi_sc_comp = df_epi_sc_comp.with_columns(
    pl.col("epi_score_normalized").round(3)
)

@st.cache_resource(ttl=1800, show_spinner=False)
def load_markets():
    return pl.read_parquet(app / 'data' / 'app_dataset_processed.parquet')
df_markets = load_markets()

@st.cache_resource(show_spinner=False)
def load_competitors():
    return pl.read_parquet(app / 'data' / 'df_competitors.parquet')

df_competitors = load_competitors()

################## APP ########################
#### SIDEBAR ####
with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; flex-direction: column; justify-content: center; height: 25vh;">
        """,
        unsafe_allow_html=True
    )
    st.header("Filtros")
    st.selectbox("Selecione a vice-presidência:", options=["Todos"] + vp)
    st.selectbox("Selecione o município:", options=["Todos"] + munic)
    st.image(app / "logo_dark.png")
    st.markdown("</div>", unsafe_allow_html=True)

    import psutil
    import os
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    st.sidebar.metric("Memory Usage", f"{memory_mb:.2f} MB")

st.title("Potencial de exportações")

st.markdown(
    "Indicador de potencial de exportações dos produtos catarinenses construído pelo Observatório FIESC.<br>"
    "Metodologia adaptada do indicador EPI (Export Potential Index) do ITC (International Trade Centre).<br>" \
    "O indicador incorpora fatores de oferta, demanda e facilidade de comércio para identificar o potencial de exportações por produto, setor e mercados.",
    unsafe_allow_html=True
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(['Visão geral', 'Produtos e mercados', 'Fornecedores', 'Mapa tarifário', 'Metodologia'])




#### TAB 1 - PRODUTOS ####
with tab1:
    ### FIRST SECTION
    col1, col2 = st.columns([2, 1])
    with col1:
        # Gerar um dicionário de cores para cada categoria de 'sc_comp'
        sc_comp_unique = df_epi_sh6['sc_comp'].unique().to_list()
        color_map = {row['sc_comp']: row['color'] for row in df_epi_sh6.select(['sc_comp', 'color']).unique().to_dicts()}

        fig = px.treemap(
            df_epi_sh6.to_pandas().head(200),
            title="Produtos (SH6):",
            path=["sh6"],
            values="epi_score_normalized",
            color="sc_comp",
            hover_data={
                "product_description_br": True,
                "sh6": True,
                "epi_score_normalized": True,
                "sc_comp": True,
                "categoria": False
            },
            color_discrete_map=color_map
        )

        fig.update_traces(marker=dict(cornerradius=5))

        fig.update_traces(
            hovertemplate="<br>".join([
                "SH6: %{label}",
                "Descrição: %{customdata[0]}",
                "Índice PE: %{customdata[2]}",
                "SC Competitiva: %{customdata[3]}"
            ])
        )

        st.plotly_chart(fig, config={"responsive": True})

    with col2:
        df_sector = df_epi_sc_comp.sort('epi_score_normalized', descending=True).head(10)
        #df_sector = df_sector.reverse()

        fig_sector = px.bar(
            df_sector.to_pandas(),
            title="Setores SC Competitiva:",
            x="epi_score_normalized",
            y="sc_comp",
            orientation="h",
            labels={"sc_comp": "", "epi_score_normalized": "Potencial de exportação"},
            color="sc_comp",
            color_discrete_map=color_map
        )

        fig_sector.update_layout(showlegend=False)

        st.plotly_chart(fig_sector, config={"responsive": True})

    ### SECOND SECTION
    col3, col4 = st.columns([2, 0.675])

    with col3:
        fig_geo = px.scatter_geo(
            df_epi_countries.to_pandas(),
            title="Potencial de mercados:",
            locations="importer",
            locationmode="ISO-3",
            color="categoria",
            hover_name="importer",
            size="epi_score_normalized",
            projection="natural earth",
            color_discrete_sequence=px.colors.qualitative.Plotly,
            size_max=50,
            hover_data={
                "importer_name": True,
                "epi_score_normalized": True,
                "categoria": False,
                "importer": False
            }
        )

        fig_geo.update_geos(
            showcountries=True,
            countrycolor="white",  
            showland=True,
            landcolor="#595959",   
            bgcolor="#0e1117",     
            showcoastlines=True,
            coastlinecolor="white", 
            countrywidth=0.1,      
            coastlinewidth=0.1
        )

        fig_geo.update_traces(
            hovertemplate="<br>".join([
            "País: %{customdata[0]}",
            "Índice PE: %{customdata[1]}"
            ])
        )

        fig_geo.update_layout(
            width=1200,
            height=600,
            legend=dict(
            title="Categoria",
            orientation="v",
            x=-0.02,
            y=1,
            bgcolor='rgba(0,0,0,0)'
            )
        )
        
        st.plotly_chart(fig_geo, config={"responsive": True})

    with col4:
        st.markdown("<div style='margin-top: 110px;'></div>", unsafe_allow_html=True)
        st.dataframe(
            df_epi_countries.select([
                pl.col('importer_name').alias('País'),
                pl.col('epi_score_normalized').alias('Índice PE'),
                pl.col('categoria').alias('Categoria')
            ]).to_pandas().head(50),
            width='stretch',
            hide_index=True
        )
    
    st.markdown("<hr style='margin-top: -50px; margin-bottom: 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='margin-top: -55px;'></div><span style='font-size:14px;'><b>Fonte:</b> CEPII (2023) e Observatório FIESC (2025).</span>", unsafe_allow_html=True)










#### TAB 2 - PRODUTOS E MERCADOS ####
with tab2:
    sh6_options = sorted([opt for opt in df_epi["sh6_product"].unique().to_list() if opt is not None])
    selected_sh6 = st.selectbox("**Selecione o código SH6:**", sh6_options, key="sh6_selectbox_tab2")

    ### Columns for layout
    col1, col2 = st.columns([0.8, 1])
    
    with col1:
        df_selected = df_epi.filter(pl.col("sh6_product") == selected_sh6).sort("epi_score_normalized", descending=True)
        df_selected_pd = df_selected.to_pandas().head(25).sort_values("epi_score_normalized", ascending=True)
        df_selected_pd_map = df_selected.to_pandas().sort_values("epi_score_normalized", ascending=False)

        df_selected_markets = df_markets.filter(pl.col("sh6_product") == selected_sh6).sort("value", descending=True)

        fig = go.Figure()

        # Bar for EPI index (primary x-axis)
        fig.add_trace(
            go.Bar(
            x=df_selected_pd["epi_score_normalized"],
            y=df_selected_pd["importer_name"],
            orientation="h",
            name="Índice PE",
            marker_color=px.colors.qualitative.Plotly[0],
            hovertemplate="País: %{y}<br>Índice PE: %{x}<extra></extra>",
            xaxis="x",
            )
        )

        # Scatter for bilateral exports (secondary x-axis)
        fig.add_trace(
            go.Scatter(
            x=df_selected_pd["bilateral_exports_sc_sh6"],
            y=df_selected_pd["importer_name"],
            mode="markers+lines",
            name="Exportações de SC",
            marker=dict(size=10, color=px.colors.qualitative.Plotly[1], symbol="circle"),
            hovertemplate="País: %{y}<br>Exportações SC: %{x}<extra></extra>",
            xaxis="x2"
            )
        )

        fig.update_layout(
            title="Índice PE e importações dos produtos catarinenses:",
            xaxis=dict(
            title="Índice PE",
            side="bottom",
            showgrid=False
            ),
            xaxis2=dict(
            title="Montante importado de Santa Catarina (US$ FOB)",
            overlaying="x",
            side="top",
            showgrid=False,
            position=0.98
            ),
            legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(0,0,0,0)'
            ),
            height=800,
            margin=dict(l=0, r=0, t=140, b=0)  # Increased top margin to 140 for more spacing below the title
        )

        st.plotly_chart(fig, config={"responsive": True})
    with col2:
        st.markdown("<div style='margin-top: 180px;'></div>", unsafe_allow_html=True)
        total_imports = df_selected_markets['value'].sum()
        st.markdown(f"**Mercado mundial do produto (2023):**<br><span style='font-size:24px; font-weight:bold;'>US$ {format_contabil(total_imports)}</span>", unsafe_allow_html=True)
        
        # Adiciona coluna de posição relativa (ranking)
        df_selected_markets = df_selected_markets.with_columns(
            (pl.arange(1, df_selected_markets.height + 1)).alias("Posição")
        )

        st.dataframe(
            df_selected_markets.select([
            pl.col('Posição'),
            pl.col('importer_name').alias("País"),
            pl.col('value_contabil').alias("Montante US$"),
            pl.col('market_share').alias("Market Share (%)"),
            pl.col('cagr_5y_adj').alias("CAGR 5 anos (%)"),
            pl.col('share_brazil').alias("Share Brasil (%)"),
            pl.col('share_sc').alias("Share SC (%)"),
            pl.col('dist').alias("Distância (km)")
            ]),
            width='stretch',
            hide_index=True
        )

        st.markdown(
            "<div style='margin-top: -15px;'></div>"
            "<span style='font-size:14px;'><b>Nota:</b> CAGR 5 anos (%) refere-se ao crescimento anual composto das importações nos últimos 5 anos.</span>",
            unsafe_allow_html=True
        )
    
    #################### MAPA ####################
    st.markdown("<div style='margin-top: 5px; margin-bottom: 10px;'></div>", unsafe_allow_html=True)
    fig_geo_prod = px.scatter_geo(
        df_selected_pd_map,
        locations="importer",
        locationmode="ISO-3",
        hover_name="importer",
        color='categoria',
        size="epi_score_normalized",
        projection="natural earth",
        color_discrete_sequence=px.colors.qualitative.Plotly,
        size_max=65,
        hover_data={
            "importer_name": True,
            "epi_score_normalized": True,
            "importer": False
        }
    )

    fig_geo_prod.update_geos(
        showcountries=True,
        countrycolor="white",  
        showland=True,
        landcolor="#595959",   
        bgcolor="#0e1117",     
        showcoastlines=True,
        coastlinecolor="white", 
        countrywidth=0.1,      
        coastlinewidth=0.1
    )

    fig_geo_prod.update_traces(
        hovertemplate="<br>".join([
        "País: %{customdata[0]}",
        "Índice PE: %{customdata[1]}"
        ])
    )

    fig_geo_prod.update_layout(
        width=1200,
        height=600,
        title=f"Distribuição geográfica do Índice PE:",
        legend=dict(
        title=None,
        orientation="h",
        x=0.33,
        y=0,
        bgcolor='rgba(0,0,0,0)'
        ),
        margin=dict(t=40)  # Reduce top margin
    )

    st.plotly_chart(fig_geo_prod, config={"responsive": True})

    st.markdown("<hr style='margin-top: -50px; margin-bottom: 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='margin-top: -55px;'></div><span style='font-size:14px;'><b>Fonte:</b> CEPII (2023) e Observatório FIESC (2025).</span>", unsafe_allow_html=True)







    
#### TAB 3 - FORNECEDORES ####
with tab3:
    # --- Cache unique values ---
    @st.cache_data(show_spinner=False)
    def get_unique_options(df: pl.DataFrame):
        """
        Return sorted unique lists for importer_name and sh6_product.
        This will only recompute when df_competitors changes.
        """
        countries = (
            df.select(pl.col("importer_name").drop_nulls().unique().sort())
            .to_series().to_list()
        )
        products = (
            df.select(pl.col("sh6_product").drop_nulls().unique().sort())
            .to_series().to_list()
        )
        return countries, products
    

    countries, products = get_unique_options(df_competitors)
    col1, col2 = st.columns([0.8, 1])

    with col1:
        sel_country = st.selectbox(
            "*Selecione o país:*",
            options=countries,
            key="country_selectbox_tab3"
        )

    with col2:
        sel_product = st.selectbox(
            "*Selecione o produto (SH6):*",
            options=products,
            key="product_selectbox_tab3"
        )

    df_competitors_filtered = (
        df_competitors
        .filter(
            (pl.col("importer_name") == sel_country) &
            (pl.col("sh6_product") == sel_product)
        )
        .sort("value", descending=True)
    )

    total_imports = df_competitors_filtered.select(pl.col("value").sum()).item()

    # ==== FIRST SECTION (fast + categorical-proof via graph_objects) ====
    import plotly.graph_objects as go
    from itertools import cycle
    import polars as pl

    col3, col4 = st.columns([2, 1.25])

    with col3:
        df_treemap_pl = (
            df_competitors_filtered
            .select([
                "exporter_name", "sh6", "value",
                "product_description_br", "value_contabil"
            ])
            .filter(pl.col("exporter_name").is_not_null() & pl.col("sh6").is_not_null())
            .head(200)
        )

        st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)


        if df_treemap_pl.height == 0:
            st.info("Sem dados para este país/produto.")
        else:
            # Optional: join your preferred colors per exporter (fallback palette if missing)
            # df_epi_sh6 must have ['exporter_name','color'] if you want custom colors
            has_colors = "df_epi_sh6" in globals() and isinstance(df_epi_sh6, pl.DataFrame) \
                        and set(["exporter_name","color"]).issubset(df_epi_sh6.columns)

            if has_colors:
                df_plot = (
                    df_treemap_pl.join(
                        df_epi_sh6.select(["exporter_name","color"]).unique(),
                        on="exporter_name",
                        how="left"
                    )
                )
            else:
                df_plot = df_treemap_pl

            # Convert columns to Python lists (no pandas/categoricals involved)
            exporters  = df_plot.get_column("exporter_name").to_list()
            sh6_arr    = df_plot.get_column("sh6").to_list()
            values     = df_plot.get_column("value").to_list()
            descr      = df_plot.get_column("product_description_br").to_list()
            v_text     = df_plot.get_column("value_contabil").to_list()

            # Colors: use provided colors if present; otherwise generate a palette
            provided_colors = df_plot.get_column("color").to_list() if has_colors else [None] * len(exporters)
            if any(c is not None for c in provided_colors):
                node_colors = [c if c is not None else "#8FA5FF" for c in provided_colors]
            else:
                # deterministic palette by exporter name
                palette = (
                    ['#23CCA1', '#E24B5E', '#EAD97F', '#4FD1C5', '#8FA5FF',
                    '#B388EB', '#FFA07A', '#7FB77E', '#F6C85F', '#9FD3C7']
                )
                cyc = cycle(palette)
                # unique exporters -> color
                uniq = {}
                node_colors = []
                for name in exporters:
                    if name not in uniq:
                        uniq[name] = next(cyc)
                    node_colors.append(uniq[name])

            # Customdata for hover: [descr, sh6, value_contabil]
            import numpy as np
            customdata = np.column_stack([descr, sh6_arr, v_text]) if exporters else np.empty((0,3))

            fig = go.Figure(
                go.Treemap(
                    labels=exporters,
                    parents=[""] * len(exporters),  # remove parent label from nodes
                    values=values,
                    branchvalues="total",
                    marker=dict(
                        colors=node_colors,
                        line=dict(width=0.5, color="rgba(255,255,255,0.15)"),
                    ),
                    tiling=dict(pad=2),
                    textinfo="label+value",
                    texttemplate="%{label}<br>%{value:.2s}",
                    hovertemplate="<br>".join([
                        "Exportador: %{label}",
                        "SH6: %{customdata[1]}",
                        "Descrição: %{customdata[0]}",
                        "Valor importado: US$ %{customdata[2]}",
                        "<extra></extra>"
                    ]),
                    customdata=customdata,
                )
            )

            fig.update_traces(marker=dict(cornerradius=5))

            fig.update_layout(
                title="Países fornecedores (2023):",
                margin=dict(l=0, r=0, t=40, b=0),
            )

            st.plotly_chart(fig, config={"responsive": True})
    
    with col4:
        st.markdown("<div style='margin-top: 0px;'></div>", unsafe_allow_html=True)
        st.markdown(f"**Total importado (2023):**<br><span style='font-size:24px; font-weight:bold;'>US$ {format_contabil(total_imports)}</span>", unsafe_allow_html=True)
        
        # Adiciona coluna de posição relativa (ranking)
        df_competitors_filtered = df_competitors_filtered.with_columns(
            (pl.arange(1, df_competitors_filtered.height + 1)).alias("Posição")
        )

        st.dataframe(
            df_competitors_filtered.select([
            pl.col('Posição'),
            pl.col('exporter_name').alias('País fornecedor'),
            pl.col('value_contabil').alias('Montante US$'),
            pl.col('importer_sh6_share').alias('Share (%)'),
            pl.col('cagr_5y_adj').alias('CAGR 5 anos (%)')
            ]).to_pandas().head(50),
            width='stretch',
            hide_index=True
        )
        st.markdown(
            "<div style='margin-top: -15px;'></div>"
            "<span style='font-size:14px;'><b>Nota:</b> CAGR 5 anos (%) refere-se ao crescimento anual composto das importações nos últimos 5 anos.</span>",
            unsafe_allow_html=True
        )

    st.markdown("<div style='margin-top: 5px; margin-bottom: 10px;'></div>", unsafe_allow_html=True)
    
    # Mapa de distribuição das importações por país para os filtros feitos
    # Use config argument instead of deprecated keyword arguments in st.plotly_chart
    fig_geo_imports = px.scatter_geo(
        df_competitors_filtered.to_pandas(),
        locations="exporter",
        locationmode="ISO-3",
        hover_name="exporter_name",
        size="value",
        projection="natural earth",
        color_discrete_sequence=px.colors.qualitative.Plotly,
        size_max=50,
        hover_data={
            "exporter_name": True,
            "value_contabil": True,
            "exporter": False
        }
    )

    fig_geo_imports.update_geos(
        showcountries=True,
        countrycolor="white",
        showland=True,
        landcolor="#595959",
        bgcolor="#0e1117",
        showcoastlines=True,
        coastlinecolor="white",
        countrywidth=0.1,
        coastlinewidth=0.1
    )

    fig_geo_imports.update_traces(
        hovertemplate="<br>".join([
            "País fornecedor: %{customdata[0]}",
            "Valor importado: US$ %{customdata[1]}"
        ])
    )

    fig_geo_imports.update_layout(
        width=1200,
        height=600,
        showlegend=False
    )

    st.plotly_chart(fig_geo_imports, config={"responsive": True})

    st.markdown("<hr style='margin-top: -50px; margin-bottom: 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='margin-top: -55px;'></div><span style='font-size:14px;'><b>Fonte:</b> CEPII (2023) e Observatório FIESC (2025).</span>", unsafe_allow_html=True)

    with tab4:
        # ===========================
        # Configurações
        # ===========================
        # Use Path para garantir caminho absoluto relativo ao app.py
        ARQ = str((app / "data" / "all_partners_tariffs_filtrado.parquet").resolve())

        # Colunas de valor/ano/produto
        VAL_COL             = "Tariff_Final"
        YEAR_COL            = "Tariff_Year"
        PRODUCT_COL         = "Product Name"

        # Colunas ISO-3 (códigos)
        REPORTER_ISO3_COL   = "Reporter"        # país que APLICA a tarifa (origem)
        PARTNER_ISO3_COL    = "Partner"         # país de DESTINO

        # Colunas de NOME (texto)
        REPORTER_NAME_COL   = "Reporter Name"   # nome do país que aplica a tarifa
        PARTNER_NAME_COL    = "Partner Name"    # nome do país de destino (filtro principal)

        # ===========================
        # Utilidades
        # ===========================
        def validar_colunas(df: pl.DataFrame, cols: List[str]) -> None:
            faltantes = [c for c in cols if c not in df.columns]
            if faltantes:
                raise KeyError(f"As colunas obrigatórias estão faltando no arquivo: {faltantes}")

        def _coerce_year(df: pl.DataFrame, year_col: str) -> pl.DataFrame:
            # Garante que o ano seja numérico para ordenar corretamente (ex.: strings "2019")
            if df.schema.get(year_col) != pl.Int64:
                df = df.with_columns(pl.col(year_col).cast(pl.Int64, strict=False))
            return df

        @st.cache_data(show_spinner=True)
        def carregar_dados(arq: str) -> pl.DataFrame:
            """
            Lê o parquet, remove tarifas nulas e mantém o registro do ano mais recente
            para cada (ReporterISO, PartnerISO, Product). Isso garante, por exemplo,
            que entre 2019 e 2023 fique apenas 2023.
            """
            df = pl.read_parquet(arq)

            colunas_necessarias = [
                REPORTER_ISO3_COL, REPORTER_NAME_COL,
                PARTNER_ISO3_COL,  PARTNER_NAME_COL,
                PRODUCT_COL, YEAR_COL, VAL_COL
            ]
            validar_colunas(df, colunas_necessarias)

            # Remove nulos no valor e garante ano numérico
            df = df.drop_nulls(subset=[VAL_COL])
            df = _coerce_year(df, YEAR_COL)

            # Mantém o ano mais recente por (ReporterISO, PartnerISO, Produto)
            # Estratégia: ordenar por ano DESC e pegar a primeira ocorrência
            df = (
                df.sort(
                    by=[REPORTER_ISO3_COL, PARTNER_ISO3_COL, PRODUCT_COL, YEAR_COL],
                    descending=[False, False, False, True]
                )
                .unique(subset=[REPORTER_ISO3_COL, PARTNER_ISO3_COL, PRODUCT_COL], keep="first")
                .select(colunas_necessarias)
            )

            if df.is_empty():
                raise ValueError("Após a seleção do ano mais recente, não há dados.")

            return df

        def _is_iso3(s: str) -> bool:
            return bool(re.fullmatch(r"[A-Z]{3}", (s or "")))

        def make_figure_and_data(
            df_filtrado: pl.DataFrame,
            pais_destino_nome: str,
            coluna_nome_reporter: str,   # ex.: "Reporter Name"
            coluna_iso_reporter: str     # ex.: "Reporter"
        ) -> Tuple[go.Figure, pl.DataFrame, pl.DataFrame]:

            df_prod = df_filtrado.drop_nulls(subset=[VAL_COL])

            # Países que aplicam tarifa > 0 (bolhas)
            df_bolhas = df_prod.filter(pl.col(VAL_COL) > 0)

            # Países com tarifa == 0 (oportunidades)
            df_zeros  = df_prod.filter(pl.col(VAL_COL) == 0)

            # Converte para pandas para o Plotly
            pdf_bolhas = df_bolhas.to_pandas()
            pdf_zeros  = df_zeros.to_pandas()

            # ---------- DETECÇÃO AUTOMÁTICA DO LOCATIONMODE ----------
            if coluna_iso_reporter in pdf_bolhas.columns:
                pdf_bolhas[coluna_iso_reporter] = (
                    pdf_bolhas[coluna_iso_reporter].astype(str).str.strip().str.upper()
                )
                iso_ratio = pdf_bolhas[coluna_iso_reporter].apply(_is_iso3).mean()
            else:
                iso_ratio = 0.0

            if iso_ratio >= 0.8:
                # Usar ISO-3
                locations_col = coluna_iso_reporter
                locationmode  = "ISO-3"
            else:
                # Usar NOME de país
                locations_col = coluna_nome_reporter
                locationmode  = "country names"

            # ---------- EVITAR BOLHAS INVISÍVEIS ----------
            # Se o valor é muito pequeno (ou todos muito próximos), adiciona um offset mínimo
            size_series = pd.to_numeric(pdf_bolhas[VAL_COL], errors="coerce").fillna(0.0)
            if len(size_series) > 0:
                min_pos = size_series[size_series > 0].min() if (size_series > 0).any() else 0.0
                if min_pos <= 0.01:
                    size_series = size_series + 0.05  # empurrão pra dar área mínima

            # ------------------ MAPA ------------------
            fig = px.scatter_geo(
                pdf_bolhas,
                locations=locations_col,
                locationmode=locationmode,
                color=VAL_COL,
                size=size_series,
                hover_name=coluna_nome_reporter,
                custom_data=[YEAR_COL, VAL_COL, coluna_nome_reporter],
                color_continuous_scale="Blues",
                projection="natural earth",
                size_max=50,
                labels={VAL_COL: "Tarifa (%)"},
            )

            fig.update_traces(
                hovertemplate=(
                    "<b>%{customdata[2]}</b><br>"
                    "Ano: %{customdata[0]:.0f}<br>"
                    "Tarifa: %{customdata[1]:.2f}%<extra></extra>"
                ),
                marker=dict(line=dict(width=0.5, color='rgba(0,0,0,0.7)')),
                selector=dict(type='scattergeo')
            )

            # ------------------ TOP 5 (usa o MESMO mode/coluna) ------------------
            pdf_top5 = pdf_bolhas.sort_values(VAL_COL, ascending=False).head(5)
            if not pdf_top5.empty:
                fig.add_trace(go.Scattergeo(
                    locations=pdf_top5[locations_col],
                    locationmode=locationmode,
                    text=[f"<b>{i+1}º</b>" for i in range(len(pdf_top5))],
                    mode="text",
                    textposition="top center",
                    textfont=dict(size=16, color="white"),
                    showlegend=False,
                ))

            # ------------------ TÍTULO E ESTILO ------------------
            produto_escolhido = df_filtrado[PRODUCT_COL][0]

            fig.update_layout(
                paper_bgcolor="#22232E",
                plot_bgcolor="#22232E",
                margin=dict(l=0, r=0, t=30, b=0),
                font=dict(color="white"),
                coloraxis_showscale=False  # Remove color legend bar
            )

            fig.update_geos(
                showcountries=True,
                countrycolor="white",
                showland=True,
                landcolor="#595959",
                bgcolor="#0e1117",  # cor do fundo do mapa e fora dele
                showcoastlines=True,
                coastlinecolor="white",
                countrywidth=0.1,
                coastlinewidth=0.1
            )
            fig.update_layout(
                paper_bgcolor="#0e1117",  # cor do fundo fora do mapa
                plot_bgcolor="#0e1117"
            )

            return fig, df_bolhas, df_zeros

        # ===========================
        # UI
        # ===========================
        try:
            df_base = carregar_dados(ARQ)

            # --- Estado inicial ---
            # Inicializa apenas se não existir no session_state
            if "pais_escolhido_nome" not in st.session_state:
                st.session_state.pais_escolhido_nome = "Brazil"  # pelo NOME
            if "produto_escolhido" not in st.session_state:
                st.session_state.produto_escolhido = None

            # --- Filtros ---
            col_filtro1, col_filtro2 = st.columns([1.5, 2.3])

            # País de destino (Partner) recebe tarifas de outros (Reporter)
            with col_filtro1:
                paises_destino_nomes = sorted(df_base[PARTNER_NAME_COL].drop_nulls().unique().to_list())

                if st.session_state.pais_escolhido_nome not in paises_destino_nomes:
                    st.session_state.pais_escolhido_nome = (
                        "Brazil" if "Brazil" in paises_destino_nomes else paises_destino_nomes[0]
                    )

                idx_pais = paises_destino_nomes.index(st.session_state.pais_escolhido_nome)
                st.selectbox("**Selecione o país:**", paises_destino_nomes, index=idx_pais, key="pais_escolhido_nome")

            # Filtra por NOME do partner
            df_filtrado_pais = df_base.filter(pl.col(PARTNER_NAME_COL) == st.session_state.pais_escolhido_nome)

            with col_filtro2:
                produtos = sorted(df_filtrado_pais[PRODUCT_COL].drop_nulls().unique().to_list())
                if not produtos:
                    st.warning(f"Não há produtos com dados de tarifa para '{st.session_state.pais_escolhido_nome}'.")
                    st.stop()

                if (st.session_state.produto_escolhido is None) or (st.session_state.produto_escolhido not in produtos):
                    st.session_state.produto_escolhido = produtos[0]

                idx_prod = produtos.index(st.session_state.produto_escolhido)
                st.selectbox("**Selecione o produto SH6:**", options=produtos, index=idx_prod, key="produto_escolhido")

            # Filtra produto e evita Reporter == Partner (por NOME)
            df_final = (
                df_filtrado_pais
                .filter(pl.col(PRODUCT_COL) == st.session_state.produto_escolhido)
                .filter(pl.col(REPORTER_NAME_COL) != st.session_state.pais_escolhido_nome)
            )

            if df_final.is_empty():
                st.warning(f"Não foram encontrados dados para '{st.session_state.produto_escolhido}' em '{st.session_state.pais_escolhido_nome}'.")
                st.stop()

            # --- Gráfico e tabelas ---
            fig, df_com_tarifa, df_sem_tarifa = make_figure_and_data(
                df_final,
                st.session_state.pais_escolhido_nome,
                REPORTER_NAME_COL,
                REPORTER_ISO3_COL
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            # Estilo das tabelas
            st.markdown("""
            <style>
                .stDataFrame, .stDataFrame [data-testid="stTable"] { background-color: #22232E !important; }
                .stDataFrame [data-testid="stTable"] .col_heading { color: white !important; background-color: #3a3c4d !important; }
                .stDataFrame [data-testid="stTable"] .cell-container { color: white !important; background-color: #22232E !important; }
            </style>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"<h5 style='text-align: center;'>Países que tarifam o país selecionado: </h5>", unsafe_allow_html=True)
                df_tabela_com_tarifa = (
                    df_com_tarifa
                    .select([REPORTER_NAME_COL, YEAR_COL, VAL_COL])
                    .rename({REPORTER_NAME_COL: "País", YEAR_COL: "Ano", VAL_COL: "Tarifa (%)"})
                    .sort(by="Tarifa (%)", descending=True)
                )
                st.data_editor(df_tabela_com_tarifa.to_pandas(), use_container_width=True, hide_index=True, disabled=True)

            with col2:
                st.markdown(f"<h5 style='text-align: center;'>Países que não tarifam o país selecionado:</h5>", unsafe_allow_html=True)
                df_tabela_sem_tarifa = (
                    df_sem_tarifa
                    .select([REPORTER_NAME_COL, YEAR_COL, VAL_COL])
                    .rename({REPORTER_NAME_COL: "País", YEAR_COL: "Ano", VAL_COL: "Tarifa (%)"})
                    .sort(by="País")
                )
                st.data_editor(df_tabela_sem_tarifa.to_pandas(), use_container_width=True, hide_index=True, disabled=True)

        except FileNotFoundError:
            st.error(f"Arquivo não encontrado: {ARQ}")
        except KeyError as e:
            st.error(f"Problema de colunas no dataset: {e}")
        # except ValueError as e:
        #     st.error(str(e))
        # except Exception as e:
        #     st.exception(e)
        
        st.markdown("<hr style='margin-top: 0px; margin-bottom: 0;'>", unsafe_allow_html=True)
        st.markdown("<div style='margin-top: 0px;'></div><span style='font-size:14px;'><b>Fonte:</b> WTIS (2023) e Observatório FIESC (2025).</span>", unsafe_allow_html=True)


