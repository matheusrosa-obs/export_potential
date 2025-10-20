"""
Reprodução das visualizações do app Streamlit em um script para uso no Jupyter.

Como usar no Jupyter:
- %run ./app/apresent.py  -> renderiza todos os gráficos com seleções padrão
- Ou importe funções/figuras e edite as variáveis de seleção no topo deste arquivo.

Este script evita dependências do Streamlit e reproduz os gráficos com máxima fidelidade.
"""

from pathlib import Path
from itertools import cycle
import numpy as np
import polars as pl
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio


# -----------------------------------------------------------------------------
# Configuração Plotly para Jupyter
# -----------------------------------------------------------------------------
try:
    # Em Jupyter, este renderer costuma funcionar bem (inclui recursos online/offline)
    pio.renderers.default = "notebook_connected"
except Exception:
    # Fallback genérico
    pio.renderers.default = "browser"


# -----------------------------------------------------------------------------
# Paths e carregamento de dados (espelha o app/app.py)
# -----------------------------------------------------------------------------
def get_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / 'app' / 'app.py').exists():
            return parent
    return Path.cwd()


PROJECT_ROOT = get_project_root()
APP_DIR = PROJECT_ROOT / 'app'
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
DATA_INTERIM = PROJECT_ROOT / 'data' / 'interim'
REFERENCES = PROJECT_ROOT / 'references'


def format_contabil(value: float) -> str:
    if value is None:
        return ""
    try:
        if value >= 1e9:
            return f"{value/1e9:,.1f} bi".replace(",", "X").replace(".", ",").replace("X", ".")
        elif value >= 1e6:
            return f"{value/1e6:,.1f} mi".replace(",", "X").replace(".", ",").replace("X", ".")
        elif value >= 1e3:
            return f"{value/1e3:,.1f} mil".replace(",", "X").replace(".", ",").replace("X", ".")
        else:
            return f"{value:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return str(value)


def format_decimal(value, decimals: int = 1) -> str:
    try:
        return f"{float(value):.{decimals}f}".replace(".", ",")
    except Exception:
        return ""


def load_data():
    # EPI scores SH6
    df_epi_sh6 = pl.read_parquet(APP_DIR / 'data' / 'epi_scores_sh6.parquet')
    df_epi_sh6 = df_epi_sh6.with_columns(pl.col("epi_score_normalized").round(3))

    # EPI scores countries
    df_epi_countries = pl.read_parquet(APP_DIR / 'data' / 'epi_scores_countries.parquet')
    df_epi_countries = df_epi_countries.with_columns(pl.col("epi_score_normalized").round(3))

    # EPI scores detalhado
    df_epi = pl.read_parquet(APP_DIR / 'data' / 'epi_scores.parquet')
    df_epi = df_epi.with_columns(pl.col("epi_score_normalized").round(3))

    # EPI por setores SC Competitiva
    df_epi_sc_comp = pl.read_parquet(APP_DIR / 'data' / 'epi_scores_sc_comp.parquet')
    df_epi_sc_comp = df_epi_sc_comp.with_columns(pl.col("epi_score_normalized").round(3))

    # Mercados mundiais (para tabela e info)
    df_markets = pl.read_parquet(APP_DIR / 'data' / 'app_dataset.parquet')
    df_markets = df_markets.select([
        pl.col('importer'),
        pl.col('country_name').alias('importer_name'),
        pl.col('sh6'),
        pl.col('description').alias('product_description_br'),
        pl.col('value'),
        pl.col('market_share'),
        pl.col('cagr_5y'),
        pl.col('share_brazil'),
        pl.col('share_sc'),
        pl.col('dist')
    ])
    df_markets = df_markets.with_columns(
        (pl.col('sh6') + " - " + pl.col('product_description_br')).alias('sh6_product')
    )
    df_markets = df_markets.with_columns(
        pl.col("value").map_elements(format_contabil).alias("value_contabil"),
        pl.col("dist").map_elements(format_contabil).alias("dist"),
    )

    # Competidores (fornecedores)
    df_competitors = pl.read_parquet(APP_DIR / 'data' / 'df_competitors.parquet')

    # Ajustes decimais em texto como no app
    df_markets = df_markets.with_columns(
        pl.col("cagr_5y").map_elements(lambda x: format_decimal(x, 1)).alias("cagr_5y_adj"),
        pl.col('market_share').map_elements(lambda x: format_decimal(x, 1)).alias('market_share'),
        pl.col('share_sc').map_elements(lambda x: format_decimal(x, 1)).alias('share_sc'),
        pl.col('share_brazil').map_elements(lambda x: format_decimal(x, 1)).alias('share_brazil'),
    )

    return df_epi_sh6, df_epi_countries, df_epi, df_epi_sc_comp, df_markets, df_competitors


# -----------------------------------------------------------------------------
# Seleções padrão para reprodução dos gráficos interativos do app
# Edite estas variáveis conforme necessidade no Jupyter.
# -----------------------------------------------------------------------------
SELECTED_SH6: str | None = "940360 - Outros móveis de madeira"  # Ex.: "010121 - Cavalos reprodutores de raça pura"
SUP_COUNTRY: str | None = "940360 - Outros móveis de madeira"   # Nome do país fornecedor
SUP_SH6: str | None = None       # Produto (sh6_product)

# -----------------------------------------------------------------------------
# Construção de gráficos (espelha o app)
# -----------------------------------------------------------------------------
def build_products_treemap(df_epi_sh6: pl.DataFrame) -> go.Figure:
    color_map = {row['sc_comp']: row['color'] for row in df_epi_sh6.select(['sc_comp', 'color']).unique().to_dicts()}
    fig = px.treemap(
        df_epi_sh6.to_pandas().head(200),
        path=["sh6"],
        values="epi_score_normalized",
        color="sc_comp",
        hover_data={
            "product_description_br": True,
            "sh6": True,
            "epi_score_normalized": True,
            "sc_comp": True,
            "categoria": False,
        },
        color_discrete_map=color_map,
    )
    fig.update_traces(marker=dict(cornerradius=5))
    fig.update_traces(
        hovertemplate="<br>".join([
            "SH6: %{label}",
            "Descrição: %{customdata[0]}",
            "Índice PE: %{customdata[2]}",
            "SC Competitiva: %{customdata[3]}",
        ]),
        textfont=dict(size=22),  # aumenta o tamanho da fonte
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        width=1600,
        height=800
    )
    return fig


def build_sectors_bar(df_epi_sc_comp: pl.DataFrame, df_epi_sh6: pl.DataFrame) -> go.Figure:
    color_map = {row['sc_comp']: row['color'] for row in df_epi_sh6.select(['sc_comp', 'color']).unique().to_dicts()}
    df_sector = df_epi_sc_comp.sort('epi_score_normalized', descending=True).head(10)
    fig = px.bar(
        df_sector.to_pandas(),
        x="epi_score_normalized",
        y="sc_comp",
        orientation="h",
        labels={"sc_comp": "", "epi_score_normalized": "Potencial de exportação"},
        color="sc_comp",
        color_discrete_map=color_map,
    )
    fig.update_layout(
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        width=800,
        height=500,
        font=dict(color='white', size=22)  # aumenta a resolução do texto
    )
    fig.update_yaxes(tickfont=dict(color='white', size=22))
    fig.update_xaxes(title_font=dict(color='white', size=22), tickfont=dict(color='white', size=22), showgrid=False)  # remove linhas de grade verticais
    return fig


def build_markets_geo(df_epi_countries: pl.DataFrame) -> go.Figure:
    fig = px.scatter_geo(
        df_epi_countries.to_pandas(),
        locations="importer",
        locationmode="ISO-3",
        color="categoria",
        hover_name="importer",
        size="epi_score_normalized",
        projection="natural earth",
        color_discrete_sequence=px.colors.qualitative.Plotly,
        size_max=100,
        hover_data={
            "importer_name": True,
            "epi_score_normalized": True,
            "categoria": False,
            "importer": False,
        },
    )
    fig.update_geos(
        showcountries=True,
        countrycolor="white",
        showland=True,
        landcolor="#595959",
        bgcolor="#1E1E29",
        showcoastlines=True,
        coastlinecolor="white",
        countrywidth=0.1,
        coastlinewidth=0.1,
        showlakes=True,
        lakecolor="#0e1117",  # cor igual ao oceano
        fitbounds="locations",  # <-- remove borda branca
        visible=False           # <-- remove borda extra
    )
    fig.update_traces(hovertemplate="<br>".join(["País: %{customdata[0]}", "Índice PE: %{customdata[1]}"]))
    fig.update_layout(
        width=1800,
        height=900,
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),  # <-- remove margens
        paper_bgcolor='rgba(0,0,0,0)',    # <-- fundo transparente
        plot_bgcolor='rgba(0,0,0,0)'      # <-- fundo transparente
    )
    return fig

def build_epi_bars_and_scatter(df_epi: pl.DataFrame, selected_sh6: str) -> go.Figure:
    df_selected = df_epi.filter(pl.col("sh6_product") == selected_sh6).sort("epi_score_normalized", descending=True)
    df_selected_pd = df_selected.to_pandas().head(25).sort_values("epi_score_normalized", ascending=True)

    fig = go.Figure()
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
    fig.add_trace(
    go.Scatter(
        x=df_selected_pd["bilateral_exports_sc_sh6"],
        y=df_selected_pd["importer_name"],
        mode="markers+lines",
        name="Exportações de SC",
        marker=dict(size=12, color=px.colors.qualitative.Plotly[1], symbol="circle"),
        hovertemplate="País: %{y}<br>Exportações SC: %{x}<extra></extra>",
        xaxis="x2",
    )
    )
    # Alinhar o zero do eixo superior ao zero do eixo inferior
    x1_min, x1_max = 0, max(df_selected_pd["epi_score_normalized"].max(), 1)
    x2_min, x2_max = 0, max(df_selected_pd["bilateral_exports_sc_sh6"].max(), 1)

    fig.update_layout(
    xaxis=dict(
        title="Índice PE",
        side="bottom",
        showgrid=False,
        title_font=dict(color='white', size=22),
        tickfont=dict(color='white', size=20),
        range=[x1_min, x1_max],
        anchor="y"
    ),
    xaxis2=dict(
        title="Montante importado de Santa Catarina (US$ FOB)",
        overlaying="x",
        side="top",
        showgrid=False,
        position=0.98,
        title_font=dict(color='white', size=22),
        tickfont=dict(color='white', size=20),
        range=[x2_min, x2_max],
        anchor="y"
    ),
    yaxis=dict(
        tickfont=dict(color='white', size=20),
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=0,
        xanchor="center",
        x=0.5,
        bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=20)
    ),
    height=800,
    margin=dict(l=0, r=0, t=140, b=0),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white', size=20)
    )
    return fig

def build_product_geo(df_epi: pl.DataFrame, selected_sh6: str) -> go.Figure:
    df_selected_pd_map = (
        df_epi.filter(pl.col("sh6_product") == selected_sh6)
        .to_pandas()
        .sort_values("epi_score_normalized", ascending=False)
    )
    fig = px.scatter_geo(
        df_selected_pd_map,
        locations="importer",
        locationmode="ISO-3",
        color="categoria",
        hover_name="importer",
        size="epi_score_normalized",
        projection="natural earth",
        color_discrete_sequence=px.colors.qualitative.Plotly,
        size_max=100,
        hover_data={
            "importer_name": True,
            "epi_score_normalized": True,
            "categoria": False,
            "importer": False,
        },
    )
    fig.update_geos(
        showcountries=True,
        countrycolor="white",
        showland=True,
        landcolor="#595959",
        bgcolor="#1E1E29",
        showcoastlines=True,
        coastlinecolor="white",
        countrywidth=0.1,
        coastlinewidth=0.1,
        showlakes=True,
        lakecolor="#0e1117",
        fitbounds="locations",
        visible=False
    )
    fig.update_traces(hovertemplate="<br>".join(["País: %{customdata[0]}", "Índice PE: %{customdata[1]}"]))
    fig.update_layout(
        width=1800,
        height=900,
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def build_suppliers_treemap(df_competitors: pl.DataFrame, df_epi_sh6: pl.DataFrame, sel_country: str, sel_product: str) -> go.Figure:
    df_competitors_filtered = (
        df_competitors
        .filter((pl.col("importer_name") == sel_country) & (pl.col("sh6_product") == sel_product))
        .sort("value", descending=True)
    )
    df_treemap_pl = (
        df_competitors_filtered
        .select(["exporter_name", "sh6", "value", "product_description_br", "value_contabil"])
        .filter(pl.col("exporter_name").is_not_null() & pl.col("sh6").is_not_null())
        .head(200)
    )
    has_colors = isinstance(df_epi_sh6, pl.DataFrame) and set(["exporter_name", "color"]).issubset(df_epi_sh6.columns)
    if has_colors:
        df_plot = df_treemap_pl.join(
            df_epi_sh6.select(["exporter_name", "color"]).unique(),
            on="exporter_name",
            how="left",
        )
    else:
        df_plot = df_treemap_pl

    exporters = df_plot.get_column("exporter_name").to_list()
    sh6_arr = df_plot.get_column("sh6").to_list()
    values = df_plot.get_column("value").to_list()
    descr = df_plot.get_column("product_description_br").to_list()
    v_text = df_plot.get_column("value_contabil").to_list()
    provided_colors = df_plot.get_column("color").to_list() if has_colors else [None] * len(exporters)
    if any(c is not None for c in provided_colors):
        node_colors = [c if c is not None else "#8FA5FF" for c in provided_colors]
    else:
        palette = ['#23CCA1', '#E24B5E', '#EAD97F', '#4FD1C5', '#8FA5FF', '#B388EB', '#FFA07A', '#7FB77E', '#F6C85F', '#9FD3C7']
        cyc = cycle(palette)
        uniq = {}
        node_colors = []
        for name in exporters:
            if name not in uniq:
                uniq[name] = next(cyc)
            node_colors.append(uniq[name])

    customdata = np.column_stack([descr, sh6_arr, v_text]) if exporters else np.empty((0, 3))
    fig = go.Figure(
        go.Treemap(
            labels=exporters,
            parents=[""] * len(exporters),
            values=values,
            branchvalues="total",
            marker=dict(colors=node_colors, line=dict(width=0.5, color="rgba(255,255,255,0.15)")),
            tiling=dict(pad=2),
            textinfo="label+value",
            texttemplate="%{label}<br>%{value:.2s}",
            hovertemplate="<br>".join([
                "Exportador: %{label}",
                "SH6: %{customdata[1]}",
                "Descrição: %{customdata[0]}",
                "Valor importado: US$ %{customdata[2]}",
                "<extra></extra>",
            ]),
            customdata=customdata,
        )
    )
    fig.update_layout(title="Países fornecedores (2023):", margin=dict(l=0, r=0, t=40, b=0))
    return fig


def build_imports_geo(df_competitors: pl.DataFrame, sel_country: str, sel_product: str) -> go.Figure:
    df_competitors_filtered = (
        df_competitors
        .filter((pl.col("importer_name") == sel_country) & (pl.col("sh6_product") == sel_product))
        .sort("value", descending=True)
    )
    fig = px.scatter_geo(
        df_competitors_filtered.to_pandas(),
        locations="exporter",
        locationmode="ISO-3",
        hover_name="exporter_name",
        size="value",
        projection="natural earth",
        color_discrete_sequence=px.colors.qualitative.Plotly,
        size_max=50,
        hover_data={"exporter_name": True, "value_contabil": True, "exporter": False},
    )
    fig.update_geos(
        showcountries=True,
        countrycolor="white",
        showland=True,
        landcolor="#595959",
        bgcolor="#0e1117",
        showcoastlines=True,
        coastlinecolor="white",
        countrywidth=0.1,
        coastlinewidth=0.1,
    )
    fig.update_traces(hovertemplate="<br>".join(["País fornecedor: %{customdata[0]}", "Valor importado: US$ %{customdata[1]}"]))
    fig.update_layout(width=1200, height=600, showlegend=False)
    return fig


# -----------------------------------------------------------------------------
# Execução direta: monta e exibe as figuras com seleções padrão
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    df_epi_sh6, df_epi_countries, df_epi, df_epi_sc_comp, df_markets, df_competitors = load_data()

    # Definições padrão inteligentes
    if SELECTED_SH6 is None:
        # Primeiro item por ordenação
        try:
            SELECTED_SH6 = df_epi.select(pl.col("sh6_product")).drop_nulls().unique().to_series()[0]
        except Exception:
            SELECTED_SH6 = None

    if SUP_COUNTRY is None:
        try:
            SUP_COUNTRY = df_competitors.select(pl.col("importer_name")).drop_nulls().unique().to_series()[0]
        except Exception:
            SUP_COUNTRY = None

    if SUP_SH6 is None:
        try:
            SUP_SH6 = df_competitors.select(pl.col("sh6_product")).drop_nulls().unique().to_series()[0]
        except Exception:
            SUP_SH6 = None

    # 1) Produtos Treemap
    fig_treemap_prod = build_products_treemap(df_epi_sh6)
    fig_treemap_prod.show()

    # 2) Setores Bar
    fig_sectors = build_sectors_bar(df_epi_sc_comp, df_epi_sh6)
    fig_sectors.show()

    # 3) Mercados Geo
    fig_markets_geo = build_markets_geo(df_epi_countries)
    fig_markets_geo.show()

    # 4) Barras + Linha por produto selecionado
    if SELECTED_SH6 is not None:
        fig_epi_combo = build_epi_bars_and_scatter(df_epi, SELECTED_SH6)
        fig_epi_combo.show()

        # Exporta a tabela usada no gráfico para .xlsx
        df_selected = df_epi.filter(pl.col("sh6_product") == SELECTED_SH6).sort("epi_score_normalized", descending=True)
        df_selected_pd = df_selected.to_pandas().head(25).sort_values("epi_score_normalized", ascending=True)
        output_path = PROJECT_ROOT / "tabela_epi_barras.xlsx"
        df_selected_pd.to_excel(output_path, index=False)
        print(f"Tabela exportada para: {output_path}")

        # Tabela de mercados (amostra similar ao app)
        df_selected_markets = df_markets.filter(pl.col("sh6_product") == SELECTED_SH6).sort("value", descending=True)
        total_imports = df_selected_markets['value'].sum()
        print(f"Mercado mundial do produto (2023): US$ {format_contabil(total_imports)}")
        df_selected_markets = df_selected_markets.with_columns((pl.arange(1, df_selected_markets.height + 1)).alias("Posição"))
        display(
            df_selected_markets.select([
                pl.col('Posição'),
                pl.col('importer_name').alias("País"),
                pl.col('value_contabil').alias("Montante US$"),
                pl.col('market_share').alias("Market Share (%)"),
                pl.col('cagr_5y_adj').alias("CAGR 5 anos (%)"),
                pl.col('share_brazil').alias("Share Brasil (%)"),
                pl.col('share_sc').alias("Share SC (%)"),
                pl.col('dist').alias("Distância (km)"),
            ]).to_pandas()
        )

        # 5) Mapa do produto
        fig_prod_geo = build_product_geo(df_epi, SELECTED_SH6)
        fig_prod_geo.show()

    # 6) Fornecedores: Treemap + Mapa
    if SUP_COUNTRY and SUP_SH6:
        fig_suppliers = build_suppliers_treemap(df_competitors, df_epi_sh6, SUP_COUNTRY, SUP_SH6)
        fig_suppliers.show()

        fig_imports_geo = build_imports_geo(df_competitors, SUP_COUNTRY, SUP_SH6)
        fig_imports_geo.show()

    print("\nDica: edite SELECTED_SH6, SUP_COUNTRY e SUP_SH6 no topo deste arquivo para refinar as figuras.")
