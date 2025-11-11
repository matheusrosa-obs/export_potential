from __future__ import annotations

import re
import numpy as np
import pandas as pd
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from typing import List, Tuple

# ===========================
# Configurações
# ===========================
st.set_page_config(page_title="Mapa de Tarifas Internacionais", layout="wide")

ARQ                 = "all_partners_tariffs_filtrado.parquet"

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
    titulo = f"Tarifas aplicadas a <b>{pais_destino_nome}</b> para o produto <i>{produto_escolhido}</i>"

    fig.update_layout(
        paper_bgcolor="#22232E",
        plot_bgcolor="#22232E",
        title=dict(text=titulo, font=dict(color="white", size=22), x=0.5, xanchor="center"),
        coloraxis_colorbar=dict(
            title=dict(text="Tarifa (%)", font=dict(color="white")),
            orientation="h", x=0.5, xanchor="center", y=-0.08, yanchor="top",
            len=0.6, thickness=12, tickcolor="white", tickfont=dict(color="white"),
            outlinecolor="rgba(255,255,255,0.2)",
        ),
        margin=dict(l=0, r=0, t=70, b=0),
        font=dict(color="white"),
    )

    fig.update_geos(
        showocean=True, oceancolor="#22232E",
        showland=True, landcolor="#595959",
        showcountries=True, countrycolor="white",
        showcoastlines=True, coastlinecolor="white",
        countrywidth=0.2, coastlinewidth=0.2,
        showframe=False, bgcolor="#22232E",
    )

    return fig, df_bolhas, df_zeros

# ===========================
# UI
# ===========================
st.markdown("<h2 style='text-align:center; color:white; margin-top:0'>Mapa Interativo de Tarifas Internacionais</h2>", unsafe_allow_html=True)

try:
    df_base = carregar_dados(ARQ)

    # --- Estado inicial ---
    if "pais_escolhido_nome" not in st.session_state:
        st.session_state.pais_escolhido_nome = "Brazil"  # pelo NOME
        st.session_state.produto_escolhido   = None

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
        st.selectbox("**País de Destino (Partner)**", paises_destino_nomes, index=idx_pais, key="pais_escolhido_nome")

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
        st.selectbox("**Produto (digite para buscar)**", options=produtos, index=idx_prod, key="produto_escolhido")

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

    st.caption(
        "O tamanho e a cor da bolha representam o valor da Tarifa. "
        "Passe o mouse para ver os detalhes. Os 5 maiores valores exibem rótulos numéricos."
    )

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
        st.markdown(f"<h5 style='text-align: center;'>Países que aplicam tarifa a {st.session_state.pais_escolhido_nome} (> 0%)</h5>", unsafe_allow_html=True)
        df_tabela_com_tarifa = (
            df_com_tarifa
            .select([REPORTER_NAME_COL, YEAR_COL, VAL_COL])
            .rename({REPORTER_NAME_COL: "País", YEAR_COL: "Ano", VAL_COL: "Tarifa (%)"})
            .sort(by="Tarifa (%)", descending=True)
        )
        st.data_editor(df_tabela_com_tarifa.to_pandas(), use_container_width=True, hide_index=True, disabled=True)

    with col2:
        st.markdown(f"<h5 style='text-align: center;'>Oportunidades para {st.session_state.pais_escolhido_nome} (Tarifa Zero)</h5>", unsafe_allow_html=True)
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
