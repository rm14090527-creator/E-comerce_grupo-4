# =============================================================================
# E-commerce: proyección de demanda, inventario predictivo y pricing
# Modelamiento Predictivo de Datos — Grupo 4
#
# Cómo correrla:
#     pip install -r requirements.txt
#     streamlit run app.py
#
# Flujo: tablas originales -> demanda proyectada (3 escenarios)
#        -> stock estimado y recomendado -> elasticidad y escenarios de precio
# =============================================================================
from __future__ import annotations

import io
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.formula.api as smf
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

try:  # Prophet es opcional: si está instalado se agrega como cuarto modelo
    import logging

    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    from prophet import Prophet

    HAY_PROPHET = True
except Exception:
    HAY_PROPHET = False

st.set_page_config(
    page_title="E-commerce · Demanda, inventario y precios",
    page_icon="🛒",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem;}
      [data-testid="stMetric"] {background:#f8f9fb; border:1px solid #e9ecef;
                                border-radius:10px; padding:10px 14px;}
      [data-testid="stMetricLabel"] p {color:#495057;}
      [data-testid="stMetricValue"] {font-size:1.55rem;}
      .nota {color:#6c757d; font-size:0.88rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_DIR = Path(__file__).parent / "data"
ESCENARIOS = ["Pesimista", "Base", "Optimista"]
COLOR = {"Pesimista": "#e8590c", "Base": "#1c7ed6", "Optimista": "#2f9e44",
         "hist": "#868e96", "real": "#212529"}
LAYOUT = dict(template="plotly_white", margin=dict(l=10, r=10, t=45, b=70),
              legend=dict(orientation="h", yanchor="top", y=-0.22, x=0))

TABLAS = {  # columnas mínimas que identifican cada tabla original
    "inventario": {"date", "product_id", "stock_on_hand"},
    "pedidos": {"order_id", "order_date", "order_status"},
    "catalogo": {"product_id", "product_name", "product_category", "product_cost"},
    "items": {"order_id", "product_id", "quantity", "gross_sales",
              "discount_amount", "net_sales"},
}
NOMBRE_ARCHIVO = {
    "pedidos": "ecommerce_sales_customer_analytics_150k.csv",
    "items": "order_items.csv",
    "catalogo": "product_catalog.csv",
    "inventario": "inventory_stock.csv",
}


def usd(x: float) -> str:
    signo = "-" if x < 0 else ""
    return f"{signo}${abs(x):,.0f}"


def usd_corto(x: float) -> str:
    for lim, suf in [(1e9, " B"), (1e6, " M"), (1e3, " k")]:
        if abs(x) >= lim:
            return f"${x / lim:,.1f}{suf}"
    return f"${x:,.0f}"


# =============================================================================
# 1. LECTURA DE LAS TABLAS ORIGINALES
# =============================================================================
def _abridor_upload(f):
    def abrir():
        f.seek(0)
        return f
    return abrir


def listar_fuentes(archivos: list) -> list:
    """Convierte CSV o ZIP (subidos o locales) en una lista de (nombre, abridor)."""
    fuentes = []
    for a in archivos:
        nombre = a.name if hasattr(a, "name") else str(a)
        if str(nombre).lower().endswith(".zip"):
            zf = zipfile.ZipFile(a if not isinstance(a, Path) else str(a))
            for m in zf.namelist():
                if m.lower().endswith(".csv") and not m.startswith("__MACOSX"):
                    fuentes.append((m, lambda zf=zf, m=m: zf.open(m)))
        elif isinstance(a, Path):
            fuentes.append((a.name, lambda a=a: open(a, "rb")))
        else:
            fuentes.append((nombre, _abridor_upload(a)))
    return fuentes


def identificar(fuentes: list) -> dict:
    """Reconoce cada tabla por sus columnas, sin depender del nombre del archivo."""
    encontradas = {}
    for _, abrir in fuentes:
        try:
            cols = set(pd.read_csv(abrir(), nrows=0).columns.str.strip())
        except Exception:
            continue
        for tipo, req in TABLAS.items():  # el orden importa: pedidos antes que items
            if req <= cols and tipo not in encontradas:
                encontradas[tipo] = abrir
                break
    return encontradas


@st.cache_data(show_spinner="Leyendo y cruzando las tablas del dataset (solo la primera vez)…",
               max_entries=2)
def procesar(_fuentes: dict, firma: str) -> dict:
    cat = pd.read_csv(_fuentes["catalogo"](), usecols=lambda c: c in {
        "product_id", "product_name", "product_category", "product_subcategory",
        "brand", "product_cost"})
    cat = cat.drop_duplicates("product_id")

    ped = pd.read_csv(_fuentes["pedidos"](),
                      usecols=["order_id", "order_date", "order_status"])
    ped["estado"] = ped["order_status"].str.strip().str.lower()
    ped["mes"] = pd.to_datetime(ped["order_date"].astype(str).str[:7] + "-01",
                                errors="coerce")
    ped = ped[["order_id", "mes", "estado"]]

    items = pd.read_csv(_fuentes["items"](), usecols=lambda c: c in {
        "order_id", "product_id", "quantity", "gross_sales", "discount_amount",
        "net_sales", "profit"})
    if "profit" not in items:
        items["profit"] = np.nan
    lin = items.merge(ped, on="order_id", how="inner")
    lin = lin.merge(cat[["product_id", "product_category"]], on="product_id", how="left")
    lin = lin.dropna(subset=["mes"])
    lin["order_id"] = lin["order_id"].astype("category").cat.codes
    lin["estado"] = lin["estado"].astype("category")
    lin["product_category"] = lin["product_category"].astype("category")

    # Panel producto-mes con pedidos completados (la demanda real entregada)
    comp = lin[lin["estado"] == "completed"]
    pm = (comp.groupby(["product_id", "mes"], as_index=False)
          .agg(unidades=("quantity", "sum"), bruto=("gross_sales", "sum"),
               descuento=("discount_amount", "sum"), neto=("net_sales", "sum"),
               utilidad=("profit", "sum")))

    # Stock promedio mensual por producto (se lee por bloques: el archivo es grande)
    partes = []
    for ch in pd.read_csv(_fuentes["inventario"](), usecols=["date", "product_id",
                          "stock_on_hand"], dtype={"date": str}, chunksize=500_000):
        ch["mes"] = ch["date"].str[:7]
        partes.append(ch.groupby(["product_id", "mes"])["stock_on_hand"]
                      .agg(["sum", "count"]))
    inv = pd.concat(partes).groupby(level=[0, 1]).sum()
    inv = (inv["sum"] / inv["count"]).rename("stock_promedio").reset_index()
    inv["mes"] = pd.to_datetime(inv["mes"] + "-01")

    meses = pd.date_range(pm["mes"].min(), pm["mes"].max(), freq="MS")
    return {"cat": cat, "lin": lin[["order_id", "product_id", "product_category", "mes",
                                    "estado", "quantity", "net_sales", "profit"]],
            "pm": pm, "inv": inv, "meses": meses}


# =============================================================================
# 2. MODELOS DE DEMANDA
# =============================================================================
def mape(real, pred) -> float:
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    m = real != 0
    return float(np.mean(np.abs((real[m] - pred[m]) / real[m])) * 100) if m.any() else np.nan


def f_holt_winters(y: pd.Series, h: int) -> np.ndarray:
    try:
        m = ExponentialSmoothing(y.values.astype(float), trend="add", seasonal="add",
                                 seasonal_periods=12).fit()
    except Exception:
        m = ExponentialSmoothing(y.values.astype(float), trend="add").fit()
    return m.forecast(h)


VARS_RF = ["mes", "lag_1", "lag_2", "lag_12", "media_3"]


def f_random_forest(y: pd.Series, h: int) -> np.ndarray:
    d = pd.DataFrame({"y": y.values.astype(float)}, index=y.index)
    d["mes"] = d.index.month
    d["lag_1"], d["lag_2"], d["lag_12"] = d.y.shift(1), d.y.shift(2), d.y.shift(12)
    d["media_3"] = d.y.shift(1).rolling(3).mean()
    d = d.dropna()
    rf = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
    rf.fit(d[VARS_RF], d["y"])
    vals, fecha, pred = list(y.values.astype(float)), y.index[-1], []
    for _ in range(h):
        fecha = fecha + pd.DateOffset(months=1)
        x = pd.DataFrame([[fecha.month, vals[-1], vals[-2], vals[-12],
                           np.mean(vals[-3:])]], columns=VARS_RF)
        p = float(rf.predict(x)[0])
        vals.append(p)
        pred.append(p)
    return np.array(pred)


def f_naive_estacional(y: pd.Series, h: int) -> np.ndarray:
    return np.tile(y.values[-12:].astype(float), int(np.ceil(h / 12)))[:h]


def f_prophet(y: pd.Series, h: int) -> np.ndarray:
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(pd.DataFrame({"ds": y.index, "y": y.values}))
    fut = m.make_future_dataframe(periods=h, freq="MS")
    return m.predict(fut)["yhat"].values[-h:]


MODELOS = {"Holt-Winters": f_holt_winters, "Random Forest": f_random_forest,
           "Naive estacional": f_naive_estacional}
if HAY_PROPHET:
    MODELOS["Prophet"] = f_prophet


@st.cache_data(show_spinner="Entrenando los modelos de demanda…", max_entries=20)
def pronosticar(wide: pd.DataFrame, h: int, val_meses: int) -> dict:
    """Valida cada modelo en los últimos val_meses y luego proyecta h meses."""
    fechas_val = wide.index[-val_meses:]
    fechas_fut = pd.date_range(wide.index[-1] + pd.DateOffset(months=1), periods=h, freq="MS")
    salida = {}
    for nombre, fn in MODELOS.items():
        val, fut = {}, {}
        for pid in wide.columns:
            y = wide[pid]
            val[pid] = np.clip(fn(y.iloc[:-val_meses], val_meses), 0, None)
            fut[pid] = np.clip(fn(y, h), 0, None)
        val = pd.DataFrame(val, index=fechas_val)
        fut = pd.DataFrame(fut, index=fechas_fut)
        real = wide.loc[fechas_val]
        salida[nombre] = {
            "val": val, "fut": fut,
            "mape_total": mape(real.sum(axis=1), val.sum(axis=1)),
            "mape_prod": {p: mape(real[p], val[p]) for p in wide.columns},
            "rmse_prod": {p: float(np.sqrt(np.mean((real[p] - val[p]) ** 2)))
                          for p in wide.columns},
        }
    return salida


# =============================================================================
# 3. INVENTARIO: regresión stock ~ demanda + producto + mes
# =============================================================================
def modelo_inventario(hist: pd.DataFrame):
    formula = "stock_promedio ~ demanda + C(mes_num)"
    if hist["product_id"].nunique() > 1:
        formula += " + C(product_id)"
    return smf.ols(formula, data=hist).fit()


# =============================================================================
# 4. PRICING: elasticidad log-log por producto
# =============================================================================
def elasticidades(pm: pd.DataFrame, cat: pd.DataFrame, ids: list) -> pd.DataFrame:
    filas = []
    for pid in ids:
        g = pm[(pm["product_id"] == pid) & (pm["unidades"] > 0)].copy()
        g["precio"] = (g["bruto"] - g["descuento"]) / g["unidades"]
        g = g[g["precio"] > 0]
        ult = g[g["mes"] > g["mes"].max() - pd.DateOffset(months=12)]
        precio = (ult["bruto"] - ult["descuento"]).sum() / ult["unidades"].sum()
        costo = float(cat.loc[cat["product_id"] == pid, "product_cost"].iloc[0])
        fila = {"product_id": pid, "precio": precio, "costo": costo,
                "margen": (precio - costo) / precio, "beta": np.nan,
                "ic_bajo": np.nan, "ic_alto": np.nan, "p": np.nan, "r2": np.nan}
        if len(g) >= 12 and np.log(g["precio"]).std() > 0:
            g["lq"], g["lp"] = np.log(g["unidades"]), np.log(g["precio"])
            r = smf.ols("lq ~ lp", data=g).fit()
            ic = r.conf_int().loc["lp"]
            fila.update(beta=r.params["lp"], ic_bajo=ic[0], ic_alto=ic[1],
                        p=r.pvalues["lp"], r2=r.rsquared)
        filas.append(fila)
    e = pd.DataFrame(filas)
    validas = e[(e["p"] < 0.10) & (e["beta"] < 0)]
    respaldo = validas["beta"].median() if len(validas) else e["beta"].median()
    usa_propia = (e["p"] < 0.10) & (e["beta"] < 0)
    e["beta_usada"] = np.where(usa_propia, e["beta"], respaldo)
    e["origen"] = np.where(usa_propia, "Propia", "Mediana del grupo")
    e["umbral"] = np.where(e["margen"] > 0, 1 / e["margen"], np.nan)
    return e


def simular_precio(e: pd.DataFrame, q0: pd.Series, delta: float) -> dict:
    """q0: unidades proyectadas por producto en el horizonte. delta: cambio de precio."""
    d = e.set_index("product_id").loc[q0.index]
    q1 = q0 * (1 + delta) ** d["beta_usada"]
    p1 = d["precio"] * (1 + delta)
    return {"unidades": float(q1.sum()), "ingreso": float((p1 * q1).sum()),
            "margen": float(((p1 - d["costo"]) * q1).sum()),
            "margen_prod": (p1 - d["costo"]) * q1, "unid_prod": q1}


# =============================================================================
# BARRA LATERAL: datos y selección de productos
# =============================================================================
st.title("🛒 E-commerce: demanda, inventario y precios")
st.markdown('<p class="nota">Del histórico de ventas a la decisión: cuánto se va a '
            'vender, cuánto stock tener y a qué precio vender. Cada pestaña usa la '
            'proyección de demanda de la anterior.</p>', unsafe_allow_html=True)

locales = sorted(DATA_DIR.glob("*.zip")) + sorted(DATA_DIR.glob("*.csv")) \
    if DATA_DIR.exists() else []

with st.sidebar:
    st.header("1 · Datos")
    opciones = (["Dataset incluido en la app"] if locales else []) + ["Subir mis archivos"]
    origen = st.radio("Origen de los datos", opciones, label_visibility="collapsed")
    if origen == "Subir mis archivos":
        subidos = st.file_uploader(
            "Sube los CSV originales o un .zip que los contenga",
            type=["csv", "zip"], accept_multiple_files=True,
            help="Se necesitan: " + ", ".join(NOMBRE_ARCHIVO.values()) +
                 ". Otros archivos (customer_master, dataset_statistics) se ignoran.")
        archivos = subidos or []
        firma = "|".join(f"{f.name}:{f.size}" for f in archivos)
    else:
        archivos = locales
        firma = "|".join(f"{p.name}:{p.stat().st_size}" for p in locales)

fuentes = identificar(listar_fuentes(archivos)) if archivos else {}
faltan = [NOMBRE_ARCHIVO[t] for t in TABLAS if t not in fuentes]
if faltan:
    st.info("**Para empezar, carga las tablas originales del dataset** en la barra "
            "lateral (CSV sueltos o un .zip).\n\n" +
            ("Faltan: " if archivos else "Se necesitan: ") +
            ", ".join(f"`{f}`" for f in faltan))
    st.stop()

D = procesar(fuentes, firma)
cat, lin, pm, inv, meses = D["cat"], D["lin"], D["pm"], D["inv"], D["meses"]
etiqueta = {r.product_id: f"{r.product_id} · {r.product_name[:38]}"
            for r in cat.itertuples()}

with st.sidebar:
    st.header("2 · Productos a analizar")
    categorias = sorted(cat["product_category"].dropna().unique())
    cats_sel = st.multiselect("Categorías", categorias, placeholder="Todas las categorías")
    cats_sel = cats_sel or categorias
    universo = cat[cat["product_category"].isin(cats_sel)][["product_id"]].merge(
        pm.groupby("product_id", as_index=False)[["unidades", "neto"]].sum(),
        on="product_id", how="left").fillna(0)
    universo = universo[universo["unidades"] > 0]
    ca, cb = st.columns(2)
    criterio = ca.radio("Mostrar", ["Más vendidos", "Menos vendidos"])
    metrica = cb.radio("Según", ["Unidades", "Ventas netas"], index=1)
    n = st.slider("Cantidad de productos", 1, 20, 5)
    with st.expander("O elegir productos a mano"):
        manual = st.multiselect("Productos", universo["product_id"].tolist(),
                                format_func=lambda p: etiqueta.get(p, p), max_selections=20,
                                placeholder="Busca por ID o nombre",
                                help="Si eliges productos aquí, reemplazan al ranking.")
    if manual:
        sel, criterio = manual, "Elegidos a mano"
    else:
        col = "neto" if metrica == "Ventas netas" else "unidades"
        sel = (universo.sort_values(col, ascending=(criterio == "Menos vendidos"))
               .head(n)["product_id"].tolist())

    st.header("3 · Proyección de demanda")
    h = st.slider("Meses a proyectar", 6, 24, 12)
    val_meses = st.slider("Meses de prueba (validación)", 3, 24, 6,
                          help="Últimos meses reales que se ocultan al modelo para medir "
                               "su error (MAPE).")

if not sel:
    st.warning("Selecciona al menos un producto en la barra lateral.")
    st.stop()

wide = (pm[pm["product_id"].isin(sel)].pivot(index="mes", columns="product_id",
                                             values="unidades")
        .reindex(index=meses, columns=sel).fillna(0))
wide.index.freq = "MS"
res = pronosticar(wide, h, val_meses)
mapes = {m: r["mape_total"] for m, r in res.items()}
mejor = min(mapes, key=mapes.get)

with st.sidebar:
    modelo = st.selectbox("Modelo", list(res), index=list(res).index(mejor),
                          help="Por defecto, el de menor error en validación.")
    st.caption(f"Histórico: {meses[0]:%b %Y} – {meses[-1]:%b %Y}. "
               f"MAPE del modelo: {mapes[modelo]:.1f}%")

    st.header("4 · Escenarios de demanda")
    modo = st.radio("Error ± para los escenarios", ["Fijo (%)", "MAPE de cada producto"])
    R = res[modelo]
    if modo == "Fijo (%)":
        err_pct = st.slider("Error ± (%)", 0, 50, 20)
        err = pd.Series(err_pct / 100, index=sel)
        err_txt = f"±{err_pct}%"
    else:
        err = pd.Series(R["mape_prod"]).reindex(sel).fillna(mapes[modelo]).clip(upper=100) / 100
        err_txt = f"±MAPE por producto ({err.min():.0%}–{err.max():.0%})"
        st.caption(f"Cada producto usa su propio error: {err.min():.0%} a {err.max():.0%} "
                   "(máximo 100%).")
    esc_sel = st.radio("Escenario activo (Inventario y Precios)", ESCENARIOS, index=1)

fut = {"Base": R["fut"], "Pesimista": (R["fut"] * (1 - err)).clip(lower=0),
       "Optimista": R["fut"] * (1 + err)}

tab0, tab1, tab2, tab3 = st.tabs(["📊 Resumen", "📈 1 · Demanda", "📦 2 · Inventario",
                                  "💲 3 · Precios"])

# =============================================================================
# PESTAÑA RESUMEN
# =============================================================================
with tab0:
    L = lin[lin["product_category"].isin(cats_sel)]
    LC = L[L["estado"] == "completed"]
    ventas, pedidos = LC["net_sales"].sum(), LC["order_id"].nunique()
    tot_ped = L["order_id"].nunique()
    k = st.columns(6)
    k[0].metric("Ventas netas", usd_corto(ventas))
    k[1].metric("Pedidos completados", f"{pedidos:,}")
    k[2].metric("Ticket promedio", usd(ventas / pedidos) if pedidos else "–")
    k[3].metric("Margen de utilidad", f"{LC['profit'].sum() / ventas:.1%}" if ventas else "–")
    k[4].metric("Tasa de devolución",
                f"{L.loc[L.estado == 'returned', 'order_id'].nunique() / tot_ped:.1%}")
    k[5].metric("Tasa de cancelación",
                f"{L.loc[L.estado == 'cancelled', 'order_id'].nunique() / tot_ped:.1%}")
    caja_escenarios = st.container()
    st.caption("Ámbito: " + ("todas las categorías" if len(cats_sel) == len(categorias)
                             else ", ".join(cats_sel)) + ". Ventas de pedidos completados.")

    c1, c2 = st.columns([1.4, 1])
    with c1:
        mens = LC.groupby("mes")["net_sales"].sum()
        mens_sel = LC[LC["product_id"].isin(sel)].groupby("mes")["net_sales"].sum()
        fig = go.Figure()
        fig.add_bar(x=mens.index, y=mens.values, name="Ámbito seleccionado",
                    marker_color="#d0ebff")
        fig.add_scatter(x=mens_sel.index, y=mens_sel.values, name="Productos analizados (eje der.)",
                        line=dict(color=COLOR["Base"], width=2.5), yaxis="y2")
        fig.update_layout(title="Ventas netas mensuales (USD)", height=380,
                          yaxis2=dict(overlaying="y", side="right", showgrid=False, rangemode="tozero"), **LAYOUT)
        st.plotly_chart(fig, width="stretch")
    with c2:
        por_cat = (lin[lin["estado"] == "completed"].groupby("product_category",
                   observed=True)["net_sales"].sum().sort_values())
        fig = go.Figure(go.Bar(
            x=por_cat.values, y=por_cat.index, orientation="h",
            marker_color=[COLOR["Base"] if c in cats_sel else "#dee2e6"
                          for c in por_cat.index]))
        fig.update_layout(title="Ventas netas por categoría", height=360, **LAYOUT)
        st.plotly_chart(fig, width="stretch")

    tabla = universo.copy()
    tabla["particip"] = tabla["neto"] / tabla["neto"].sum()
    tabla = tabla.sort_values("neto", ascending=False)
    tabla["acum"] = tabla["particip"].cumsum()
    tabla["ABC"] = pd.cut(tabla["acum"], [0, .8, .95, 1.0001], labels=["A", "B", "C"])
    tabla["ranking"] = np.arange(1, len(tabla) + 1)
    tabla = tabla[tabla["product_id"].isin(sel)].merge(
        cat[["product_id", "product_name", "product_category"]], on="product_id")
    st.markdown(f"**Productos analizados ({len(sel)})** — {criterio.lower()} del ámbito")
    st.dataframe(
        tabla.assign(particip=tabla["particip"] * 100)[
            ["ranking", "product_id", "product_name", "product_category", "unidades",
             "neto", "particip", "ABC"]],
        hide_index=True, width="stretch",
        column_config={
            "ranking": "Puesto", "product_id": "ID", "product_name": "Producto",
            "product_category": "Categoría",
            "unidades": st.column_config.NumberColumn("Unidades 2021-25", format="%,d"),
            "neto": st.column_config.NumberColumn("Ventas netas", format="$%,.0f"),
            "particip": st.column_config.NumberColumn("% ventas del ámbito", format="%.2f%%"),
            "ABC": "Clase ABC"},
    )

# =============================================================================
# PESTAÑA 1: DEMANDA
# =============================================================================
with tab1:
    st.subheader("¿Cuánto se va a vender?")
    tot = {s: fut[s].sum(axis=1) for s in ESCENARIOS}
    prom_hist = wide.sum(axis=1).iloc[-12:].mean()
    k = st.columns(4)
    k[0].metric(f"Error de validación · {modelo}", f"{mapes[modelo]:.1f}%",
                help=f"Error porcentual medio del modelo en los últimos {val_meses} meses "
                     "reales (serie total de los productos analizados).")
    for i, s in enumerate(ESCENARIOS):
        k[i + 1].metric(f"Demanda {s.lower()} · {h} meses", f"{tot[s].sum():,.0f} u.",
                        f"{tot[s].mean() / prom_hist - 1:+.1%} vs. último año",
                        help="Promedio mensual proyectado frente al promedio mensual de los últimos 12 meses reales.")

    ver = st.selectbox("Serie a graficar", ["Total de productos analizados"] + sel,
                       format_func=lambda p: etiqueta.get(p, p))
    if ver.startswith("Total"):
        hist_s, serie = wide.sum(axis=1), {s: tot[s] for s in ESCENARIOS}
        val_s = R["val"].sum(axis=1)
    else:
        hist_s, serie = wide[ver], {s: fut[s][ver] for s in ESCENARIOS}
        val_s = R["val"][ver]

    fig = go.Figure()
    fig.add_scatter(x=hist_s.index, y=hist_s.values, name="Histórico",
                    line=dict(color=COLOR["hist"], width=2))
    fig.add_scatter(x=val_s.index, y=val_s.values, name="Ajuste en validación",
                    line=dict(color=COLOR["Base"], width=1.5, dash="dot"))
    x = list(serie["Base"].index)
    fig.add_scatter(x=x + x[::-1],
                    y=list(serie["Optimista"].values) + list(serie["Pesimista"].values[::-1]),
                    fill="toself", fillcolor="rgba(28,126,214,0.12)",
                    line=dict(width=0), name="Rango de escenarios", hoverinfo="skip")
    for s in ESCENARIOS:
        fig.add_scatter(x=x, y=serie[s].values, name=s, mode="lines+markers",
                        line=dict(color=COLOR[s], width=3.5 if s == esc_sel else 1.6,
                                  dash=None if s == esc_sel else "dash"),
                        marker=dict(size=5))
    fig.add_vline(x=hist_s.index[-1], line_dash="dot", line_color="#adb5bd")
    fig.update_layout(title=f"Demanda mensual (unidades) · {modelo}", height=430,
                      hovermode="x unified", **LAYOUT)
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.markdown("**Proyección por producto**")
        por_prod = pd.DataFrame({
            "Producto": sel,
            "Últimos 12 meses": wide.iloc[-12:].sum().values,
            **{f"{s} ({h} m)": fut[s].sum().values for s in ESCENARIOS},
            "MAPE producto": [R["mape_prod"][p] for p in sel],
        })
        st.dataframe(por_prod, hide_index=True, width="stretch", column_config={
            **{c: st.column_config.NumberColumn(format="%,.0f")
               for c in por_prod.columns[1:-1]},
            "MAPE producto": st.column_config.NumberColumn(format="%.0f%%")})
    with c2:
        st.markdown(f"**Comparación de modelos (validación {val_meses} meses)**")
        comp_m = pd.DataFrame({"Modelo": list(mapes), "MAPE total": list(mapes.values()),
                               "MAPE promedio por producto": [
                                   np.nanmean(list(res[m]["mape_prod"].values())) for m in res]
                               }).sort_values("MAPE total")
        st.dataframe(comp_m, hide_index=True, width="stretch", column_config={
            c: st.column_config.NumberColumn(format="%.1f%%") for c in comp_m.columns[1:]})
        st.caption("El error baja al agregar productos: la demanda individual es pequeña y "
                   "volátil, la total es más estable.")

    salida = pd.concat([fut[s].stack().rename(s) for s in ESCENARIOS], axis=1).reset_index()
    salida.columns = ["mes", "product_id"] + ESCENARIOS
    salida["mes"] = salida["mes"].dt.strftime("%Y-%m")
    st.download_button("⬇️ Descargar proyección (CSV)", salida.round(2).to_csv(index=False),
                       "proyeccion_demanda.csv", "text/csv")

# Demanda proyectada en formato largo, insumo de inventario y precios
largo = {s: fut[s].stack().rename("demanda").reset_index()
         .rename(columns={"level_0": "mes"}) for s in ESCENARIOS}
for s in ESCENARIOS:
    largo[s].columns = ["mes", "product_id", "demanda"]

# =============================================================================
# PESTAÑA 2: INVENTARIO
# =============================================================================
with tab2:
    st.subheader("¿Cuánto stock tener?")
    st.markdown('<p class="nota">Una regresión aprende cómo se ha movido el stock según '
                'la demanda, el producto y el mes. Con la demanda proyectada estima el stock '
                'que habría <b>si se mantiene la práctica actual</b> y lo compara con el '
                '<b>stock recomendado</b> por una política de reposición mensual con stock '
                'de seguridad.</p>', unsafe_allow_html=True)

    hist = (wide.stack().rename("demanda").reset_index())
    hist.columns = ["mes", "product_id", "demanda"]
    hist = hist.merge(inv, on=["product_id", "mes"], how="inner")
    hist["mes_num"] = hist["mes"].dt.month
    mod_inv = modelo_inventario(hist)

    c1, c2 = st.columns(2)
    nivel = c1.select_slider("Nivel de servicio objetivo", [0.90, 0.95, 0.975, 0.99],
                             value=0.95, format_func=lambda v: f"{v:.1%}")
    lead = c2.slider("Tiempo de reposición del proveedor (días)", 3, 60, 10)
    esc_inv = esc_sel
    z = {0.90: 1.282, 0.95: 1.645, 0.975: 1.960, 0.99: 2.326}[nivel]

    sigma = pd.Series(R["rmse_prod"])  # error mensual del pronóstico por producto
    ss = z * sigma * np.sqrt(lead / 30)
    costos = cat.set_index("product_id")["product_cost"]

    resumen, detalle = {}, {}
    for s in ESCENARIOS:
        f = largo[s].copy()
        f["mes_num"] = f["mes"].dt.month
        f["stock_estimado"] = np.clip(mod_inv.predict(f), 0, None)
        f["ss"] = f["product_id"].map(ss)
        f["stock_recomendado"] = f["ss"] + f["demanda"] / 2  # stock medio con pedido mensual
        g = f.groupby("product_id").agg(demanda=("demanda", "mean"),
                                        estimado=("stock_estimado", "mean"),
                                        recomendado=("stock_recomendado", "mean"),
                                        ss=("ss", "first"))
        g["rop"] = g["demanda"] * lead / 30 + g["ss"]
        g["maximo"] = g["ss"] + g["demanda"]
        g["cobertura"] = np.where(g["demanda"] > 0, g["estimado"] / g["demanda"] * 30, np.nan)
        g["estado"] = np.select([g["estimado"] < g["rop"], g["estimado"] > g["maximo"]],
                                ["🔴 Riesgo de quiebre", "🟠 Sobrestock"], "🟢 Adecuado")
        g["brecha_usd"] = (g["estimado"] - g["recomendado"]) * costos.reindex(g.index)
        detalle[s], resumen[s] = g, f

    k = st.columns(3)
    for i, s in enumerate(ESCENARIOS):
        g = detalle[s]
        br = g["brecha_usd"].sum()
        k[i].metric(
            ("▶ " if s == esc_sel else "") + f"{s} · stock estimado vs. recomendado",
            f"{g['estimado'].sum():,.0f} u. vs. {g['recomendado'].sum():,.0f} u.",
            f"{'Exceso' if br > 0 else 'Faltante'} de {usd(abs(br))} a costo",
            delta_color="inverse" if br > 0 else "normal")
    st.caption("Exceso = capital inmovilizado frente a la política recomendada. "
               "Faltante = inversión adicional necesaria para cubrir el nivel de servicio.")

    hist_tot = hist.groupby("mes")["stock_promedio"].sum()
    fig = go.Figure()
    fig.add_scatter(x=hist_tot.index, y=hist_tot.values, name="Stock histórico",
                    line=dict(color=COLOR["hist"], width=2))
    for s in ESCENARIOS:
        t = resumen[s].groupby("mes")["stock_estimado"].sum()
        fig.add_scatter(x=t.index, y=t.values, name=f"Estimado {s.lower()}",
                        line=dict(color=COLOR[s], width=3.5 if s == esc_sel else 1.6))
    rb = resumen[esc_sel].groupby("mes")["stock_recomendado"].sum()
    fig.add_scatter(x=rb.index, y=rb.values, name=f"Recomendado ({esc_sel.lower()})",
                    line=dict(color="#7048e8", width=2.5, dash="dash"))
    fig.add_vline(x=hist_tot.index[-1], line_dash="dot", line_color="#adb5bd")
    fig.update_layout(title="Stock promedio mensual (unidades, suma de productos)",
                      height=420, hovermode="x unified", **LAYOUT)
    st.plotly_chart(fig, width="stretch")

    g = detalle[esc_inv].reset_index()
    g["Producto"] = g["product_id"]
    st.markdown(f"**Política recomendada por producto · escenario {esc_inv.lower()}**")
    st.dataframe(
        g[["Producto", "demanda", "estimado", "ss", "rop", "recomendado", "cobertura",
           "estado", "brecha_usd"]],
        hide_index=True, width="stretch", column_config={
            "demanda": st.column_config.NumberColumn("Demanda / mes", format="%.1f"),
            "estimado": st.column_config.NumberColumn("Stock estimado", format="%.1f",
                help="Stock promedio que predice la regresión con la práctica actual."),
            "ss": st.column_config.NumberColumn("Stock de seguridad", format="%.1f",
                help="z × error mensual del pronóstico × √(días de reposición / 30)"),
            "rop": st.column_config.NumberColumn("Punto de reorden", format="%.1f",
                help="Pedir cuando el stock baje de este nivel."),
            "recomendado": st.column_config.NumberColumn("Stock recomendado", format="%.1f"),
            "cobertura": st.column_config.NumberColumn("Cobertura (días)", format="%.0f"),
            "estado": "Diagnóstico",
            "brecha_usd": st.column_config.NumberColumn("Exceso (+) / faltante (−)",
                                                        format="$%,.0f")})

    with st.expander("Ver el modelo de regresión de inventario"):
        b = mod_inv.params["demanda"]
        m = st.columns(4)
        m[0].metric("R²", f"{mod_inv.rsquared:.3f}")
        m[1].metric("Error típico (RMSE)", f"{np.sqrt(mod_inv.mse_resid):.2f} u.")
        m[2].metric("Efecto de la demanda", f"{b:+.3f} u. de stock",
                    help="Cambio en el stock promedio por cada unidad adicional de demanda.")
        m[3].metric("p-valor de la demanda", f"{mod_inv.pvalues['demanda']:.4f}")
        st.latex(r"Stock_{it} = \beta_0 + \beta_1 \cdot Demanda_{it} + "
                 r"\gamma_{producto} + \delta_{mes} + \varepsilon_{it}")
        st.caption(f"{int(mod_inv.nobs)} observaciones producto-mes. Un coeficiente negativo "
                   "indica que en los meses de más venta el stock se agota antes de reponerse.")

# =============================================================================
# PESTAÑA 3: PRECIOS
# =============================================================================
with tab3:
    st.subheader("¿A qué precio vender?")
    e = elasticidades(pm, cat, sel)
    if e["beta_usada"].isna().all():
        st.warning("No hay suficiente variación de precio para estimar la elasticidad.")
        st.stop()

    c1, c2 = st.columns([1, 1.25])
    with c1:
        ee = e.dropna(subset=["beta"]).sort_values("beta", ascending=False)
        fig = go.Figure()
        fig.add_scatter(
            x=ee["beta"], y=ee["product_id"], mode="markers",
            marker=dict(size=12, color=[COLOR["Base"] if p < 0.10 else "#adb5bd"
                                        for p in ee["p"]]),
            error_x=dict(type="data", symmetric=False, array=ee["ic_alto"] - ee["beta"],
                         arrayminus=ee["beta"] - ee["ic_bajo"], color="#a5d8ff"),
            hovertemplate="%{y}<br>Elasticidad: %{x:.2f}<extra></extra>", showlegend=False)
        fig.add_vline(x=-1, line_dash="dash", line_color="#868e96",
                      annotation_text="−1: límite elástico", annotation_position="top left")
        fig.update_layout(title="Elasticidad precio por producto (IC 95%)",
                          height=120 + 45 * len(ee), xaxis_title="β  (gris = no significativa)",
                          **LAYOUT)
        st.plotly_chart(fig, width="stretch")
        st.caption("Modelo log-log por producto: log(Q) = α + β·log(P), donde β es la "
                   "elasticidad. Un descuento sube el margen solo si |β| > 1 / margen.")
    with c2:
        tab_e = e.rename(columns={"product_id": "Producto"})
        st.dataframe(
            tab_e[["Producto", "beta", "p", "beta_usada", "origen", "precio", "margen",
                   "umbral"]],
            hide_index=True, width="stretch", column_config={
                "beta": st.column_config.NumberColumn("Elasticidad", format="%.2f"),
                "p": st.column_config.NumberColumn("p-valor", format="%.3f"),
                "beta_usada": st.column_config.NumberColumn("β usada", format="%.2f"),
                "origen": "Origen β",
                "precio": st.column_config.NumberColumn("Precio efectivo", format="$%,.0f"),
                "margen": st.column_config.NumberColumn("Margen", format="percent"),
                "umbral": st.column_config.NumberColumn("Umbral 1/margen", format="%.2f")})
        rentable = (e["beta_usada"].abs() > e["umbral"]).sum()
        st.info(f"**{rentable} de {len(e)}** productos tienen una elasticidad mayor a su "
                "umbral, es decir, solo en esos un descuento aumentaría el margen.")

    st.markdown("---")
    c1, c2 = st.columns([1, 1.3])
    with c1:
        delta = st.slider("Simular un cambio de precio (%)", -30, 30, -15, step=5) / 100
        q0 = {s: fut[s].sum() for s in ESCENARIOS}  # unidades proyectadas por producto
        filas = []
        for s in ESCENARIOS:
            a, b_ = simular_precio(e, q0[s], 0), simular_precio(e, q0[s], delta)
            filas.append({"Escenario": s, "Unidades": b_["unidades"],
                          "Δ unidades": b_["unidades"] / a["unidades"] - 1,
                          "Ingreso": b_["ingreso"], "Δ ingreso": b_["ingreso"] / a["ingreso"] - 1,
                          "Margen bruto": b_["margen"],
                          "Δ margen": b_["margen"] / a["margen"] - 1})
        sim = pd.DataFrame(filas)
        st.dataframe(sim[["Escenario", "Unidades", "Ingreso", "Margen bruto"]],
                     hide_index=True, width="stretch", column_config={
            "Unidades": st.column_config.NumberColumn(format="%,.0f"),
            "Ingreso": st.column_config.NumberColumn(format="$%,.0f"),
            "Margen bruto": st.column_config.NumberColumn(format="$%,.0f")})
        dm, du, di = sim.loc[sim["Escenario"] == esc_sel,
                             ["Δ margen", "Δ unidades", "Δ ingreso"]].iloc[0]
        (st.error if dm < 0 else st.success)(
            f"Con {delta:+.0%} en el precio: unidades {du:+.1%}, ingreso {di:+.1%} y "
            f"**margen {dm:+.1%}** (igual en los tres escenarios, en porcentaje). "
            f"El inventario tendría que ajustarse {du:+.1%}.")
    with c2:
        deltas = np.arange(-20, 21, 5) / 100
        fig = go.Figure()
        mejores = {}
        for s in ESCENARIOS:
            ms = [simular_precio(e, q0[s], d)["margen"] for d in deltas]
            i = int(np.argmax(ms))
            mejores[s] = (deltas[i], ms[i], ms[list(deltas).index(0)])
            fig.add_scatter(x=deltas * 100, y=ms, name=s, mode="lines+markers",
                            line=dict(color=COLOR[s], width=3.5 if s == esc_sel else 1.6))
            fig.add_scatter(x=[deltas[i] * 100], y=[ms[i]], mode="markers",
                            marker=dict(symbol="star", size=16, color=COLOR[s]),
                            showlegend=False, hoverinfo="skip")
        fig.add_vline(x=0, line_dash="dot", line_color="#adb5bd")
        fig.update_layout(title=f"Margen bruto proyectado ({h} meses) según el cambio de precio",
                          xaxis_title="Cambio de precio (%)", yaxis_title="USD", height=380,
                          **LAYOUT)
        st.plotly_chart(fig, width="stretch")

    d_b, m_b, m0_b = mejores[esc_sel]
    if m_b / m0_b - 1 < 0.01:
        st.info(f"**Recomendación (escenario {esc_sel.lower()}): mantener el precio.** El mejor cambio "
                f"evaluado ({d_b:+.0%}) mejora el margen menos de 1%, una diferencia que "
                "está dentro del error del modelo.")
    else:
        st.success(
          f"**Recomendación (escenario {esc_sel.lower()}):** un cambio de precio de **{d_b:+.0%}** "
          f"maximiza el margen dentro del rango evaluado (±20%): {usd(m_b)} frente a "
          f"{usd(m0_b)} sin cambios ({m_b / m0_b - 1:+.1%}). "
          f"En los tres escenarios de demanda la mejor opción es "
          f"{', '.join(f'{s.lower()} {mejores[s][0]:+.0%}' for s in ESCENARIOS)}.")
    st.caption("Limitaciones: el precio solo varía por los descuentos, así que la elasticidad "
               "es una asociación, no un efecto causal. Fuera de ±20% el modelo extrapola. "
               "Se usa la elasticidad propia si es significativa (p < 0,10); si no, la "
               "mediana del grupo.")

# =============================================================================
# TABLA COMPARATIVA DE ESCENARIOS (se muestra en la pestaña Resumen)
# =============================================================================
with caja_escenarios:
    filas = []
    for s_ in ESCENARIOS:
        g = detalle[s_]
        d_, m_, m0_ = mejores[s_]
        filas.append({
            "Escenario": ("▶ " if s_ == esc_sel else "") + s_,
            "Demanda proyectada": float(fut[s_].values.sum()),
            "Stock estimado (prom.)": g["estimado"].sum(),
            "Stock recomendado (prom.)": g["recomendado"].sum(),
            "Exceso (+) / faltante (−)": g["brecha_usd"].sum(),
            "Productos en riesgo de quiebre": int((g["estado"] == "🔴 Riesgo de quiebre").sum()),
            "Margen sin cambiar precio": m0_,
            "Mejor cambio de precio": f"{d_:+.0%}" if m_ / m0_ - 1 >= 0.01 else "Mantener",
            "Margen con ese precio": m_,
        })
    st.markdown(f"**Los 3 escenarios de un vistazo** · demanda {err_txt} · "
                f"{h} meses · {len(sel)} producto(s). ▶ = escenario seleccionado")
    st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch", column_config={
        "Demanda proyectada": st.column_config.NumberColumn("Demanda (u.)", format="%,.0f"),
        "Stock estimado (prom.)": st.column_config.NumberColumn(
            "Stock estimado", format="%,.0f", help="Stock promedio mensual con la práctica actual."),
        "Stock recomendado (prom.)": st.column_config.NumberColumn(
            "Stock recomendado", format="%,.0f"),
        "Exceso (+) / faltante (−)": st.column_config.NumberColumn(
            "Exceso / faltante", format="$%,.0f"),
        "Productos en riesgo de quiebre": st.column_config.NumberColumn("En riesgo"),
        "Margen sin cambiar precio": st.column_config.NumberColumn(
            "Margen actual", format="$%,.0f"),
        "Mejor cambio de precio": "Mejor precio",
        "Margen con ese precio": st.column_config.NumberColumn(
            "Margen óptimo", format="$%,.0f"),
    })
