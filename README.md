# E-commerce: demanda, inventario y precios

App de Streamlit que integra las tres dimensiones del proyecto de Modelamiento Predictivo de Datos:

1. **Demanda**: Holt-Winters, Random Forest con rezagos y naive estacional. Se valida con los últimos 12 meses (MAPE) y se proyectan 3 escenarios (pesimista, base y optimista, ± error).
2. **Inventario**: una regresión `stock ~ demanda + producto + mes` estima el stock con la demanda proyectada, y se compara con una política recomendada (stock de seguridad, punto de reorden y nivel de servicio).
3. **Precios**: elasticidad log-log por producto. Se simulan cambios de precio sobre la demanda proyectada de cada escenario.

## Correr en tu computadora

```
pip install -r requirements.txt
streamlit run app.py
```

Si usas Anaconda y el comando `streamlit` falla, usa `python -m streamlit run app.py`.

La carpeta `data/` ya trae las 4 tablas originales comprimidas (`dataset_ecommerce.zip`), así que la app abre con datos sin subir nada. Desde la barra lateral también puedes subir tus propios CSV o un .zip con ellos.

Tablas que usa: `ecommerce_sales_customer_analytics_150k.csv`, `order_items.csv`, `product_catalog.csv` e `inventory_stock.csv`. Si subes `customer_master.csv` o `dataset_statistics.csv`, se ignoran.

## Publicar (GitHub + Streamlit Community Cloud)

1. Crea un repositorio en GitHub y sube **todo el contenido de esta carpeta**, incluidas `data/` y `.streamlit/`. El zip pesa ~23 MB, por debajo del límite de 25 MB de la subida web de GitHub.
2. Entra a https://share.streamlit.io con tu cuenta de GitHub → **Create app** → elige el repositorio, la rama `main` y el archivo `app.py` → **Deploy**.
3. Copia la URL pública (`https://<nombre>.streamlit.app`) y genera el QR con cualquier generador de QR.

La primera carga tarda unos segundos porque lee los CSV. Después queda en caché.

## Opcional: Prophet

Si agregas `prophet` a `requirements.txt`, la app lo detecta y lo suma como cuarto modelo. No viene incluido porque hace la instalación más lenta y pesada en Streamlit Cloud.
