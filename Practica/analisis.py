# %%
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# %%

# Establecemos el working directory
os.chdir(".\\Data\\2023-Data")

# %%

# Lectura de datos
df = pd.read_excel("data.xlsx", header=None)

# =================================
# 2. Analisis de datos faltantes
# =================================

missing_count = df.isna().sum()
missing_percent = (missing_count / len(df)) * 100
missing_summary = pd.DataFrame(
    {"Valores faltantes": missing_count, "Porcentaje": missing_percent.round(2)}
)

print("\n--- Datos faltantes por columna ---")
print(missing_summary)


# Deteccion de outliers

# Para reemplazar los NA + espacio(s) por NA
df = df.replace("NA", "NaN")
df = df.replace("NA ", "NaN")
df = df.replace("NA  ", "NaN")

# Seleccionamos solamente los datos de 13CVPDB para cada columna
isotope_columns = [df[df.columns[i]].iloc[10:] for i in range(1, len(df.columns))]

df_isotopes = pd.DataFrame(isotope_columns).T
df_isotopes = pd.DataFrame(df_isotopes, dtype="float")
df_isotopes = df_isotopes.set_axis(df.iloc[0][1:], axis=1)
df_isotopes = df_isotopes.set_index(df[0].iloc[10:])
print(df_isotopes)
# Valores no registrados en los años de medicion
for i in range(len(isotope_columns)):
    col = isotope_columns[i]
    missing_count = col.isna().sum()
    data = pd.DataFrame(col.dropna(), dtype="float")
    # Seleccionar los años con datos
    years_with_data = df[0].iloc[data.index]
    years_data = 1 + years_with_data.max() - years_with_data.min()
    missing_data = years_data - len(data)
    missing_percent = (missing_data / years_data) * 100
    print(
        df[1 + i].iloc[0],
        ": ",
        round(missing_percent, 2),
        " percent missing",
        years_data,
        " years of data ",
        missing_data,
        " years missing",
    )

# Tests de normalidad para cada locacion
for col_name in df_isotopes:
    shapiro_test = stats.shapiro(df_isotopes[col_name].dropna())
    if shapiro_test.pvalue < 0.05:
        print(col_name, " se descarta normalidad con p value ", shapiro_test.pvalue)
    else:
        print(col_name, " no hay evidencia contra la hipótesis nula.")

# Metodo 1: Z-score
print("\nOutlier detection using Z-score method (|Z| > 3):")
zscore_outliers = {}

for i in range(len(isotope_columns)):
    # Eliminamos los NA para los calculos
    col = isotope_columns[i]
    data = pd.DataFrame(col.dropna(), dtype="float")

    # Calculamos los z-scores y seleccionamos los posibles outliers
    z_scores = np.abs(stats.zscore(data))
    outliers = data[z_scores > 3]
    outliers = pd.DataFrame(outliers).dropna()
    zscore_outliers[i] = []

    # Imprimimos los posibles outliers (si hay), y las fechas a las que corresponden
    if len(outliers) > 0:
        outliersyears = df[0].iloc[outliers.index].values
        zscore_outliers[i] = list(outliersyears)
        print(f"{df[i].iloc[0]}: {len(outliers)} outliers")
        print(f"  Outlier values:\n{outliers}")
        print(f"  Outlier years: {outliersyears}")

# Imprime scattler plot con regresion lineal y posibles outliers usando hatmatrix
for columncode in df_isotopes.columns:
    # Extraer datos (quitando NA)
    xd = df_isotopes[columncode].dropna()
    X = xd.index.to_numpy(dtype=float)
    y = xd.to_numpy(dtype=float)
    # Añadir intercepto
    X1 = np.column_stack([np.ones(X.shape[0]), X])
    # Calcular beta con mínimos cuadrados
    beta_hat, *_ = np.linalg.lstsq(X1, y, rcond=None)
    # Calcular matriz sombrero H
    H = X1 @ np.linalg.inv(X1.T @ X1) @ X1.T
    leverages = np.diag(H)
    # Regla práctica de corte
    n, p = X1.shape
    threshold = 2 * p / n
    # --- Visualización ---
    plt.figure(figsize=(8, 5))
    # Resaltar puntos con leverage alto
    outliers = leverages > threshold
    plt.scatter(
        X[np.invert(outliers)],
        y[np.invert(outliers)],
        c="blue",
        alpha=0.6,
        label="Datos",
    )
    plt.scatter(X[outliers], y[outliers], c="red", alpha=0.6, label="Posible outlier")
    plt.plot(X, X1 @ beta_hat, c="red", label="Recta ajustada")
    plt.title(
        f"Relación entre año y carbono en {columncode} con posibles outliers (Hat Matrix)"
    )
    plt.xlabel("Año")
    plt.ylabel("δ¹³C (‰, VPDB)")
    plt.legend()
    plt.show()

# Metodo 2: IQR rango intercuantilico
print("\nOutlier detection using IQR method:")
iqr_outliers = {}

for i in range(len(isotope_columns)):
    # Eliminamos los NA para los calculos
    col = isotope_columns[i]
    data = pd.DataFrame(col.dropna(), dtype="float")

    # Calculamos el rango intercuartilico y seleccionamos los posibles outliers
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = data[(data < lower_bound) | (data > upper_bound)]
    outliers = pd.DataFrame(outliers).dropna()
    iqr_outliers[i] = []

    # Imprimimos los posibles outliers (si hay), y las fechas a las que corresponden
    if len(outliers) > 0:
        outliersyears = df[0].iloc[outliers.index].values
        iqr_outliers[i] = list(outliersyears)
        print(f"{df[i].iloc[0]}: {len(outliers)} outliers")
        print(f"  Outlier values:\n {outliers.values}")
        print(f"  Outlier years: {outliersyears}")

# Visualizamos los outliers posibles
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Z-score
zscore_counts = [len(zscore_outliers[i]) for i in range(len(df.columns) - 1)]
axes[0].bar([df[i + 1].iloc[0] for i in range(len(isotope_columns))], zscore_counts)
axes[0].set_title("Outliers detectados por Z-score (|Z| > 3)", fontweight="bold")
axes[0].set_ylabel("Número de outliers")
axes[0].tick_params(axis="x", rotation=45)

# IQR
iqr_counts = [len(iqr_outliers[i]) for i in range(len(df.columns) - 1)]
axes[1].bar([df[i + 1].iloc[0] for i in range(len(isotope_columns))], iqr_counts)
axes[1].set_title("Outliers detectados por RIQ", fontweight="bold")
axes[1].set_ylabel("Número de outliers")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()

# Inconsistencias o codificacion ambigua
duplicate_years = df[0].iloc[10:].duplicated()
if duplicate_years.any():
    print(f"{duplicate_years.sum()} fechas duplicadas encontradas")
    print(df["Year CE"][duplicate_years].values)
else:
    print("No hay fechas duplicadas.")


# Imprimimos el numero de diferentes codigos de zonas
print(f"\nNumber of different codes: {len(df.iloc[:, 1:].iloc[0].unique())}")

# Imprimimos el numero de diferentes paises
print(f"\nNumber of different countries: {len(df.iloc[:, 1:].iloc[2].unique())}")

# Imprimimos los distintos paises que hay
print(f"\nDistinct countries: {df.iloc[:, 1:].iloc[2].unique()}")

# Imprimimos el numero de diferentes especies
print(f"\nNumber of different species: {len(df.iloc[:, 1:].iloc[5].unique())}")

# Imprimimos las distintas especies que hay
print(f"\nDistinct species: {df.iloc[:, 1:].iloc[5].unique()}")


# =================================
# 3. IMPUTACION DE DATOS
# =================================

# Porcentaje de datos faltantes por rango de tiempo de cuando iniciaron las mediciones a cuando terminaron
for i in range(len(isotope_columns)):
    col = isotope_columns[i]
    data = col.dropna()
    years_with_data = df[0].iloc[data.index]
    # print(len(years_with_data)  )
    total_years = years_with_data.max() - years_with_data.min() + 1

    missing_years = total_years - data.shape[0]
    missing_percentage = (missing_years / total_years) * 100
    print(
        f"{df[i+1].iloc[0]}: {missing_percentage:.2f}% missing data "
        + f"({missing_years} out of {total_years} years)"
    )


# Graficar cada columna por separado
for col in df_isotopes.columns:
    # Quitar los valores NaN para esa columna
    serie = df_isotopes[col].dropna()

    plt.figure(figsize=(6, 4))
    plt.plot(serie.index, serie.values, linestyle="none", marker="o", label=col)
    plt.title(f"Serie: {col}")
    plt.xlabel("Índice")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid(True)
    plt.show()


# Reemplazo con media
df_mean_imput = df_isotopes.copy()

for nameip in df_isotopes.columns:
    col = df_isotopes[nameip]
    data = col.dropna()
    years_with_data = data.index

    # Seleccionar los años dentro del rango de datos y que son NaN
    missing_years = df_isotopes.index[
        df_isotopes.index.isin(range(years_with_data.min(), years_with_data.max() + 1))
        & df_isotopes[nameip].isna()
    ]

    # Imputar con la media de la columna solo en los años faltantes
    df_mean_imput.loc[missing_years, nameip] = df_isotopes[nameip].mean()
    print(len(df_mean_imput[nameip].dropna()))

for nameip in df_isotopes.columns:
    col_original = df_isotopes[nameip]
    col_imputed = df_mean_imput[nameip]

    # Índices de los imputados
    imputed_idx = col_original[col_original.isna()].index

    plt.figure(figsize=(10, 4))

    # Graficar la serie completa imputada
    plt.plot(
        col_imputed.index, col_imputed.values, label="Serie imputada", color="blue"
    )

    # Resaltar los puntos imputados con la media
    plt.scatter(
        imputed_idx,
        col_imputed.loc[imputed_idx],
        color="red",
        marker="o",
        label="Imputado (media)",
    )

    plt.title(f"Serie {nameip} con imputación por media")
    plt.xlabel("Año")
    plt.ylabel("Valor")
    plt.legend()
    plt.tight_layout()
    plt.show()


# Reemplazo con interpolacion lineal
df_interp_imput = df_isotopes.copy()

for nameip in df_isotopes.columns:
    col = df_isotopes[nameip]
    data = col.dropna()
    years_with_data = data.index

    # Solo interpolar dentro del rango válido de años
    mask = df_isotopes.index.isin(
        range(years_with_data.min(), years_with_data.max() + 1)
    )

    # Interpolar únicamente en ese rango
    df_interp_imput.loc[mask, nameip] = col.loc[mask].interpolate(method="linear")

    print(len(df_interp_imput[nameip].dropna()))

# Graficar resultados
for nameip in df_isotopes.columns:
    col_original = df_isotopes[nameip]
    col_imputed = df_interp_imput[nameip]

    # Índices de los imputados
    imputed_idx = col_original[col_original.isna()].index

    plt.figure(figsize=(10, 4))

    # Graficar la serie completa imputada
    plt.plot(
        col_imputed.index,
        col_imputed.values,
        label="Serie imputada (interpolación)",
        color="blue",
    )

    # Resaltar los puntos imputados con interpolación
    plt.scatter(
        imputed_idx,
        col_imputed.loc[imputed_idx],
        color="red",
        marker="o",
        label="Imputado (interp.)",
    )

    plt.title(f"Serie {nameip} con imputación por interpolación lineal")
    plt.xlabel("Año")
    plt.ylabel("Valor")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Reemplazo con simulación normal

from scipy.stats import anderson  # Prueba de normalidad Anderson-Darling

# Creamos un arreglo vacío para almacenar los estadísticos
# Diccionario para almacenar los resultados
results_dict = {
    "column": [],
    "statistic": [],
    "critical_values": [],
    "significance_levels": [],
}

# Iteramos por cada columna
for col in df_isotopes.columns:
    data = df_isotopes[col].dropna()  # Eliminamos NaN
    result = anderson(data)

    results_dict["column"].append(col)
    results_dict["statistic"].append(result.statistic)
    results_dict["critical_values"].append(result.critical_values)
    results_dict["significance_levels"].append(result.significance_level)

# Convertimos el diccionario a DataFrame
results_df = pd.DataFrame(results_dict)


print(results_df)

# IMPUTAMOS CON EN LAS COLUMNAS QUE SEA POSIBLE MEDIANTE UNA NORMAL.
plt.ion()
df_isotopes2 = df_isotopes.copy()

for cv, sig, nameip in zip(
    results_df.critical_values, results_df.statistic, results_df.column
):
    print(f"{nameip}. Estadístico {sig} -> Valor crítico: {cv[2]}")

    if sig < cv[2]:
        print("No se rechaza normalidad (al 5%)")

        # Columna actual
        col = df_isotopes[nameip]
        data = col.dropna()

        # Calcular parámetros de la distribución normal
        mu, sigma = data.mean(), data.std()

        # Identificar años faltantes dentro del rango observado
        years_with_data = data.index
        missing_years = df_isotopes.index[
            df_isotopes.index.isin(
                range(years_with_data.min(), years_with_data.max() + 1)
            )
            & df_isotopes[nameip].isna()
        ]

        # Generar valores imputados con normal(mu, sigma)
        np.random.seed(42)
        imputed_values = np.random.normal(loc=mu, scale=sigma, size=len(missing_years))

        # Reemplazar en el DataFrame
        df_isotopes2.loc[missing_years, nameip] = imputed_values

        # --- GRAFICO COMPARATIVO ---
        plt.figure(figsize=(8, 5))
        plt.hist(
            col.dropna(),
            bins=20,
            color="black",
            edgecolor="black",
            alpha=0.6,
            label="Antes de imputar",
        )
        plt.hist(
            df_isotopes2[nameip].dropna(),
            bins=20,
            color="salmon",
            edgecolor="black",
            alpha=0.6,
            label="Después de imputar",
        )
        plt.title(f"{nameip} - Comparación antes vs después")
        plt.xlabel("Valor δ13C (‰ VPDB)")
        plt.ylabel("Frecuencia")
        plt.legend()
        plt.show()

    else:
        print("Se rechaza normalidad (al 5%)")


# ===============================
# 4. Codificacion y escalamiento
# ===============================

# Codificacion de paises
country_names = pd.get_dummies(df.iloc[2, 1:], drop_first=False)
print(country_names)
# Codificacion de especies
species_names = pd.get_dummies(df.iloc[5, 1:], drop_first=False)
print(species_names)

# Escalamiento de variables geograficas: altitud, latitud y longitud
from sklearn.preprocessing import StandardScaler, MinMaxScaler

z_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()
df_geographic = (
    df.iloc[[3, 4, 8], 1:]
    .set_axis(df.iloc[0, 1:], axis=1)
    .set_axis(df.iloc[[3, 4, 8], 0], axis=0)
)

df_geographic.iloc[2, 9] = 60  # Correccion de dato faltante con investigado

df_geo_T = df_geographic.T
df_geo_T.index.name = "Site"
df_geo_T.reset_index(inplace=True)
num_cols = ["Latitude", "Longitude", "elevation a.s.l."]

# --- Z-score ---
scaler = StandardScaler()
df_zscore = df_geo_T.copy()
df_zscore[num_cols] = scaler.fit_transform(df_geo_T[num_cols])

# --- Min-max ---
minmax = MinMaxScaler()
df_minmax = df_geo_T.copy()
df_minmax[num_cols] = minmax.fit_transform(df_geo_T[num_cols])

print("\nZ-score:\n", df_zscore.head())
print("\nMin-max:\n", df_minmax.head())


# Para transformar las variables categoricas

mdata = df.iloc[:10]  # filas con info de sitios, species, coordenadas,
data = df.iloc[10:].copy()  # filas con años y mediciones

# La primera columna de data tiene los años reales
data_numeric = data.copy()
data_numeric.columns = ["Year"] + list(
    data_numeric.columns[1:]
)  # renombrar primera columna como Year

#  convertir la columna Year a int
data_numeric["Year"] = pd.to_numeric(data_numeric["Year"], errors="coerce")


# Pasar a formato largo

df_long = data_numeric.melt(
    id_vars=["Year"],  # columna fija (años)
    var_name="Site Code",  # nombres de las columnas originales
    value_name="Valor",  # valores de cada celda
)


# Crear un diccionario Site Code
species_map = dict(
    zip(mdata.iloc[5, 2:], mdata.iloc[5, 2:])
)  # fila 5 contiene especies
df_long["Species"] = df_long["Site Code"].map(species_map)

df_long["Site Code"] = df_long["Site Code"].astype(str)
df_long["Species"] = df_long["Species"].astype(str)
df_dummies = pd.get_dummies(df_long, columns=["Site Code", "Species"], drop_first=False)

dummies_cols = [
    col
    for col in df_dummies.columns
    if col.startswith("Site Code_") or col.startswith("Species_")
]
new_names = {col: col.split("_", 1)[1] for col in dummies_cols}
df_dummies = df_dummies.rename(columns=new_names)

# Convertir a 0/1
df_dummies[list(new_names.values())] = df_dummies[list(new_names.values())].astype(int)
#  mostrar las primeras filas
print(df_dummies.iloc[360:390])

# PARA ESCALAR VARIABLES NUMERICAS
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Columnas que quieres escalar
cols_to_scale = ["WOB ", "WIN", "REN ", "NIE2"]

# Min-Max Scaling (0 a 1)
scaler_minmax = MinMaxScaler()
df_isotopes_minmax = df_isotopes.copy()
df_isotopes_minmax[cols_to_scale] = scaler_minmax.fit_transform(
    df_isotopes_minmax[cols_to_scale]
)

# Standard
scaler_std = StandardScaler()
df_isotopes_std = df_isotopes.copy()
df_isotopes_std[cols_to_scale] = scaler_std.fit_transform(
    df_isotopes_std[cols_to_scale]
)
# Ahora df_isotopes tiene esas columnas escaladas
print(df_isotopes[cols_to_scale].head())
# Columnas escaladas
cols_scaled = ["WOB ", "WIN", "REN ", "NIE2"]


# Graficar Min-Max
for col in cols_to_scale:
    plt.figure(figsize=(10, 4))
    plt.plot(
        df_isotopes_minmax.index,
        df_isotopes_minmax[col],
        color="blue",
        marker="o",
        linestyle="-",
    )
    plt.title(f"{col} - Min-Max Scaled (0 a 1)")
    plt.xlabel("Año")
    plt.ylabel("Valor escalado")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Graficar Standard
for col in cols_to_scale:
    plt.figure(figsize=(10, 4))
    plt.plot(
        df_isotopes_std.index,
        df_isotopes_std[col],
        color="green",
        marker="o",
        linestyle="-",
    )
    plt.title(f"{col} - Standard Scaled (media=0, std=1)")
    plt.xlabel("Año")
    plt.ylabel("Valor escalado")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ================================
# 5. Visualizacion exploratoria
# ================================

columnnumber = 2  # Del 0 al 24
columncode = df_isotopes.columns[columnnumber]
sitename = df.iloc[1][columnnumber + 1]

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Histograma
sns.histplot(df_isotopes[columncode].dropna(), kde=False, ax=axes[0], color="skyblue")
axes[0].set_title(f"Histograma de {sitename}")

axes[0].set_xlabel("δ¹³C (‰, VPDB)")
# Densidad kernel
sns.kdeplot(df_isotopes[columncode].dropna(), ax=axes[1], color="green", fill=True)
axes[1].set_title(f"Densidad kernel estimada de {sitename}")

axes[1].set_xlabel("δ¹³C (‰, VPDB)")

plt.show()

# Histograma con densidad estimada encima

fig, ax = plt.subplots(figsize=(15, 6))

sns.histplot(
    df_isotopes[columncode].dropna(), kde=True, ax=ax, color="skyblue", stat="density"
)

ax.set_title(f"Histograma y densidad kernel de {sitename}")
plt.xlabel("δ¹³C (‰, VPDB)")
plt.show()

# Graficas de dispersion para mismas especies pero de diferentes sitios

# Hacemos un diccionario con las diferentes especies y columnas
different_species = df.iloc[:, 1:].iloc[5].value_counts().to_dict()

for species in different_species:
    subdata_columns = df.columns[df.iloc[5] == species]
    subdata = df.iloc[10:, subdata_columns]
    subdata = subdata.set_axis(df.iloc[0, subdata_columns], axis=1)
    subdata = subdata.set_index(df.iloc[10:, 0])

    plt.figure(figsize=(8, 5))
    for location in subdata.columns:
        # Extraer datos (quitando NA)
        data_withoutna = subdata[location].dropna()
        X = data_withoutna.index.to_numpy(dtype=float)
        y = data_withoutna.to_numpy(dtype=float)
        plt.scatter(X, y, alpha=0.6, label=location)
    plt.title(f"Datos de la especie {species}")
    plt.ylabel("δ¹³C (‰, VPDB)")
    plt.xlabel("Año")
    plt.legend()
    plt.show()


# Relacion entre fecha y carbono (datos imputados con media)

columnnumber = 3
columncode = df_mean_imput.columns[columnnumber]
sitename = df.iloc[1][columnnumber + 1]

# Extraer datos (quitando NA)
imputated_data_withoutna = df_mean_imput[columncode].dropna()
X = imputated_data_withoutna.index.to_numpy(dtype=float)
y = imputated_data_withoutna.to_numpy(dtype=float)

# Añadir intercepto
X1 = np.column_stack([np.ones(X.shape[0]), X])

# Calcular beta con mínimos cuadrados
beta_hat, *_ = np.linalg.lstsq(X1, y, rcond=None)

# Calcular matriz sombrero H
H = X1 @ np.linalg.inv(X1.T @ X1) @ X1.T
leverages = np.diag(H)

# Regla práctica de corte
n, p = X1.shape
threshold = 2 * p / n

# --- Visualización ---
plt.figure(figsize=(8, 5))

# Resaltar puntos con leverage alto
outliers = leverages > threshold

plt.scatter(
    X[np.invert(outliers)], y[np.invert(outliers)], c="blue", alpha=0.6, label="Datos"
)
plt.scatter(X[outliers], y[outliers], c="red", alpha=0.6, label="Posible outlier")
plt.plot(X, X1 @ beta_hat, c="red", label="Recta ajustada")
plt.title(
    f"Relación entre año y carbono en {sitename} con posibles outliers (Hat Matrix)"
)
plt.xlabel("Año")
plt.ylabel("δ¹³C (‰, VPDB)")
plt.legend()
plt.show()

# ========================================================
# Deteccion de outliers o valores extremos por dos metodos

columnnumber = 3
columncode = df_isotopes.columns[columnnumber]
sitename = df.iloc[1][columnnumber + 1]

# Z-score

plt.figure(figsize=(8, 5))
data_without_na = df_isotopes[columncode].dropna()
years = list(data_without_na.index)

x_no_outlier = [year for year in years if not year in zscore_outliers[columnnumber]]
x_outlier = [year for year in years if year in zscore_outliers[columnnumber]]

y_no_outlier = [data_without_na[year] for year in x_no_outlier]
y_outlier = [data_without_na[year] for year in x_outlier]

plt.title(f"Posibles outliers en {sitename} con Z-score")
plt.scatter(x_no_outlier, y_no_outlier, c="blue", alpha=0.6)
plt.scatter(
    x_outlier,
    y_outlier,
    facecolors="red",
    edgecolors="r",
    alpha=0.6,
    label="Posibles outliers",
)
plt.ylabel("δ¹³C (‰, VPDB)")
plt.xlabel("Año")
plt.legend()
plt.show()

# IQR

plt.figure(figsize=(8, 5))
data_without_na = df_isotopes[columncode].dropna()
years = list(data_without_na.index)

x_no_outlier = [year for year in years if not year in iqr_outliers[columnnumber]]
x_outlier = [year for year in years if year in iqr_outliers[columnnumber]]

y_no_outlier = [data_without_na[year] for year in x_no_outlier]
y_outlier = [data_without_na[year] for year in x_outlier]


plt.title(f"Posibles outliers en {sitename} con RIQ")
plt.scatter(x_no_outlier, y_no_outlier, c="blue", alpha=0.6)
plt.scatter(
    x_outlier,
    y_outlier,
    facecolors="red",
    edgecolors="r",
    alpha=0.6,
    label="Posibles outliers",
)
plt.ylabel("δ¹³C (‰, VPDB)")
plt.xlabel("Año")
plt.legend()
plt.show()
