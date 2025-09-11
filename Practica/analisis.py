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

# Analisis de datos faltantes

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
axes[0].set_title("Outliers detected by Z-score method (|Z| > 3)", fontweight="bold")
axes[0].set_ylabel("Number of outliers")
axes[0].tick_params(axis="x", rotation=45)

# IQR
iqr_counts = [len(iqr_outliers[i]) for i in range(len(df.columns) - 1)]
axes[1].bar([df[i + 1].iloc[0] for i in range(len(isotope_columns))], iqr_counts)
axes[1].set_title("Outliers detected by IQR method", fontweight="bold")
axes[1].set_ylabel("Number of outliers")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()

# Inconsistencias o codificacion ambigua

# Revisamos si hay fechas duplicadas
duplicate_years = df[0].iloc[10:].duplicated()
if duplicate_years.any():
    print(f"Warning: {duplicate_years.sum()} duplicate years found!")
    print(df["Year CE"][duplicate_years].values)
else:
    print("No duplicate years found.")

# Imprimimos el rango de fechas disponibles para cada localizacion
print("\nYear range for each site:")

for i in range(len(isotope_columns)):
    # Remove NA values for calculation
    col = isotope_columns[i]
    data = pd.DataFrame(col.dropna(), dtype="float")

    if len(data) > 0:
        years_with_data = df[0].iloc[data.index]
        print(
            f"{df[i].iloc[0]}: {years_with_data.min()} - {years_with_data.max()} "
            + f"({len(data)} years of data)"
        )

# Imprimimos el numero de diferentes codigos de zonas
print(f"\nNumber of different codes: {len(df.iloc[:, 1:].iloc[0].unique())}")

# Imprimimos el numero de diferentes paises
print(f"\nNumber of different countries: {len(df.iloc[:, 1:].iloc[2].unique())}")

# Imprimimos el numero de diferentes especies
print(f"\nNumber of different species: {len(df.iloc[:, 1:].iloc[5].unique())}")

# 3. IMPUTACION DE DATOS

# Reemplazo con media
df_mean_imput = df_isotopes.fillna(df_isotopes.mean())

# Reemplazo con interpolacion lineal
df_linear_imput = df_isotopes.interpolate(method="linear")

# Para guardarlos en archivos de Excel
# df_mean_imput.to_excel("meanimputation.xlsx")
# df_linear_imput.to_excel("linearimputation.xlsx")


# 4. Codificacion y escalamiento

# 5. Visualizacion exploratoria

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

ax.set_title(f"Histograma y densidad kernel: {sitename}")

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


plt.title(f"Posibles outlierss en {sitename} con RIQ")
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

# %%
