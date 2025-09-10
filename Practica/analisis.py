# %%
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# %%

# Establecemos el working directory
os.chdir(".\\Practica\\Data\\2023-Data")

# Lectura de datos
df = pd.read_excel("data.xlsx", header=None)
print(df)
# Se quitaron columnas que no aportaban informacion
df1 = df.drop([0, 1, 2, 3, 4, 5, 6, 7, 8])
print(df1)


# SE TOMAN LAS COLUMNAS DEL PRIMER AÑO AL ULTIMO AÑO Y SE TRANSPONEN
df2 = df.iloc[[6, 7]].T
# Se vuelve indexar
df2.columns = df2.iloc[0]  # asigna la primera fila como nombres de columnas
df2 = df2[
    1:
]  # TOMA A PARTIR DE EL SEGUNDO RENGLON POR QUE CON LA LINEA ANTERIOR SE CLONO EL PRIMERO
df2 = df2.reset_index(drop=True)  # Se resetan los indices
df2.index.name = None  # Se quita el nombre del indice
print(df2)

# SE ACOMODAN LOS INDICES DE LAS COLUMNAS SELECCIONADAS
print(df1)
df1.columns = df1.iloc[0]  # asigna la primera fila como nombres de columnas
df1 = df1[
    1:
]  # TOMA A PARTIR DE EL SEGUNDO RENGLON POR QUE CON LA LINEA ANTERIOR SE CLONO EL PRIMERO
df1 = df1.reset_index(drop=True)  # Se resetan los indices
df1.index.name = None  # Se quita el nombre del indice
print(df1)
# Se concatenaron los df para tener los nombres de las columnas correctos
df = pd.concat([df2, df1], axis=1)
print(df)

# Ya que hay columnas repetidas, se renombran


def make_unique_columns(df):
    """
    Renombra columnas duplicadas en un DataFrame agregando sufijos .1, .2, etc.
    """
    cols = pd.Series(df.columns, dtype="string")
    for dup in cols[cols.duplicated()].unique():
        dup_idx = cols[cols == dup].index.tolist()
        for i, idx in enumerate(dup_idx):
            if i == 0:
                continue  # el primero se queda igual
            cols[idx] = f"{dup}.{i}"
    df.columns = cols
    return df


df = make_unique_columns(df)
print(df)

# Se analiza el tipo de datos que tiene el df
print(df.dtypes)
df.info()


# =====================================
# Vuelvo a importar. Hay que ponerse de acuerdo con los nombres de variables
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
print(f"\nNumber of different codes: {len(df[1:].iloc[0].unique())}")

# Imprimimos el numero de diferentes paises
print(f"\nNumber of different countries: {len(df[1:].iloc[2].unique())}")

# Imprimimos el numero de diferentes especies
print(f"\nNumber of different species: {len(df[1:].iloc[5].unique())}")

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

columnnumber = 5  # Del 0 al 24
columncode = df_isotopes.columns[columnnumber]
sitename = df.iloc[1][columnnumber + 1]

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Histograma
sns.histplot(df_isotopes[columncode].dropna(), kde=False, ax=axes[0], color="skyblue")
axes[0].set_title(f"Histograma: {sitename}")

# Densidad kernel
sns.kdeplot(df_isotopes[columncode].dropna(), ax=axes[1], color="green", fill=True)
axes[1].set_title(f"Densidad kernel: {sitename}")

plt.show()

# Histograma con densidad estimada encima

fig, ax = plt.subplots(figsize=(15, 6))

sns.histplot(
    df_isotopes[columncode].dropna(), kde=True, ax=ax, color="skyblue", stat="density"
)

ax.set_title(f"Histograma y densidad kernel: {sitename}")

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
plt.ylabel("Carbono")
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

plt.title(f"Possible outliers on {sitename} with Z-score")
plt.scatter(x_no_outlier, y_no_outlier, c="blue", alpha=0.6)
plt.scatter(
    x_outlier,
    y_outlier,
    facecolors="red",
    edgecolors="r",
    alpha=0.6,
    label="Possible outliers",
)
plt.ylabel(f"{sitename}")
plt.xlabel("Carbón")
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


plt.title(f"Possible outliers on {sitename} with IQR")
plt.scatter(x_no_outlier, y_no_outlier, c="blue", alpha=0.6)
plt.scatter(
    x_outlier,
    y_outlier,
    facecolors="red",
    edgecolors="r",
    alpha=0.6,
    label="Possible outliers",
)
plt.ylabel(f"{sitename}")
plt.xlabel("Carbón")
plt.legend()
plt.show()
