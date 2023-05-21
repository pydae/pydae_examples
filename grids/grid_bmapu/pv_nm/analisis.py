import pandas as pd
import numpy as np
filename='inv10(3)withoutsetpoint_50ms.csv'
df = pd.read_csv(filename)
df['tiempo_ejecucion'] = df['tiempo_ejecucion']/1000000000
print(df['tiempo_ejecucion'].describe())
df_nuevo = df[df['tiempo_ejecucion'] > 0.05].dropna()
print(df_nuevo['tiempo_ejecucion'].size)