import pandas as pd
import os
from pathlib import Path

def cargar_y_limpiar_dataset():
        # Ver si hay carpeta csv
    ruta_csv = Path("csv")
    if not ruta_csv.exists():
        print("ERROR: No existe la carpeta 'csv'")
        print("Archivos en carpeta actual:")
        for item in Path(".").iterdir():
            print(f"  - {item.name}")
        return None
    
    print("OK Carpeta 'csv' encontrada")
    
    # Buscar archivos CSV en la carpeta csv
    archivos_csv = list(ruta_csv.glob("*.csv"))
    print(f"Archivos CSV encontrados: {len(archivos_csv)}")
    
    if not archivos_csv:
        print("ERROR: No se encontraron archivos CSV")
        print("Archivos en carpeta 'csv':")
        for item in ruta_csv.iterdir():
            print(f"  - {item.name}")
        return None
    
    dataframes = []
    
    # Procesar cada archivo CSV
    for archivo_csv in archivos_csv:
        try:
            print(f"Leyendo: {archivo_csv.name}")
            
            # Leer el CSV
            df = pd.read_csv(archivo_csv)
            
            # El nombre del archivo es el artista
            nombre_artista = archivo_csv.stem  # para quitar la extensión 
        
            
        except Exception as e:
            print(f"   ERROR en {archivo_csv.name}: {e}")
    
    if not dataframes:
        print("ERROR: No se pudieron leer archivos CSV")
        return None
    
    # Unir todos los DataFrames
    df_completo = pd.concat(dataframes, ignore_index=True)
    print(f"DATASET CRUDO: {len(df_completo)} canciones cargadas")
    print(f"COLUMNAS: {df_completo.columns.tolist()}")
    
    return df_completo

def limpiar_datos(df_completo):
    # Buscar columna de letras
    columna_lyric = None
    for col in df_completo.columns:
        if 'lyric' in col.lower():
            columna_lyric = col
            break
    
    if not columna_lyric:
        print("no se encuentra columna de letras")
        return None
    
    print(f"Columna de letras: '{columna_lyric}'")
    
    # 1. Eliminar canciones sin letra
    inicial = len(df_completo)
    df_limpio = df_completo.dropna(subset=[columna_lyric])
    print(f"OK Sin letras vacias: {len(df_limpio)} (eliminadas: {inicial - len(df_limpio)})")
    
    # 2. Filtrar letras muy cortas
    inicial = len(df_limpio)
    df_limpio = df_limpio[df_limpio[columna_lyric].str.len() >= 100]
    print(f"OK Sin letras cortas: {len(df_limpio)} (eliminadas: {inicial - len(df_limpio)})")
    
    # 3. Eliminar duplicados
    if 'Artist' in df_limpio.columns and 'Title' in df_limpio.columns:
        inicial = len(df_limpio)
        df_limpio = df_limpio.drop_duplicates(subset=['Artist', 'Title'])
        print(f"OK Sin duplicados: {len(df_limpio)} (eliminados: {inicial - len(df_limpio)})")
    
    # 4. Limpieza de texto
    def limpiar_texto(texto):
        if pd.isna(texto):
            return ""
        texto = str(texto)
        texto = texto.replace('\r', ' ').replace('\n', ' ')
        texto = ' '.join(texto.split())
        return texto.strip()
    
    df_limpio[columna_lyric] = df_limpio[columna_lyric].apply(limpiar_texto)
    
    # 5. Renombrar columnas
    df_final = df_limpio.rename(columns={columna_lyric: 'Lyric'})
    
    # 6. Seleccionar columnas finales
    columnas_finales = ['Lyric']
    if 'Artist' in df_final.columns:
        columnas_finales.append('Artist')
    if 'Title' in df_final.columns:
        columnas_finales.append('Title')
    
    df_final = df_final[columnas_finales]
    
    print(f"DATASET LIMPIO: {len(df_final)} canciones listas")
    return df_final

def crear_dataset_completo(df_limpio):
    # Usar todas las canciones disponibles
    print(f"U: {len(df_limpio)}")
    
    if 'Artist' in df_limpio.columns:
        artistas = df_limpio['Artist'].nunique()
        print(f"De {artistas} artistas diferentes")
        
        #  distribución por artista
        print("DISTRIBUCION POR ARTISTA:")
        distribucion = df_limpio['Artist'].value_counts().head(10)
        for artista, count in distribucion.items():
            print(f"  - {artista}: {count} canciones")
    
    return df_limpio

def guardar_resultado(df):    
    # Crear carpeta 
    os.makedirs("data/processed", exist_ok=True)
    
    # Guardar 
    ruta_guardado = "data/processed/dataset_canciones.csv"
    df.to_csv(ruta_guardado, index=False, encoding='utf-8')
    
    print(f"ARCHIVO GUARDADO: {ruta_guardado}")
    
    # Estadisticas
    print(f"ESTADISTICAS FINALES:")
    print(f"  - Canciones totales: {len(df)}")
    
    if 'Artist' in df.columns:
        artistas = df['Artist'].nunique()
        print(f"  - Artistas unicos: {artistas}")
    
    # Mostrar algunos ejemplos
    print(f"\nEJEMPLOS DE CANCIONES:")
    for i in range(min(3, len(df))):
        titulo = df.iloc[i]['Title'] if 'Title' in df.columns else "Sin titulo"
        artista = df.iloc[i]['Artist'] if 'Artist' in df.columns else "Desconocido"
        letra_preview = df.iloc[i]['Lyric'][:80] + "..." if len(df.iloc[i]['Lyric']) > 80 else df.iloc[i]['Lyric']
        
        print(f"  {i+1}. '{titulo}'")
        print(f"     Artista: {artista}")
        print(f"     Letra: '{letra_preview}'")
        print()

# EJECUCION PRINCIPAL 
if __name__ == "__main__":    
    try:
        # 1. Cargar datos
        df_completo = cargar_y_limpiar_dataset()
        if df_completo is None:
            raise Exception("No se pudieron cargar los datos")
        
        # 2. Limpiar datos
        df_limpio = limpiar_datos(df_completo)
        if df_limpio is None or len(df_limpio) == 0:
            raise Exception("No hay datos")
        
        # 3. Usar TODAS las canciones (cambio principal)
        df_final = crear_dataset_completo(df_limpio)
        
        # 4. Guardar
        guardar_resultado(df_final)
        
    except Exception as e:
        print(f"ERROR: {e}")
    
    input("\nSalir...")