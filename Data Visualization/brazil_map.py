import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import imageio.v2 as imageio
import os
import shutil
import geobr

def create_dual_scale_regional_map_gif():
    """
    Generates a GIF of a choropleth map of Brazil's regions from 2014 onwards,
    using a dual color scale to better visualize variations across all regions.
    """
    print("Iniciando a criação do mapa com escala dupla de cores...")

    # --- 1. Load Geographic and Consumption Data ---
    print("[1/6] Carregando dados geográficos e de consumo...")
    try:
        gdf_states = geobr.read_state(year=2020)
        gdf_regions = gdf_states.dissolve(by='name_region')
        gdf_regions.reset_index(inplace=True)
        gdf_regions.rename(columns={'name_region': 'NM_REGIAO'}, inplace=True)
    except Exception as e:
        print(f"ERRO: Não foi possível carregar dados geográficos via 'geobr'. Causa: {e}")
        return

    try:
        consumption_data = pd.read_csv("Consumo_Mensal.csv", delimiter=',')
    except FileNotFoundError:
        print("ERRO: Arquivo 'Consumo_Mensal.csv' não encontrado.")
        return

    # --- 2. Process and Map Consumption Data ---
    print("[2/6] Processando e mapeando os dados de consumo...")
    consumption_data.columns = [col.replace('Mwmed_', '') for col in consumption_data.columns]
    consumption_data['Data'] = pd.to_datetime(consumption_data['Data'], format='%m/%d/%Y')
    consumption_data = consumption_data[consumption_data['Data'].dt.year >= 2014].copy()
    
    melted_data = consumption_data.melt(
        id_vars=['Data'],
        value_vars=['Nordeste', 'Norte', 'Sudeste/CO', 'Sul'],
        var_name='Regiao_CSV',
        value_name='Consumo'
    )
    melted_data['ano_mes'] = melted_data['Data'].dt.to_period('M')

    def standardize_region_name(name):
        return name.upper().replace('-', ' ').replace('  ', ' ').strip()
    
    gdf_regions['NM_REGIAO_STD'] = gdf_regions['NM_REGIAO'].apply(standardize_region_name)
    region_mapping_dict = {'Nordeste': ['NORDESTE'], 'Norte': ['NORTE'], 'Sul': ['SUL'], 'Sudeste/CO': ['SUDESTE', 'CENTRO OESTE']}
    
    expanded_rows = []
    for _, row in melted_data.iterrows():
        mapped_regions = region_mapping_dict.get(row['Regiao_CSV'], [])
        for region_name in mapped_regions:
            new_row = row.to_dict(); new_row['NM_REGIAO_STD'] = region_name; expanded_rows.append(new_row)
    expanded_df = pd.DataFrame(expanded_rows)
    merged_data = gdf_regions.merge(expanded_df, on='NM_REGIAO_STD', how='left')

    # --- 3. Calculate Separate Color Scales ---
    print("[3/6] Calculando as escalas de cores separadas...")
    seco_consumo = merged_data[merged_data['NM_REGIAO_STD'].isin(['SUDESTE', 'CENTRO OESTE'])]['Consumo']
    outras_consumo = merged_data[~merged_data['NM_REGIAO_STD'].isin(['SUDESTE', 'CENTRO OESTE'])]['Consumo']
    
    norm_seco = Normalize(vmin=seco_consumo.min(), vmax=seco_consumo.max())
    norm_outras = Normalize(vmin=outras_consumo.min(), vmax=outras_consumo.max())
    
    cmap_seco = plt.get_cmap('Blues')
    cmap_outras = plt.get_cmap('YlOrRd')

    # --- 4. Prepare for GIF Generation ---
    print("[4/6] Preparando a geração dos frames do GIF...")
    output_gif_path = 'mapa_consumo_regional_escala_dupla.gif'
    temp_dir = 'temp_regional_map_frames'
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    sorted_months = sorted(merged_data['ano_mes'].dropna().unique())
    image_paths = []

    # --- 5. Generate a Map for Each Month with Dual Scales ---
    for i, month in enumerate(sorted_months):
        print(f"  - Gerando frame {i+1}/{len(sorted_months)}: {month}")
        month_data = merged_data[merged_data['ano_mes'] == month]
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 12))
        ax.axis('off')
        ax.set_title(f'Consumo de Energia por Região (Escala Dupla)\n{month}', fontdict={'fontsize': '20', 'fontweight': '3'})
        
        # Manually plot each region with its corresponding color scale
        for idx, region_row in gdf_regions.iterrows():
            region_name = region_row['NM_REGIAO_STD']
            data_row = month_data[month_data['NM_REGIAO_STD'] == region_name]
            
            color = 'lightgrey' # Default color if no data
            if not data_row.empty:
                consumo_val = data_row['Consumo'].iloc[0]
                if pd.notna(consumo_val):
                    if region_name in ['SUDESTE', 'CENTRO OESTE']:
                        color = cmap_seco(norm_seco(consumo_val))
                    else:
                        color = cmap_outras(norm_outras(consumo_val))
            
            gpd.GeoSeries([region_row['geometry']]).plot(ax=ax, color=color, edgecolor='white', linewidth=1.2)

        # Add dual color bars
        cax_outras = fig.add_axes([0.85, 0.25, 0.03, 0.5])
        cbar_outras = plt.colorbar(plt.cm.ScalarMappable(norm=norm_outras, cmap=cmap_outras), cax=cax_outras)
        cbar_outras.set_label('Consumo - Norte, Nordeste, Sul (Mw.med)', fontsize=12)
        
        cax_seco = fig.add_axes([0.1, 0.25, 0.03, 0.5])
        cbar_seco = plt.colorbar(plt.cm.ScalarMappable(norm=norm_seco, cmap=cmap_seco), cax=cax_seco)
        cbar_seco.set_label('Consumo - Sudeste & Centro-Oeste (Mw.med)', fontsize=12)
        cax_seco.yaxis.set_ticks_position('left')
        cax_seco.yaxis.set_label_position('left')

        frame_path = os.path.join(temp_dir, f'frame_{month}.png')
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        image_paths.append(frame_path)

    # --- 6. Create GIF from frames ---
    print("\n[6/6] Montando o arquivo GIF final...")
    with imageio.get_writer(output_gif_path, mode='I', fps=8) as writer:
        for filename in image_paths:
            writer.append_data(imageio.imread(filename))

    shutil.rmtree(temp_dir)
    print(f"\nProcesso concluído! O GIF foi salvo como: '{output_gif_path}'")

# Execute the function
create_dual_scale_regional_map_gif()