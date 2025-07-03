# -*- coding: utf-8 -*-
"""
MERGED SCRIPT for Global and Local Energy Consumption Forecasting (v8 - Final Refinement)

This version refines the outputs based on user feedback for maximum clarity.

Key Improvements in this version:
- Heatmaps are now split into four distinct files: Global MAPE, Global RMSE, Local MAPE, and Local RMSE.
- The various "race" GIFs have been consolidated into a single, superior "Total Competition" GIF
  for each region, showing all 7 models competing together.
- Added individual GIF creation for each model's predictions over time per region.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import imageio.v2 as imageio
import os
import shutil
import warnings

# --- Model Imports ---
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.api import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# --- Initial Setup & Configuration ---
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', UserWarning)

# --- 1. MASTER DIRECTORY SETUP ---
master_output_dir = "Resultados_Analise_Final"
print("="*60)
print("     INICIANDO ANÁLISE COMPARATIVA DE MODELOS (VERSÃO REFINADA)     ")
print("="*60)
if os.path.exists(master_output_dir):
    shutil.rmtree(master_output_dir)
    print(f"[INFO] Diretório de resultados anterior '{master_output_dir}' removido.")
os.makedirs(master_output_dir, exist_ok=True)

dir_structure = {
    "comparativo_geral": os.path.join(master_output_dir, "1_Comparativo_Geral_Barras"),
    "competicao_total": os.path.join(master_output_dir, "2_Competicao_Total_Regional"),
    "analise_global": os.path.join(master_output_dir, "3_Analise_Global"),
    "analise_local": os.path.join(master_output_dir, "4_Analise_Local_por_Regiao"),
    "tabelas": os.path.join(master_output_dir, "Tabelas"),
    "temp_frames": os.path.join(master_output_dir, "temp_frames_para_gifs"),
    "analise_individual_modelos": os.path.join(master_output_dir, "5_Analise_Individual_Modelos")
}
for region in ['Nordeste', 'Norte', 'Sudeste_CO', 'Sul']:
    dir_structure[f"local_{region}"] = os.path.join(dir_structure["analise_local"], region)
for path in dir_structure.values():
    os.makedirs(path, exist_ok=True)
print(f"\n[INFO] Estrutura de diretórios criada dentro de '{master_output_dir}'.")

# --- 2. DATA LOADING AND PREPARATION ---
print("[INFO] Lendo e preparando os dados...")
try:
    monthly_data = pd.read_csv("Consumo_Mensal.csv")
except FileNotFoundError:
    print(f"\n[ERRO] Arquivo 'Consumo_Mensal.csv' não encontrado.")
    exit()
monthly_data_reset = monthly_data.reset_index()
monthly_data_reset.columns = [col.replace('Mwmed_', '') for col in monthly_data_reset.columns]
long_data = monthly_data_reset.melt(
    id_vars=['Data'], value_vars=['Nordeste', 'Norte', 'Sudeste/CO', 'Sul'],
    var_name='Region', value_name='Average_Consumption'
)
long_data['Data'] = pd.to_datetime(long_data['Data'], format='%m/%d/%Y')
long_data.sort_values(by=['Data'], inplace=True)
print("[INFO] Criando features de engenharia de dados...")
long_data['month'] = long_data['Data'].dt.month
long_data['year'] = long_data['Data'].dt.year
long_data['Consumption_lag_1'] = long_data.groupby('Region')['Average_Consumption'].shift(1)
long_data['Consumption_lag_12'] = long_data.groupby('Region')['Average_Consumption'].shift(12)
long_data['Consumption_rolling_mean_3'] = long_data.groupby('Region')['Average_Consumption'].rolling(window=3).mean().reset_index(level=0, drop=True)
long_data.dropna(inplace=True)
region_dummies = pd.get_dummies(long_data['Region'], prefix='Region', drop_first=True)
long_data_encoded = pd.concat([long_data, region_dummies], axis=1)

# --- 3. CONFIGURATION AND PARAMETERS ---
print("[INFO] Configurando parâmetros de controle para o treinamento.")
min_year, max_year = long_data['year'].min(), long_data['year'].max()
min_train_years, min_test_years = 2, 1
regions = long_data['Region'].unique()
model_types_local = ['LR', 'RF', 'Prophet', 'HW', 'SARIMA']
model_types_global = ['Global_LR', 'Global_RF']
target_col = 'Average_Consumption'
features_local = ['month', 'year', 'Consumption_lag_1', 'Consumption_lag_12', 'Consumption_rolling_mean_3']
prophet_regressors = ['Consumption_lag_1', 'Consumption_lag_12', 'Consumption_rolling_mean_3']
features_global = features_local + list(region_dummies.columns)

sarima_params = {
    'Nordeste': {'order': (3, 0, 1), 'seasonal_order': (0, 1, 1, 12), 'trend': 'c'},
    'Norte': {'order': (0, 1, 0), 'seasonal_order': (2, 0, 0, 12), 'trend': None},
    'Sudeste_CO': {'order': (2, 1, 0), 'seasonal_order': (0, 1, 1, 12), 'trend': None},
    'Sul': {'order': (1, 0, 0), 'seasonal_order': (2, 1, 0, 12), 'trend': 'c'}
}

# --- 4. MAIN TRAINING AND EVALUATION LOOP ---
print("\n" + "="*60)
print("      INICIANDO TREINAMENTO E AVALIAÇÃO DE TODOS OS MODELOS      ")
print("="*60)
yearly_metrics = {m: {model: {r: {} for r in regions} for model in model_types_local} for m in ['MAPE', 'RMSE']}
yearly_metrics_global = {m: {model: {} for model in model_types_global} for m in ['MAPE', 'RMSE']}

# Initialize image_filenames with correct structure for local and global models
all_model_types = model_types_local + model_types_global
image_filenames = {
    'Ultimate_Race_Bar': [],
    'Competition': {r: [] for r in regions},
    'Individual_Models': {model: {r: [] for r in regions} for model in all_model_types}
}

for train_end_year in range(min_year + min_train_years - 1, max_year - min_test_years + 1):
    print(f"\n--- Processando Ano de Treino: {train_end_year} ---")
    split_date = pd.to_datetime(f'{train_end_year + 1}-01-01')
    train_data_global = long_data_encoded[long_data_encoded['Data'] < split_date].copy()
    test_data_global = long_data_encoded[long_data_encoded['Data'] >= split_date].copy()
    train_data_local = long_data[long_data['Data'] < split_date].copy()
    test_data_local = long_data[long_data['Data'] >= split_date].copy()
    if train_data_global.empty or test_data_global.empty: # Ensure test data exists for global models
        print(f"[AVISO] Dados de treino ou teste globais vazios para {train_end_year}. Pulando este ano.")
        continue

    # --- Global Model Training ---
    X_train_g, y_train_g = train_data_global[features_global], train_data_global[target_col]
    X_test_g, y_test_g = test_data_global[features_global], test_data_global[target_col]

    # Global_LR
    model_glr = LinearRegression().fit(X_train_g, y_train_g)
    y_pred_glr = model_glr.predict(X_test_g)
    yearly_metrics_global['MAPE']['Global_LR'][train_end_year] = mean_absolute_percentage_error(y_test_g, y_pred_glr) * 100
    yearly_metrics_global['RMSE']['Global_LR'][train_end_year] = np.sqrt(mean_squared_error(y_test_g, y_pred_glr))
    test_data_global.loc[:, 'Pred_Global_LR'] = y_pred_glr

    # Global_RF
    model_grf = RandomForestRegressor(random_state=42).fit(X_train_g, y_train_g)
    y_pred_grf = model_grf.predict(X_test_g)
    yearly_metrics_global['MAPE']['Global_RF'][train_end_year] = mean_absolute_percentage_error(y_test_g, y_pred_grf) * 100
    yearly_metrics_global['RMSE']['Global_RF'][train_end_year] = np.sqrt(mean_squared_error(y_test_g, y_pred_grf))
    test_data_global.loc[:, 'Pred_Global_RF'] = y_pred_grf
    
    # --- Individual Global Model GIF Frames ---
    for model_name_global, y_pred_global_all_regions in {'Global_LR': y_pred_glr, 'Global_RF': y_pred_grf}.items():
        # Ensure y_pred_global_all_regions is aligned with test_data_global for slicing by region
        temp_test_data_global = test_data_global.copy()
        temp_test_data_global['Pred_Current_Global'] = y_pred_global_all_regions

        for region in regions:
            region_str = region.replace('/', '_')
            plt.figure(figsize=(12, 6))
            
            # Plot historical actual data for the region
            region_actual_data = long_data[long_data['Region'] == region]
            plt.plot(region_actual_data['Data'], region_actual_data['Average_Consumption'], label='Real', color='black', alpha=0.8, linewidth=2)
            
            # Plot global model predictions for the current region
            region_test_data_global = temp_test_data_global[temp_test_data_global['Region'] == region]
            
            if not region_test_data_global.empty:
                plt.plot(region_test_data_global['Data'], region_test_data_global['Pred_Current_Global'], label=f'Previsão {model_name_global}', linestyle='--', color='red')
            else:
                print(f"[AVISO] Sem dados de teste globais para {region} em {train_end_year}.")
                plt.text(0.5, 0.5, 'Sem dados de teste', transform=plt.gca().transAxes,
                         fontsize=12, color='gray', ha='center', va='center')
            
            plt.title(f'{model_name_global} Performance: {region} (Treino até {train_end_year})', fontsize=14)
            plt.xlabel('Data'); plt.ylabel('Consumo Médio'); plt.legend(); plt.grid(True, alpha=0.5)
            plt.axvline(x=split_date, color='r', linestyle='-', label='Início do Teste')
            plt.tight_layout()
            
            # CONSISTENT FILENAME GENERATION
            # Use model_name_global directly as it's the key in image_filenames['Individual_Models']
            filename = os.path.join(dir_structure['temp_frames'], f"individual_model_{model_name_global}_{region_str}_{train_end_year}.png")
            plt.savefig(filename)
            plt.close() # Close figure immediately
            image_filenames['Individual_Models'][model_name_global][region].append(filename)

    # --- Local Model Training ---
    all_local_preds = {}
    for region in regions:
        region_train = train_data_local[train_data_local['Region'] == region]
        region_test = test_data_local[test_data_local['Region'] == region]
        
        if region_train.empty or region_test.empty:
            print(f"[AVISO] Dados de treino ou teste locais vazios para {region} em {train_end_year}. Pulando modelos locais para esta região/ano.")
            continue
        
        X_train_l, y_train_l = region_train[features_local], region_train[target_col]
        X_test_l, y_test_l = region_test[features_local], region_test[target_col]
        y_preds = {}
        
        # LR
        try:
            y_preds['LR'] = LinearRegression().fit(X_train_l, y_train_l).predict(X_test_l)
        except Exception as e:
            y_preds['LR'] = np.full(len(y_test_l), np.nan)
            print(f"[AVISO] Falha LR para {region} em {train_end_year}: {e}")
        
        # RF
        try:
            y_preds['RF'] = RandomForestRegressor(random_state=42).fit(X_train_l, y_train_l).predict(X_test_l)
        except Exception as e:
            y_preds['RF'] = np.full(len(y_test_l), np.nan)
            print(f"[AVISO] Falha RF para {region} em {train_end_year}: {e}")
        
        # HW
        ts_train = region_train.set_index('Data')[target_col]
        try:
            # Ensure ts_train has enough data points for seasonal_periods
            if len(ts_train) >= 2 * 12: # At least two full years for seasonal_periods=12
                model_hw = ExponentialSmoothing(ts_train, trend='add', seasonal='add', seasonal_periods=12, initialization_method="estimated").fit()
                y_preds['HW'] = model_hw.forecast(len(y_test_l))
            else:
                y_preds['HW'] = np.full(len(y_test_l), np.nan)
                print(f"[AVISO] Dados insuficientes para HW em {region} em {train_end_year}.")
        except Exception as e:
            y_preds['HW'] = np.full(len(y_test_l), np.nan)
            print(f"[AVISO] Falha HW para {region} em {train_end_year}: {e}")

        # SARIMA
        try:
            params = sarima_params[region.replace('/', '_')]
            # Check if ts_train has enough data for SARIMA
            if len(ts_train) > max(params['order'][0], params['seasonal_order'][0] * params['seasonal_order'][3]):
                model_sarima = SARIMAX(ts_train, order=params['order'], seasonal_order=params['seasonal_order'], trend=params['trend']).fit(disp=False)
                y_preds['SARIMA'] = model_sarima.forecast(len(y_test_l))
            else:
                y_preds['SARIMA'] = np.full(len(y_test_l), np.nan)
                print(f"[AVISO] Dados insuficientes para SARIMA em {region} em {train_end_year}.")
        except Exception as e:
            y_preds['SARIMA'] = np.full(len(y_test_l), np.nan)
            print(f"[AVISO] Falha SARIMA para {region} em {train_end_year}: {e}")

        # Prophet
        try:
            prophet_train_df = region_train[['Data', target_col] + prophet_regressors].rename(columns={'Data': 'ds', target_col: 'y'})
            if not prophet_train_df.empty:
                model_prophet = Prophet(weekly_seasonality=False, daily_seasonality=False)
                for regressor in prophet_regressors: model_prophet.add_regressor(regressor)
                model_prophet.fit(prophet_train_df)
                y_preds['Prophet'] = model_prophet.predict(region_test[['Data'] + prophet_regressors].rename(columns={'Data': 'ds'}))['yhat'].values
            else:
                y_preds['Prophet'] = np.full(len(y_test_l), np.nan)
                print(f"[AVISO] Dados de treino Prophet vazios para {region} em {train_end_year}.")
        except Exception as e:
            y_preds['Prophet'] = np.full(len(y_test_l), np.nan)
            print(f"[AVISO] Falha Prophet para {region} em {train_end_year}: {e}")

        all_local_preds[region] = y_preds

        for model_name, y_pred in y_preds.items():
            if not np.isnan(y_pred).any():
                yearly_metrics['MAPE'][model_name][region][train_end_year] = mean_absolute_percentage_error(y_test_l, y_pred) * 100
                yearly_metrics['RMSE'][model_name][region][train_end_year] = np.sqrt(mean_squared_error(y_test_l, y_pred))

                # --- Individual Local Model GIF Frames ---
                region_str = region.replace('/', '_')
                plt.figure(figsize=(12, 6))
                plt.plot(long_data[long_data['Region'] == region]['Data'], long_data[long_data['Region'] == region]['Average_Consumption'], label='Real', color='black', alpha=0.8, linewidth=2)
                plt.plot(region_test['Data'], y_pred, label=f'Previsão Local {model_name}', linestyle='--', color='blue')
                plt.title(f'Local {model_name} Performance: {region} (Treino até {train_end_year})', fontsize=14)
                plt.xlabel('Data'); plt.ylabel('Consumo Médio'); plt.legend(); plt.grid(True, alpha=0.5)
                plt.axvline(x=split_date, color='r', linestyle='-', label='Início do Teste')
                plt.tight_layout()
                
                # CONSISTENT FILENAME GENERATION
                # Use model_name directly as it's the key in image_filenames['Individual_Models']
                filename = os.path.join(dir_structure['temp_frames'], f"individual_model_{model_name}_{region_str}_{train_end_year}.png")
                plt.savefig(filename)
                plt.close() # Close figure immediately
                image_filenames['Individual_Models'][model_name][region].append(filename)
            else:
                print(f"[AVISO] Previsões NaN para Local {model_name} em {region} no ano {train_end_year}. Frame não gerado.")


    # --- GENERATE VISUALIZATION FRAMES ---
    # Bar chart race
    ultimate_mapes = {m: yearly_metrics_global['MAPE'][m].get(train_end_year, np.nan) for m in model_types_global}
    for model in model_types_local:
        local_mapes_for_year = [yearly_metrics['MAPE'][model][r].get(train_end_year, np.nan) for r in regions]
        ultimate_mapes[f'Local {model}'] = np.nanmean(local_mapes_for_year) if local_mapes_for_year else np.nan
    
    race_df = pd.DataFrame(list(ultimate_mapes.items()), columns=['Modelo', 'MAPE']).sort_values('MAPE').dropna()
    race_df['Tipo'] = race_df['Modelo'].apply(lambda x: 'Global' if 'Global' in x else 'Local')
    plt.figure(figsize=(14, 8))
    sns.barplot(x='MAPE', y='Modelo', data=race_df, hue='Tipo', palette={'Global': '#d53e4f', 'Local': '#3288bd'}, dodge=False)
    plt.title(f"Comparativo Geral (Barras) - Performance Média (Treino até {train_end_year})", fontsize=16)
    plt.xlabel("MAPE Médio (%)"); plt.ylabel("Modelo"); plt.tight_layout()
    filename = os.path.join(dir_structure['temp_frames'], f'ultimate_race_bar_{train_end_year}.png')
    plt.savefig(filename)
    plt.close() # Close figure immediately
    image_filenames['Ultimate_Race_Bar'].append(filename)

    # Total Competition GIF Frames
    for region, preds_this_region in all_local_preds.items():
        region_str = region.replace('/', '_')
        plt.figure(figsize=(16, 8))
        plt.plot(long_data[long_data['Region']==region]['Data'], long_data[long_data['Region']==region]['Average_Consumption'], label='Real', color='black', alpha=0.9, linewidth=2)
        # Plot local models
        for model_name, y_pred in preds_this_region.items():
            if not np.isnan(y_pred).any():
                plt.plot(test_data_local[test_data_local['Region']==region]['Data'], y_pred, label=f'Local {model_name}', linestyle='--')
        # Plot global models
        for model_name_global in model_types_global:
            pred_col_name = f'Pred_{model_name_global}'
            global_pred_region = test_data_global[test_data_global['Region'] == region]
            if not global_pred_region.empty and pred_col_name in global_pred_region.columns: # Check if column exists
                plt.plot(global_pred_region['Data'], global_pred_region[pred_col_name], label=model_name_global, linestyle=':')
        
        plt.title(f'Competição Total de Modelos: {region} (Treino até {train_end_year})', fontsize=16)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1)); plt.grid(True, alpha=0.5); plt.axvline(x=split_date, color='r', linestyle='-')
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        filename = os.path.join(dir_structure['temp_frames'], f"competition_{region_str}_{train_end_year}.png")
        plt.savefig(filename)
        plt.close() # Close figure immediately
        image_filenames['Competition'][region].append(filename)


# --- 5. FINAL OUTPUT GENERATION ---
print("\n" + "="*60)
print("             FINALIZANDO - GERANDO ARQUIVOS DE SAÍDA             ")
print("="*60)

def create_gif(image_list, output_path, fps=1.5):
    if image_list:
        try:
            # Filter out None/empty paths if any, although with current fixes, lists should only have valid paths
            valid_images = [imageio.imread(f) for f in image_list if os.path.exists(f)]
            if valid_images:
                imageio.mimsave(output_path, valid_images, fps=fps)
                print(f"  [SUCESSO] GIF gerado: {os.path.relpath(output_path)}")
            else:
                print(f"  [AVISO] Nenhuma imagem válida encontrada para criar GIF: {os.path.relpath(output_path)}")
        except Exception as e:
            print(f"  [ERRO] Falha ao criar GIF {os.path.relpath(output_path)}. Causa: {e}")
    else:
        print(f"  [AVISO] Lista de imagens vazia para criar GIF: {os.path.relpath(output_path)}")


# --- Generate GIFs ---
print("[INFO] Gerando arquivos GIF...")
create_gif(image_filenames['Ultimate_Race_Bar'], os.path.join(dir_structure['comparativo_geral'], 'comparativo_geral_barras.gif'))
for region in regions:
    region_str = region.replace('/', '_')
    create_gif(image_filenames['Competition'][region], os.path.join(dir_structure['competicao_total'], f'competicao_total_{region_str}.gif'))

# Generate individual model GIFs
for model_type_key in all_model_types: # Iterate through the keys used in image_filenames['Individual_Models']
    for region in regions:
        region_str = region.replace('/', '_')
        
        # Determine the correct directory name for the model on disk
        # This is for the FOLDER name, can be prefixed for clarity
        if model_type_key.startswith('Global_'):
            model_dir_name = model_type_key 
        else: # Local models will have folder names like 'Local_LR'
            model_dir_name = f"Local_{model_type_key}" 
        
        output_sub_dir = os.path.join(dir_structure['analise_individual_modelos'], model_dir_name)
        os.makedirs(output_sub_dir, exist_ok=True)
        
        # Determine the GIF filename, also prefixed for clarity
        gif_name = f"{model_dir_name}_individual_{region_str}.gif"
        
        # Pass the list of image paths associated with this model_type_key and region
        create_gif(image_filenames['Individual_Models'][model_type_key][region], os.path.join(output_sub_dir, gif_name))


# --- Generate Final Tables and Heatmaps ---
print("[INFO] Gerando tabelas e mapas de calor finais...")
final_metrics = []
# Ensure final_year is calculated based on available data, not just max_year
# It should be the last year for which 'test data' was generated and metrics collected
# In your loop, 'train_end_year' goes up to max_year - min_test_years.
# So, the final test year starts from (max_year - min_test_years) + 1.
# Let's say final_year_for_metrics refers to the train_end_year that produced the *last* set of predictions.
final_year_for_metrics = max_year - min_test_years # This is the last train_end_year processed

for model_type in model_types_global:
    # Use .get() with a default of np.nan in case a metric is missing for the final year
    mape_val = yearly_metrics_global['MAPE'][model_type].get(final_year_for_metrics, np.nan)
    rmse_val = yearly_metrics_global['RMSE'][model_type].get(final_year_for_metrics, np.nan)
    final_metrics.append({'Modelo': model_type, 'Regiao': 'Global', 'MAPE': mape_val, 'RMSE': rmse_val})

for model in model_types_local:
    for region in regions:
        mape_val = yearly_metrics['MAPE'][model][region].get(final_year_for_metrics, np.nan)
        rmse_val = yearly_metrics['RMSE'][model][region].get(final_year_for_metrics, np.nan)
        final_metrics.append({'Modelo': f'Local {model}', 'Regiao': region, 'MAPE': mape_val, 'RMSE': rmse_val})

metrics_df = pd.DataFrame(final_metrics).dropna(subset=['MAPE', 'RMSE'], how='all') # Drop rows only if both MAPE and RMSE are NaN
metrics_df.to_csv(os.path.join(dir_structure['tabelas'], 'metricas_finais.csv'), index=False) # Add index=False
print(f"  [SUCESSO] Tabela de métricas salva em: {os.path.relpath(dir_structure['tabelas'])}")

# Separate heatmaps for Global and Local scopes
global_metrics_pivot_df = metrics_df[metrics_df['Modelo'].str.contains('Global')].copy()
if not global_metrics_pivot_df.empty:
    global_metrics_pivot_df['Regiao'] = 'Global' # Ensure a consistent 'Regiao' for pivoting
global_metrics_df_for_heatmap = global_metrics_pivot_df.set_index('Modelo')

local_metrics_df = metrics_df[~metrics_df['Modelo'].str.contains('Global')].copy()

def plot_heatmap(df, metric, scope, output_dir, final_year_val):
    try:
        if df.empty:
            print(f"  [AVISO] DataFrame vazio para mapa de calor de {metric} ({scope}). Pulando geração.")
            return

        # For global models, we want 'Global' as the only region column
        if scope == 'Global':
            # Drop the 'Regiao' column before pivoting if it's all 'Global', as it's redundant for the heatmap
            pivot_data = df[['Modelo', metric]].set_index('Modelo')
            # Add a dummy column for heatmap visual if needed, or just plot as a bar chart if only one 'region'
            # For a true heatmap, you need at least two dimensions. If only 'Global' region, it's essentially a list.
            # We will use 'Global' as a column for consistency with heatmap function.
            pivot_data.columns = ['Global']
        else: # For local models, pivot by actual regions
            pivot_data = df.pivot_table(index='Modelo', columns='Regiao', values=metric)
            if pivot_data.empty: # Check after pivot as well
                print(f"  [AVISO] DataFrame pivotado vazio para mapa de calor de {metric} ({scope}). Pulando geração.")
                return

        # Dynamic sizing, ensuring min size
        fig_width = max(8, pivot_data.shape[1] * 2) if pivot_data.shape[1] > 0 else 8
        fig_height = max(6, pivot_data.shape[0] * 1.5) if pivot_data.shape[0] > 0 else 6

        plt.figure(figsize=(min(fig_width, 15), min(fig_height, 10))) 
        sns.heatmap(pivot_data, annot=True, fmt=".2f", cmap="YlOrRd", linewidths=.5)
        plt.title(f'{metric} Final - Modelos {scope} (Teste a partir de {final_year_val+1})', fontsize=16)
        plt.xlabel('Região'); plt.ylabel('Modelo'); plt.yticks(rotation=0); plt.tight_layout()
        heatmap_path = os.path.join(output_dir, f"heatmap_{scope.lower()}_{metric}.png")
        plt.savefig(heatmap_path)
        plt.close()
        print(f"  [SUCESSO] Mapa de calor de {metric} ({scope}) gerado.")
    except Exception as e:
        print(f"  [ERRO] Não foi possível gerar o mapa de calor de {metric} ({scope}). Erro: {e}")

# Pass the final_year_for_metrics to the plotting function
plot_heatmap(global_metrics_df_for_heatmap, 'MAPE', 'Global', dir_structure['analise_global'], final_year_for_metrics)
plot_heatmap(global_metrics_df_for_heatmap, 'RMSE', 'Global', dir_structure['analise_global'], final_year_for_metrics)
plot_heatmap(local_metrics_df, 'MAPE', 'Local', dir_structure['analise_local'], final_year_for_metrics)
plot_heatmap(local_metrics_df, 'RMSE', 'Local', dir_structure['analise_local'], final_year_for_metrics)

# --- Final Cleanup ---
if os.path.exists(dir_structure['temp_frames']):
    shutil.rmtree(dir_structure['temp_frames'])
    print(f"\n[INFO] Diretório de frames temporários removido.")
print("\n" + "="*60)
print("                    PROCESSAMENTO CONCLUÍDO!                    ")
print(f"   Todos os resultados foram salvos em '{master_output_dir}'    ")
print("="*60)