import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import config

def plot_results():
    experiments_dir = 'experiments'
    tradernet_dir = os.path.join(experiments_dir, 'tradernet')
    integrated_dir = os.path.join(experiments_dir, 'integrated')
    
    # Collect data
    data = []
    
    # 1. TraderNet Results
    if os.path.exists(tradernet_dir):
        for agent_name in os.listdir(tradernet_dir):
            agent_dir = os.path.join(tradernet_dir, agent_name)
            if not os.path.isdir(agent_dir): continue
            
            for filename in os.listdir(agent_dir):
                if filename.endswith('_eval_cumul_pnls.csv'):
                    # filename format: {dataset}_{reward_fn}_eval_cumul_pnls.csv
                    # e.g. DOGEUSDT_Market-Limit Orders_eval_cumul_pnls.csv
                    parts = filename.split('_eval_cumul_pnls.csv')[0]
                    # This splitting is fragile if dataset name has underscores, but standard is DOGEUSDT
                    # Let's assume dataset is the first part if we split by known reward functions?
                    # Or just use the whole prefix as "Scenario"
                    
                    filepath = os.path.join(agent_dir, filename)
                    df = pd.read_csv(filepath)
                    
                    # Add to data
                    for i, row in df.iterrows():
                        data.append({
                            'Step': i,
                            'Cumulative PnL': row['cumulative_pnl'],
                            'Agent': agent_name,
                            'Type': 'Standard'
                        })

    # 2. Integrated Results
    if os.path.exists(integrated_dir):
        for agent_name in os.listdir(integrated_dir):
            agent_dir = os.path.join(integrated_dir, agent_name)
            if not os.path.isdir(agent_dir): continue
            
            for filename in os.listdir(agent_dir):
                if filename.endswith('_eval_cumul_pnls.csv'):
                    filepath = os.path.join(agent_dir, filename)
                    df = pd.read_csv(filepath)
                    
                    for i, row in df.iterrows():
                        data.append({
                            'Step': i,
                            'Cumulative PnL': row['cumulative_pnl'],
                            'Agent': f"{agent_name} (Integrated)",
                            'Type': 'Integrated'
                        })
    
    if not data:
        print("No result files found in experiments/")
        return

    df_all = pd.DataFrame(data)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_all, x='Step', y='Cumulative PnL', hue='Agent', style='Type')
    plt.title('Cumulative PnL Comparison (Evaluation)')
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative PnL')
    plt.grid(True)
    
    output_path = os.path.join(experiments_dir, 'comparison_plot.png')
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    plot_results()
