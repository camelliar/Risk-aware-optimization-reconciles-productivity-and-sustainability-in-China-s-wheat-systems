
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- 设置一个漂亮的绘图风格 ---
sns.set_theme(style="whitegrid")


class UpscalingFramework:
    """
    一个封装了从站点优化到栅格尺度上卷(Upscaling)整个工作流的类。
    """

    def __init__(self):
        self.model_n = None
        self.model_irr = None
        self.training_columns = None
        print("初始化风险权衡上卷框架 (Risk-Aware Upscaling Framework)...")

    def _prepare_data(self, df):
        """内部辅助函数：对数据进行预处理（One-Hot编码）。"""
        df_processed = pd.get_dummies(df, columns=['AEZ', 'Scenario'], prefix=['AEZ', 'Scenario'])
        if self.training_columns is not None:
            # 确保预测数据的列与训练数据完全一致
            df_processed = df_processed.reindex(columns=self.training_columns, fill_value=0)
        return df_processed

    def train_models(self, site_data_path, show_evaluation=True):
        """
        加载站点优化结果，并分别训练氮肥和灌溉的预测模型。

        参数:
        - site_data_path (str): 站点优化结果CSV文件的路径。
        - show_evaluation (bool): 是否显示模型评估图表。
        """
        print("\n--- 步骤 1: 正在从站点数据训练机器学习模型 ---")
        site_df = pd.read_csv(site_data_path)

        # 1. 数据预处理
        features_df = self._prepare_data(site_df)
        self.training_columns = features_df.drop(columns=['Site_ID', 'Optimal_N', 'Optimal_Irrigation']).columns

        # 2. 训练氮肥模型 (N Model)
        X_n = features_df[self.training_columns]
        y_n = features_df['Optimal_N']
        self.model_n = self._train_single_model(X_n, y_n, "Optimal Nitrogen", show_evaluation)

        # 3. 训练灌溉模型 (Irrigation Model)
        X_irr = features_df[self.training_columns]
        y_irr = features_df['Optimal_Irrigation']
        self.model_irr = self._train_single_model(X_irr, y_irr, "Optimal Irrigation", show_evaluation)

    def _train_single_model(self, X, y, target_name, show_plot):
        """内部辅助函数：训练并评估单个XGBoost模型。"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5,
                                 random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        print(f"  - {target_name} 模型训练完成. R² = {r2:.3f}")

        if show_plot:
            self._plot_evaluation(y_test, y_pred, target_name, r2)

        return model

    def predict_for_grid(self, grid_data_path):
        """
        加载栅格数据，并为所有情景进行预测，返回最终的处方数据框。

        参数:
        - grid_data_path (str): 栅格预测因子CSV文件的路径。
        """
        if not self.model_n or not self.model_irr:
            raise RuntimeError("模型尚未训练，请先调用 train_models() 方法。")

        print("\n--- 步骤 2: 正在将模型应用于栅格尺度数据 ---")
        grid_df = pd.read_csv(grid_data_path)

        # 为每个栅格点扩展所有情景
        scenarios = ['T+10%', 'T+5%', 'T0', 'T-5%', 'T-10%']
        grid_expanded_df = pd.concat([grid_df.assign(Scenario=s) for s in scenarios], ignore_index=True)

        # 数据预处理
        grid_processed = self._prepare_data(grid_expanded_df)

        # 进行预测
        print(f"  - 正在为 {len(grid_df)} 个栅格点和 {len(scenarios)} 种情景生成处方...")
        grid_expanded_df['Predicted_N'] = self.model_n.predict(grid_processed)
        grid_expanded_df['Predicted_Irrigation'] = self.model_irr.predict(grid_processed)

        print("  - 栅格尺度预测完成。")
        return grid_expanded_df

    @staticmethod
    def _plot_evaluation(y_true, y_pred, title, r2):
        """静态辅助函数：绘制模型评估的散点图。"""
        plt.figure(figsize=(6, 6))
        plt.scatter(y_true, y_pred, alpha=0.5, label=f'$R^2 = {r2:.3f}$')
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--', color='red', lw=2, label='1:1 Line')
        plt.xlabel("Observed Optimal Value")
        plt.ylabel("Predicted Optimal Value")
        plt.title(f"Model Evaluation: {title}")
        plt.legend()
        plt.show()


# ====================================================================
# --- 主程序：用几行代码讲述整个故事 ---
# ====================================================================

if __name__ == "__main__":
    from generate_mock_data import generate_mock_data  # 假设您将数据生成代码封装在一个函数里

    generate_mock_data()

    # 1. 初始化我们的框架
    framework = UpscalingFramework()

    # 2. 从站点数据训练模型，并展示评估结果
    framework.train_models(site_data_path='site_optimization_results.csv', show_evaluation=True)

    # 3. 将训练好的模型应用于全国的栅格数据
    grid_predictions = framework.predict_for_grid(grid_data_path='grid_predictors.csv')

    # 4. 展示最终的成果
    print("\n--- 最终成果: 全国尺度优化处方 (预览) ---")
    print(grid_predictions.head())

    # (可选) 成果可视化：以T0情景为例，绘制简单的结果分布图
    t0_results = grid_predictions[grid_predictions['Scenario'] == 'T0']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(t0_results['Predicted_N'], ax=axes[0], kde=True, bins=30).set_title('Distribution of Optimal N (T0)')
    sns.histplot(t0_results['Predicted_Irrigation'], ax=axes[1], kde=True, bins=30).set_title(
        'Distribution of Optimal Irrigation (T0)')
    plt.tight_layout()
    plt.show()