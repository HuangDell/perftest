import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
from abc import ABC, abstractmethod

class BaseAnalyzer(ABC):
    def __init__(self, ft_value: int, thre_value: int, version: str, analysis_type: str):
        self.ft_value = ft_value
        self.thre_value = thre_value
        self.version = version
        self.resources_dir = f"./out/prototype/{version}"
        self.output_dir = os.path.join(self.resources_dir, analysis_type)
        self.base_filename = f"prototype_ft_{ft_value}_thre_{thre_value}_{version}"
        
        os.makedirs(self.resources_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.log_file = os.path.join(self.resources_dir, f"{self.base_filename}.log")
        self.png_file = os.path.join(self.output_dir, f"{self.base_filename}.png")

        self.init_version()


    def init_version(self):
        if(self.version=='v5'):
            self.version='QP = 2'
        elif (self.version == 'v6'):
            self.version='QP = 3'
        pass


    @abstractmethod
    def extract_data(self) -> Tuple:
        """从日志文件中提取数据"""
        pass

    def get_statistics(self, data: np.ndarray) -> Dict:
        """计算统计数据"""
        if len(data) == 0:
            return {}
        return {
            'mean': np.mean(data),
            'median': np.median(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
            'q1': np.percentile(data, 25),
            'q3': np.percentile(data, 75),
            'iqr': np.percentile(data, 75) - np.percentile(data, 25)
        }

    def create_single_boxplot(self, ax, data: np.ndarray, stats: Dict, title: str, ylabel: str):
        """创建单个箱线图"""
        sns.boxplot(data=data, ax=ax, width=0.5, fliersize=5, linewidth=2, color='skyblue',
                   flierprops={'markerfacecolor': 'red', 'marker': 'o', 'markeredgecolor': 'darkred'},
                   medianprops={'color': 'red', 'linewidth': 2},
                   boxprops={'facecolor': 'skyblue', 'alpha': 0.7},
                   whiskerprops={'color': 'darkblue', 'linewidth': 2},
                   capprops={'color': 'darkblue', 'linewidth': 2})

        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, pad=10)
        ax.tick_params(axis='both', labelsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)

        if stats:
            stats_text = "\n".join([
                f"Statistical Analysis:",
                f"━━━━━━━━━━━━━━━━━━",
                f"Mean: {stats['mean']:.2f}",
                f"Median: {stats['median']:.2f}",
                f"Std Dev: {stats['std']:.2f}",
                f"━━━━━━━━━━━━━━━━━━",
                f"Min: {stats['min']:.2f}",
                f"Max: {stats['max']:.2f}",
                f"Q1: {stats['q1']:.2f}",
                f"Q3: {stats['q3']:.2f}",
                f"IQR: {stats['iqr']:.2f}"
            ])
            ax.text(1.02, 0.98, stats_text,
                   transform=ax.transAxes,
                   fontsize=10,
                   verticalalignment='top',
                   bbox=dict(facecolor='white', alpha=0.9, edgecolor='lightgray', boxstyle='round,pad=0.8'))

    def create_plot(self, data_list: List[np.ndarray], stats_list: List[Dict], 
                   titles: List[str], ylabels: List[str], plot_title: str):
        """创建多子图的箱线图"""
        sns.set_style("whitegrid")
        sns.set_palette("husl")

        num_plots = len(data_list)
        fig, axes = plt.subplots(1, num_plots, figsize=(7*num_plots, 6))
        if num_plots == 1:
            axes = [axes]

        for ax, data, stats, title, ylabel in zip(axes, data_list, stats_list, titles, ylabels):
            self.create_single_boxplot(ax, data, stats, title, ylabel)

        plt.suptitle(f'{plot_title}\n(flowlet timeout={self.ft_value}, threshold={self.thre_value}, {self.version})',
                    fontsize=16, fontweight='bold', y=1.05)
        plt.tight_layout()
        plt.savefig(self.png_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()

    def print_statistics(self, stats_list: List[Dict], labels: List[str]):
        """打印统计信息"""
        print(f"\nAnalysis complete!")
        print(f"Plot saved as: {self.png_file}")
        print("\nStatistical Summary:")
        for stats, label in zip(stats_list, labels):
            if stats:
                print(f"\n{label} Stats:")
                for key, value in stats.items():
                    print(f"{key}: {value:.2f}")

    @abstractmethod
    def process_data(self):
        """处理数据的主函数"""
        pass

class FCTAnalyzer(BaseAnalyzer):
    def __init__(self, ft_value: int, thre_value: int, version: str):
        super().__init__(ft_value, thre_value, version, "fct")
        self.avg_fct_pattern = r"Average FCT:\s+(\d+\.\d+)\s+usec"
        self.fct_99th_pattern = r"99th FCT:\s+(\d+\.\d+)\s+usec"

    def extract_data(self) -> Tuple[np.ndarray, np.ndarray]:
        average_fct = []
        fct_99 = []
        with open(self.log_file, 'r') as f:
            content = f.read()
            avg_fct_matches = re.finditer(self.avg_fct_pattern, content)
            fct_99_matches = re.finditer(self.fct_99th_pattern, content)
            
            for match in avg_fct_matches:
                average_fct.append(float(match.group(1)))
            for match in fct_99_matches:
                fct_99.append(float(match.group(1)))
        
        return np.array(average_fct), np.array(fct_99)

    def process_data(self):
        print(f"Extracting data from {self.log_file}")
        avg_data, fct_99 = self.extract_data()
        
        avg_stats = self.get_statistics(avg_data)
        fct_99_stats = self.get_statistics(fct_99)
        
        self.create_plot(
            data_list=[avg_data, fct_99],
            stats_list=[avg_stats, fct_99_stats],
            titles=['Average FCT', '99th FCT'],
            ylabels=['Average FCT (usec)', '99th Percentile FCT (usec)'],
            plot_title='FCT Analysis'
        )
        
        self.print_statistics(
            stats_list=[avg_stats, fct_99_stats],
            labels=['Average FCT', '99th FCT']
        )

class BandwidthAnalyzer(BaseAnalyzer):
    def __init__(self, ft_value: int, thre_value: int, version: str):
        super().__init__(ft_value, thre_value, version, "bandwidth")
        self.pattern = r'65536\s+\d+\s+\d+\.\d+\s+(\d+\.\d+)' 

    def extract_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        average_bw= []
        with open(self.log_file, 'r') as f:
            content = f.read()
            matches = re.finditer(self.pattern, content)
            
            for match in matches:
                average_bw.append(float(match.group(1)))

        
        return np.array(average_bw)

    def process_data(self):
        print(f"Extracting data from {self.log_file}")
        avg_data= self.extract_data()
        
        avg_stats = self.get_statistics(avg_data)
        data_list = [avg_data]
        stats_list = [avg_stats]
        titles = ['Average Bandwidth']
        ylabels = ['Bandwidth (MB/sec)']
            
        self.create_plot(data_list, stats_list, titles, ylabels, 'Performance Analysis')
        self.print_statistics(
            stats_list=stats_list,
            labels=['Average Bandwidth']
        )

if __name__ == "__main__":
    ft_value = 7500
    thre_value = 6000
    version = "v5"
    
    fct = FCTAnalyzer(ft_value, thre_value, version)
    fct.process_data()

    band = BandwidthAnalyzer(ft_value, thre_value, version)
    band.process_data()