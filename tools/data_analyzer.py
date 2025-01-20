import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
from collections import defaultdict 
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
        elif (self.version == 'v7'):
            self.version='QP = 4'
        elif (self.version == 'v8'):
            self.version='QP = 6'
        elif (self.version == 'v9'):
            self.version='QP = 8'
        elif (self.version == 'v4'):
            self.version='QP = 1'

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

class PerformanceAnalyzer:  
    def __init__(self, data: Dict):  
        self.data = data  
        
    def calculate_improvements(self, metric: str, base_params: str, ref_params: str, ft_value: int) -> Dict:  
        """  
        计算相对于基准配置的性能改进  
        metric: 'avg_bw', 'avg_fct', 或 'fct_99'  
        """  
        if base_params not in self.data or ref_params not in self.data:  
            return None  
            
        base_value = self.data[base_params][metric]  
        ref_value = self.data[ref_params][metric]  
        
        improvements = {}  
        # 只分析具有相同ft值的配置  
        for params, values in self.data.items():  
            if values['ft_value'] != ft_value:  
                continue  
                
            if params in [base_params, ref_params]:  
                continue  
                
            current_value = values[metric]  
            # 对于带宽，更大值表示更好；对于FCT，更小值表示更好  
            if metric == 'avg_bw':  
                imp_base = ((current_value - base_value) / base_value) * 100  
                imp_ref = ((current_value - ref_value) / ref_value) * 100  
            else:  # FCT metrics  
                imp_base = ((base_value - current_value) / base_value) * 100  
                imp_ref = ((ref_value - current_value) / ref_value) * 100  
                
            improvements[params] = {  
                'vs_base': imp_base,  
                'vs_ref': imp_ref,  
                'value': current_value,  
                'ft': values['ft_value'],  
                'thre': values['thre_value']  
            }  
            
        return improvements  

    def find_extreme_improvements(self, improvements: Dict) -> Tuple[Dict, Dict]:  
        """找出最大提升和最大降低的配置"""  
        if not improvements:  
            return None, None  
            
        # vs base  
        max_imp_base = max(improvements.items(), key=lambda x: x[1]['vs_base'])  
        min_imp_base = min(improvements.items(), key=lambda x: x[1]['vs_base'])  
        
        # vs ref  
        max_imp_ref = max(improvements.items(), key=lambda x: x[1]['vs_ref'])  
        min_imp_ref = min(improvements.items(), key=lambda x: x[1]['vs_ref'])  
        
        return {  
            'vs_base': {  
                'max': max_imp_base,  
                'min': min_imp_base  
            }  
        }, {  
            'vs_ref': {  
                'max': max_imp_ref,  
                'min': min_imp_ref  
            }  
        }  
    
class GroupedAnalyzer:  
    def __init__(self, base_dir: str = "./out/prototype"):  
        self.base_dir = base_dir  
        self.log_pattern = re.compile(r"prototype_ft_(\d+)_thre_(\d+)_(v\d+)\.log$")  
    
    def analyze_performance_improvements(self, version_data: Dict):  
        """分析性能改进"""  
        for version, data in version_data.items():  
            if not data:  
                continue  
                
            print(f"\nPerformance Analysis for {self._get_version_display(version)}:")  
            print("=" * 80)  
            
            # 找出基准配置和参考配置  
            base_params = None  
            ref_params_5000 = None  
            ref_params_7500 = None  
            
            for params, values in data.items():  
                if values['ft_value'] == 0 and values['thre_value'] == 0:  
                    base_params = params  
                elif values['ft_value'] == 5000 and values['thre_value'] == 0:  
                    ref_params_5000 = params  
                elif values['ft_value'] == 7500 and values['thre_value'] == 0:  
                    ref_params_7500 = params  
            
            if not base_params:  
                print(f"Warning: Could not find base configuration (ft=0, thre=0) for {version}")  
                continue  
                
            analyzer = PerformanceAnalyzer(data)  
            metrics = {  
                'avg_bw': 'Average Bandwidth',  
                'avg_fct': 'Average FCT',  
                'fct_99': '99th Percentile FCT'  
            }  
            
            for metric, metric_name in metrics.items():  
                print(f"\n{metric_name}:")  
                print("-" * 40)  
                
                # 分析 ft=5000 的配置  
                if ref_params_5000:  
                    print("\nFor configurations with ft=5000:")  
                    improvements = analyzer.calculate_improvements(  
                        metric, base_params, ref_params_5000, ft_value=5000)  
                    
                    if improvements:  
                        base_imps, ref_imps = analyzer.find_extreme_improvements(improvements)  
                        
                        # Print improvements vs base (ft=0, thre=0)  
                        max_base = base_imps['vs_base']['max']  
                        min_base = base_imps['vs_base']['min']  
                        print(f"\nCompared to baseline (ft=0, thre=0):")  
                        print(f"  Maximum improvement: {max_base[1]['vs_base']:.2f}% "  
                              f"(ft={max_base[1]['ft']}, thre={max_base[1]['thre']})")  
                        print(f"  Maximum degradation: {min_base[1]['vs_base']:.2f}% "  
                              f"(ft={min_base[1]['ft']}, thre={min_base[1]['thre']})")  
                        
                        # Print improvements vs reference (ft=5000, thre=0)  
                        max_ref = ref_imps['vs_ref']['max']  
                        min_ref = ref_imps['vs_ref']['min']  
                        print(f"\nCompared to reference (ft=5000, thre=0):")  
                        print(f"  Maximum improvement: {max_ref[1]['vs_ref']:.2f}% "  
                              f"(ft={max_ref[1]['ft']}, thre={max_ref[1]['thre']})")  
                        print(f"  Maximum degradation: {min_ref[1]['vs_ref']:.2f}% "  
                              f"(ft={min_ref[1]['ft']}, thre={min_ref[1]['thre']})")  
                
                # 分析 ft=7500 的配置  
                if ref_params_7500:  
                    print("\nFor configurations with ft=7500:")  
                    improvements = analyzer.calculate_improvements(  
                        metric, base_params, ref_params_7500, ft_value=7500)  
                    
                    if improvements:  
                        base_imps, ref_imps = analyzer.find_extreme_improvements(improvements)  
                        
                        # Print improvements vs base (ft=0, thre=0)  
                        max_base = base_imps['vs_base']['max']  
                        min_base = base_imps['vs_base']['min']  
                        print(f"\nCompared to baseline (ft=0, thre=0):")  
                        print(f"  Maximum improvement: {max_base[1]['vs_base']:.2f}% "  
                              f"(ft={max_base[1]['ft']}, thre={max_base[1]['thre']})")  
                        print(f"  Maximum degradation: {min_base[1]['vs_base']:.2f}% "  
                              f"(ft={min_base[1]['ft']}, thre={min_base[1]['thre']})")  
                        
                        # Print improvements vs reference (ft=7500, thre=0)  
                        max_ref = ref_imps['vs_ref']['max']  
                        min_ref = ref_imps['vs_ref']['min']  
                        print(f"\nCompared to reference (ft=7500, thre=0):")  
                        print(f"  Maximum improvement: {max_ref[1]['vs_ref']:.2f}% "  
                              f"(ft={max_ref[1]['ft']}, thre={max_ref[1]['thre']})")  
                        print(f"  Maximum degradation: {min_ref[1]['vs_ref']:.2f}% "  
                              f"(ft={min_ref[1]['ft']}, thre={min_ref[1]['thre']})")  
    
    def _get_version_display(self, version: str) -> str:  
        version_map = {  
            'v4': 'QP = 1',
            'v5': 'QP = 2',  
            'v6': 'QP = 3',  
            'v7': 'QP = 4',  
            'v8': 'QP = 6',  
            'v9': 'QP = 8'  
        }  
        return version_map.get(version, version)  

    def extract_data_for_version(self) -> Dict:  
        """Extract and group data by version"""  
        version_data = defaultdict(lambda: defaultdict(list))  
        
        for root, _, files in os.walk(self.base_dir):  
            for file in files:  
                if file.endswith('.log'):  
                    match = self.log_pattern.match(file)  
                    if match:  
                        ft_value = int(match.group(1))  
                        thre_value = int(match.group(2))  
                        version = match.group(3)  
                        log_path = os.path.join(root, file)  
                        
                        # Extract FCT data  
                        fct_analyzer = FCTAnalyzer(ft_value, thre_value, version)  
                        avg_fct, fct_99 = fct_analyzer.extract_data()  
                        
                        # Extract bandwidth data  
                        bw_analyzer = BandwidthAnalyzer(ft_value, thre_value, version)  
                        avg_bw = bw_analyzer.extract_data()  
                        
                        # Store data with parameters  
                        params = f"{ft_value}/{thre_value}"  
                        version_data[version][params] = {  
                            'ft_value': ft_value,  
                            'thre_value': thre_value,  
                            'avg_fct': np.mean(avg_fct),  
                            'avg_bw': np.mean(avg_bw),  
                            'fct_99': np.mean(fct_99)
                        }  
        
        return version_data  

    def create_grouped_plots(self):  
        """Create grouped bar plots for each version"""  
        version_data = self.extract_data_for_version()  

        # First, analyze performance improvements  
        self.analyze_performance_improvements(version_data)
        
        for version, data in version_data.items():  
            if not data:  
                continue  
                
            # Prepare data for plotting  
            params = []  
            avg_fcts = []  
            avg_bws = []  
            fct_99s = []
            
            # Sort by ft_value and thre_value  
            sorted_params = sorted(data.items(),   
                                 key=lambda x: (x[1]['ft_value'], x[1]['thre_value']))  
            
            for param, values in sorted_params:  
                if(values['ft_value']==0):
                    params.append('ECMP')
                else:
                    params.append(f"{values['ft_value']}\n{values['thre_value']}")  
                avg_fcts.append(values['avg_fct'])  
                fct_99s.append(values['fct_99'])
                avg_bws.append(values['avg_bw'])  
            
 # Create figure with three subplots  
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))  
            
            # Plot Average Bandwidth (Top)  
            bars1 = ax1.bar(params, avg_bws, color='lightgreen')  
            ax1.set_title(f'Average Bandwidth ({self._get_version_display(version)})',   
                         fontsize=14, pad=20)  
            ax1.set_xlabel('Flowlet Timeout / Threshold', fontsize=12)  
            ax1.set_ylabel('Bandwidth (MB/sec)', fontsize=12)  
            ax1.grid(True, linestyle='--', alpha=0.7)  
            
            # Add value labels on bars  
            for bar in bars1:  
                height = bar.get_height()  
                ax1.text(bar.get_x() + bar.get_width()/2., height,  
                        f'{height:.2f}',  
                        ha='center', va='bottom', rotation=0)  

            # Plot Average FCT (Middle)  
            bars2 = ax2.bar(params, avg_fcts, color='skyblue')  
            ax2.set_title(f'Average FCT ({self._get_version_display(version)})',   
                         fontsize=14, pad=20)  
            ax2.set_xlabel('Flowlet Timeout / Threshold', fontsize=12)  
            ax2.set_ylabel('Average FCT (usec)', fontsize=12)  
            ax2.grid(True, linestyle='--', alpha=0.7)  
            
            # Add value labels on bars  
            for bar in bars2:  
                height = bar.get_height()  
                ax2.text(bar.get_x() + bar.get_width()/2., height,  
                        f'{height:.2f}',  
                        ha='center', va='bottom', rotation=0)  
            
            # Plot 99th Percentile FCT (Bottom)  
            bars3 = ax3.bar(params, fct_99s, color='salmon')  
            ax3.set_title(f'99th Percentile FCT ({self._get_version_display(version)})',   
                         fontsize=14, pad=20)  
            ax3.set_xlabel('Flowlet Timeout / Threshold', fontsize=12)  
            ax3.set_ylabel('99th Percentile FCT (usec)', fontsize=12)  
            ax3.grid(True, linestyle='--', alpha=0.7)  
            
            # Add value labels on bars  
            for bar in bars3:  
                height = bar.get_height()  
                ax3.text(bar.get_x() + bar.get_width()/2., height,  
                        f'{height:.2f}',  
                        ha='center', va='bottom', rotation=0)  
            
            # Adjust layout and save  
            plt.tight_layout()  
            output_dir = os.path.join(self.base_dir, version, "grouped_analysis")  
            os.makedirs(output_dir, exist_ok=True)  
            plt.savefig(os.path.join(output_dir, f'grouped_analysis_{version}.png'),   
                       dpi=300, bbox_inches='tight')  
            plt.close()  
            
            print(f"Created grouped analysis plot for {self._get_version_display(version)}")  

class AnalysisManager:  
    def __init__(self, base_dir: str = "./out/prototype"):  
        self.base_dir = base_dir  
        self.log_pattern = re.compile(r"prototype_ft_(\d+)_thre_(\d+)_(v\d+)\.log$")  
    
    def scan_logs(self) -> List[Tuple[int, int, str]]:  
        """扫描目录下的所有日志文件并提取参数"""  
        log_params = []  
        for root, _, files in os.walk(self.base_dir):  
            for file in files:  
                if file.endswith('.log'):  
                    match = self.log_pattern.match(file)  
                    if match:  
                        ft_value = int(match.group(1))  
                        thre_value = int(match.group(2))  
                        version = match.group(3)  
                        log_params.append((ft_value, thre_value, version))  
        return log_params  

    def analyze_all(self):  
        """分析所有找到的日志文件"""  
        log_params = self.scan_logs()  
        if not log_params:  
            print("No log files found matching the expected pattern.")  
            return  

        print(f"Found {len(log_params)} log files to analyze.")  
        for ft_value, thre_value, version in log_params:  
            print(f"\nAnalyzing: ft={ft_value}, threshold={thre_value}, version={version}")  
            
            try:  
                # 分析 FCT  
                fct = FCTAnalyzer(ft_value, thre_value, version)  
                fct.process_data()  
                
                # 分析带宽  
                band = BandwidthAnalyzer(ft_value, thre_value, version)  
                band.process_data()  
                
            except Exception as e:  
                print(f"Error analyzing file with parameters: "  
                      f"ft={ft_value}, threshold={thre_value}, version={version}")  
                print(f"Error details: {str(e)}")  
    
    # 以version为组进行分析绘制
    def analyze_group(self):
        print("\nGenerating grouped analysis plots...")  
        grouped_analyzer = GroupedAnalyzer(self.base_dir)  
        grouped_analyzer.create_grouped_plots()  
        


def main():  
    # 创建分析管理器并运行分析  
    manager = AnalysisManager()  
    # manager.analyze_all()  
    manager.analyze_group()

if __name__ == "__main__":  
    main()  

# if __name__ == "__main__":
#     ft_value = 7500
#     thre_value = 6000
#     version = "v5"
    
#     fct = FCTAnalyzer(ft_value, thre_value, version)
#     fct.process_data()

#     band = BandwidthAnalyzer(ft_value, thre_value, version)
#     band.process_data()