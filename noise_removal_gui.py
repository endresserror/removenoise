import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import pywt
import time
from sklearn.metrics import mean_squared_error
import os
import json
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading

# MultiModalNoiseRemoval class definition (moved from noise_removal.py)
class MultiModalNoiseRemoval:
    """
    Class for removing noise from multi-modal side-channel data from IoT devices
    Implements multiple methods and can evaluate speed and accuracy
    """
    
    def __init__(self, config_path=None):
        """Initialize function"""
        # Default settings
        self.config = {
            'moving_avg_window': 5,
            'lowpass_cutoff': 0.1,
            'median_kernel': 5,
            'wavelet_family': 'db4',
            'wavelet_level': 3,
            'kalman_process_variance': 1e-5,
            'kalman_measurement_variance': 0.1
        }
        
        # Load settings file if exists
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config.update(json.load(f))
        
        self.results = {}  # Store processing results and performance metrics
    
    def load_data(self, file_path, column_name=None):
        """Data loading function"""
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            data = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel.")
        
        # Use specific column if specified
        if column_name and column_name in data.columns:
            self.data = data[column_name].values
        else:
            # Extract only numeric data
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError("No numeric data found in the file.")
            self.data = data[numeric_cols[0]].values
        
        self.time_axis = np.arange(len(self.data))
        print(f"Loaded {len(self.data)} data points.")
        return self.data
    
    def moving_average_filter(self, data=None, window_size=None):
        """Moving average filter for noise removal"""
        if data is None:
            data = self.data
        if window_size is None:
            window_size = self.config['moving_avg_window']
        
        start_time = time.time()
        
        # Calculate moving average
        window = np.ones(window_size) / window_size
        filtered_data = np.convolve(data, window, mode='same')
        
        # Edge processing (first and last window_size/2 points are inaccurate in convolution)
        half_window = window_size // 2
        filtered_data[:half_window] = data[:half_window]
        filtered_data[-half_window:] = data[-half_window:]
        
        processing_time = time.time() - start_time
        
        return filtered_data, processing_time
    
    def lowpass_filter(self, data=None, cutoff=None, fs=1.0):
        """Lowpass filter for noise removal"""
        if data is None:
            data = self.data
        if cutoff is None:
            cutoff = self.config['lowpass_cutoff']
        
        start_time = time.time()
        
        # Design Butterworth lowpass filter
        b, a = signal.butter(4, cutoff, 'low', fs=fs)
        
        # Zero-phase filtering to prevent phase shift
        filtered_data = signal.filtfilt(b, a, data)
        
        processing_time = time.time() - start_time
        
        return filtered_data, processing_time
    
    def median_filter(self, data=None, kernel_size=None):
        """Median filter for noise removal"""
        if data is None:
            data = self.data
        if kernel_size is None:
            kernel_size = self.config['median_kernel']
        
        start_time = time.time()
        
        filtered_data = signal.medfilt(data, kernel_size=kernel_size)
        
        processing_time = time.time() - start_time
        
        return filtered_data, processing_time
    
    def wavelet_denoising(self, data=None, wavelet=None, level=None):
        """Wavelet transform for noise removal"""
        if data is None:
            data = self.data
        if wavelet is None:
            wavelet = self.config['wavelet_family']
        if level is None:
            level = self.config['wavelet_level']
        
        start_time = time.time()
        
        # Wavelet decomposition
        coeffs = pywt.wavedec(data, wavelet, level=level)
        
        # Calculate threshold (universal threshold)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(data)))
        
        # Threshold processing (soft threshold)
        new_coeffs = [coeffs[0]]  # Keep approximation coefficients
        for i in range(1, len(coeffs)):
            new_coeffs.append(pywt.threshold(coeffs[i], threshold, 'soft'))
        
        # Inverse wavelet transform
        filtered_data = pywt.waverec(new_coeffs, wavelet)
        
        # Adjust data length (inverse transform may change length)
        if len(filtered_data) > len(data):
            filtered_data = filtered_data[:len(data)]
        
        processing_time = time.time() - start_time
        
        return filtered_data, processing_time
    
    def kalman_filter(self, data=None, process_var=None, measurement_var=None):
        """Kalman filter for noise removal"""
        if data is None:
            data = self.data
        if process_var is None:
            process_var = self.config['kalman_process_variance']
        if measurement_var is None:
            measurement_var = self.config['kalman_measurement_variance']
        
        start_time = time.time()
        
        # Initialize Kalman filter
        n = len(data)
        filtered_data = np.zeros(n)
        
        # Initial state
        x_est = data[0]
        p_est = 1.0
        
        # Filtering
        for i in range(n):
            # Prediction step
            x_pred = x_est
            p_pred = p_est + process_var
            
            # Update step
            k = p_pred / (p_pred + measurement_var)
            x_est = x_pred + k * (data[i] - x_pred)
            p_est = (1 - k) * p_pred
            
            filtered_data[i] = x_est
        
        processing_time = time.time() - start_time
        
        return filtered_data, processing_time
    
    def process_all_methods(self, progress_callback=None):
        """Process data with all filtering methods"""
        if not hasattr(self, 'data') or self.data is None:
            raise ValueError("No data to process. Load data first.")
        
        self.results = {}
        total_methods = 5
        
        # Moving average filter
        if progress_callback:
            progress_callback(0.1)
        
        filtered_data, proc_time = self.moving_average_filter()
        self.results['Moving Average'] = {
            'filtered_data': filtered_data,
            'processing_time': proc_time
        }
        
        # Median filter
        if progress_callback:
            progress_callback(0.3)
        
        filtered_data, proc_time = self.median_filter()
        self.results['Median Filter'] = {
            'filtered_data': filtered_data,
            'processing_time': proc_time
        }
        
        # Lowpass filter
        if progress_callback:
            progress_callback(0.5)
        
        filtered_data, proc_time = self.lowpass_filter()
        self.results['Lowpass Filter'] = {
            'filtered_data': filtered_data,
            'processing_time': proc_time
        }
        
        # Wavelet denoising
        if progress_callback:
            progress_callback(0.7)
        
        filtered_data, proc_time = self.wavelet_denoising()
        self.results['Wavelet Filter'] = {
            'filtered_data': filtered_data,
            'processing_time': proc_time
        }
        
        # Kalman filter
        if progress_callback:
            progress_callback(0.9)
        
        filtered_data, proc_time = self.kalman_filter()
        self.results['Kalman Filter'] = {
            'filtered_data': filtered_data,
            'processing_time': proc_time
        }
        
        if progress_callback:
            progress_callback(1.0)
    
    def evaluate_methods(self, ground_truth=None):
        """Evaluate performance of each filtering method"""
        if not self.results:
            raise ValueError("No results to evaluate. Run process_all_methods first.")
        
        # If no ground truth provided, use average of all methods as reference
        if ground_truth is None:
            print("No ground truth provided. Using average of all methods as reference.")
            all_filtered = np.array([r['filtered_data'] for r in self.results.values()])
            ground_truth = np.mean(all_filtered, axis=0)
        
        # Calculate MSE and SNR for each method
        for method_name, result in self.results.items():
            filtered_data = result['filtered_data']
            
            # Mean Squared Error
            mse = mean_squared_error(ground_truth, filtered_data)
            
            # Signal-to-Noise Ratio (SNR)
            noise = filtered_data - ground_truth
            signal_power = np.mean(ground_truth ** 2)
            noise_power = np.mean(noise ** 2)
            snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
            
            # Store metrics
            self.results[method_name]['mse'] = mse
            self.results[method_name]['snr'] = snr
    
    def plot_results(self, show_original=True, show_metrics=True):
        """Visualize results"""
        if not self.results:
            raise ValueError("No results to plot. Run process_all_methods first.")
        
        # Plot settings
        plt.figure(figsize=(15, 10))
        
        # Original data display
        if show_original:
            plt.subplot(2, 1, 1)
            plt.plot(self.time_axis, self.data, 'k-', label='Original', alpha=0.5)
            plt.title('Original Signal with Noise')
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.grid(True)
            plt.legend()
        
        # Filtering results display
        plt.subplot(2, 1, 2)
        
        # Plot results for each method
        for name, result in self.results.items():
            label = f"{name}"
            if show_metrics and 'mse' in result and 'snr' in result:
                label += f" (MSE: {result['mse']:.4f}, SNR: {result['snr']:.2f}dB, Time: {result['processing_time']:.3f}s)"
            plt.plot(self.time_axis, result['filtered_data'], label=label)
        
        plt.title('Noise Removal Results')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def save_results(self, output_dir='./results'):
        """Save processing results"""
        if not self.results:
            raise ValueError("No results to save. Run process_all_methods first.")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save original data
        original_df = pd.DataFrame({
            'time': self.time_axis,
            'original': self.data
        })
        original_df.to_csv(os.path.join(output_dir, 'original_data.csv'), index=False)
        
        # Save filtered data for each method
        for method_name, result in self.results.items():
            filtered_df = pd.DataFrame({
                'time': self.time_axis,
                'filtered': result['filtered_data']
            })
            
            # Add metrics if available
            if 'mse' in result:
                filtered_df['mse'] = result['mse']
            if 'snr' in result:
                filtered_df['snr'] = result['snr']
            
            # Create safe filename
            safe_name = method_name.lower().replace(' ', '_')
            filtered_df.to_csv(os.path.join(output_dir, f'{safe_name}_result.csv'), index=False)
        
        # Save summary
        summary = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'config': self.config,
            'metrics': {name: {k: v for k, v in result.items() if k != 'filtered_data'} 
                        for name, result in self.results.items()}
        }
        
        with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Results saved to {output_dir}")


class NoiseRemovalGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("IoT Side-Channel Data Noise Removal Tool")
        self.root.geometry("1000x800")
        self.root.minsize(800, 600)
        
        # Initialize noise removal object
        self.nr = MultiModalNoiseRemoval()
        
        # Variables to hold data
        self.clean_signal = None
        self.noisy_signal = None
        self.time_axis = None
        self.has_ground_truth = False
        
        # Progress bar variable
        self.progress_var = tk.DoubleVar()
        self.progress_var.set(0)
        
        # Language resources
        self.translations = {
            "en": {
                "app_title": "IoT Side-Channel Data Noise Removal Tool",
                "settings_tab": "Settings",
                "results_tab": "Results",
                "filter_comparison_tab": "Filter Comparison",
                "file_settings": "File Settings",
                "browse_button": "Browse...",
                "signal_data_column": "Signal Data Column:",
                "reference_signal_column": "Reference Signal Column (Optional):",
                "generate_sample": "Generate Sample Data",
                "filter_parameters": "Filter Parameters",
                "moving_avg_window": "Moving Average Window Size:",
                "lowpass_cutoff": "Lowpass Cutoff Frequency:",
                "median_kernel": "Median Kernel Size:",
                "wavelet_settings": "Wavelet Settings",
                "wavelet_family": "Wavelet Family:",
                "wavelet_level": "Decomposition Level:",
                "kalman_settings": "Kalman Filter Settings",
                "process_variance": "Process Variance:",
                "measurement_variance": "Measurement Variance:",
                "run_noise_removal": "Run Noise Removal",
                "save_results": "Save Results",
                "upload_results": "Upload Results",
                "local_file_analysis": "Local File Analysis",
                "status_ready": "Ready",
                "select_file_dialog_title": "Select File",
                "loading_file_status": "Loading file",
                "error_dialog_title": "Error",
                "unsupported_file_format": "Unsupported file format.",
                "no_numeric_data": "No numeric data found in the file.",
                "select_columns_title": "Select Columns",
                "select_columns_prompt": "Select columns to display:",
                # ... more translations ...
            },
            "ja": {
                "app_title": "IoTサイドチャネルデータ ノイズ除去ツール",
                "settings_tab": "設定",
                "results_tab": "結果",
                "filter_comparison_tab": "フィルタ比較",
                "file_settings": "ファイル設定",
                "browse_button": "参照...",
                "signal_data_column": "信号データ列:",
                "reference_signal_column": "参照信号列 (オプション):",
                "generate_sample": "サンプルデータ生成",
                "filter_parameters": "フィルタパラメータ",
                "moving_avg_window": "移動平均ウィンドウサイズ:",
                "lowpass_cutoff": "ローパスカットオフ周波数:",
                "median_kernel": "メディアンカーネルサイズ:",
                "wavelet_settings": "ウェーブレット設定",
                "wavelet_family": "ウェーブレットファミリー:",
                "wavelet_level": "分解レベル:",
                "kalman_settings": "カルマンフィルタ設定",
                "process_variance": "プロセス分散:",
                "measurement_variance": "測定分散:",
                "run_noise_removal": "ノイズ除去実行",
                "save_results": "結果保存",
                "upload_results": "結果アップロード",
                "local_file_analysis": "ローカルファイル分析",
                "status_ready": "準備完了",
                "select_file_dialog_title": "ファイルを選択",
                "loading_file_status": "ファイル読み込み中",
                "error_dialog_title": "エラー",
                "unsupported_file_format": "サポートされていないファイル形式です。",
                "no_numeric_data": "ファイルに数値データが含まれていません。",
                "select_columns_title": "データ列選択",
                "select_columns_prompt": "表示する列を選択してください:",
                # ... more translations ...
            }
        }
        
        # Current language
        self.current_language = tk.StringVar(value="en")
        
        # Create menu
        self.menu_bar = tk.Menu(root)
        self.root.config(menu=self.menu_bar)
        
        # Language menu
        self.language_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Language", menu=self.language_menu)
        self.language_menu.add_radiobutton(label="English", variable=self.current_language, 
                                          value="en", command=self.update_language)
        self.language_menu.add_radiobutton(label="日本語", variable=self.current_language, 
                                          value="ja", command=self.update_language)
        
        # Build GUI
        self.setup_ui()
    
    def get_text(self, key):
        """Get translated text for the current language"""
        return self.translations[self.current_language.get()].get(key, key)
    
    def update_language(self):
        """Update UI text based on selected language"""
        # Update window title
        self.root.title(self.get_text("app_title"))
        
        # Update tab names
        self.tab_control.tab(0, text=self.get_text("settings_tab"))
        self.tab_control.tab(1, text=self.get_text("results_tab"))
        self.tab_control.tab(2, text=self.get_text("filter_comparison_tab"))
        
        # Update all labels, buttons, etc.
        self.file_frame.config(text=self.get_text("file_settings"))
        self.browse_button.config(text=self.get_text("browse_button"))
        # ... update all other UI elements ...
        
        # Update status bar
        self.status_var.set(self.get_text("status_ready"))
    
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Tab control
        tab_control = ttk.Notebook(main_frame)
        self.tab_control = tab_control
        
        # Tab 1: Settings tab
        settings_tab = ttk.Frame(tab_control)
        tab_control.add(settings_tab, text=self.get_text('settings_tab'))
        
        # Tab 2: Results tab
        results_tab = ttk.Frame(tab_control)
        tab_control.add(results_tab, text=self.get_text('results_tab'))
        
        # Tab 3: Comparison tab
        compare_tab = ttk.Frame(tab_control)
        tab_control.add(compare_tab, text=self.get_text('filter_comparison_tab'))
        
        tab_control.pack(expand=True, fill=tk.BOTH)
        
        # Explicitly select the first tab (Settings tab)
        tab_control.select(settings_tab)
        
        # Settings tab content
        self._setup_settings_tab(settings_tab)
        
        # Results tab content
        self._setup_results_tab(results_tab)
        
        # Comparison tab content
        self._setup_compare_tab(compare_tab)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set(self.get_text("status_ready"))
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Add progress bar above the status bar
        self.progress_bar = ttk.Progressbar(
            self.root, 
            variable=self.progress_var, 
            orient=tk.HORIZONTAL,
            length=100, 
            mode='determinate'
        )
        self.progress_bar.pack(side=tk.BOTTOM, fill=tk.X, before=status_bar)
    
    def _setup_settings_tab(self, parent):
        # Upper control frame
        control_frame = ttk.LabelFrame(parent, text=self.get_text("file_settings"), padding=10)
        control_frame.pack(fill=tk.X, pady=5)
        
        # Input file selection
        file_frame = ttk.Frame(control_frame)
        file_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(file_frame, text="入力ファイル:").grid(row=0, column=0, padx=5, sticky=tk.W)
        self.file_path_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path_var, width=50).grid(row=0, column=1, padx=5)
        self.browse_button = ttk.Button(file_frame, text=self.get_text("browse_button"), command=self.browse_file)
        self.browse_button.grid(row=0, column=2, padx=5)
        
        # Column selection (signal data)
        ttk.Label(file_frame, text=self.get_text("signal_data_column")).grid(row=1, column=0, padx=5, sticky=tk.W)
        self.signal_column_var = tk.StringVar()
        self.signal_column_combo = ttk.Combobox(file_frame, textvariable=self.signal_column_var, width=20)
        self.signal_column_combo.grid(row=1, column=1, padx=5, sticky=tk.W)
        
        # Ground truth (reference signal) availability
        ttk.Label(file_frame, text=self.get_text("reference_signal_column")).grid(row=2, column=0, padx=5, sticky=tk.W)
        self.ground_truth_column_var = tk.StringVar()
        self.ground_truth_combo = ttk.Combobox(file_frame, textvariable=self.ground_truth_column_var, width=20)
        self.ground_truth_combo.grid(row=2, column=1, padx=5, sticky=tk.W)
        ttk.Label(file_frame, text="(オプション)").grid(row=2, column=2, padx=5, sticky=tk.W)
        
        # Sample data button
        ttk.Button(file_frame, text=self.get_text("generate_sample"), command=self.generate_sample_data).grid(row=3, column=0, columnspan=3, pady=10)
        
        # Parameter settings frame
        param_frame = ttk.LabelFrame(parent, text=self.get_text("filter_parameters"), padding=10)
        param_frame.pack(fill=tk.X, pady=5)
        
        # Parameter grid
        params_grid = ttk.Frame(param_frame)
        params_grid.pack(fill=tk.X)
        
        # Moving average window size
        ttk.Label(params_grid, text=self.get_text("moving_avg_window")).grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        self.moving_avg_var = tk.IntVar(value=5)
        ttk.Spinbox(params_grid, from_=3, to=21, increment=2, textvariable=self.moving_avg_var, width=5).grid(row=0, column=1, padx=5, pady=2)
        
        # Lowpass cutoff
        ttk.Label(params_grid, text=self.get_text("lowpass_cutoff")).grid(row=0, column=2, padx=5, pady=2, sticky=tk.W)
        self.lowpass_var = tk.DoubleVar(value=0.1)
        ttk.Spinbox(params_grid, from_=0.01, to=0.5, increment=0.01, textvariable=self.lowpass_var, width=5).grid(row=0, column=3, padx=5, pady=2)
        
        # Median kernel size
        ttk.Label(params_grid, text=self.get_text("median_kernel")).grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        self.median_var = tk.IntVar(value=5)
        ttk.Spinbox(params_grid, from_=3, to=21, increment=2, textvariable=self.median_var, width=5).grid(row=1, column=1, padx=5, pady=2)
        
        # Wavelet family
        ttk.Label(params_grid, text=self.get_text("wavelet_family")).grid(row=1, column=2, padx=5, pady=2, sticky=tk.W)
        self.wavelet_var = tk.StringVar(value='db4')
        wavelet_combo = ttk.Combobox(params_grid, textvariable=self.wavelet_var, width=5)
        wavelet_combo['values'] = ('db1', 'db2', 'db4', 'db8', 'sym4', 'coif3', 'haar')
        wavelet_combo.grid(row=1, column=3, padx=5, pady=2)
        
        # Wavelet level
        ttk.Label(params_grid, text=self.get_text("wavelet_level")).grid(row=2, column=0, padx=5, pady=2, sticky=tk.W)
        self.wavelet_level_var = tk.IntVar(value=3)
        ttk.Spinbox(params_grid, from_=1, to=8, textvariable=self.wavelet_level_var, width=5).grid(row=2, column=1, padx=5, pady=2)
        
        # Kalman process variance
        ttk.Label(params_grid, text=self.get_text("process_variance")).grid(row=2, column=2, padx=5, pady=2, sticky=tk.W)
        self.kalman_process_var = tk.DoubleVar(value=1e-5)
        ttk.Entry(params_grid, textvariable=self.kalman_process_var, width=8).grid(row=2, column=3, padx=5, pady=2)
        
        # Kalman measurement variance
        ttk.Label(params_grid, text=self.get_text("measurement_variance")).grid(row=3, column=0, padx=5, pady=2, sticky=tk.W)
        self.kalman_meas_var = tk.DoubleVar(value=0.1)
        ttk.Entry(params_grid, textvariable=self.kalman_meas_var, width=8).grid(row=3, column=1, padx=5, pady=2)
        
        # Processing button frame
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=10)
        
        # Run processing button
        self.process_button = ttk.Button(button_frame, text=self.get_text("run_noise_removal"), command=self.process_data)
        self.process_button.pack(side=tk.LEFT, padx=5)
    
    def _setup_results_tab(self, parent):
        # Result display Matplotlib canvas
        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Button frame
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=5)
        
        # Save button and upload button
        self.save_button = ttk.Button(button_frame, text=self.get_text("save_results"), command=self.save_results)
        self.save_button.pack(side=tk.RIGHT, padx=5)
        
        # Add upload button
        self.upload_button = ttk.Button(button_frame, text=self.get_text("upload_results"), command=self.upload_results)
        self.upload_button.pack(side=tk.RIGHT, padx=5)
        
        # Add local file load button
        self.load_local_button = ttk.Button(button_frame, text=self.get_text("local_file_analysis"), command=self.load_local_file)
        self.load_local_button.pack(side=tk.RIGHT, padx=5)
        
        # Initially disable save and upload buttons
        self.save_button.config(state=tk.DISABLED)
        self.upload_button.config(state=tk.DISABLED)
    
    def _setup_compare_tab(self, parent):
        # Comparison graph frame
        compare_frame = ttk.Frame(parent)
        compare_frame.pack(fill=tk.BOTH, expand=True)
        
        # Comparison graph
        self.compare_figure = Figure(figsize=(10, 8), dpi=100)
        self.compare_canvas = FigureCanvasTkAgg(self.compare_figure, compare_frame)
        self.compare_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def browse_file(self):
        """ファイル選択ダイアログを表示"""
        filetypes = [
            ('CSVファイル', '*.csv'),
            ('Excelファイル', '*.xlsx *.xls'),
            ('すべてのファイル', '*.*')
        ]
        file_path = filedialog.askopenfilename(filetypes=filetypes)
        if file_path:
            self.file_path_var.set(file_path)
            self.status_var.set(f"ファイル選択: {file_path}")
            self.load_file_columns(file_path)
    
    def load_file_columns(self, file_path):
        """ファイルから列情報を読み込んでコンボボックスを更新"""
        try:
            if (file_path.endswith('.csv')):
                data = pd.read_csv(file_path)
            elif (file_path.endswith('.xlsx') or file_path.endswith('.xls')):
                data = pd.read_excel(file_path)
            else:
                messagebox.showerror("エラー", "サポートされていないファイル形式です。")
                return
            
            # 数値列のみ取得
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_cols:
                messagebox.showerror("エラー", "ファイルに数値データが含まれていません。")
                return
            
            
            self.signal_column_combo['values'] = numeric_cols
            self.ground_truth_combo['values'] = [''] + numeric_cols
            
            
            self.signal_column_var.set(numeric_cols[0])
            self.ground_truth_column_var.set('')
            
            self.status_var.set(f"ファイル読み込み完了: {len(data)}行, {len(numeric_cols)}列の数値データ")
            
        except Exception as e:
            messagebox.showerror("エラー", f"ファイル読み込み中にエラーが発生しました: {str(e)}")
    
    def generate_sample_data(self):
        """サンプルデータの生成と保存"""
        try:
            
            np.random.seed(42)
            t = np.linspace(0, 10, 1000)
            
            
            clean_signal = np.sin(2 * np.pi * 0.5 * t) + signal.sawtooth(2 * np.pi * 0.2 * t)
            
            
            noise = 0.5 * np.random.normal(0, 1, len(t))
            noisy_signal = clean_signal + noise
            
            
            sample_df = pd.DataFrame({
                'time': t,
                'clean': clean_signal,
                'noisy': noisy_signal
            })
            
            
            sample_dir = './sample_data'
            os.makedirs(sample_dir, exist_ok=True)
            sample_path = os.path.join(sample_dir, 'sample_signal.csv')
            
            sample_df.to_csv(sample_path, index=False)
            
            
            self.file_path_var.set(sample_path)
            self.load_file_columns(sample_path)
            
            
            self.signal_column_var.set('noisy')
            self.ground_truth_column_var.set('clean')
            
            
            self.preview_data(t, clean_signal, noisy_signal)
            
            self.status_var.set(f"サンプルデータ生成完了: {sample_path}")
            messagebox.showinfo("サンプルデータ", f"サンプルデータを生成しました: {sample_path}\n信号列: 'noisy'\n参照信号列: 'clean'")
            
        except Exception as e:
            messagebox.showerror("エラー", f"サンプルデータ生成中にエラーが発生しました: {str(e)}")
    
    def preview_data(self, time, clean=None, noisy=None):
        """データのプレビュー表示"""
        self.figure.clear()
        
        ax = self.figure.add_subplot(111)
        if noisy is not None:
            ax.plot(time, noisy, 'k-', label='Noisy Signal', alpha=0.7)
        if clean is not None:
            ax.plot(time, clean, 'g-', label='Clean Signal', alpha=0.7)
        
        ax.set_title('データプレビュー')
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')
        ax.grid(True)
        ax.legend()
        
        self.canvas.draw()
    
    def process_data(self):
        """データ処理実行（非同期処理）"""
        if not self.file_path_var.get():
            messagebox.showerror("エラー", "処理するファイルを選択してください。")
            return
        
        
        self.process_button.config(state=tk.DISABLED)
        self.status_var.set("処理中...")
        self.progress_var.set(0)
        
       
        threading.Thread(target=self._process_data_thread, daemon=True).start()
    
    def _process_data_thread(self):
       
        try:
            # パラメータの更新
            self.nr.config.update({
                'moving_avg_window': self.moving_avg_var.get(),
                'lowpass_cutoff': self.lowpass_var.get(),
                'median_kernel': self.median_var.get(),
                'wavelet_family': self.wavelet_var.get(),
                'wavelet_level': self.wavelet_level_var.get(),
                'kalman_process_variance': self.kalman_process_var.get(),
                'kalman_measurement_variance': self.kalman_meas_var.get()
            })
            
           
            self.progress_var.set(20)
            self.root.update_idletasks()
            
            
            file_path = self.file_path_var.get()
            signal_column = self.signal_column_var.get()
            ground_truth_column = self.ground_truth_column_var.get()
            
            
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                data = pd.read_excel(file_path)
            else:
                messagebox.showerror("エラー", "サポートされていないファイル形式です。")
                return
            
            
            if 'time' in data.columns:
                self.time_axis = data['time'].values
            else:
                self.time_axis = np.arange(len(data))
            
            
            if signal_column and signal_column in data.columns:
                self.noisy_signal = data[signal_column].values
            else:
                messagebox.showerror("エラー", "指定された信号列が見つかりません。")
                return
            
            
            if ground_truth_column and ground_truth_column in data.columns:
                self.clean_signal = data[ground_truth_column].values
                self.has_ground_truth = True
            else:
                self.clean_signal = None
                self.has_ground_truth = False
            
            
            self.nr.data = self.noisy_signal
            self.nr.time_axis = self.time_axis
            
            self.progress_var.set(40)
            self.root.update_idletasks()
            
            
            self.nr.process_all_methods(progress_callback=self._update_progress)
            
            
            if self.has_ground_truth:
                self.nr.evaluate_methods(ground_truth=self.clean_signal)
            else:
                self.nr.evaluate_methods()  
            
            self.progress_var.set(90)
            self.root.update_idletasks()
            
            
            self.root.after(0, self._finish_processing)
            
        except Exception as e:
            
            self.root.after(0, lambda: self._show_error(str(e)))
    
    def _update_progress(self, value):
        """進捗状況を更新するコールバック (40-90%の範囲で使用)"""
        
        progress = 40 + value * 50
        self.progress_var.set(progress)
        self.root.update_idletasks()
    
    def _finish_processing(self):
        """処理完了後のUI更新"""
        self.display_results()
        self.save_button.config(state=tk.NORMAL)
        self.upload_button.config(state=tk.NORMAL)  
        self.progress_var.set(100)
        self.status_var.set("処理完了。結果を確認してください。")
        
        
        for tab_id, tab_text in enumerate(self.tab_control.tabs()):
            if self.tab_control.tab(tab_id, "text") == self.get_text("results_tab"):
                self.tab_control.select(tab_id)
                break
    
    def _show_error(self, error_msg):
        """エラー表示 (UIスレッドで実行)"""
        messagebox.showerror("エラー", f"処理中にエラーが発生しました: {error_msg}")
        self.status_var.set(f"エラー: {error_msg}")
        self.process_button.config(state=tk.NORMAL)
        self.progress_var.set(0)
    
    def display_results(self):
        """処理結果の表示"""
        self.figure.clear()
        
        
        ax1 = self.figure.add_subplot(211)
        ax1.plot(self.time_axis, self.noisy_signal, 'k-', label='Noisy Signal', alpha=0.5)
        if self.has_ground_truth:
            ax1.plot(self.time_axis, self.clean_signal, 'g-', label='Clean Signal (Reference)', alpha=0.7)
        ax1.set_title('Original Signal')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True)
        ax1.legend()
        
        
        ax2 = self.figure.add_subplot(212)
        
        
        for name, result in self.nr.results.items():
            label = f"{name}"
            if 'mse' in result and 'snr' in result:
                label += f" (MSE: {result['mse']:.4f}, SNR: {result['snr']:.2f}dB)"
            ax2.plot(self.time_axis, result['filtered_data'], label=label)
        
        ax2.set_title('Noise Removal Results')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Amplitude')
        ax2.grid(True)
        ax2.legend()
        
        self.figure.tight_layout()
        self.canvas.draw()
        
        
        if 'mse' in next(iter(self.nr.results.values())):
            best_method = min(self.nr.results.items(), key=lambda x: x[1]['mse'])[0]
            messagebox.showinfo("処理結果", f"最も性能の良い手法: {best_method}\n\n" + 
                              "\n".join([f"{name}: MSE = {r['mse']:.6f}, SNR = {r['snr']:.2f}dB, 処理時間 = {r['processing_time']:.3f}秒" 
                                        for name, r in sorted(self.nr.results.items(), key=lambda x: x[1]['mse'])]))
    
    def save_results(self):
        """結果の保存"""
        try:
            
            output_dir = filedialog.askdirectory(title="結果の保存先を選択")
            if not output_dir:
                return
            
            
            self.nr.save_results(output_dir=output_dir)
            
            self.status_var.set(f"結果を保存しました: {output_dir}")
            messagebox.showinfo("保存完了", f"処理結果を以下に保存しました:\n{output_dir}")
            
        except Exception as e:
            messagebox.showerror("エラー", f"結果保存中にエラーが発生しました: {str(e)}")

    def upload_results(self):
        """結果データまたはローカルファイルをサーバーにアップロードする機能"""
       
        upload_dialog = tk.Toplevel(self.root)
        upload_dialog.title("結果アップロード設定")
        upload_dialog.geometry("500x450")
        upload_dialog.transient(self.root)  
        upload_dialog.grab_set()  
        
        
        ttk.Label(upload_dialog, text="処理結果またはローカルファイルをクラウドサーバーにアップロードします。\n"
                                     "アップロードしたデータは研究目的で利用される場合があります。", 
                  wraplength=480).pack(pady=10, padx=10)
        
        
        source_frame = ttk.LabelFrame(upload_dialog, text="アップロードソース", padding=10)
        source_frame.pack(fill=tk.X, padx=10, pady=5)
        
        source_var = tk.StringVar(value="result")
        ttk.Radiobutton(source_frame, text="現在の処理結果をアップロード", variable=source_var, 
                       value="result").pack(anchor=tk.W, padx=5, pady=2)
        ttk.Radiobutton(source_frame, text="ローカルファイルからアップロード", variable=source_var, 
                       value="file").pack(anchor=tk.W, padx=5, pady=2)
        
       
        file_frame = ttk.Frame(source_frame)
        file_path_var = tk.StringVar()
        
        def toggle_file_frame(*args):
            if source_var.get() == "file":
                file_frame.pack(fill=tk.X, padx=5, pady=5)
            else:
                file_frame.pack_forget()
        
        
        source_var.trace("w", toggle_file_frame)
        
        
        ttk.Label(file_frame, text="アップロードファイル:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(file_frame, textvariable=file_path_var, width=30).grid(row=0, column=1, padx=5, pady=5)
        
        def browse_upload_file():
            filetypes = [
                ('CSVファイル', '*.csv'),
                ('JSONファイル', '*.json'),
                ('テキストファイル', '*.txt'),
                ('すべてのファイル', '*.*')
            ]
            file_path = filedialog.askopenfilename(filetypes=filetypes, parent=upload_dialog)
            if file_path:
                file_path_var.set(file_path)
        
        ttk.Button(file_frame, text="参照...", command=browse_upload_file).grid(row=0, column=2, padx=5, pady=5)
        
        
        server_frame = ttk.LabelFrame(upload_dialog, text="サーバー設定", padding=10)
        server_frame.pack(fill=tk.X, padx=10, pady=5)
        
        
        ttk.Label(server_frame, text="サーバーURL:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        server_url_var = tk.StringVar(value="https://example.com/api/upload")
        ttk.Entry(server_frame, textvariable=server_url_var, width=40).grid(row=0, column=1, padx=5, pady=5)
        
        
        ttk.Label(server_frame, text="API Key:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        api_key_var = tk.StringVar()
        ttk.Entry(server_frame, textvariable=api_key_var, show="*", width=40).grid(row=1, column=1, padx=5, pady=5)
        
        
        method_vars = {}
        data_frame = ttk.LabelFrame(upload_dialog, text="アップロードデータ設定", padding=10)
        
        def setup_method_options():
            if source_var.get() == "result":
                data_frame.pack(fill=tk.X, padx=10, pady=5)
                
                if hasattr(self, 'nr') and self.nr.results:
                    
                    ttk.Label(data_frame, text="アップロードする手法:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
                    
                    
                    for i, method_name in enumerate(self.nr.results.keys()):
                        if method_name not in method_vars:
                            method_vars[method_name] = tk.BooleanVar(value=True)
                        ttk.Checkbutton(data_frame, text=method_name, variable=method_vars[method_name]).grid(
                            row=i//2 + 1, column=i%2, padx=5, pady=2, sticky=tk.W)
                else:
                    ttk.Label(data_frame, text="処理結果がありません。先にノイズ除去処理を実行してください。").grid(
                        row=0, column=0, columnspan=2, padx=5, pady=5)
            else:
                data_frame.pack_forget()
        
        
        source_var.trace("w", lambda *args: setup_method_options())
        
        
        meta_frame = ttk.LabelFrame(upload_dialog, text="メタデータ", padding=10)
        meta_frame.pack(fill=tk.X, padx=10, pady=5)
        
        
        ttk.Label(meta_frame, text="メタデータ (JSON):").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        metadata_text = tk.Text(meta_frame, height=5, width=50)
        metadata_text.grid(row=1, column=0, columnspan=2, padx=5, pady=5)
        metadata_text.insert(tk.END, '{\n  "device": "example-device",\n  "purpose": "research"\n}')
        
        
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(upload_dialog, variable=progress_var, orient=tk.HORIZONTAL, length=100, mode='determinate')
        progress_bar.pack(fill=tk.X, padx=10, pady=5)
        
        
        status_var = tk.StringVar(value="アップロード設定を入力してください")
        status_label = ttk.Label(upload_dialog, textvariable=status_var, anchor=tk.W)
        status_label.pack(fill=tk.X, padx=10, pady=5)
        
        
        def do_upload():
            
            if not server_url_var.get():
                messagebox.showerror("エラー", "サーバーURLを入力してください。", parent=upload_dialog)
                return
            
            
            if source_var.get() == "result":
                if not hasattr(self, 'nr') or not self.nr.results:
                    messagebox.showerror("エラー", "アップロードする処理結果がありません。", parent=upload_dialog)
                    return
                
                
                selected_methods = [name for name, var in method_vars.items() if var.get()]
                if not selected_methods:
                    messagebox.showerror("エラー", "アップロードする手法を少なくとも1つ選択してください。", parent=upload_dialog)
                    return
            else:  
                if not file_path_var.get():
                    messagebox.showerror("エラー", "アップロードするファイルを選択してください。", parent=upload_dialog)
                    return
                
                
                if not os.path.exists(file_path_var.get()):
                    messagebox.showerror("エラー", "指定されたファイルが見つかりません。", parent=upload_dialog)
                    return
            
            try:
                
                metadata_str = metadata_text.get("1.0", tk.END)
                try:
                    metadata = json.loads(metadata_str)
                except json.JSONDecodeError:
                    messagebox.showerror("エラー", "メタデータのJSON形式が不正です。", parent=upload_dialog)
                    return
                
                
                if source_var.get() == "result":
                    confirm_msg = f"以下の設定で処理結果をアップロードします。よろしいですか？\n\n" \
                                 f"・サーバー: {server_url_var.get()}\n" \
                                 f"・選択手法: {', '.join(selected_methods)}\n"
                else:
                    confirm_msg = f"以下の設定でファイルをアップロードします。よろしいですか？\n\n" \
                                 f"・サーバー: {server_url_var.get()}\n" \
                                 f"・ファイル: {file_path_var.get()}\n"
                
                
                confirm = messagebox.askyesno("確認", confirm_msg, parent=upload_dialog)
                if not confirm:
                    return
                
                
                upload_button.config(state=tk.DISABLED)
                cancel_button.config(state=tk.DISABLED)
                
                
                progress_var.set(10)
                status_var.set("アップロード準備中...")
                upload_dialog.update_idletasks()
                
                
                time.sleep(0.5)  
                progress_var.set(30)
                
                if source_var.get() == "result":
                    status_var.set("処理結果データ変換中...")
                    
                    
                else:
                    status_var.set("ファイル読み込み中...")
                    
                    
                
                upload_dialog.update_idletasks()
                
                
                time.sleep(1)
                progress_var.set(60)
                status_var.set("サーバーに送信中...")
                upload_dialog.update_idletasks()
                
                
                time.sleep(0.8) 
                progress_var.set(100)
                status_var.set("アップロード完了！")
                upload_dialog.update_idletasks()
                
                messagebox.showinfo("完了", 
                    f"データのアップロードが完了しました。\n\n"
                    f"・アップロード先: {server_url_var.get()}\n"
                    f"・アップロード手法数: {len(selected_methods)}",
                    parent=upload_dialog)
                
                upload_dialog.destroy()
                
            except Exception as e:
                messagebox.showerror("エラー", f"アップロード中にエラーが発生しました: {str(e)}", parent=upload_dialog)
                status_var.set(f"エラー: {str(e)}")
                upload_button.config(state=tk.NORMAL)
                cancel_button.config(state=tk.NORMAL)
        
        button_frame = ttk.Frame(upload_dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        upload_button = ttk.Button(button_frame, text="アップロード実行", command=do_upload)
        upload_button.pack(side=tk.RIGHT, padx=5)
        
        cancel_button = ttk.Button(button_frame, text="キャンセル", command=upload_dialog.destroy)
        cancel_button.pack(side=tk.RIGHT, padx=5)
        upload_dialog.focus_set()

    def load_local_file(self):
        """Load local file directly for analysis in results tab"""
        filetypes = [
            ('CSV Files', '*.csv'),
            ('Excel Files', '*.xlsx *.xls'),
            ('All Files', '*.*')
        ]
        file_path = filedialog.askopenfilename(filetypes=filetypes, title=self.get_text("select_file_dialog_title"))
        if not file_path:
            return
        
        try:
            self.status_var.set(self.get_text("loading_file_status") + f": {file_path}")
            self.progress_var.set(20)
            self.root.update_idletasks()

            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                data = pd.read_excel(file_path)
            else:
                messagebox.showerror(self.get_text("error_dialog_title"), 
                                    self.get_text("unsupported_file_format"))
                return
            
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                messagebox.showerror(self.get_text("error_dialog_title"), 
                                    self.get_text("no_numeric_data"))
                return
            
            select_dialog = tk.Toplevel(self.root)
            select_dialog.title(self.get_text("select_columns_title"))
            select_dialog.geometry("400x300")
            select_dialog.transient(self.root)
            select_dialog.grab_set()
            
            ttk.Label(select_dialog, text=self.get_text("select_columns_prompt")).pack(pady=10, padx=10)
            
            columns_frame = ttk.Frame(select_dialog)
            columns_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            columns_listbox = tk.Listbox(columns_frame, selectmode=tk.MULTIPLE, height=10)
            columns_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            scrollbar = ttk.Scrollbar(columns_frame, orient=tk.VERTICAL, command=columns_listbox.yview)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            columns_listbox.config(yscrollcommand=scrollbar.set)
            
            for col in numeric_cols:
                columns_listbox.insert(tk.END, col)
            
            time_frame = ttk.Frame(select_dialog)
            time_frame.pack(fill=tk.X, padx=10, pady=5)
            
            ttk.Label(time_frame, text="時間軸として使用する列:").pack(side=tk.LEFT)
            time_var = tk.StringVar(value="index")
            time_combo = ttk.Combobox(time_frame, textvariable=time_var, width=15)
            time_combo['values'] = ['index'] + numeric_cols
            time_combo.pack(side=tk.LEFT, padx=5)
            
            btn_frame = ttk.Frame(select_dialog)
            btn_frame.pack(fill=tk.X, padx=10, pady=10)
            
            def plot_selected():
                selected_indices = columns_listbox.curselection()
                if not selected_indices:
                    messagebox.showerror(self.get_text("error_dialog_title"), 
                                        self.get_text("select_columns_prompt"), parent=select_dialog)
                    return
                
                selected_columns = [numeric_cols[i] for i in selected_indices]

                if time_var.get() == "index":
                    time_data = np.arange(len(data))
                else:
                    time_data = data[time_var.get()].values

                select_dialog.destroy()
                
                
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                
                for col in selected_columns:
                    ax.plot(time_data, data[col].values, label=col)
                
                ax.set_title('ローカルファイルデータ')
                ax.set_xlabel('Time')
                ax.set_ylabel('Value')
                ax.grid(True)
                ax.legend()
                
                self.figure.tight_layout()
                self.canvas.draw()
                
                
                self.status_var.set(f"ローカルファイルを表示: {os.path.basename(file_path)}")
                self.progress_var.set(0)
            
            ttk.Button(btn_frame, text="プロット", command=plot_selected).pack(side=tk.RIGHT, padx=5)
            ttk.Button(btn_frame, text="キャンセル", command=select_dialog.destroy).pack(side=tk.RIGHT, padx=5)
            
            
            self.progress_var.set(0)
            self.status_var.set(f"ファイル読み込み完了: {os.path.basename(file_path)}")
            
        except Exception as e:
            messagebox.showerror(self.get_text("error_dialog_title"), 
                                f"{self.get_text('loading_file_status')} {str(e)}")
            self.progress_var.set(0)
            self.status_var.set("エラーが発生しました")



if __name__ == "__main__":
    root = tk.Tk()
    app = NoiseRemovalGUI(root)
    root.mainloop()