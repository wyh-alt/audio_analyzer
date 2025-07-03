import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QTableWidget, 
                            QTableWidgetItem, QPushButton, QVBoxLayout, QHBoxLayout, 
                            QWidget, QLabel, QHeaderView, QMessageBox, QProgressBar,
                            QStatusBar, QGroupBox, QSizePolicy, QSpinBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMimeData, QUrl, QThreadPool, QRunnable, QObject, QMutex
from PyQt5.QtGui import QDragEnterEvent, QDropEvent, QFont, QIcon
import librosa
import soundfile as sf
import threading
import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

# 创建信号类，用于线程间通信
class WorkerSignals(QObject):
    """
    定义工作线程的信号
    """
    result = pyqtSignal(dict)
    progress = pyqtSignal(int, int)  # 修改为发送文件索引和总文件数
    finished = pyqtSignal()
    error = pyqtSignal(str)

class AudioAnalysisWorker(QRunnable):
    """单个音频文件的分析工作线程"""
    
    def __init__(self, file_path, file_index, total_files):
        super().__init__()
        self.file_path = file_path
        self.file_index = file_index
        self.total_files = total_files
        self.signals = WorkerSignals()
        self.is_cancelled = False
        
    def run(self):
        if self.is_cancelled:
            return
            
        try:
            result = self.analyze_audio(self.file_path)
            self.signals.result.emit(result)
        except Exception as e:
            if not self.is_cancelled:
                error_result = {
                    'filename': os.path.basename(self.file_path),
                    'filepath': self.file_path,
                    'audio_type': '分析错误',
                    'channels': '-',
                    'sample_rate': '-',
                    'duration': '-',
                    'error': str(e)
                }
                self.signals.result.emit(error_result)
        
        # 更新进度 - 发送文件索引和总文件数，而不是百分比
        if not self.is_cancelled:
            self.signals.progress.emit(self.file_index + 1, self.total_files)
    
    def cancel(self):
        """取消分析任务"""
        self.is_cancelled = True
            
    def analyze_audio(self, file_path):
        """分析音频文件并判断声道类型"""
        try:
            # 加载音频文件
            y, sr = librosa.load(file_path, mono=False, sr=None)
            
            # 获取文件信息
            with sf.SoundFile(file_path) as sf_file:
                channels = sf_file.channels
                sample_rate = sf_file.samplerate
                duration = len(sf_file) / sf_file.samplerate
            
            # 判断声道类型
            audio_type = "未知"
            
            if channels == 1:
                audio_type = "单声道"
            elif channels == 2:
                # 检查是否为假立体声
                # 如果是数组则表示多声道，否则是单声道已经被librosa转换
                if isinstance(y, np.ndarray) and y.ndim > 1:
                    # 计算左右声道的相似度
                    correlation = np.corrcoef(y[0], y[1])[0, 1]
                    if correlation > 0.98:  # 设定阈值，高度相似认为是假立体声
                        audio_type = "假立体声"
                    else:
                        audio_type = "立体声"
                else:
                    audio_type = "单声道"
            elif channels == 6:
                audio_type = "5.1环绕声"
            elif channels == 8:
                audio_type = "7.1环绕声"
            else:
                audio_type = f"{channels}声道"
                
            return {
                'filename': os.path.basename(file_path),
                'filepath': file_path,
                'audio_type': audio_type,
                'channels': channels,
                'sample_rate': f"{sample_rate/1000:.1f}kHz",
                'duration': f"{duration:.2f}秒",
                'error': None
            }
        except Exception as e:
            raise Exception(f"分析失败: {str(e)}")


class AudioAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("音频声道检测工具")
        self.setMinimumSize(900, 600)
        self.setAcceptDrops(True)
        
        # 设置应用图标
        try:
            if os.path.exists("app_icon.png"):
                self.setWindowIcon(QIcon("app_icon.png"))
            elif os.path.exists("app_icon.ico"):
                self.setWindowIcon(QIcon("app_icon.ico"))
            elif os.path.exists("app_icon_256x256.png"):
                self.setWindowIcon(QIcon("app_icon_256x256.png"))
        except Exception:
            # 如果加载图标失败，忽略错误继续运行
            pass
        
        self.results = []
        self.thread_pool = QThreadPool()
        # 设置默认线程数为CPU核心数的一半
        default_threads = max(2, multiprocessing.cpu_count() // 2)
        self.thread_pool.setMaxThreadCount(default_threads)
        self.completed_tasks = 0
        self.total_tasks = 0
        self.active_workers = []
        self.is_analyzing = False
        self.progress_mutex = QMutex()  # 添加互斥锁保护进度更新
        self.current_progress = 0  # 当前进度值
        
        self.init_ui()
    
    def init_ui(self):
        """初始化UI界面"""
        # 创建中央部件和主布局
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # 创建控制按钮区域
        controls_layout = QHBoxLayout()
        
        self.select_btn = QPushButton("选择音频文件")
        self.select_btn.clicked.connect(self.select_files)
        
        self.select_folder_btn = QPushButton("选择文件夹")
        self.select_folder_btn.clicked.connect(self.select_folder)
        
        self.stop_btn = QPushButton("停止分析")
        self.stop_btn.clicked.connect(self.stop_analysis)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("background-color: #d9534f;")
        
        self.export_btn = QPushButton("导出Excel")
        self.export_btn.clicked.connect(self.export_excel)
        self.export_btn.setEnabled(False)
        
        # 添加线程数控制
        thread_layout = QHBoxLayout()
        thread_label = QLabel("线程数:")
        self.thread_spinbox = QSpinBox()
        self.thread_spinbox.setMinimum(1)
        self.thread_spinbox.setMaximum(multiprocessing.cpu_count())
        self.thread_spinbox.setValue(self.thread_pool.maxThreadCount())
        self.thread_spinbox.valueChanged.connect(self.update_thread_count)
        
        thread_layout.addWidget(thread_label)
        thread_layout.addWidget(self.thread_spinbox)
        
        # 添加控件到布局
        controls_layout.addWidget(self.select_btn)
        controls_layout.addWidget(self.select_folder_btn)
        controls_layout.addWidget(self.stop_btn)
        controls_layout.addLayout(thread_layout)
        controls_layout.addStretch()
        controls_layout.addWidget(self.export_btn)
        
        # 创建进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        # 创建表格
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["文件名", "音频类型", "声道数", "采样率", "时长", "文件路径"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        
        # 创建统计信息区域
        self.stats_group = QGroupBox("统计信息")
        stats_layout = QHBoxLayout()
        self.stats_label = QLabel("未分析任何文件")
        stats_layout.addWidget(self.stats_label)
        self.stats_group.setLayout(stats_layout)
        
        # 创建状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("准备就绪，请选择音频文件或拖放文件到窗口")
        
        # 添加所有组件到主布局
        main_layout.addLayout(controls_layout)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.table)
        main_layout.addWidget(self.stats_group)
        
        # 设置中央部件
        self.setCentralWidget(central_widget)
    
    def update_thread_count(self, value):
        """更新线程池的最大线程数"""
        self.thread_pool.setMaxThreadCount(value)
        self.status_bar.showMessage(f"线程数已设置为 {value}")
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """处理拖拽进入事件"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event: QDropEvent):
        """处理文件拖放事件"""
        if self.is_analyzing:
            QMessageBox.warning(self, "警告", "正在进行分析，请等待当前分析完成或点击停止")
            return
            
        file_paths = []
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if os.path.isdir(path):
                self.process_folder(path, file_paths)
            else:
                file_paths.append(path)
        
        if file_paths:
            self.analyze_audio_files(file_paths)
    
    def process_folder(self, folder_path, file_paths):
        """处理文件夹，提取所有音频文件"""
        audio_extensions = ['.mp3', '.wav', '.flac', '.ogg', '.aac', '.m4a', '.wma']
        for root, _, files in os.walk(folder_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in audio_extensions):
                    file_paths.append(os.path.join(root, file))
    
    def select_files(self):
        """选择音频文件"""
        if self.is_analyzing:
            QMessageBox.warning(self, "警告", "正在进行分析，请等待当前分析完成或点击停止")
            return
            
        file_dialog = QFileDialog()
        file_paths, _ = file_dialog.getOpenFileNames(
            self, 
            "选择音频文件",
            "",
            "音频文件 (*.mp3 *.wav *.flac *.ogg *.aac *.m4a *.wma);;所有文件 (*)"
        )
        
        if file_paths:
            self.analyze_audio_files(file_paths)
    
    def select_folder(self):
        """选择文件夹"""
        if self.is_analyzing:
            QMessageBox.warning(self, "警告", "正在进行分析，请等待当前分析完成或点击停止")
            return
            
        folder_dialog = QFileDialog()
        folder_path = folder_dialog.getExistingDirectory(self, "选择文件夹")
        
        if folder_path:
            file_paths = []
            self.process_folder(folder_path, file_paths)
            
            if file_paths:
                self.analyze_audio_files(file_paths)
            else:
                QMessageBox.information(self, "提示", "所选文件夹中未找到音频文件")
    
    def stop_analysis(self):
        """停止当前分析任务"""
        if not self.is_analyzing:
            return
            
        reply = QMessageBox.question(
            self, 
            "确认停止", 
            "确定要停止当前分析任务吗？\n已分析的结果将被保留。",
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # 取消所有工作线程
            for worker in self.active_workers:
                if hasattr(worker, 'cancel'):
                    worker.cancel()
            
            # 清空工作线程列表
            self.active_workers = []
            
            # 重置线程池
            self.thread_pool.clear()
            
            # 更新UI状态
            self.is_analyzing = False
            self.select_btn.setEnabled(True)
            self.select_folder_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.thread_spinbox.setEnabled(True)
            
            if self.results:
                self.export_btn.setEnabled(True)
            
            self.status_bar.showMessage("分析已停止")
            self.progress_bar.setVisible(False)
            
            # 更新统计信息
            self.update_statistics()
    
    def analyze_audio_files(self, file_paths):
        """分析音频文件"""
        if not file_paths:
            return
        
        if self.is_analyzing:
            return
            
        self.is_analyzing = True
        
        # 清空表格和结果
        self.table.setRowCount(0)
        self.results = []
        self.completed_tasks = 0
        self.total_tasks = len(file_paths)
        self.active_workers = []
        self.current_progress = 0  # 重置当前进度
        
        # 预先创建表格行
        self.table.setRowCount(self.total_tasks)
        for i in range(self.total_tasks):
            for j in range(6):
                if j == 0:
                    self.table.setItem(i, j, QTableWidgetItem("分析中..."))
                else:
                    self.table.setItem(i, j, QTableWidgetItem(""))
        
        # 显示进度条
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        
        # 禁用按钮
        self.select_btn.setEnabled(False)
        self.select_folder_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.thread_spinbox.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        # 更新状态栏
        self.status_bar.showMessage(f"正在使用 {self.thread_pool.maxThreadCount()} 个线程分析 {self.total_tasks} 个文件...")
        
        # 使用线程池处理文件
        for i, file_path in enumerate(file_paths):
            worker = AudioAnalysisWorker(file_path, i, self.total_tasks)
            
            # 连接信号
            worker.signals.result.connect(self.handle_result)
            worker.signals.progress.connect(self.update_progress)
            
            # 添加到活动工作线程列表
            self.active_workers.append(worker)
            
            # 提交到线程池
            self.thread_pool.start(worker)
    
    def handle_result(self, result):
        """处理单个文件的分析结果"""
        self.results.append(result)
        self.completed_tasks += 1
        
        # 更新表格中的结果
        for row, res in enumerate(self.results):
            filename = res.get('filename', '')
            
            # 查找对应的行
            for i in range(self.table.rowCount()):
                if self.table.item(i, 0) and (self.table.item(i, 0).text() == "分析中..." or self.table.item(i, 0).text() == filename):
                    self.table.setItem(i, 0, QTableWidgetItem(res['filename']))
                    self.table.setItem(i, 1, QTableWidgetItem(res['audio_type']))
                    self.table.setItem(i, 2, QTableWidgetItem(str(res['channels'])))
                    self.table.setItem(i, 3, QTableWidgetItem(str(res['sample_rate'])))
                    self.table.setItem(i, 4, QTableWidgetItem(str(res['duration'])))
                    self.table.setItem(i, 5, QTableWidgetItem(res['filepath']))
                    
                    # 根据音频类型设置不同颜色
                    if res['audio_type'] == "假立体声":
                        for col in range(6):
                            self.table.item(i, col).setBackground(Qt.yellow)
                    elif res['audio_type'] == "立体声":
                        for col in range(6):
                            self.table.item(i, col).setBackground(Qt.green)
                    elif res['audio_type'] == "单声道":
                        for col in range(6):
                            self.table.item(i, col).setBackground(Qt.cyan)
                    elif "环绕声" in res['audio_type']:
                        for col in range(6):
                            self.table.item(i, col).setBackground(Qt.magenta)
                    elif res['audio_type'] == "分析错误":
                        for col in range(6):
                            self.table.item(i, col).setBackground(Qt.red)
                    break
        
        # 检查是否所有任务都已完成
        if self.completed_tasks >= self.total_tasks and self.is_analyzing:
            self.analysis_finished()
    
    def update_progress(self, completed_files, total_files):
        """更新进度条 - 使用互斥锁确保进度只增不减"""
        # 使用互斥锁保护进度更新
        self.progress_mutex.lock()
        try:
            # 计算当前百分比
            progress_percent = int(completed_files * 100 / total_files)
            
            # 确保进度只增不减
            if progress_percent > self.current_progress:
                self.current_progress = progress_percent
                self.progress_bar.setValue(self.current_progress)
                
                # 更新状态栏
                self.status_bar.showMessage(f"已完成 {self.completed_tasks}/{self.total_tasks} 个文件分析 ({self.current_progress}%)")
        finally:
            self.progress_mutex.unlock()
    
    def update_status(self, message):
        """更新状态栏消息"""
        self.status_bar.showMessage(message)
    
    def update_statistics(self):
        """更新统计信息"""
        if not self.results:
            self.stats_label.setText("未分析任何文件")
            return
            
        # 统计不同音频类型的数量
        stats = {}
        for result in self.results:
            audio_type = result['audio_type']
            if audio_type not in stats:
                stats[audio_type] = 0
            stats[audio_type] += 1
        
        # 构建统计文本
        stats_text = f"共分析 {len(self.results)} 个文件: "
        stats_items = []
        for audio_type, count in stats.items():
            stats_items.append(f"{audio_type}: {count}个")
        
        self.stats_label.setText(stats_text + ", ".join(stats_items))
    
    def analysis_finished(self):
        """分析完成后的处理"""
        # 隐藏进度条
        self.progress_bar.setVisible(False)
        
        # 启用按钮
        self.select_btn.setEnabled(True)
        self.select_folder_btn.setEnabled(True)
        self.thread_spinbox.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.is_analyzing = False
        
        # 更新统计信息
        self.update_statistics()
        
        if self.results:
            self.export_btn.setEnabled(True)
            self.status_bar.showMessage(f"分析完成，使用了 {self.thread_pool.maxThreadCount()} 个线程")
        else:
            self.status_bar.showMessage("未找到音频文件")
    
    def export_excel(self):
        """导出结果到Excel文件"""
        if not self.results:
            return
            
        # 弹出保存对话框
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(
            self,
            "保存Excel文件",
            "音频分析结果.xlsx",
            "Excel文件 (*.xlsx)"
        )
        
        if not file_path:
            return
            
        try:
            # 创建DataFrame
            df = pd.DataFrame(self.results)
            
            # 选择要导出的列并重命名
            df = df[['filename', 'audio_type', 'channels', 'sample_rate', 'duration', 'filepath']]
            df.columns = ["文件名", "音频类型", "声道数", "采样率", "时长", "文件路径"]
            
            # 保存到Excel
            df.to_excel(file_path, index=False, sheet_name="音频分析结果")
            
            QMessageBox.information(self, "导出成功", f"分析结果已成功导出至\n{file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "导出失败", f"导出Excel文件时出错：\n{str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # 使用Fusion风格，外观更现代
    
    # 设置应用程序字体
    font = QFont("Microsoft YaHei UI", 9)
    app.setFont(font)
    
    # 应用程序样式表
    app.setStyleSheet("""
        QMainWindow, QDialog {
            background-color: #f5f5f5;
        }
        QTableWidget {
            gridline-color: #d0d0d0;
            selection-background-color: #0078d7;
        }
        QTableWidget::item:selected {
            color: white;
        }
        QPushButton {
            background-color: #0078d7;
            color: white;
            border: none;
            padding: 6px 12px;
            border-radius: 4px;
        }
        QPushButton:hover {
            background-color: #1683d8;
        }
        QPushButton:pressed {
            background-color: #006cc1;
        }
        QPushButton:disabled {
            background-color: #cccccc;
            color: #666666;
        }
        QProgressBar {
            border: 1px solid #d0d0d0;
            border-radius: 4px;
            text-align: center;
        }
        QProgressBar::chunk {
            background-color: #0078d7;
        }
        QGroupBox {
            border: 1px solid #d0d0d0;
            border-radius: 4px;
            margin-top: 6px;
            padding-top: 6px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;
        }
        QHeaderView::section {
            background-color: #f0f0f0;
            padding: 4px;
            border: 1px solid #d0d0d0;
            border-left: none;
        }
        QSpinBox {
            border: 1px solid #d0d0d0;
            border-radius: 4px;
            padding: 2px;
        }
    """)
    
    window = AudioAnalyzerApp()
    window.show()
    sys.exit(app.exec_()) 