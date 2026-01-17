#!/usr/bin/env python3
"""
RainbowLuminaCore Web Dashboard

Real-time monitoring dashboard for distributed photonic computing
Built with Flask and vanilla JavaScript

Features:
- Live pipeline execution visualization
- Tile utilization charts
- Performance metrics
- Load balancing status
"""

import asyncio
import json
import time
import threading
from datetime import datetime
from flask import Flask, render_template, jsonify, request
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import random


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class TileStatus:
    """瓦片状态"""
    tile_id: str
    status: str = "idle"  # idle, processing, completed, error
    current_layer: int = -1
    utilization: float = 0.0
    temperature: float = 25.0
    tasks_completed: int = 0
    avg_task_time: float = 0.0
    last_update: float = field(default_factory=time.time)


@dataclass
class PipelineMetrics:
    """流水线指标"""
    total_batches: int = 0
    completed_batches: int = 0
    throughput: float = 0.0
    avg_latency: float = 0.0
    pipeline_efficiency: float = 0.0


@dataclass
class LoadBalanceEvent:
    """负载均衡事件"""
    timestamp: str
    tile_id: str
    event_type: str  # rebalance, imbalance_detected, threshold_exceeded
    details: str


# ============================================================================
# Dashboard State
# ============================================================================

class DashboardState:
    """仪表盘状态管理"""

    def __init__(self, num_tiles: int = 4):
        self.num_tiles = num_tiles
        self.tiles: Dict[str, TileStatus] = {
            f"Tile-{i}": TileStatus(tile_id=f"Tile-{i}")
            for i in range(num_tiles)
        }
        self.metrics = PipelineMetrics()
        self.events: List[LoadBalanceEvent] = []
        self.start_time = time.time()
        self.running = True

        # 模拟数据生成
        self._start_simulation()

    def _start_simulation(self):
        """启动模拟数据生成"""
        def simulate():
            batch_num = 0
            while self.running:
                # 随机更新瓦片状态
                for tile_id, tile in self.tiles.items():
                    if random.random() < 0.3:
                        tile.status = "processing"
                        tile.current_layer = random.randint(0, 11)
                        tile.utilization = random.uniform(0.3, 0.9)
                        tile.temperature = random.uniform(35.0, 75.0)
                        tile.tasks_completed += 1
                        tile.avg_task_time = random.uniform(20, 80)
                        tile.last_update = time.time()
                    else:
                        tile.status = "idle"
                        tile.utilization = 0.0
                        tile.current_layer = -1

                # 更新指标
                batch_num += 1
                self.metrics.total_batches = batch_num
                self.metrics.completed_batches = int(batch_num * random.uniform(0.8, 0.95))
                self.metrics.throughput = batch_num / max(1, time.time() - self.start_time) * 60
                self.metrics.avg_latency = random.uniform(50, 150)
                self.metrics.pipeline_efficiency = random.uniform(0.7, 0.95)

                # 偶尔生成负载均衡事件
                if random.random() < 0.02:
                    event = LoadBalanceEvent(
                        timestamp=datetime.now().strftime("%H:%M:%S"),
                        tile_id=random.choice(list(self.tiles.keys())),
                        event_type="imbalance_detected",
                        details=f"Load imbalance: {random.uniform(15, 35):.1f}%"
                    )
                    self.events.insert(0, event)
                    if len(self.events) > 50:
                        self.events.pop()

                time.sleep(1.0)

        threading.Thread(target=simulate, daemon=True).start()

    def get_state(self) -> Dict[str, Any]:
        """获取当前状态"""
        return {
            "tiles": {
                tile_id: {
                    "tile_id": tile.tile_id,
                    "status": tile.status,
                    "current_layer": tile.current_layer,
                    "utilization": tile.utilization,
                    "temperature": tile.temperature,
                    "tasks_completed": tile.tasks_completed,
                    "avg_task_time": tile.avg_task_time
                }
                for tile_id, tile in self.tiles.items()
            },
            "metrics": {
                "total_batches": self.metrics.total_batches,
                "completed_batches": self.metrics.completed_batches,
                "throughput": self.metrics.throughput,
                "avg_latency": self.metrics.avg_latency,
                "pipeline_efficiency": self.metrics.pipeline_efficiency,
                "uptime": time.time() - self.start_time
            },
            "events": [
                {
                    "timestamp": e.timestamp,
                    "tile_id": e.tile_id,
                    "event_type": e.event_type,
                    "details": e.details
                }
                for e in self.events[:20]
            ],
            "timestamp": datetime.now().isoformat()
        }

    def stop(self):
        """停止模拟"""
        self.running = False


# ============================================================================
# Flask App
# ============================================================================

app = Flask(__name__, template_folder='templates')
dashboard = DashboardState(num_tiles=4)


@app.route('/')
def index():
    """主页面"""
    return render_template('dashboard.html')


@app.route('/api/state')
def api_state():
    """获取当前状态 (用于轮询)"""
    return jsonify(dashboard.get_state())


@app.route('/api/tiles')
def api_tiles():
    """获取瓦片状态"""
    state = dashboard.get_state()
    return jsonify(state["tiles"])


@app.route('/api/metrics')
def api_metrics():
    """获取性能指标"""
    state = dashboard.get_state()
    return jsonify(state["metrics"])


@app.route('/api/events')
def api_events():
    """获取事件日志"""
    state = dashboard.get_state()
    return jsonify(state["events"])


@app.route('/api/rebalance', methods=['POST'])
def api_rebalance():
    """触发负载均衡"""
    # 模拟触发重平衡
    event = LoadBalanceEvent(
        timestamp=datetime.now().strftime("%H:%M:%S"),
        tile_id="All",
        event_type="rebalance",
        details="Manual rebalancing triggered"
    )
    dashboard.events.insert(0, event)
    return jsonify({"status": "success", "message": "Rebalancing initiated"})


@app.route('/api/control', methods=['POST'])
def api_control():
    """控制命令"""
    action = request.json.get('action')
    if action == 'pause':
        # 暂停模拟
        pass
    elif action == 'resume':
        # 恢复模拟
        pass
    elif action == 'reset':
        # 重置统计
        dashboard.start_time = time.time()
        for tile in dashboard.tiles.values():
            tile.tasks_completed = 0
            tile.avg_task_time = 0.0
        dashboard.metrics = PipelineMetrics()
    return jsonify({"status": "success"})


def run_dashboard(host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
    """运行仪表盘"""
    print(f"\n🌈 RainbowLuminaCore Dashboard")
    print(f"=" * 50)
    print(f"   Dashboard: http://{host}:{port}")
    print(f"   API Endpoints:")
    print(f"     - /api/state   (full state)")
    print(f"     - /api/tiles   (tile status)")
    print(f"     - /api/metrics (performance)")
    print(f"     - /api/events  (event log)")
    print(f"   Controls: POST /api/rebalance, /api/control")
    print(f"=" * 50)
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_dashboard(host='0.0.0.0', port=5000)
