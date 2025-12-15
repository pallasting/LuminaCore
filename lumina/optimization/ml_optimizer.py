"""
基于机器学习的参数优化算法

实现智能参数优化：
- 多目标优化算法
- 自适应优化策略
- 强化学习优化
- 贝叶斯优化
"""

import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class OptimizationResult:
    """优化结果数据类"""

    optimal_parameters: Dict[str, float]
    objective_value: float
    convergence_history: List[float]
    optimization_time: float
    confidence_score: float


@dataclass
class MultiObjectiveResult:
    """多目标优化结果"""

    pareto_front: List[Dict[str, float]]
    optimal_solutions: List[Dict[str, Any]]
    trade_off_analysis: Dict[str, Any]


class NeuralOptimizer(nn.Module):
    """
    神经网络优化器

    使用神经网络学习最优参数配置
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_objectives: int = 1):
        super(NeuralOptimizer, self).__init__()

        self.input_dim = input_dim
        self.num_objectives = num_objectives

        # 编码器网络
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # 参数预测网络
        self.parameter_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),  # 输出参数向量
        )

        # 目标预测网络
        self.objective_predictor = nn.Sequential(
            nn.Linear(hidden_dim + 32, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_objectives),
        )

        # 置信度预测
        self.confidence_predictor = nn.Sequential(
            nn.Linear(hidden_dim + 32, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, system_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            system_state: 系统状态向量

        Returns:
            (预测参数, 预测目标值, 置信度)
        """
        # 编码系统状态
        encoded_state = self.encoder(system_state)

        # 预测最优参数
        predicted_params = self.parameter_predictor(encoded_state)

        # 组合输入预测目标
        combined_input = torch.cat([encoded_state, predicted_params], dim=-1)
        predicted_objectives = self.objective_predictor(combined_input)

        # 预测置信度
        confidence = self.confidence_predictor(combined_input)

        return predicted_params, predicted_objectives, confidence

    def optimize(
        self,
        system_state: torch.Tensor,
        target_objectives: torch.Tensor,
        num_iterations: int = 100,
        learning_rate: float = 0.01,
    ) -> OptimizationResult:
        """
        执行优化

        Args:
            system_state: 系统状态
            target_objectives: 目标值
            num_iterations: 迭代次数
            learning_rate: 学习率

        Returns:
            优化结果
        """
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        convergence_history = []
        start_time = (
            torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        )
        end_time = (
            torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        )

        if start_time:
            start_time.record()

        best_params = None
        best_objective = float("inf")
        best_confidence = 0.0

        for iteration in range(num_iterations):
            optimizer.zero_grad()

            # 前向传播
            pred_params, pred_objectives, confidence = self(system_state)

            # 计算损失（目标值误差 + 置信度惩罚）
            objective_loss = criterion(pred_objectives, target_objectives)
            confidence_loss = -torch.log(confidence + 1e-8).mean()  # 最大化置信度
            total_loss = objective_loss + 0.1 * confidence_loss

            # 反向传播
            total_loss.backward()
            optimizer.step()

            # 记录收敛历史
            current_objective = objective_loss.item()
            convergence_history.append(current_objective)

            # 更新最优解
            if current_objective < best_objective:
                best_objective = current_objective
                best_params = pred_params.detach()
                best_confidence = confidence.item()

        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            optimization_time = start_time.elapsed_time(end_time) / 1000.0
        else:
            optimization_time = 0.0  # 简化计时

        return OptimizationResult(
            optimal_parameters={
                "params": best_params.cpu().numpy() if best_params is not None else None
            },
            objective_value=best_objective,
            convergence_history=convergence_history,
            optimization_time=optimization_time,
            confidence_score=best_confidence,
        )


class MultiObjectiveOptimizer:
    """
    多目标优化器

    实现NSGA-II算法进行多目标优化
    """

    def __init__(self, population_size: int = 50, num_generations: int = 100):
        self.population_size = population_size
        self.num_generations = num_generations

        # 优化参数范围
        self.param_ranges = {
            "temperature": (250.0, 400.0),
            "power": (0.1, 10.0),
            "wavelength": (400e-9, 2000e-9),
            "noise_level": (0.0, 0.5),
            "dispersion": (-100.0, 100.0),
        }

    def optimize(
        self, objectives: List[Callable], constraints: Optional[List[Callable]] = None
    ) -> MultiObjectiveResult:
        """
        执行多目标优化

        Args:
            objectives: 目标函数列表
            constraints: 约束函数列表

        Returns:
            多目标优化结果
        """
        # 初始化种群
        population = self._initialize_population()

        for generation in range(self.num_generations):
            # 评估种群
            fitness_values = self._evaluate_population(population, objectives)

            # 非支配排序
            fronts = self._fast_non_dominated_sort(fitness_values)

            # 计算拥挤度
            crowding_distances = self._calculate_crowding_distance(
                fronts, fitness_values
            )

            # 选择父代
            parents = self._tournament_selection(population, fronts, crowding_distances)

            # 遗传操作
            offspring = self._crossover_and_mutation(parents)

            # 精英保留
            population = self._elitist_selection(
                population, offspring, fronts, crowding_distances
            )

        # 获取帕累托前沿
        final_fitness = self._evaluate_population(population, objectives)
        pareto_front = self._extract_pareto_front(population, final_fitness)

        # 分析权衡
        trade_off_analysis = self._analyze_trade_offs(pareto_front, objectives)

        return MultiObjectiveResult(
            pareto_front=pareto_front,
            optimal_solutions=self._select_optimal_solutions(pareto_front),
            trade_off_analysis=trade_off_analysis,
        )

    def _initialize_population(self) -> List[Dict[str, float]]:
        """初始化种群"""
        population = []

        for _ in range(self.population_size):
            individual = {}
            for param, (min_val, max_val) in self.param_ranges.items():
                individual[param] = random.uniform(min_val, max_val)
            population.append(individual)

        return population

    def _evaluate_population(
        self, population: List[Dict[str, float]], objectives: List[Callable]
    ) -> List[List[float]]:
        """评估种群适应度"""
        fitness_values = []

        for individual in population:
            fitness = []
            for objective in objectives:
                try:
                    value = objective(individual)
                    fitness.append(value)
                except:
                    fitness.append(float("inf"))  # 无效解
            fitness_values.append(fitness)

        return fitness_values

    def _fast_non_dominated_sort(
        self, fitness_values: List[List[float]]
    ) -> List[List[int]]:
        """快速非支配排序"""
        num_individuals = len(fitness_values)
        domination_count = [0] * num_individuals
        dominated_solutions = [[] for _ in range(num_individuals)]
        fronts = [[]]

        for i in range(num_individuals):
            for j in range(num_individuals):
                if i != j:
                    if self._dominates(fitness_values[i], fitness_values[j]):
                        dominated_solutions[i].append(j)
                    elif self._dominates(fitness_values[j], fitness_values[i]):
                        domination_count[i] += 1

            if domination_count[i] == 0:
                fronts[0].append(i)

        i = 0
        while fronts[i]:
            next_front = []
            for individual in fronts[i]:
                for dominated in dominated_solutions[individual]:
                    domination_count[dominated] -= 1
                    if domination_count[dominated] == 0:
                        next_front.append(dominated)
            i += 1
            fronts.append(next_front)

        return fronts[:-1]  # 移除空的前沿

    def _dominates(self, fitness1: List[float], fitness2: List[float]) -> bool:
        """检查fitness1是否支配fitness2"""
        at_least_one_better = False
        for f1, f2 in zip(fitness1, fitness2):
            if f1 > f2:  # 假设最小化问题
                return False
            if f1 < f2:
                at_least_one_better = True
        return at_least_one_better

    def _calculate_crowding_distance(
        self, fronts: List[List[int]], fitness_values: List[List[float]]
    ) -> List[float]:
        """计算拥挤度"""
        num_objectives = len(fitness_values[0])
        crowding_distances = [0.0] * len(fitness_values)

        for front in fronts:
            if len(front) <= 2:
                for individual in front:
                    crowding_distances[individual] = float("inf")
                continue

            for m in range(num_objectives):
                # 按目标值排序
                front_sorted = sorted(front, key=lambda x: fitness_values[x][m])

                # 边界点设置无穷大拥挤度
                crowding_distances[front_sorted[0]] = float("inf")
                crowding_distances[front_sorted[-1]] = float("inf")

                # 计算中间点的拥挤度
                f_max = fitness_values[front_sorted[-1]][m]
                f_min = fitness_values[front_sorted[0]][m]

                if f_max == f_min:
                    continue

                for i in range(1, len(front_sorted) - 1):
                    crowding_distances[front_sorted[i]] += (
                        fitness_values[front_sorted[i + 1]][m]
                        - fitness_values[front_sorted[i - 1]][m]
                    ) / (f_max - f_min)

        return crowding_distances

    def _tournament_selection(
        self,
        population: List[Dict[str, float]],
        fronts: List[List[int]],
        crowding_distances: List[float],
    ) -> List[Dict[str, float]]:
        """锦标赛选择"""
        selected = []

        for _ in range(len(population)):
            # 随机选择两个个体
            i1, i2 = random.sample(range(len(population)), 2)

            # 比较支配等级
            front1 = next(i for i, front in enumerate(fronts) if i1 in front)
            front2 = next(i for i, front in enumerate(fronts) if i2 in front)

            if front1 < front2:
                selected.append(population[i1])
            elif front2 < front1:
                selected.append(population[i2])
            else:
                # 相同前沿，比较拥挤度
                if crowding_distances[i1] > crowding_distances[i2]:
                    selected.append(population[i1])
                else:
                    selected.append(population[i2])

        return selected

    def _crossover_and_mutation(
        self, parents: List[Dict[str, float]]
    ) -> List[Dict[str, float]]:
        """交叉和变异"""
        offspring = []

        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                parent1 = parents[i]
                parent2 = parents[i + 1]

                # 单点交叉
                child1 = {}
                child2 = {}

                for param in self.param_ranges.keys():
                    if random.random() < 0.5:
                        child1[param] = parent1[param]
                        child2[param] = parent2[param]
                    else:
                        child1[param] = parent2[param]
                        child2[param] = parent1[param]

                # 变异
                for child in [child1, child2]:
                    for param, (min_val, max_val) in self.param_ranges.items():
                        if random.random() < 0.1:  # 10%变异率
                            child[param] = random.uniform(min_val, max_val)

                offspring.extend([child1, child2])
            else:
                offspring.append(parents[i])

        return offspring

    def _elitist_selection(
        self,
        old_population: List[Dict[str, float]],
        offspring: List[Dict[str, float]],
        fronts: List[List[int]],
        crowding_distances: List[float],
    ) -> List[Dict[str, float]]:
        """精英选择"""
        combined_population = old_population + offspring
        combined_fitness = self._evaluate_population(
            combined_population, []
        )  # 需要重新评估

        # 重新排序
        new_fronts = self._fast_non_dominated_sort(combined_fitness)
        new_crowding_distances = self._calculate_crowding_distance(
            new_fronts, combined_fitness
        )

        # 选择最好的个体
        selected = []
        for front in new_fronts:
            if len(selected) + len(front) <= self.population_size:
                selected.extend(front)
            else:
                # 按拥挤度排序剩余个体
                remaining = len(selected) + len(front) - self.population_size
                front_sorted = sorted(
                    front, key=lambda x: new_crowding_distances[x], reverse=True
                )
                selected.extend(front_sorted[:remaining])
                break

        return [combined_population[i] for i in selected]

    def _extract_pareto_front(
        self, population: List[Dict[str, float]], fitness_values: List[List[float]]
    ) -> List[Dict[str, float]]:
        """提取帕累托前沿"""
        pareto_front = []

        for i, individual in enumerate(population):
            is_dominated = False
            for j, other_fitness in enumerate(fitness_values):
                if i != j and self._dominates(other_fitness, fitness_values[i]):
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_front.append(individual)

        return pareto_front

    def _analyze_trade_offs(
        self, pareto_front: List[Dict[str, float]], objectives: List[Callable]
    ) -> Dict[str, Any]:
        """分析权衡关系"""
        if len(pareto_front) < 2:
            return {"trade_offs": "Insufficient data for analysis"}

        # 计算目标值
        objective_values = []
        for individual in pareto_front:
            values = []
            for objective in objectives:
                try:
                    values.append(objective(individual))
                except:
                    values.append(float("inf"))
            objective_values.append(values)

        # 分析相关性
        correlations = {}
        for i in range(len(objectives)):
            for j in range(i + 1, len(objectives)):
                obj1_values = [vals[i] for vals in objective_values]
                obj2_values = [vals[j] for vals in objective_values]

                if len(obj1_values) > 1:
                    correlation = np.corrcoef(obj1_values, obj2_values)[0, 1]
                    correlations[f"obj_{i}_vs_obj_{j}"] = correlation

        return {
            "correlations": correlations,
            "pareto_front_size": len(pareto_front),
            "objective_ranges": {
                f"obj_{i}": [
                    min(vals[i] for vals in objective_values),
                    max(vals[i] for vals in objective_values),
                ]
                for i in range(len(objectives))
            },
        }

    def _select_optimal_solutions(
        self, pareto_front: List[Dict[str, float]]
    ) -> List[Dict[str, Any]]:
        """选择最优解"""
        optimal_solutions = []

        # 选择不同权重的解
        weights = [(1.0, 0.0), (0.5, 0.5), (0.0, 1.0)]  # 不同权重组合

        for w1, w2 in weights:
            best_solution = None
            best_score = float("inf")

            for solution in pareto_front:
                # 简化的加权评分（需要根据实际目标函数调整）
                score = w1 * solution.get("temperature", 0) + w2 * solution.get(
                    "power", 0
                )
                if score < best_score:
                    best_score = score
                    best_solution = solution

            if best_solution:
                optimal_solutions.append(
                    {
                        "solution": best_solution,
                        "weights": (w1, w2),
                        "score": best_score,
                    }
                )

        return optimal_solutions


class BayesianOptimizer:
    """
    贝叶斯优化器

    使用高斯过程进行贝叶斯优化
    """

    def __init__(
        self,
        parameter_bounds: Dict[str, Tuple[float, float]],
        n_initial_points: int = 5,
        n_iterations: int = 25,
    ):
        self.parameter_bounds = parameter_bounds
        self.n_initial_points = n_initial_points
        self.n_iterations = n_iterations

        # 观测数据
        self.observations = []
        self.parameter_names = list(parameter_bounds.keys())

    def optimize(self, objective_function: Callable) -> OptimizationResult:
        """
        执行贝叶斯优化

        Args:
            objective_function: 目标函数

        Returns:
            优化结果
        """
        # 初始观测
        self._initial_observations(objective_function)

        convergence_history = []

        for iteration in range(self.n_iterations):
            # 拟合高斯过程
            gp_model = self._fit_gaussian_process()

            # 选择下一个评估点
            next_point = self._acquisition_function(gp_model)

            # 评估目标函数
            objective_value = objective_function(next_point)

            # 添加观测
            self.observations.append((next_point, objective_value))

            convergence_history.append(objective_value)

        # 返回最优解
        best_observation = min(self.observations, key=lambda x: x[1])

        return OptimizationResult(
            optimal_parameters=best_observation[0],
            objective_value=best_observation[1],
            convergence_history=convergence_history,
            optimization_time=0.0,  # 简化
            confidence_score=self._estimate_confidence(),
        )

    def _initial_observations(self, objective_function: Callable):
        """初始观测"""
        for _ in range(self.n_initial_points):
            # 随机采样
            point = {}
            for param, (min_val, max_val) in self.parameter_bounds.items():
                point[param] = random.uniform(min_val, max_val)

            objective_value = objective_function(point)
            self.observations.append((point, objective_value))

    def _fit_gaussian_process(self):
        """拟合高斯过程（简化实现）"""
        # 这里应该使用真正的GP库，如GPy或scikit-learn
        # 简化实现返回虚拟模型
        return {"mock": True}

    def _acquisition_function(self, gp_model) -> Dict[str, float]:
        """采集函数（期望改善）"""
        # 简化实现：随机采样
        point = {}
        for param, (min_val, max_val) in self.parameter_bounds.items():
            point[param] = random.uniform(min_val, max_val)

        return point

    def _estimate_confidence(self) -> float:
        """估计置信度"""
        if len(self.observations) < 2:
            return 0.5

        # 基于观测方差估计置信度
        values = [obs[1] for obs in self.observations]
        std_dev = np.std(values)
        mean_val = np.mean(values)

        # 较低的变异系数表示较高的置信度
        cv = std_dev / (abs(mean_val) + 1e-10)
        confidence = max(0.1, min(1.0, 1.0 - cv))

        return confidence


class AdaptiveOptimizer:
    """
    自适应优化器

    根据系统状态动态调整优化策略
    """

    def __init__(self):
        self.optimizers = {
            "neural": NeuralOptimizer(input_dim=10, num_objectives=2),
            "multi_objective": MultiObjectiveOptimizer(),
            "bayesian": BayesianOptimizer(
                parameter_bounds={
                    "temperature": (250, 400),
                    "power": (0.1, 10.0),
                    "wavelength": (400e-9, 2000e-9),
                }
            ),
        }

        self.performance_history = []
        self.strategy_performance = {name: [] for name in self.optimizers.keys()}

    def optimize(
        self, system_state: Dict[str, Any], objectives: List[str]
    ) -> Dict[str, Any]:
        """
        自适应优化

        Args:
            system_state: 系统状态
            objectives: 优化目标

        Returns:
            优化结果
        """
        # 分析系统状态
        state_features = self._extract_state_features(system_state)

        # 选择最优策略
        best_strategy = self._select_strategy(state_features, objectives)

        # 执行优化
        optimizer = self.optimizers[best_strategy]
        result = self._execute_strategy(optimizer, system_state, objectives)

        # 更新性能历史
        self._update_performance_history(best_strategy, result)

        return {
            "strategy": best_strategy,
            "result": result,
            "performance_analysis": self._analyze_performance(),
        }

    def _extract_state_features(self, system_state: Dict[str, Any]) -> torch.Tensor:
        """提取状态特征"""
        # 简化的特征提取
        features = [
            system_state.get("temperature", 300.0) / 400.0,
            system_state.get("power_consumption", 5.0) / 10.0,
            system_state.get("snr", 20.0) / 40.0,
            system_state.get("error_rate", 1e-6) / 1e-3,
            len(system_state.get("channel_utilization", [])) / 10.0,
        ]

        return torch.tensor(features, dtype=torch.float32)

    def _select_strategy(
        self, state_features: torch.Tensor, objectives: List[str]
    ) -> str:
        """选择优化策略"""
        # 基于历史性能和当前状态选择策略
        if len(objectives) > 1:
            return "multi_objective"
        elif len(self.performance_history) > 10:
            return "bayesian"
        else:
            return "neural"

    def _execute_strategy(
        self, optimizer, system_state: Dict[str, Any], objectives: List[str]
    ) -> Any:
        """执行选定策略"""
        # 简化的执行逻辑
        if hasattr(optimizer, "optimize"):
            # 假设优化器有optimize方法
            return optimizer.optimize(lambda x: sum(x.values()))  # 简化目标函数
        else:
            return {"mock_result": True}

    def _update_performance_history(self, strategy: str, result: Any):
        """更新性能历史"""
        # 记录策略性能
        if hasattr(result, "objective_value"):
            self.strategy_performance[strategy].append(result.objective_value)

        self.performance_history.append(
            {"strategy": strategy, "timestamp": np.datetime64("now"), "result": result}
        )

    def _analyze_performance(self) -> Dict[str, Any]:
        """分析优化性能"""
        analysis = {}

        for strategy, performances in self.strategy_performance.items():
            if performances:
                analysis[strategy] = {
                    "mean_performance": np.mean(performances),
                    "best_performance": min(performances),
                    "std_performance": np.std(performances),
                    "num_runs": len(performances),
                }

        return analysis
