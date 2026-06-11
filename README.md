# StockMind · 股票心智

> 越用越强的股票分析 Agent — 基于 Hermes Agent 自进化思想

## 核心特性

- **五维度分析**：基本面、技术面、情绪面、新闻面、宏观面
- **自进化学习**：每次决策后自动提取经验，下次复用
- **反思机制**：24h 后回看判断，评估准确率，持续改进
- **技能库**：FTS5 全文检索，像 Hermes Agent 一样积累技能文档
- **多 LLM 支持**：MiniMax / OpenAI / DeepSeek
- **Web 仪表盘**：暗色主题，实时分析，决策历史，技能可视化

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 设置 API Key
export MINIMAX_API_KEY="your-key"
# 或
export OPENAI_API_KEY="your-key"

# 3. 运行
python main.py interactive

# 4. 或启动 Web 界面
python main.py web
```

## 使用方式

### 命令行

```bash
python main.py analyze AAPL      # 分析股票
python main.py reflect            # 反思历史决策
python main.py report             # 查看表现报告
python main.py suggest            # 获取改进建议
python main.py watchlist add TSLA # 添加关注
python main.py web                # 启动 Web 界面
```

### 交互模式

```bash
python main.py interactive

StockMind> analyze AAPL
StockMind> reflect
StockMind> report
StockMind> skills list
StockMind> watchlist analyze
StockMind> quit
```

## 架构

```
stockmind/
├── config.py              # 配置文件
├── main.py                # 主入口 + CLI + 交互模式
├── web_dashboard.py       # Web 仪表盘
├── core/
│   ├── database.py        # SQLite + FTS5 数据层
│   ├── data_fetcher.py    # 股票数据获取（yfinance）
│   ├── llm_client.py      # LLM 统一客户端
│   ├── skill_system.py    # 技能库（Hermes Agent 模式）
│   ├── analyzer.py        # 五维度分析引擎
│   └── reflector.py       # 反思 + 学习模块
├── data/                  # 数据库文件（自动创建）
├── skills/                # 技能文档（自动创建）
└── requirements.txt
```

## 自进化原理

```
决策 → 反思 → 提取技能 → 复用技能 → 更好的决策
  ↑                                    │
  └────────────────────────────────────┘
```

1. **决策**：Agent 分析股票，给出买/卖/持有建议
2. **反思**：24h 后回看，判断是否正确
3. **提取技能**：从成功/失败中提取可复用经验
4. **复用技能**：下次分析时检索相关技能，作为上下文
5. **越用越强**：技能库不断增长，决策质量持续提升

## License

MIT
