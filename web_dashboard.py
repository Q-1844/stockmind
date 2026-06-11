"""
StockMind · 股票心智 — Web 仪表盘
基于 Python 内置 http.server 的轻量级 Web 界面，零外部依赖
"""
import json
import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

from config import WEB_HOST, WEB_PORT
from core import database as db

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 嵌入式 HTML 仪表盘（自包含单文件，暗色主题 + 琥珀金强调色）
# ═══════════════════════════════════════════════════════════════

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>StockMind · 股票心智</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{--bg:#0f1117;--surface:#1a1d27;--card:#222633;--border:#2e3348;
  --text:#e4e6f0;--dim:#8b8fa3;--accent:#f0a500;--accent2:#d4920b;
  --green:#22c55e;--red:#ef4444;--blue:#3b82f6;--radius:8px}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  background:var(--bg);color:var(--text);min-height:100vh}
.app{display:grid;grid-template-columns:240px 1fr;grid-template-rows:auto 1fr auto;
  min-height:100vh;gap:0}
/* 顶栏 */
.header{grid-column:1/-1;background:var(--surface);border-bottom:1px solid var(--border);
  padding:12px 24px;display:flex;align-items:center;justify-content:space-between}
.header h1{font-size:18px;font-weight:700;color:var(--accent)}
.header .status{font-size:13px;color:var(--dim)}
/* 侧栏 */
.sidebar{background:var(--surface);border-right:1px solid var(--border);
  padding:16px;overflow-y:auto;display:flex;flex-direction:column;gap:16px}
.sidebar h2{font-size:13px;text-transform:uppercase;color:var(--dim);letter-spacing:1px;margin-bottom:8px}
.watchlist-item{display:flex;align-items:center;justify-content:space-between;
  padding:8px 10px;border-radius:var(--radius);cursor:pointer;transition:background .15s}
.watchlist-item:hover{background:var(--card)}
.watchlist-item .sym{font-weight:600;font-size:14px}
.watchlist-item .del{color:var(--red);cursor:pointer;opacity:0;transition:opacity .15s;font-size:12px}
.watchlist-item:hover .del{opacity:1}
.nav-btn{display:block;width:100%;text-align:left;padding:8px 10px;border:none;
  background:none;color:var(--text);font-size:14px;border-radius:var(--radius);cursor:pointer}
.nav-btn:hover{background:var(--card)}
.nav-btn.active{background:var(--accent);color:#000;font-weight:600}
.add-form{display:flex;gap:6px}
.add-form input{flex:1;padding:6px 10px;border:1px solid var(--border);border-radius:var(--radius);
  background:var(--bg);color:var(--text);font-size:13px;outline:none}
.add-form input:focus{border-color:var(--accent)}
.add-form button{padding:6px 12px;border:none;border-radius:var(--radius);
  background:var(--accent);color:#000;font-weight:600;cursor:pointer;font-size:13px}
/* 主区域 */
.main{padding:24px;overflow-y:auto;display:flex;flex-direction:column;gap:20px}
.panel{background:var(--card);border-radius:var(--radius);padding:20px;border:1px solid var(--border)}
.panel h3{font-size:15px;margin-bottom:12px;color:var(--accent)}
/* 分析面板 */
.analysis-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.analysis-main{grid-column:1/-1}
.action-badge{display:inline-block;padding:4px 14px;border-radius:20px;font-weight:700;font-size:14px}
.action-buy{background:rgba(34,197,94,.15);color:var(--green)}
.action-sell{background:rgba(239,68,68,.15);color:var(--red)}
.action-hold{background:rgba(139,143,163,.15);color:var(--dim)}
.conf-bar{height:8px;border-radius:4px;background:var(--border);overflow:hidden;margin-top:6px}
.conf-fill{height:100%;border-radius:4px;transition:width .4s}
.dim-row{display:flex;align-items:center;gap:10px;padding:6px 0;border-bottom:1px solid var(--border)}
.dim-row:last-child{border:none}
.dim-label{width:60px;font-size:13px;color:var(--dim)}
.dim-score{width:50px;font-size:13px;font-weight:600}
.dim-reason{flex:1;font-size:12px;color:var(--dim)}
/* 决策历史表格 */
.decision-table{width:100%;border-collapse:collapse;font-size:13px}
.decision-table th{text-align:left;padding:8px 10px;border-bottom:2px solid var(--border);color:var(--dim);font-weight:600}
.decision-table td{padding:8px 10px;border-bottom:1px solid var(--border)}
.decision-table tr:hover{background:rgba(240,165,0,.04)}
/* 技能卡片 */
.skills-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}
.skill-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:14px}
.skill-card .name{font-weight:600;font-size:14px;margin-bottom:6px}
.skill-card .badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600;margin-right:4px}
.badge-fundamental{background:rgba(59,130,246,.15);color:var(--blue)}
.badge-technical{background:rgba(168,85,247,.15);color:#a855f7}
.badge-sentiment{background:rgba(236,72,153,.15);color:#ec4899}
.badge-news{background:rgba(245,158,11,.15);color:#f59e0b}
.badge-macro{background:rgba(20,184,166,.15);color:#14b8a6}
.skill-card .meta{font-size:12px;color:var(--dim);margin-top:6px}
/* 底栏 */
.footer{grid-column:1/-1;background:var(--surface);border-top:1px solid var(--border);
  padding:10px 24px;font-size:12px;color:var(--dim);display:flex;justify-content:space-between}
/* 加载动画 */
.loading{text-align:center;padding:40px;color:var(--dim)}
.spinner{display:inline-block;width:24px;height:24px;border:3px solid var(--border);
  border-top-color:var(--accent);border-radius:50%;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
/* 响应式 */
@media(max-width:768px){
  .app{grid-template-columns:1fr;grid-template-rows:auto auto 1fr auto}
  .sidebar{border-right:none;border-bottom:1px solid var(--border);flex-direction:row;flex-wrap:wrap;gap:8px}
  .analysis-grid{grid-template-columns:1fr}
}
</style>
</head>
<body>
<div class="app">
  <header class="header">
    <h1>🧠 StockMind · 股票心智</h1>
    <span class="status" id="statusText">已连接</span>
  </header>
  <aside class="sidebar">
    <div>
      <h2>📋 关注列表</h2>
      <div id="watchlist"></div>
      <div class="add-form" style="margin-top:10px">
        <input id="addSymbol" placeholder="输入代码如 AAPL" maxlength="10">
        <button onclick="addWatchlist()">添加</button>
      </div>
    </div>
    <div>
      <h2>🧭 导航</h2>
      <button class="nav-btn active" onclick="showPanel('analysis')">📊 分析面板</button>
      <button class="nav-btn" onclick="showPanel('decisions')">📝 决策历史</button>
      <button class="nav-btn" onclick="showPanel('skills')">🧠 技能库</button>
      <button class="nav-btn" onclick="showPanel('report')">📈 性能报告</button>
    </div>
  </aside>
  <main class="main">
    <div id="panelAnalysis" class="panel">
      <h3>📊 股票分析</h3>
      <div id="analysisContent"><p style="color:var(--dim)">从关注列表选择股票或输入代码开始分析</p></div>
    </div>
    <div id="panelDecisions" class="panel" style="display:none">
      <h3>📝 决策历史</h3>
      <div id="decisionsContent"><div class="loading"><span class="spinner"></span></div></div>
    </div>
    <div id="panelSkills" class="panel" style="display:none">
      <h3>🧠 技能库</h3>
      <div style="margin-bottom:12px">
        <input id="skillQuery" placeholder="搜索技能..." style="padding:6px 12px;border:1px solid var(--border);border-radius:var(--radius);background:var(--bg);color:var(--text);width:240px;outline:none">
        <button onclick="searchSkills()" style="padding:6px 14px;border:none;border-radius:var(--radius);background:var(--accent);color:#000;font-weight:600;cursor:pointer;margin-left:6px">搜索</button>
      </div>
      <div id="skillsContent"><div class="loading"><span class="spinner"></span></div></div>
    </div>
    <div id="panelReport" class="panel" style="display:none">
      <h3>📈 性能报告</h3>
      <div id="reportContent"><div class="loading"><span class="spinner"></span></div></div>
    </div>
  </main>
  <footer class="footer">
    <span>StockMind · 越用越强的股票分析 Agent</span>
    <span id="refreshTimer">自动刷新: 60s</span>
  </footer>
</div>
<script>
const API={analyze:s=>`/api/analyze?symbol=${s}`,reflect:d=>`/api/reflect?days=${d}`,
  report:'/api/report',suggest:'/api/suggest',watchlist:'/api/watchlist',
  watchlistAdd:'/api/watchlist/add',watchlistRemove:'/api/watchlist/remove',
  decisions:(s,l)=>`/api/decisions?symbol=${s||''}&limit=${l||10}`,
  skills:q=>`/api/skills?query=${q||''}`,health:'/api/health'};
let currentSymbol='',refreshInterval=null,refreshCount=60;
const dimLabels={fundamental:'基本面',technical:'技术面',sentiment:'情绪面',news:'新闻面',macro:'宏观面'};
const actionLabels={buy:'买入',sell:'卖出',hold:'观望'};

async function api(url,opts={}){
  try{const r=await fetch(url,opts);return await r.json()}
  catch(e){return{error:e.message}}
}

async function loadWatchlist(){
  const data=await api(API.watchlist);
  const el=document.getElementById('watchlist');
  if(data.error||!data.items){el.innerHTML='<p style="color:var(--dim);font-size:13px">暂无关注股票</p>';return}
  el.innerHTML=data.items.map(i=>`<div class="watchlist-item" onclick="analyze('${i.symbol}')">
    <span class="sym">${i.symbol}</span><span class="del" onclick="event.stopPropagation();removeWatchlist('${i.symbol}')">✕</span></div>`).join('');
}

async function addWatchlist(){
  const inp=document.getElementById('addSymbol'),sym=inp.value.trim().toUpperCase();
  if(!sym)return;
  await api(API.watchlistAdd,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({symbol:sym})});
  inp.value='';loadWatchlist();
}

async function removeWatchlist(sym){
  await api(API.watchlistRemove,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({symbol:sym})});
  loadWatchlist();
}

async function analyze(symbol){
  currentSymbol=symbol;
  const el=document.getElementById('analysisContent');
  el.innerHTML='<div class="loading"><span class="spinner"></span><p style="margin-top:8px">正在分析 '+symbol+' ...</p></div>';
  const data=await api(API.analyze(symbol));
  if(data.error){el.innerHTML=`<p style="color:var(--red)">分析失败: ${data.error}</p>`;return}
  renderAnalysis(data,el);
  resetRefresh();
}

function renderAnalysis(d,el){
  const act=d.action||'hold',conf=d.confidence||0,risk=d.risk_level||'medium';
  const confColor=conf>=0.7?'var(--green)':conf>=0.4?'var(--accent)':'var(--red)';
  const riskLabel={low:'低',medium:'中',high:'高'}[risk]||risk;
  const riskColor={low:'var(--green)',medium:'var(--accent)',high:'var(--red)'}[risk]||'var(--dim)';
  let dimsHtml='';
  const dims=d.dimensions||{};
  for(const key of['fundamental','technical','sentiment','news','macro']){
    const dm=dims[key]||{},sc=dm.score||0;
    const scColor=sc>0.1?'var(--green)':sc<-0.1?'var(--red)':'var(--dim)';
    dimsHtml+=`<div class="dim-row"><span class="dim-label">${dimLabels[key]||key}</span>
      <span class="dim-score" style="color:${scColor}">${sc>0?'+':''}${sc.toFixed(2)}</span>
      <span class="dim-reason">${dm.reason||'-'}</span></div>`;
  }
  el.innerHTML=`
    <div class="analysis-grid">
      <div class="analysis-main" style="display:flex;align-items:center;gap:16px;flex-wrap:wrap">
        <span style="font-size:22px;font-weight:800">${d.symbol||''}</span>
        <span class="action-badge action-${act}">${actionLabels[act]||act}</span>
        <span style="font-size:13px;color:var(--dim)">信心度: <b style="color:${confColor}">${(conf*100).toFixed(0)}%</b></span>
        <span style="font-size:13px;color:var(--dim)">风险: <b style="color:${riskColor}">${riskLabel}</b></span>
        ${d.price_target?`<span style="font-size:13px;color:var(--dim)">目标价: <b>$${d.price_target.toFixed(2)}</b></span>`:''}
      </div>
      <div style="grid-column:1/-1">
        <div class="conf-bar"><div class="conf-fill" style="width:${conf*100}%;background:${confColor}"></div></div>
      </div>
      <div style="grid-column:1/-1">
        <p style="font-size:13px;margin-bottom:10px;line-height:1.6">${d.reason||''}</p>
      </div>
      <div style="grid-column:1/-1">${dimsHtml}</div>
    </div>`;
}

async function loadDecisions(symbol,limit){
  const el=document.getElementById('decisionsContent');
  el.innerHTML='<div class="loading"><span class="spinner"></span></div>';
  const data=await api(API.decisions(symbol||'',limit||20));
  if(data.error||!data.items||data.items.length===0){el.innerHTML='<p style="color:var(--dim)">暂无决策记录</p>';return}
  el.innerHTML=`<table class="decision-table"><thead><tr>
    <th>日期</th><th>股票</th><th>行动</th><th>信心度</th><th>原因</th></tr></thead><tbody>
    ${data.items.map(d=>{const act=d.action||'hold';return`<tr>
      <td>${d.date||''}</td><td>${d.symbol||''}</td>
      <td><span class="action-badge action-${act}" style="font-size:12px;padding:2px 8px">${actionLabels[act]||act}</span></td>
      <td>${((d.confidence||0)*100).toFixed(0)}%</td>
      <td style="max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${d.reason||''}</td></tr>`}).join('')}
    </tbody></table>`;
}

async function loadSkills(query){
  const el=document.getElementById('skillsContent');
  el.innerHTML='<div class="loading"><span class="spinner"></span></div>';
  const data=await api(API.skills(query||''));
  if(data.error||!data.items||data.items.length===0){el.innerHTML='<p style="color:var(--dim)">暂无技能数据</p>';return}
  el.innerHTML=`<div class="skills-grid">${data.items.map(s=>{
    const cat=s.category||'fundamental';
    return`<div class="skill-card">
      <div class="name">${s.name||''}</div>
      <span class="badge badge-${cat}">${dimLabels[cat]||cat}</span>
      <div class="meta">使用 ${s.usage_count||0} 次 · 成功率 ${((s.success_rate||0)*100).toFixed(0)}%</div>
    </div>`}).join('')}</div>`;
}

function searchSkills(){loadSkills(document.getElementById('skillQuery').value.trim())}

async function loadReport(){
  const el=document.getElementById('reportContent');
  el.innerHTML='<div class="loading"><span class="spinner"></span></div>';
  const data=await api(API.report);
  if(data.error){el.innerHTML=`<p style="color:var(--red)">加载失败: ${data.error}</p>`;return}
  const acc=data.overall_accuracy||0;
  const accColor=acc>=0.6?'var(--green)':acc>=0.4?'var(--accent)':'var(--red)';
  const trendMap={improving:'📈 上升',declining:'📉 下降',stable:'➡️ 稳定'};
  let byActionHtml='';
  if(data.by_action){for(const[a,s]of Object.entries(data.by_action)){
    byActionHtml+=`<div class="dim-row"><span class="dim-label">${actionLabels[a]||a}</span>
      <span class="dim-score">${(s.accuracy*100).toFixed(0)}%</span>
      <span class="dim-reason">${s.count||0} 次</span></div>`}}
  let bySymbolHtml='';
  if(data.by_symbol){for(const[s,st]of Object.entries(data.by_symbol)){
    bySymbolHtml+=`<div class="dim-row"><span class="dim-label">${s}</span>
      <span class="dim-score">${(st.accuracy*100).toFixed(0)}%</span>
      <span class="dim-reason">${st.count||0} 次</span></div>`}}
  el.innerHTML=`
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:16px">
      <div style="background:var(--surface);padding:14px;border-radius:var(--radius);text-align:center">
        <div style="font-size:24px;font-weight:800;color:${accColor}">${(acc*100).toFixed(1)}%</div>
        <div style="font-size:12px;color:var(--dim)">整体准确率</div></div>
      <div style="background:var(--surface);padding:14px;border-radius:var(--radius);text-align:center">
        <div style="font-size:24px;font-weight:800">${data.total_decisions||0}</div>
        <div style="font-size:12px;color:var(--dim)">总决策数</div></div>
      <div style="background:var(--surface);padding:14px;border-radius:var(--radius);text-align:center">
        <div style="font-size:24px;font-weight:800">${data.skills_learned||0}</div>
        <div style="font-size:12px;color:var(--dim)">已学技能</div></div>
      <div style="background:var(--surface);padding:14px;border-radius:var(--radius);text-align:center">
        <div style="font-size:24px;font-weight:800">${trendMap[data.recent_trend]||'-'}</div>
        <div style="font-size:12px;color:var(--dim)">近期趋势</div></div>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
      <div><h4 style="font-size:13px;color:var(--dim);margin-bottom:8px">按行动类型</h4>${byActionHtml||'<p style="color:var(--dim)">暂无数据</p>'}</div>
      <div><h4 style="font-size:13px;color:var(--dim);margin-bottom:8px">按股票</h4>${bySymbolHtml||'<p style="color:var(--dim)">暂无数据</p>'}</div>
    </div>`;
}

function showPanel(name){
  ['analysis','decisions','skills','report'].forEach(n=>{
    document.getElementById('panel'+n.charAt(0).toUpperCase()+n.slice(1)).style.display=n===name?'block':'none'});
  document.querySelectorAll('.nav-btn').forEach(b=>b.classList.remove('active'));
  const labels={analysis:'📊 分析面板',decisions:'📝 决策历史',skills:'🧠 技能库',report:'📈 性能报告'};
  document.querySelectorAll('.nav-btn').forEach(b=>{if(b.textContent.trim()===labels[name])b.classList.add('active')});
  if(name==='decisions')loadDecisions(currentSymbol,20);
  if(name==='skills')loadSkills('');
  if(name==='report')loadReport();
}

function resetRefresh(){refreshCount=60;if(refreshInterval)clearInterval(refreshInterval);
  refreshInterval=setInterval(()=>{refreshCount--;document.getElementById('refreshTimer').textContent=`自动刷新: ${refreshCount}s`;
    if(refreshCount<=0&&currentSymbol){analyze(currentSymbol)}},1000)}

document.getElementById('addSymbol').addEventListener('keydown',e=>{if(e.key==='Enter')addWatchlist()});
document.getElementById('skillQuery').addEventListener('keydown',e=>{if(e.key==='Enter')searchSkills()});
loadWatchlist();loadSkills('');
</script>
</body>
</html>"""


# ═══════════════════════════════════════════════════════════════
# 请求处理器
# ═══════════════════════════════════════════════════════════════

class _RequestHandler(BaseHTTPRequestHandler):
    """StockMind Web API 请求处理器"""

    agent = None  # 类变量，由 StockMindWebDashboard 注入

    # ─── 通用响应工具 ──────────────────────────────────────────

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self._cors_headers()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self._cors_headers()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length > 0 else b""

    def _parse_json_body(self) -> dict:
        try:
            raw = self._read_body()
            return json.loads(raw) if raw else {}
        except (json.JSONDecodeError, ValueError):
            return {}

    # ─── 路由分发 ──────────────────────────────────────────────

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        params = parse_qs(parsed.query)

        if path == "" or path == "/":
            self._send_html(DASHBOARD_HTML)

        elif path == "/api/analyze":
            symbol = params.get("symbol", [""])[0]
            if not symbol:
                self._send_json({"error": "缺少 symbol 参数"}, 400)
                return
            try:
                result = self.agent.analyze(symbol)
                self._send_json(result)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        elif path == "/api/reflect":
            days = int(params.get("days", ["1"])[0])
            try:
                results = self.agent.reflect(days_ago=days)
                self._send_json({"items": results})
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        elif path == "/api/report":
            try:
                report_data = self.agent.report()
                self._send_json(report_data)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        elif path == "/api/suggest":
            try:
                suggestions = self.agent.suggest()
                self._send_json({"items": suggestions})
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        elif path == "/api/watchlist":
            try:
                items = db.get_watchlist()
                self._send_json({"items": items})
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        elif path == "/api/decisions":
            symbol = params.get("symbol", [""])[0]
            limit = int(params.get("limit", ["10"])[0])
            try:
                if symbol:
                    items = db.get_decisions_by_symbol(symbol, limit=limit)
                else:
                    items = db.get_decisions(limit=limit)
                self._send_json({"items": items})
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        elif path == "/api/skills":
            query = params.get("query", [""])[0]
            try:
                if query:
                    items = db.search_skills(query)
                else:
                    items = db.get_all_skills()
                self._send_json({"items": items})
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        elif path == "/api/health":
            self._send_json({"status": "ok", "service": "StockMind Web Dashboard"})

        else:
            self._send_json({"error": f"未知路由: {path}"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/api/watchlist/add":
            body = self._parse_json_body()
            symbol = body.get("symbol", "").strip().upper()
            if not symbol:
                self._send_json({"error": "缺少 symbol"}, 400)
                return
            try:
                rid = db.add_to_watchlist(symbol)
                if rid == -1:
                    self._send_json({"message": f"{symbol} 已在关注列表中"}, 200)
                else:
                    self._send_json({"message": f"已添加 {symbol}", "id": rid})
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        elif path == "/api/watchlist/remove":
            body = self._parse_json_body()
            symbol = body.get("symbol", "").strip().upper()
            if not symbol:
                self._send_json({"error": "缺少 symbol"}, 400)
                return
            try:
                removed = db.remove_from_watchlist(symbol)
                if removed:
                    self._send_json({"message": f"已移除 {symbol}"})
                else:
                    self._send_json({"message": f"{symbol} 不在关注列表中"}, 200)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        else:
            self._send_json({"error": f"未知路由: {path}"}, 404)

    def log_message(self, format, *args):
        logger.debug(f"[Web] {args[0] if args else ''}")


# ═══════════════════════════════════════════════════════════════
# Web 仪表盘主类
# ═══════════════════════════════════════════════════════════════

class StockMindWebDashboard:
    """StockMind Web 仪表盘：在后台线程中启动 HTTP 服务"""

    def __init__(self, agent, host: str = None, port: int = None):
        self.agent = agent
        self.host = host or WEB_HOST
        self.port = port or WEB_PORT
        self._server: HTTPServer = None
        self._thread: threading.Thread = None

    def start(self):
        """在后台线程中启动 Web 服务器"""
        if self._server is not None:
            logger.warning("[Web] 服务器已在运行")
            return

        _RequestHandler.agent = self.agent

        self._server = HTTPServer((self.host, self.port), _RequestHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

        logger.info(f"[Web] 仪表盘已启动: http://{self.host}:{self.port}")
        print(f"  🌐 Web 仪表盘: http://localhost:{self.port}")

    def stop(self):
        """停止 Web 服务器"""
        if self._server is not None:
            self._server.shutdown()
            self._server = None
            self._thread = None
            logger.info("[Web] 仪表盘已停止")
