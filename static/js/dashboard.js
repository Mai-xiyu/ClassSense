/**
 * AI课堂读空气 —— 教师仪表盘前端逻辑
 */

// ===== 状态 =====
let ws = null;
let chart = null;
let isRunning = false;
let classActive = false;
let currentSessionId = null;
let timelineData = { labels: [], scores: [] };
let eventLog = [];
let hasLoggedDataArrival = false;
const MAX_TIMELINE_POINTS = 300;
const MAX_LOG_ITEMS = 50;

// 行为颜色映射
const BEHAVIOR_COLORS = {
    focused: '#16a34a',
    head_down: '#ca8a04',
    lying_down: '#dc2626',
    hand_raised: '#2563eb',
    looking_away: '#7c3aed',
};
const BEHAVIOR_CN = {
    focused: '专注',
    head_down: '低头',
    lying_down: '趴桌',
    hand_raised: '举手',
    looking_away: '扭头',
};

// ===== 初始化 =====
document.addEventListener('DOMContentLoaded', async () => {
    // 检查是否有进行中的课堂
    const resp = await fetch('/api/status');
    const status = await resp.json();
    if (status.is_active) {
        classActive = true;
        currentSessionId = status.session_id;
        enterClassMode();
    }
    connectWebSocket();
});

// ===== 上课/下课控制 =====
async function startClass() {
    const btn = document.getElementById('btnStart');
    btn.disabled = true;
    btn.textContent = '启动中...';

    const name = document.getElementById('className').value.trim() || undefined;
    try {
        const resp = await fetch('/api/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name }),
        });
        const data = await resp.json();
        if (data.error) {
            alert(data.error);
            btn.disabled = false;
            btn.textContent = '开始检测';
            return;
        }
        classActive = true;
        currentSessionId = data.session_id;
        enterClassMode(true);
        // 重置数据
        timelineData = { labels: [], scores: [] };
        if (chart) {
            chart.data.labels = [];
            chart.data.datasets[0].data = [];
            chart.update('none');
        }
    } catch (e) {
        alert('启动失败: ' + e.message);
        btn.disabled = false;
        btn.textContent = '开始检测';
    }
}

async function stopClass() {
    if (!confirm('确定要下课吗？将停止检测并生成课后报告。')) return;

    const btn = document.getElementById('btnStop');
    btn.disabled = true;
    btn.textContent = '保存中...';

    try {
        const resp = await fetch('/api/stop', { method: 'POST' });
        const data = await resp.json();
        if (data.error) {
            alert(data.error);
            btn.disabled = false;
            btn.textContent = '结束课程';
            return;
        }
        // 跳转到课后报告
        window.location.href = `/report/${data.session_id}`;
    } catch (e) {
        alert('下课失败: ' + e.message);
        btn.disabled = false;
        btn.textContent = '结束课程';
    }
}

function enterClassMode(isFreshStart = false) {
    document.getElementById('startOverlay').style.display = 'none';
    document.getElementById('mainContent').style.display = '';
    document.getElementById('bottomBar').style.display = '';
    document.getElementById('classInfo').textContent =
        `课堂 #${currentSessionId} 进行中`;
    initChart();
    document.getElementById('statusText').textContent = '检测中';
    document.getElementById('statusDot').classList.add('active');

    const list = document.getElementById('logList');
    list.innerHTML = '';
    eventLog = [];
    hasLoggedDataArrival = false;
    if (isFreshStart) {
        addLog(0, '课程已开始，等待实时数据接入', '#2563eb');
    } else {
        addLog(0, '已进入进行中的课堂', '#2563eb');
    }
}

// ===== 历史记录 =====
async function showHistory() {
    const modal = document.getElementById('historyModal');
    const body = document.getElementById('historyList');
    modal.style.display = 'flex';
    body.innerHTML = '加载中...';

    try {
        const resp = await fetch('/api/sessions');
        const sessions = await resp.json();
        if (sessions.length === 0) {
            body.innerHTML = '<p style="color:var(--text-muted);text-align:center">暂无历史记录</p>';
            return;
        }
        let html = `<table class="history-table"><thead><tr>
            <th>课堂</th><th>时间</th><th>专注度</th><th>报告</th>
        </tr></thead><tbody>`;
        for (const s of sessions) {
            const time = s.start_time ? new Date(s.start_time).toLocaleString('zh-CN') : '--';
            let badge = '--';
            if (s.avg_attention != null) {
                const score = Math.round(s.avg_attention);
                const cls = score >= 70 ? 'score-high' : score >= 40 ? 'score-mid' : 'score-low';
                badge = `<span class="score-badge ${cls}">${score}%</span>`;
            }
            const reportLink = s.end_time
                ? `<a href="/report/${s.id}">查看报告</a>`
                : '<span style="color:var(--text-muted)">进行中</span>';
            html += `<tr>
                <td>${s.name}</td>
                <td style="color:var(--text-muted)">${time}</td>
                <td>${badge}</td>
                <td>${reportLink}</td>
            </tr>`;
        }
        html += '</tbody></table>';
        body.innerHTML = html;
    } catch (e) {
        body.innerHTML = `<p style="color:var(--red)">加载失败: ${e.message}</p>`;
    }
}

function closeHistory(event) {
    if (event.target === document.getElementById('historyModal')) {
        document.getElementById('historyModal').style.display = 'none';
    }
}

// ===== WebSocket =====
function connectWebSocket() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${location.host}/ws/attention`);

    ws.onopen = () => {
        isRunning = true;
        if (classActive) {
            document.getElementById('statusDot').classList.add('active');
            document.getElementById('statusText').textContent = '检测中';
        }
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        // 收到下课通知 → 跳转报告页
        if (data.type === 'class_ended') {
            window.location.href = `/report/${data.session_id}`;
            return;
        }
        // Agent 流式消息
        if (data.type === 'agent_start') {
            onAgentStart(data.agent);
            return;
        }
        if (data.type === 'agent_chunk') {
            onAgentChunk(data.agent, data.content);
            return;
        }
        if (data.type === 'agent_done') {
            onAgentDone(data.agent, data.content);
            return;
        }
        if (data.type === 'agent_error') {
            onAgentError(data.agent, data.error);
            return;
        }
        if (classActive) {
            updateDashboard(data);
        }
    };

    ws.onclose = () => {
        isRunning = false;
        document.getElementById('statusDot').classList.remove('active');
        document.getElementById('statusText').textContent = '已断开';
        // 仅在课堂进行中时重连
        if (classActive) {
            setTimeout(connectWebSocket, 3000);
        }
    };

    ws.onerror = () => {
        ws.close();
    };
}

// ===== 更新仪表盘 =====
function updateDashboard(data) {
    const score = data.smoothed_score || 0;
    const total = data.total_people || 0;
    const behaviors = data.behavior_counts || {};
    const elapsed = data.elapsed_seconds || 0;

    if (!hasLoggedDataArrival) {
        addLog(elapsed, '已接入实时数据流', '#16a34a');
        hasLoggedDataArrival = true;
    }

    // 1. 更新专注度环形指标
    updateScoreRing(score);

    // 2. 更新统计数字
    document.getElementById('totalPeople').textContent = total;
    document.getElementById('elapsedTime').textContent = formatTime(elapsed);

    const focusedCount = (behaviors.focused || 0) + (behaviors.hand_raised || 0);
    document.getElementById('focusedCount').textContent = focusedCount;

    // 3. 更新时间线图表
    updateTimeline(elapsed, score);

    // 4. 更新行为分布
    updateBehaviorBars(behaviors, total);

    // 5. 检测事件并记录日志
    checkEvents(data);
}

// ===== 专注度环形 =====
function updateScoreRing(score) {
    const circle = document.getElementById('scoreCircle');
    const number = document.getElementById('scoreNumber');
    const label = document.getElementById('scoreLabel');

    // 圆环进度 (周长 ≈ 440)
    const circumference = 440;
    const offset = circumference - (score / 100) * circumference;
    circle.style.strokeDashoffset = offset;

    let color, labelText;
    if (score >= 70) {
        color = '#16a34a'; labelText = '课堂状态良好';
    } else if (score >= 40) {
        color = '#ca8a04'; labelText = '注意力有所下降';
    } else {
        color = '#dc2626'; labelText = '需要调整节奏';
    }
    circle.style.stroke = color;
    number.textContent = Math.round(score);
    label.textContent = labelText;
    label.style.color = color;
}

// ===== 时间线图表 =====
function initChart() {
    const ctx = document.getElementById('timelineChart').getContext('2d');
    
    chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: '专注度 %',
                data: [],
                borderColor: '#16a34a',
                backgroundColor: 'rgba(22, 163, 74, 0.06)',
                fill: true,
                tension: 0.3,
                pointRadius: 0,
                pointHoverRadius: 5,
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: '#16a34a',
                pointHoverBorderWidth: 2,
                borderWidth: 2,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            animation: { duration: 400, easing: 'easeOutQuart' },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { color: '#a1a1aa', maxTicksLimit: 6, font: { size: 11 } },
                    border: { display: false }
                },
                y: {
                    min: 0, max: 100,
                    grid: { color: '#f0f0f2' },
                    ticks: { color: '#a1a1aa', stepSize: 25, font: { size: 11 } },
                    border: { display: false }
                },
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#18181b',
                    titleColor: '#a1a1aa',
                    bodyColor: '#ffffff',
                    borderWidth: 0,
                    padding: { top: 6, bottom: 6, left: 10, right: 10 },
                    titleFont: { size: 11, weight: '500' },
                    bodyFont: { size: 14, weight: '600' },
                    displayColors: false,
                    cornerRadius: 6,
                    callbacks: {
                        label: function(context) { return context.parsed.y + '%'; }
                    }
                },
            },
        },
    });
}

function updateTimeline(elapsed, score) {
    const label = formatTime(elapsed);
    timelineData.labels.push(label);
    timelineData.scores.push(score);

    if (timelineData.labels.length > MAX_TIMELINE_POINTS) {
        timelineData.labels.shift();
        timelineData.scores.shift();
    }

    chart.data.labels = timelineData.labels;
    chart.data.datasets[0].data = timelineData.scores;

    const ds = chart.data.datasets[0];
    if (score >= 70) {
        ds.borderColor = '#16a34a';
        ds.backgroundColor = 'rgba(22, 163, 74, 0.06)';
        ds.pointHoverBorderColor = '#16a34a';
    } else if (score >= 40) {
        ds.borderColor = '#ca8a04';
        ds.backgroundColor = 'rgba(202, 138, 4, 0.06)';
        ds.pointHoverBorderColor = '#ca8a04';
    } else {
        ds.borderColor = '#dc2626';
        ds.backgroundColor = 'rgba(220, 38, 38, 0.06)';
        ds.pointHoverBorderColor = '#dc2626';
    }
    
    chart.update('none');
}

// ===== 行为分布条 =====
function updateBehaviorBars(behaviors, total) {
    const container = document.getElementById('behaviorBars');
    if (total === 0) return;

    const allBehaviors = ['focused', 'head_down', 'lying_down', 'hand_raised', 'looking_away'];
    let html = '';

    for (const beh of allBehaviors) {
        const count = behaviors[beh] || 0;
        const pct = Math.round((count / total) * 100);
        const color = BEHAVIOR_COLORS[beh];
        const label = BEHAVIOR_CN[beh];

        html += `
            <div class="behavior-row">
                <span class="behavior-label">${label}</span>
                <div class="behavior-bar-track">
                    <div class="behavior-bar-fill" style="width:${pct}%;background:${color}"></div>
                </div>
                <span class="behavior-count" style="color:${color}">${count}</span>
            </div>
        `;
    }
    container.innerHTML = html;
}

// ===== 事件日志 =====
function checkEvents(data) {
    const score = data.smoothed_score || 0;
    const elapsed = data.elapsed_seconds || 0;
    const behaviors = data.behavior_counts || {};

    // 专注度骤降
    if (timelineData.scores.length > 5) {
        const recent = timelineData.scores.slice(-5);
        const prev = timelineData.scores.slice(-10, -5);
        if (prev.length === 5) {
            const recentAvg = recent.reduce((a, b) => a + b, 0) / 5;
            const prevAvg = prev.reduce((a, b) => a + b, 0) / 5;
            if (prevAvg - recentAvg > 20) {
                addLog(elapsed, '专注度快速下降', '#ca8a04');
            }
        }
    }

    // 大面积趴桌
    const total = data.total_people || 0;
    const lying = behaviors.lying_down || 0;
    if (total > 0 && lying / total > 0.2) {
        addLog(elapsed, `${lying}人趴桌（${Math.round(lying/total*100)}%）`, '#dc2626');
    }

    // 有人举手
    const raised = behaviors.hand_raised || 0;
    if (raised > 0) {
        addLog(elapsed, `${raised}人举手`, '#2563eb');
    }
}

function addLog(elapsed, message, color) {
    const list = document.getElementById('logList');

    // 避免短时间内重复
    if (eventLog.length > 0) {
        const last = eventLog[eventLog.length - 1];
        if (last.message === message && elapsed - last.elapsed < 10) return;
    }

    eventLog.push({ elapsed, message, color });
    if (eventLog.length > MAX_LOG_ITEMS) eventLog.shift();

    const item = document.createElement('li');
    item.className = 'log-item';
    item.innerHTML = `
        <span class="log-time">${formatTime(elapsed)}</span>
        <span class="log-dot" style="background:${color}"></span>
        <span>${message}</span>
    `;

    list.insertBefore(item, list.firstChild);
    if (list.children.length > MAX_LOG_ITEMS) {
        list.removeChild(list.lastChild);
    }
}

// ===== 工具函数 =====
function formatTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

// ===== Agent 面板 =====
function onAgentStart(agent) {
    const el = agent === 'student' ? document.getElementById('studentContent')
                                   : document.getElementById('teacherContent');
    el.textContent = '';
    el.classList.add('agent-typing');
    document.getElementById('agentStatusBar').innerHTML =
        '<span class="typing-dot"></span> <span>AI 正在分析中...</span>';
    document.getElementById('btnAnalyze').disabled = true;
}

function onAgentChunk(agent, content) {
    const el = agent === 'student' ? document.getElementById('studentContent')
                                   : document.getElementById('teacherContent');
    el.textContent += content;
}

function onAgentDone(agent, content) {
    const el = agent === 'student' ? document.getElementById('studentContent')
                                   : document.getElementById('teacherContent');
    el.textContent = content;
    el.classList.remove('agent-typing');
    const now = new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
    document.getElementById('agentStatusBar').innerHTML =
        `<span>上次分析: ${now}</span>`;
    document.getElementById('btnAnalyze').disabled = false;
}

function onAgentError(agent, error) {
    const el = agent === 'student' ? document.getElementById('studentContent')
                                   : document.getElementById('teacherContent');
    el.innerHTML = `<span style="color:var(--red)">${error}</span>`;
    el.classList.remove('agent-typing');
    document.getElementById('btnAnalyze').disabled = false;
}

async function triggerAnalysis() {
    const btn = document.getElementById('btnAnalyze');
    btn.disabled = true;
    try {
        const resp = await fetch('/api/agent/analyze', { method: 'POST' });
        const data = await resp.json();
        if (data.error) {
            alert(data.error);
            btn.disabled = false;
        }
        // 结果会通过 WebSocket 推送，不需要在这里处理
    } catch (e) {
        alert('分析请求失败: ' + e.message);
        btn.disabled = false;
    }
}

// 页面加载时检查 Agent 状态
async function checkAgentStatus() {
    try {
        const resp = await fetch('/api/agent/status');
        const data = await resp.json();
        if (!data.configured) {
            document.getElementById('agentStatusBar').innerHTML =
                '<span><a href="/settings" style="color:var(--purple)">配置大模型</a> 后即可启用 AI 助教分析</span>';
            document.getElementById('btnAnalyze').style.display = 'none';
        }
    } catch (e) { /* ignore */ }
}
checkAgentStatus();
