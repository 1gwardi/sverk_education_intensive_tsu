"""
Центральный веб-сервер управления роем (Ноутбук: 192.168.1.76)
Реальный полёт: приём снимков, координат, результатов детекции.
"""
import json
import threading
import math
import base64
import os
from http.server import HTTPServer, BaseHTTPRequestHandler

MARKER_IDS = list(range(25, 50))
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'captures')
os.makedirs(UPLOAD_DIR, exist_ok=True)

state = {
    'excluded_markers': [],
    'division_axis': 'y',
    'mission_status': 'idle',
    'drones': {
        'drone_1': {
            'online': False, 'start_marker': None,
            'route_ready': False, 'route_accepted': False, 'route': [],
            'results': {'Красный': 0, 'Жёлтый': 0, 'Синий': 0},
            'items': [],  # Найденные объекты с координатами и снимками
            'status': 'Офлайн', 'telemetry': {'x': 0, 'y': 0, 'z': 0}
        },
        'drone_2': {
            'online': False, 'start_marker': None,
            'route_ready': False, 'route_accepted': False, 'route': [],
            'results': {'Красный': 0, 'Жёлтый': 0, 'Синий': 0},
            'items': [],
            'status': 'Офлайн', 'telemetry': {'x': 0, 'y': 0, 'z': 0}
        }
    }
}
state_lock = threading.Lock()


def get_marker_coords(marker_id):
    col = (marker_id - 25) % 5
    row = (marker_id - 25) // 5
    return -2 + col, 2 - row


def optimize_route(route_points):
    if len(route_points) <= 2:
        return route_points
    optimized = [route_points[0]]
    remaining = route_points[1:]
    while remaining:
        last = optimized[-1]
        nearest = min(remaining, key=lambda p: math.hypot(p['x'] - last['x'], p['y'] - last['y']))
        optimized.append(nearest)
        remaining.remove(nearest)
    return optimized


def build_routes():
    d1, d2 = state['drones']['drone_1'], state['drones']['drone_2']
    if d1['start_marker'] is None or d2['start_marker'] is None:
        return False

    d1_start_x, d1_start_y = get_marker_coords(d1['start_marker'])
    d2_start_x, d2_start_y = get_marker_coords(d2['start_marker'])

    available_markers = []
    for marker_id in MARKER_IDS:
        if marker_id in state['excluded_markers']:
            continue
        if marker_id == d1['start_marker'] or marker_id == d2['start_marker']:
            continue
        x, y = get_marker_coords(marker_id)
        available_markers.append({'marker_id': marker_id, 'x': x, 'y': y})

    axis = state['division_axis']
    d1_markers = []
    d2_markers = []

    if axis == 'y':
        for marker in available_markers:
            if marker['x'] < 0:
                d1_markers.append(marker)
            elif marker['x'] > 0:
                d2_markers.append(marker)
            else:
                if abs(marker['y'] - d1_start_y) < abs(marker['y'] - d2_start_y):
                    d1_markers.append(marker)
                else:
                    d2_markers.append(marker)
    else:
        for marker in available_markers:
            if marker['y'] > 0:
                d1_markers.append(marker)
            elif marker['y'] < 0:
                d2_markers.append(marker)
            else:
                if abs(marker['x'] - d1_start_x) < abs(marker['x'] - d2_start_x):
                    d1_markers.append(marker)
                else:
                    d2_markers.append(marker)

    if abs(len(d1_markers) - len(d2_markers)) > 2:
        if len(d1_markers) > len(d2_markers):
            d1_markers.sort(key=lambda m: abs(m['x']) if axis == 'y' else abs(m['y']))
            transfer_count = (len(d1_markers) - len(d2_markers)) // 2
            for _ in range(transfer_count):
                if d1_markers:
                    d2_markers.append(d1_markers.pop())
        else:
            d2_markers.sort(key=lambda m: abs(m['x']) if axis == 'y' else abs(m['y']))
            transfer_count = (len(d2_markers) - len(d1_markers)) // 2
            for _ in range(transfer_count):
                if d2_markers:
                    d1_markers.append(d2_markers.pop())

    r1 = [{'x': d1_start_x, 'y': d1_start_y, 'z': 1.2, 'marker_id': d1['start_marker']}]
    r2 = [{'x': d2_start_x, 'y': d2_start_y, 'z': 1.2, 'marker_id': d2['start_marker']}]

    for marker in d1_markers:
        r1.append({'x': marker['x'], 'y': marker['y'], 'z': 1.2, 'marker_id': marker['marker_id']})
    for marker in d2_markers:
        r2.append({'x': marker['x'], 'y': marker['y'], 'z': 1.2, 'marker_id': marker['marker_id']})

    r1 = optimize_route(r1)
    r2 = optimize_route(r2)

    d1['route'] = r1
    d2['route'] = r2
    d1['route_ready'] = True
    d2['route_ready'] = True
    d1['status'] = 'Маршрут готов к выдаче'
    d2['status'] = 'Маршрут готов к выдаче'
    return True


HTML_PAGE = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<title>Управление роем дронов</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'Segoe UI', sans-serif; background: #f0f2f5; padding: 20px; }
  .container { max-width: 1400px; margin: 0 auto; display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; }
  .card { background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
  h2 { margin-top: 0; color: #333; border-bottom: 2px solid #667eea; padding-bottom: 10px; font-size: 1.3em; }
  .grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; margin: 20px 0; }
  .marker-btn { aspect-ratio: 1; border: 2px solid #ccc; border-radius: 8px; background: #f5f5f5;
    font-size: 1.1em; font-weight: bold; cursor: pointer; transition: all 0.2s;
    display: flex; align-items: center; justify-content: center; }
  .marker-btn:hover { transform: scale(1.05); }
  .marker-btn.excluded { background: #ffebee; border-color: #e53935; color: #c62828; text-decoration: line-through; }
  .marker-btn.route-d1 { background: #e3f2fd; border-color: #2196f3; color: #0d47a1; box-shadow: 0 0 8px rgba(33,150,243,0.4); }
  .marker-btn.route-d2 { background: #e8f5e9; border-color: #4caf50; color: #1b5e20; box-shadow: 0 0 8px rgba(76,175,80,0.4); }
  .marker-btn.start-d1 { background: #bbdefb; border-color: #1565c0; color: #0d47a1; font-weight: 900; }
  .marker-btn.start-d2 { background: #c8e6c9; border-color: #2e7d32; color: #1b5e20; font-weight: 900; }
  .legend { display: flex; gap: 10px; margin-top: 10px; font-size: 0.8em; color: #666; flex-wrap: wrap; }
  .legend-item { display: flex; align-items: center; gap: 5px; }
  .legend-box { width: 14px; height: 14px; border-radius: 4px; border: 2px solid; }
  .axis-selector { display: flex; gap: 10px; margin: 15px 0; }
  .axis-btn { flex: 1; padding: 12px; border: 2px solid #ccc; border-radius: 8px; background: white;
    font-size: 0.95em; font-weight: bold; cursor: pointer; transition: all 0.2s; }
  .axis-btn.active { background: #667eea; color: white; border-color: #667eea; }
  .axis-btn:hover:not(.active) { background: #f0f0f0; }
  .route-sequence { margin-top: 15px; padding: 10px; background: #fafafa; border-radius: 8px; font-size: 0.85em; }
  .route-sequence strong { display: block; margin-bottom: 5px; }
  .seq-d1 { color: #1565c0; } .seq-d2 { color: #2e7d32; }
  .controls { display: flex; flex-direction: column; gap: 10px; margin-top: 20px; }
  button.action-btn { width: 100%; padding: 14px; border: none; border-radius: 8px; font-size: 1em; font-weight: bold; cursor: pointer; color: white; }
  .btn-takeoff { background: #ff9800; }
  .btn-send-route { background: #2196f3; }
  .btn-reset { background: #f44336; }
  button:disabled { background: #ccc; cursor: not-allowed; opacity: 0.7; }
  .phase-indicator { margin-top: 15px; padding: 12px; border-radius: 8px; font-weight: bold; text-align: center; font-size: 0.95em; }
  .phase-idle { background: #eceff1; color: #546e7a; }
  .phase-takeoff { background: #fff3e0; color: #e65100; animation: pulse 1.5s infinite; }
  .phase-ready { background: #e8f5e9; color: #1b5e20; }
  .phase-started { background: #c8e6c9; color: #1b5e20; animation: pulse 1.5s infinite; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.7; } }
  .status-hint { margin-top: 10px; font-size: 0.85em; color: #666; text-align: center; }
  .drone-status { margin-top: 15px; padding: 12px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #ccc; }
  .drone-status.online { border-left-color: #4caf50; }
  .drone-status h3 { margin: 0 0 10px 0; font-size: 1.05em; display: flex; justify-content: space-between; align-items: center; }
  .status-dot { font-size: 0.8em; padding: 3px 10px; border-radius: 12px; background: #eee; color: #666; }
  .status-dot.active { background: #e8f5e9; color: #2e7d32; font-weight: bold; }
  .drone-info { font-size: 0.9em; margin: 8px 0; color: #555; }
  .drone-info span { font-weight: 600; color: #333; }
  .results { display: flex; gap: 12px; margin-top: 12px; font-weight: bold; font-size: 0.95em; }
  .res-red { color: #e53935; } .res-yellow { color: #fbc02d; } .res-blue { color: #1e88e5; }
  .total-results { margin-top: 20px; padding: 15px; background: #e8eaf6; border-radius: 8px; text-align: center; }
  .total-results h3 { color: #3f51b5; margin-bottom: 10px; font-size: 1.1em; }
  .total-counts { display: flex; justify-content: space-around; font-size: 1.2em; font-weight: bold; }
  .items-section { margin-top: 20px; }
  .items-section h3 { font-size: 1em; margin-bottom: 10px; color: #333; }
  .items-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; max-height: 400px; overflow-y: auto; }
  .item-card { background: #f8f9fa; border-radius: 8px; padding: 8px; border-left: 4px solid #667eea; }
  .item-card img { width: 100%; height: 100px; object-fit: cover; border-radius: 6px; margin-bottom: 5px; background: #ddd; }
  .item-info { font-size: 0.75em; color: #555; }
  .item-info b { color: #333; }
</style>
</head>
<body>
<div class="container">
  <div class="card">
    <h2>🗺️ Карта ArUco-меток (5×5, ID 25–49)</h2>
    <p style="font-size: 0.9em; color: #666; margin-bottom: 10px;">Нажмите на метку, чтобы исключить её</p>
    <div class="grid" id="markerGrid"></div>
    <div class="legend">
      <div class="legend-item"><div class="legend-box" style="background:#ffebee; border-color:#e53935;"></div> Исключена</div>
      <div class="legend-item"><div class="legend-box" style="background:#bbdefb; border-color:#1565c0;"></div> Старт D1</div>
      <div class="legend-item"><div class="legend-box" style="background:#c8e6c9; border-color:#2e7d32;"></div> Старт D2</div>
      <div class="legend-item"><div class="legend-box" style="background:#e3f2fd; border-color:#2196f3;"></div> Маршрут D1</div>
      <div class="legend-item"><div class="legend-box" style="background:#e8f5e9; border-color:#4caf50;"></div> Маршрут D2</div>
    </div>
  </div>
  <div class="card">
    <h2>⚙️ Управление миссией</h2>
    <div class="axis-selector">
      <button class="axis-btn" id="axisY" onclick="setAxis('y')">↕️ По оси Y (вдоль)</button>
      <button class="axis-btn" id="axisX" onclick="setAxis('x')">↔️ По оси X (поперёк)</button>
    </div>
    <div class="route-sequence">
      <strong class="seq-d1">🚁 Маршрут Drone 1 (ID меток):</strong>
      <span id="d1-route-seq">-</span>
    </div>
    <div class="route-sequence">
      <strong class="seq-d2">🚁 Маршрут Drone 2 (ID меток):</strong>
      <span id="d2-route-seq">-</span>
    </div>
    <div id="phase-indicator" class="phase-indicator phase-idle"> Ожидание начала</div>
    <div class="controls">
      <button class="action-btn btn-takeoff" id="takeoffBtn" onclick="sendTakeoff()" disabled>🛫 Отправить команду взлёта</button>
      <button class="action-btn btn-send-route" id="sendRouteBtn" onclick="sendRoute()" disabled>📤 Отправить путь</button>
      <button class="action-btn btn-reset" onclick="resetAll()">🔄 Сброс</button>
    </div>
    <p id="status-hint" class="status-hint">⚠️ Ожидание: дроны должны быть в сети</p>
  </div>
  <div class="card">
    <h2>🚁 Статус дронов</h2>
    <div id="d1-card" class="drone-status">
      <h3>Drone 1 (192.168.1.7) <span id="d1-online" class="status-dot">● Офлайн</span></h3>
      <div class="drone-info">Статус: <span id="d1-status">Офлайн</span></div>
      <div class="drone-info">Стартовая метка: <span id="d1-start">-</span></div>
      <div class="drone-info">Маршрут: <span id="d1-route">Не сгенерирован</span></div>
      <div class="results">
        <span class="res-red">К: <span id="d1-r">0</span></span>
        <span class="res-yellow">Ж: <span id="d1-y">0</span></span>
        <span class="res-blue">С: <span id="d1-b">0</span></span>
      </div>
    </div>
    <div id="d2-card" class="drone-status" style="margin-top: 15px;">
      <h3>Drone 2 (192.168.1.15) <span id="d2-online" class="status-dot">● Офлайн</span></h3>
      <div class="drone-info">Статус: <span id="d2-status">Офлайн</span></div>
      <div class="drone-info">Стартовая метка: <span id="d2-start">-</span></div>
      <div class="drone-info">Маршрут: <span id="d2-route">Не сгенерирован</span></div>
      <div class="results">
        <span class="res-red">К: <span id="d2-r">0</span></span>
        <span class="res-yellow">Ж: <span id="d2-y">0</span></span>
        <span class="res-blue">С: <span id="d2-b">0</span></span>
      </div>
    </div>
    <div class="total-results">
      <h3>📊 Общий итог</h3>
      <div class="total-counts">
        <span class="res-red">К: <span id="total-r">0</span></span>
        <span class="res-yellow">Ж: <span id="total-y">0</span></span>
        <span class="res-blue">С: <span id="total-b">0</span></span>
      </div>
    </div>
    <div class="items-section">
      <h3>📸 Найденные объекты</h3>
      <div id="items-container" class="items-grid"></div>
    </div>
  </div>
</div>
<script>
const grid = document.getElementById('markerGrid');
for (let i = 25; i <= 49; i++) {
  const btn = document.createElement('button');
  btn.className = 'marker-btn'; btn.textContent = i;
  btn.onclick = () => toggleMarker(i, btn);
  grid.appendChild(btn);
}
function toggleMarker(id, btn) {
  fetch('/api/toggle_marker', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({id: id}) })
  .then(() => updateUI());
}
function setAxis(axis) {
  fetch('/api/set_axis', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({axis: axis}) })
  .then(() => updateUI());
}
function sendTakeoff() {
  if(!confirm('Отправить команду ВЗЛЁТА дронам?')) return;
  fetch('/api/send_takeoff', {method: 'POST'}).then(() => updateUI());
}
function sendRoute() {
  if(!confirm('Отправить путь дронам? Они начнут выполнение.')) return;
  fetch('/api/send_route', {method: 'POST'}).then(() => updateUI());
}
function resetAll() { fetch('/api/reset', {method: 'POST'}).then(() => location.reload()); }
function updateUI() {
  fetch('/api/state').then(r => r.json()).then(data => {
    document.getElementById('axisY').classList.toggle('active', data.division_axis === 'y');
    document.getElementById('axisX').classList.toggle('active', data.division_axis === 'x');
    const d1_ids = data.drones.drone_1.route.map(wp => wp.marker_id).filter(id => id !== undefined);
    const d2_ids = data.drones.drone_2.route.map(wp => wp.marker_id).filter(id => id !== undefined);
    const d1_start = data.drones.drone_1.start_marker;
    const d2_start = data.drones.drone_2.start_marker;
    document.querySelectorAll('.marker-btn').forEach(btn => {
      const id = parseInt(btn.textContent);
      btn.className = 'marker-btn';
      if (data.excluded_markers.includes(id)) btn.classList.add('excluded');
      if (d1_start === id) btn.classList.add('start-d1');
      if (d2_start === id) btn.classList.add('start-d2');
      if (d1_ids.includes(id)) btn.classList.add('route-d1');
      if (d2_ids.includes(id)) btn.classList.add('route-d2');
    });
    document.getElementById('d1-route-seq').textContent = d1_ids.length > 0 ? d1_ids.join(' → ') : '-';
    document.getElementById('d2-route-seq').textContent = d2_ids.length > 0 ? d2_ids.join(' → ') : '-';
    const takeoffBtn = document.getElementById('takeoffBtn');
    const sendRouteBtn = document.getElementById('sendRouteBtn');
    const hint = document.getElementById('status-hint');
    const phase = document.getElementById('phase-indicator');
    const bothOnline = data.drones.drone_1.online && data.drones.drone_2.online;
    const bothAccepted = data.drones.drone_1.route_accepted && data.drones.drone_2.route_accepted;
    takeoffBtn.disabled = !(bothOnline && data.mission_status === 'idle');
    sendRouteBtn.disabled = !(data.mission_status === 'ready' && bothAccepted);
    phase.className = 'phase-indicator';
    if (data.mission_status === 'idle') {
      phase.classList.add('phase-idle');
      if (bothOnline) { phase.textContent = '✅ Дроны в сети. Готовы к взлёту.'; hint.textContent = 'Выберите ось деления и нажмите "Отправить команду взлёта".'; hint.style.color = '#2e7d32'; }
      else { phase.textContent = '⏸ Ожидание подключения дронов'; hint.textContent = '⚠️ Ожидание: дроны должны быть в сети.'; hint.style.color = '#666'; }
    } else if (data.mission_status === 'takeoff') {
      phase.classList.add('phase-takeoff');
      phase.textContent = '🛫 Дроны взлетают и ищут стартовые метки...';
      hint.textContent = 'Ожидание координат от обоих дронов...';
      hint.style.color = '#e65100';
    } else if (data.mission_status === 'ready') {
      phase.classList.add('phase-ready');
      phase.textContent = '✅ Маршруты готовы и подтверждены!';
      hint.textContent = 'Нажмите "Отправить путь", чтобы дроны начали выполнение.';
      hint.style.color = '#1b5e20';
    } else if (data.mission_status === 'started') {
      phase.classList.add('phase-started');
      phase.textContent = ' Миссия выполняется!';
      hint.textContent = 'Дроны работают автономно.';
      hint.style.color = '#1b5e20';
    }
    ['1', '2'].forEach(i => {
      const d = data.drones['drone_' + i];
      const card = document.getElementById('d'+i+'-card');
      const onlineEl = document.getElementById('d'+i+'-online');
      if (d.online) { onlineEl.textContent = '● В сети'; onlineEl.classList.add('active'); card.classList.add('online'); }
      else { onlineEl.textContent = '● Офлайн'; onlineEl.classList.remove('active'); card.classList.remove('online'); }
      document.getElementById('d'+i+'-status').textContent = d.status;
      document.getElementById('d'+i+'-start').textContent = d.start_marker !== null ? d.start_marker : '-';
      let routeText = 'Не сгенерирован';
      if (d.route_accepted) routeText = '✅ Принят (' + d.route.length + ' точек)';
      else if (d.route_ready) routeText = '⏳ Ожидает подтверждения...';
      document.getElementById('d'+i+'-route').textContent = routeText;
      document.getElementById('d'+i+'-r').textContent = d.results['Красный'];
      document.getElementById('d'+i+'-y').textContent = d.results['Жёлтый'];
      document.getElementById('d'+i+'-b').textContent = d.results['Синий'];
    });
    document.getElementById('total-r').textContent = data.drones.drone_1.results['Красный'] + data.drones.drone_2.results['Красный'];
    document.getElementById('total-y').textContent = data.drones.drone_1.results['Жёлтый'] + data.drones.drone_2.results['Жёлтый'];
    document.getElementById('total-b').textContent = data.drones.drone_1.results['Синий'] + data.drones.drone_2.results['Синий'];

    // Найденные объекты
    const itemsContainer = document.getElementById('items-container');
    const allItems = [...data.drones.drone_1.items, ...data.drones.drone_2.items];
    if (allItems.length === 0) {
      itemsContainer.innerHTML = '<p style="color:#999; font-size:0.85em;">Пока ничего не найдено</p>';
    } else {
      itemsContainer.innerHTML = allItems.map(item => 
        '<div class="item-card">' +
          (item.image_url ? '<img src="' + item.image_url + '" alt="' + item.color + '">' : '') +
          '<div class="item-info"><b>' + item.color + '</b><br>x=' + item.x.toFixed(2) + ', y=' + item.y.toFixed(2) + '</div>' +
        '</div>'
      ).join('');
    }
  });
}
setInterval(updateUI, 1000); updateUI();
</script>
</body>
</html>"""


class RequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            self._send(200, 'text/html; charset=utf-8', HTML_PAGE.encode('utf-8'))
        elif self.path == '/api/state':
            with state_lock:
                response = {
                    'excluded_markers': state['excluded_markers'],
                    'division_axis': state['division_axis'],
                    'mission_status': state['mission_status'],
                    'drones': state['drones']
                }
            self._send(200, 'application/json', json.dumps(response).encode('utf-8'))
        elif self.path in ('/api/drone/drone_1/get_route', '/api/drone/drone_2/get_route'):
            drone_id = self.path.split('/')[-2]
            with state_lock:
                d = state['drones'][drone_id]
                self._send(200, 'application/json',
                           json.dumps({'route': d['route'] if d['route_ready'] else []}).encode('utf-8'))
        elif self.path.startswith('/static/captures/'):
            filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.path.lstrip('/'))
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    self._send(200, 'image/jpeg', f.read())
            else:
                self._send(404, 'text/plain', b'Not found')
        else:
            self._send(404, 'text/plain', b'Not found')

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(length).decode('utf-8')) if length > 0 else {}

        if self.path == '/api/toggle_marker':
            with state_lock:
                mid = body.get('id')
                if mid in state['excluded_markers']:
                    state['excluded_markers'].remove(mid)
                else:
                    state['excluded_markers'].append(mid)
            self._send(200, 'application/json', b'{"status":"ok"}')

        elif self.path == '/api/set_axis':
            with state_lock:
                axis = body.get('axis', 'y')
                if axis in ('x', 'y'):
                    state['division_axis'] = axis
            self._send(200, 'application/json', b'{"status":"ok"}')

        elif self.path == '/api/send_takeoff':
            with state_lock:
                d1, d2 = state['drones']['drone_1'], state['drones']['drone_2']
                if not (d1['online'] and d2['online']):
                    self._send(400, 'application/json', b'{"error":"Drones not online"}')
                    return
                if state['mission_status'] != 'idle':
                    self._send(400, 'application/json', b'{"error":"Mission already started"}')
                    return
                state['mission_status'] = 'takeoff'
                d1['status'] = 'Команда взлёта получена'
                d2['status'] = 'Команда взлёта получена'
            self._send(200, 'application/json', b'{"status":"ok"}')

        elif self.path == '/api/send_route':
            with state_lock:
                d1, d2 = state['drones']['drone_1'], state['drones']['drone_2']
                if state['mission_status'] != 'ready':
                    self._send(400, 'application/json', b'{"error":"Routes not ready"}')
                    return
                if not (d1['route_accepted'] and d2['route_accepted']):
                    self._send(400, 'application/json', b'{"error":"Routes not accepted"}')
                    return
                state['mission_status'] = 'started'
                d1['status'] = 'Выполнение миссии'
                d2['status'] = 'Выполнение миссии'
            self._send(200, 'application/json', b'{"status":"ok"}')

        elif self.path in ('/api/drone/drone_1/online', '/api/drone/drone_2/online'):
            drone_id = self.path.split('/')[-2]
            with state_lock:
                state['drones'][drone_id]['online'] = True
                if state['drones'][drone_id]['status'] == 'Офлайн':
                    state['drones'][drone_id]['status'] = 'В сети (ожидание команды взлёта)'
            self._send(200, 'application/json', b'{"status":"ok"}')

        elif self.path in ('/api/drone/drone_1/report_start', '/api/drone/drone_2/report_start'):
            drone_id = self.path.split('/')[-2]
            with state_lock:
                d = state['drones'][drone_id]
                d['start_marker'] = body.get('marker_id')
                d['status'] = 'Стартовая метка найдена, ожидание второго дрона'
                d1, d2 = state['drones']['drone_1'], state['drones']['drone_2']
                if d1['start_marker'] is not None and d2['start_marker'] is not None:
                    if build_routes():
                        state['mission_status'] = 'ready'
            self._send(200, 'application/json', b'{"status":"ok"}')

        elif self.path in ('/api/drone/drone_1/accept_route', '/api/drone/drone_2/accept_route'):
            drone_id = self.path.split('/')[-2]
            with state_lock:
                state['drones'][drone_id]['route_accepted'] = True
                state['drones'][drone_id]['status'] = 'Маршрут принят, ожидание команды'
            self._send(200, 'application/json', b'{"status":"ok"}')

        elif self.path in ('/api/drone/drone_1/report_finish', '/api/drone/drone_2/report_finish'):
            drone_id = self.path.split('/')[-2]
            with state_lock:
                state['drones'][drone_id]['results'] = body.get('results', {})
                state['drones'][drone_id]['status'] = 'Миссия завершена'
            self._send(200, 'application/json', b'{"status":"ok"}')

        elif self.path in ('/api/drone/drone_1/report_item', '/api/drone/drone_2/report_item'):
            drone_id = self.path.split('/')[-2]
            with state_lock:
                item = body.get('item', {})
                if 'image_base64' in item:
                    img_data = base64.b64decode(item['image_base64'])
                    filename = f"{drone_id}_item_{len(state['drones'][drone_id]['items'])}_{int(time.time() * 1000)}.jpg"
                    filepath = os.path.join(UPLOAD_DIR, filename)
                    with open(filepath, 'wb') as f:
                        f.write(img_data)
                    item['image_url'] = f'/static/captures/{filename}'
                    del item['image_base64']
                state['drones'][drone_id]['items'].append(item)
            self._send(200, 'application/json', b'{"status":"ok"}')

        elif self.path in ('/api/drone/drone_1/telemetry', '/api/drone/drone_1/update_status',
                           '/api/drone/drone_2/telemetry', '/api/drone/drone_2/update_status'):
            drone_id = self.path.split('/')[-2]
            with state_lock:
                if 'status' in body: state['drones'][drone_id]['status'] = body['status']
                if 'x' in body or 'y' in body or 'z' in body:
                    state['drones'][drone_id]['telemetry'].update({
                        'x': body.get('x', 0), 'y': body.get('y', 0), 'z': body.get('z', 0)
                    })
            self._send(200, 'application/json', b'{"status":"ok"}')

        elif self.path == '/api/reset':
            with state_lock:
                state['excluded_markers'] = []
                state['division_axis'] = 'y'
                state['mission_status'] = 'idle'
                for d in state['drones'].values():
                    d.update({
                        'online': False, 'start_marker': None, 'route_ready': False,
                        'route_accepted': False, 'route': [], 'items': [],
                        'results': {'Красный': 0, 'Жёлтый': 0, 'Синий': 0},
                        'status': 'Офлайн', 'telemetry': {'x': 0, 'y': 0, 'z': 0}
                    })
            self._send(200, 'application/json', b'{"status":"ok"}')
        else:
            self._send(404, 'text/plain', b'Not found')

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header('Content-type', ctype)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)


if __name__ == '__main__':
    import time

    server = HTTPServer(('0.0.0.0', 8000), RequestHandler)
    print("✅ Сервер запущен: http://192.168.1.76:8000")
    server.serve_forever()