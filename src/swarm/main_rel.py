"""
Клиент дрона — РЕАЛЬНЫЙ ПОЛЁТ.
Логика: взлёт → поиск стартовой метки → получение маршрута → облёт → детекция → посадка.
"""
import sys, time, json, math, base64, os
import numpy as np
import cv2
import urllib.request
import sverk_interfaces
from aruco_det_loc.msg import MarkerArray

SERVER_URL = os.environ.get('MISSION_SERVER_URL', 'http://192.168.1.76:8000').rstrip('/')
VALID_MARKER_IDS = set(range(25, 50))

# Параметры полёта
HEIGHT = 1.5
SPEED = 0.8
FOV = 65.0
MERGE = 0.23
NAMES = ('Красный', 'Жёлтый', 'Синий')

def post_json(url, data):
    try:
        req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'),
                                     headers={'Content-Type': 'application/json'}, method='POST')
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        print(f"  [сеть] {url}: {e}")
        return None

def get_json(url):
    try:
        with urllib.request.urlopen(urllib.request.Request(url), timeout=2.0) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        print(f"  [сеть] {url}: {e}")
        return None

def components(mask):
    """Поиск связных компонент на маске половинного разрешения."""
    m = mask[::2, ::2]
    h, w = m.shape
    seen = np.zeros((h, w), dtype=np.uint8)
    for sy, sx in zip(*np.nonzero(m)):
        if seen[sy, sx]:
            continue
        stack = [(int(sy), int(sx))]
        seen[sy, sx] = 1
        pts = []
        while stack:
            y, x = stack.pop()
            pts.append((y, x))
            for yy, xx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                if 0 <= yy < h and 0 <= xx < w and m[yy,xx] and not seen[yy,xx]:
                    seen[yy,xx] = 1
                    stack.append((yy,xx))
        if len(pts) < 35 or len(pts) > 3000:
            continue
        a = np.asarray(pts)
        y0,x0 = a.min(axis=0); y1,x1 = a.max(axis=0)
        bw,bh = x1-x0+1,y1-y0+1
        if bw < 5 or bh < 5 or min(bw,bh)/max(bw,bh) < 0.45:
            continue
        if len(pts)/(bw*bh) < 0.55:
            continue
        if x0 == 0 or y0 == 0 or x1 == w-1 or y1 == h-1:
            continue
        yield float(a[:,1].mean()*2), float(a[:,0].mean()*2)

def detect(frame):
    """Детекция цветов через HSV (устойчиво к освещению). frame — RGB."""
    # Конвертируем RGB в HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    # Красный (два диапазона)
    mask_r1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
    mask_r2 = cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
    mask_red = cv2.bitwise_or(mask_r1, mask_r2)

    # Жёлтый
    mask_yellow = cv2.inRange(hsv, (20, 100, 100), (35, 255, 255))

    # Синий
    mask_blue = cv2.inRange(hsv, (100, 100, 100), (130, 255, 255))

    masks = (mask_red, mask_yellow, mask_blue)
    for color, mask in enumerate(masks):
        for u, v in components(mask):
            yield color, u, v

def remember(items, color, x, y, drone_id, frame_no):
    """Запомнить объект, сливая близкие обнаружения."""
    best = None
    for item in items:
        if item['color'] == color and math.hypot(item['x']-x, item['y']-y) < MERGE:
            best = item
            break
    if best is None:
        items.append(dict(color=color, x=x, y=y, n=1, frames={(drone_id, frame_no)}, announced=False))
        best = items[-1]
    elif (drone_id, frame_no) not in best['frames']:
        best['frames'].add((drone_id, frame_no))
        best['n'] += 1
        k = best['n']
        best['x'] += (x - best['x']) / k
        best['y'] += (y - best['y']) / k
    return best

def send_item_to_server(drone_id, color, x, y, frame, u, v):
    """Отправить найденный объект со снимком на сервер."""
    try:
        h, w = frame.shape[:2]
        # Вырезаем область вокруг объекта
        pad = 30
        x1, y1 = max(0, int(u) - pad), max(0, int(v) - pad)
        x2, y2 = min(w, int(u) + pad), min(h, int(v) + pad)
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return

        # Конвертируем RGB в BGR для сохранения
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        item = {
            'color': NAMES[color],
            'x': float(x),
            'y': float(y),
            'image_base64': img_base64
        }
        post_json(f'{SERVER_URL}/api/drone/{drone_id}/report_item', {'item': item})
    except Exception as e:
        print(f"  [WARN] Ошибка отправки снимка: {e}")

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ('drone_1', 'drone_2'):
        print("Использование: python3 main.py <drone_1|drone_2>")
        return

    drone_id = sys.argv[1]
    print(f"[{drone_id}] Инициализация интерфейса...")
    drone = sverk_interfaces.init(Nodename=drone_id)

    items = []  # Найденные объекты
    last_frames = {}
    takeoff_attempted = False

    try:
        # 1. РЕГИСТРАЦИЯ В СЕТИ
        print(f"[{drone_id}] Статус: Регистрация в сети...")
        post_json(f'{SERVER_URL}/api/drone/{drone_id}/online', {})
        post_json(f'{SERVER_URL}/api/drone/{drone_id}/telemetry', {'status': 'В сети (ожидание команды взлёта)'})

        # 2. ОЖИДАНИЕ КОМАНДЫ ВЗЛЁТА
        print(f"[{drone_id}] Статус: Ожидание команды взлёта от оператора...")
        while True:
            state = get_json(f'{SERVER_URL}/api/state')
            if state and state.get('mission_status') == 'takeoff':
                break
            time.sleep(0.5)

        # 3. ВЗЛЁТ
        print(f"[{drone_id}] ✅ Команда взлёта получена! Взлёт...")
        post_json(f'{SERVER_URL}/api/drone/{drone_id}/telemetry', {'status': 'Взлёт'})
        takeoff_attempted = True
        drone.control.navigate_wait(x=0, y=0, z=HEIGHT, speed=0.6, frame_id='body', auto_arm=True, timeout=25)
        time.sleep(2.0)  # Стабилизация

        # 4. ПОИСК СТАРТОВОЙ МЕТКИ (с обходом QoS-проблемы sverk_interfaces)
        print(f"[{drone_id}] Статус: Поиск стартовой метки...")
        post_json(f'{SERVER_URL}/api/drone/{drone_id}/telemetry', {'status': 'Поиск стартовой метки'})

        import rclpy
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

        start_id = None
        last_msg = None

        # Создаём правильный QoS-профиль (BEST_EFFORT), как у издателя ArUco
        marker_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Отдельный узел с собственным executor; не используем приватные поля
        # sverk_interfaces и не вращаем узел, который библиотека уже обслуживает.
        node = rclpy.create_node('marker_start_' + drone_id)

        def marker_callback(msg):
            nonlocal last_msg
            last_msg = msg

        # Создаём нативную подписку в обход проблемной обёртки
        sub = node.create_subscription(MarkerArray, '/aruco/det/markers', marker_callback, marker_qos)

        print(f"[{drone_id}] Ожидание данных с камеры (до 5 сек)...")
        start_time = time.time()
        while time.time() - start_time < 5.0:
            rclpy.spin_once(node, timeout_sec=0.1)
            if last_msg is not None and last_msg.markers:
                telemetry = drone.control.get_telemetry(frame_id='aruco_map')
                visible = [m.id for m in last_msg.markers if m.id in VALID_MARKER_IDS]
                if (visible and telemetry.connected and
                        math.isfinite(telemetry.x) and math.isfinite(telemetry.y)):
                    start_id = min(visible, key=lambda mid: math.hypot(
                        telemetry.x - ((mid - 25) % 5 - 2),
                        telemetry.y - (2 - (mid - 25) // 5)))
            if start_id is not None:
                break
            time.sleep(0.1)

        # Очищаем подписку
        node.destroy_subscription(sub)
        node.destroy_node()

        if start_id is None:
            raise RuntimeError('Стартовая метка или положение не определены; маршрут не отправлен')

        print(f"[{drone_id}] ✅ Найдена стартовая метка: {start_id}")
        post_json(f'{SERVER_URL}/api/drone/{drone_id}/report_start', {'marker_id': start_id})
        # 5. ОЖИДАНИЕ МАРШРУТА
        print(f"[{drone_id}] Статус: Ожидание маршрута от сервера...")
        post_json(f'{SERVER_URL}/api/drone/{drone_id}/telemetry', {'status': 'Стартовая метка найдена, ожидание маршрута'})
        route = []
        while not route:
            resp = get_json(f'{SERVER_URL}/api/drone/{drone_id}/get_route')
            if resp:
                route = resp.get('route', [])
            if not route:
                time.sleep(0.5)
        print(f"[{drone_id}] Маршрут получен: {len(route)} точек")

        # 6. ПОДТВЕРЖДЕНИЕ МАРШРУТА
        print(f"[{drone_id}] Статус: Подтверждение получения маршрута...")
        post_json(f'{SERVER_URL}/api/drone/{drone_id}/accept_route', {})
        post_json(f'{SERVER_URL}/api/drone/{drone_id}/telemetry', {'status': 'Маршрут принят, ожидание команды'})

        # 7. ОЖИДАНИЕ КОМАНДЫ СТАРТ
        print(f"[{drone_id}] Статус: Ожидание команды 'Отправить путь' от оператора...")
        while True:
            state = get_json(f'{SERVER_URL}/api/state')
            if state and state.get('mission_status') == 'started':
                break
            time.sleep(0.5)

                # 8. ВЫПОЛНЕНИЕ МИССИИ
        print(f"[{drone_id}] ✅ КОМАНДА ПОЛУЧЕНА! Начало выполнения миссии...")
        post_json(f'{SERVER_URL}/api/drone/{drone_id}/telemetry', {'status': 'Выполнение миссии'})

        frame_no = 0
        for i, wp in enumerate(route):
            print(f"[{drone_id}] Точка {i+1}/{len(route)}: marker={wp.get('marker_id', '?')}, x={wp['x']:.2f}, y={wp['y']:.2f}")
            post_json(f'{SERVER_URL}/api/drone/{drone_id}/telemetry', {'status': f'Облёт: точка {i+1}/{len(route)}'})

            # Навигация к точке
            drone.control.navigate_wait(
                x=wp['x'], y=wp['y'], z=wp['z'],
                speed=SPEED, frame_id='aruco_map',
                tolerance=0.3, timeout=15.0
            )
            time.sleep(1.0)  # Стабилизация над точкой

            # === ИСПРАВЛЕНИЕ ЗДЕСЬ ===
            # Получаем кадр (возвращает BGR)
            frame_bgr = drone.image.take_picture(timeout=1.0)

            if isinstance(frame_bgr, np.ndarray):
                # Конвертируем BGR в RGB, чтобы функция detect() работала корректно
                frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_no += 1
                signature = hash(frame.tobytes())

                if last_frames.get(drone_id) != signature:
                    last_frames[drone_id] = signature
                    
                    # Получаем телеметрию для расчёта координат
                    p = drone.control.get_telemetry(frame_id='aruco_map')
                    h, w = frame.shape[:2]
                    focal = w / (2 * math.tan(math.radians(FOV) / 2))
                    
                    # Детекция цветов
                    for color, u, v in detect(frame):
                        # Пересчёт пиксельных координат в мировые
                        bx = -(v - h/2) * p.z / focal
                        by = -(u - w/2) * p.z / focal
                        x = p.x + bx
                        y = p.y + by
                        
                        # Проверяем, что объект в пределах поля
                        if -2.15 <= x <= 2.15 and -2.15 <= y <= 2.15:
                            best = remember(items, color, x, y, drone_id, frame_no)
                            if best['n'] >= 3 and not best['announced']:
                                print(f"[{drone_id}] {NAMES[color]}: x={best['x']:.2f}, y={best['y']:.2f}")
                                best['announced'] = True
                                send_item_to_server(drone_id, color, best['x'], best['y'], frame, u, v)
            else:
                print(f"[{drone_id}] ⚠️ Кадр не получен")

        # 9. ПОСАДКА
        print(f"[{drone_id}] Статус: Посадка...")
        post_json(f'{SERVER_URL}/api/drone/{drone_id}/telemetry', {'status': 'Посадка'})
        drone.control.land(timeout=15.0)
        time.sleep(2.0)

        # 10. ОТПРАВКА ИТОГОВ
        final_counts = {'Красный': 0, 'Жёлтый': 0, 'Синий': 0}
        for item in items:
            if item['n'] >= 3:
                final_counts[NAMES[item['color']]] += 1
        
        print(f"[{drone_id}] Отправка итогов: {final_counts}")
        post_json(f'{SERVER_URL}/api/drone/{drone_id}/report_finish', {'results': final_counts})
        print(f"[{drone_id}] ✅ Миссия успешно завершена!")

    except KeyboardInterrupt:
        print(f"\n[{drone_id}] Прервано оператором.")
        post_json(f'{SERVER_URL}/api/drone/{drone_id}/telemetry', {'status': 'Прервано оператором'})
    except Exception as e:
        print(f"[{drone_id}] КРИТИЧЕСКАЯ ОШИБКА: {e}")
    finally:
        try:
            if takeoff_attempted:
                drone.control.land()
        except:
            pass
        drone.close()

if __name__ == '__main__':
    main()
