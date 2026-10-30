"""Симулятор ТГУ: два дрона, 2 + 3 ряда, результат в консоли."""
import time
STARTED = time.monotonic()
import math
import cv2
import numpy as np
import sverk_interfaces

VERSION = "2026-10-28.5-rows"
HEIGHT, SPEED, FOV = 1.6, 0.8, 65.0
SEARCH_END, END = STARTED + 94, STARTED + 118
ROWS, EDGE, MERGE = (-1.6, -0.8, 0.0, 0.8, 1.6), 2.0, 0.16
NAMES = ("Красный", "Жёлтый", "Зелёный")
aruco = (cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000))
    if hasattr(cv2, "aruco") and hasattr(cv2.aruco, "ArucoDetector") else None)


def check(result):
    ok = result.get("success", False) if isinstance(result, dict) else getattr(result, "success", False)
    if not ok:
        raise RuntimeError("API не подтвердил команду: " + str(result))


def routes(positions):
    """Два нижних ряда и три верхних; первые точки ближе к своим стартам."""
    order = sorted(range(2), key=lambda i: positions[i].y)
    if not positions[order[0]].y < -0.52 or not positions[order[1]].y > -0.28:
        raise RuntimeError("Для разделения 2+3 старты должны находиться по разные стороны Y=-0.4")
    paths = [None, None]
    for index, rows in zip(order, (ROWS[:2], ROWS[2:])):
        rows = sorted(rows, reverse=abs(positions[index].y-rows[-1]) < abs(positions[index].y-rows[0]))
        ends = [-EDGE, EDGE] if positions[index].x < 0 else [EDGE, -EDGE]
        paths[index] = [(x, y) for row, y in enumerate(rows) for x in (ends if row % 2 == 0 else ends[::-1])]
    return paths


def detect(frame):
    r, g, b = frame.astype(np.int16).transpose(2, 0, 1)
    masks = ((r > 100) & (r > g*1.45) & (r > b*1.45),
             (r > 120) & (g > 110) & (b < r*.65) & (b < g*.65) & (abs(r-g) < 100),
             (g > 90) & (g > r*1.45) & (g > b*1.3))
    h, w = frame.shape[:2]
    for color, mask in enumerate(masks):
        count, _, stats, centers = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)
        for i in range(1, count):
            x, y, bw, bh, area = stats[i]
            if (100 <= area <= 12000 and min(bw, bh)/max(bw, bh) > .6
                    and area/(bw*bh) > .65 and x > 1 and y > 1 and x+bw < w-1 and y+bh < h-1):
                yield color, centers[i]


def projection(frame, p, k):
    """Пиксели -> map; при видимых ArUco уточняем преобразование по кадру."""
    c, s, z = math.cos(p.yaw), math.sin(p.yaw), p.z
    body = np.array([[0, -z/k[1, 1], .0655+z*k[1, 2]/k[1, 1]],
                     [-z/k[0, 0], 0, z*k[0, 2]/k[0, 0]]])
    result = np.array([[c, -s], [s, c]]) @ body
    result[:, 2] += (p.x, p.y)
    if aruco is None:
        return result
    corners, ids, _ = aruco.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
    anchors = [] if ids is None else [(np.asarray(q).reshape(4, 2).mean(0),
        [-1.6+.8*(int(i) % 5), -1.6+.8*(int(i) // 5)])
        for q, i in zip(corners, ids.ravel()) if 0 <= int(i) < 25]
    if len(anchors) == 1:
        uv, xy = anchors[0]
        result[:, 2] += np.array(xy)-result @ np.r_[uv, 1.]
    elif len(anchors) >= 2:
        mat, values = [], []
        for (u, v), (x, y) in anchors:
            mat.extend([[u, v, 1, 0], [-v, u, 0, 1]])
            values.extend([x, y])
        coef, _, rank, _ = np.linalg.lstsq(mat, values, rcond=None)
        if rank == 4 and np.max(np.abs(np.asarray(mat) @ coef-values)) < .08:
            a, b, tx, ty = coef
            result = np.array([[a, b, tx], [b, -a, ty]])
    return result


def sample(d, k, items):
    frame = d.camera.read_numpy(encoding="rgb8", timeout=.25)
    if not isinstance(frame, np.ndarray):
        return  # None / JsNull вместо кадра
    p = d.control.get_telemetry(frame_id="map")
    if not all(math.isfinite(v) for v in (p.x, p.y, p.z, p.yaw)) or p.z < .5:
        return
    matrix, used = projection(frame, p, k), set()
    for color, uv in detect(frame):
        x, y = matrix @ np.r_[uv, 1.]
        if max(abs(x), abs(y)) > 2.4:
            continue
        candidates = [(math.hypot(t["x"]-x, t["y"]-y), i) for i, t in enumerate(items)
                      if t["color"] == color and i not in used]
        distance, i = min(candidates, default=(math.inf, -1))
        if distance >= MERGE:
            i = len(items)
            items.append(dict(color=color, x=float(x), y=float(y), n=0))
        used.add(i)
        t = items[i]
        t["n"] += 1
        t["x"] += (x-t["x"])/t["n"]
        t["y"] += (y-t["y"])/t["n"]
        if t["n"] == 3:
            print(f"{d.drone_id}: {NAMES[color]} x={t['x']:.2f}, y={t['y']:.2f}", flush=True)
    return True


def main():
    print("TGU_SOLUTION", VERSION, flush=True)
    drones, items, complete = [], [], False
    try:
        for name in ("drone_1", "drone_2"):
            drones.append(sverk_interfaces.init(drone_id=name))
        matrices = []
        for d in drones:
            check(d.camera.configure(width=320, height=240, fov=FOV, frame_rate=10))
            k = np.asarray(d.camera.get_camera_info(timeout=1.).K, dtype=float).reshape(3, 3)
            if not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
                raise RuntimeError("Нет параметров камеры")
            matrices.append(k)
        for d in drones:
            if SEARCH_END-time.monotonic() < 1:
                raise TimeoutError("Исчерпан бюджет взлёта")
            check(d.control.navigate_wait(x=0, y=0, z=HEIGHT, speed=.6, frame_id="body",
                auto_arm=True, timeout=min(25., SEARCH_END-time.monotonic())))
        paths = routes([d.control.get_telemetry(frame_id="map") for d in drones])
        indices, sent, arrived, next_frame = [0, 0], [None, None], [None, None], [0., 0.]
        valid, last_good = [0, 0], [time.monotonic()]*2
        for d, path in zip(drones, paths):
            print(d.drone_id, "ряды:", list(dict.fromkeys(y for _, y in path)), flush=True)
        while time.monotonic() < SEARCH_END:
            for i, d in enumerate(drones):
                if indices[i] == len(paths[i]):
                    continue
                now = time.monotonic()
                x, y = paths[i][indices[i]]
                if sent[i] is None:
                    check(d.control.navigate(x=x, y=y, z=HEIGHT, yaw=0., speed=SPEED, frame_id="map"))
                    sent[i], arrived[i], valid[i] = now, None, 0
                if now >= next_frame[i]:
                    if sample(d, matrices[i], items):
                        valid[i], last_good[i] = valid[i]+1, time.monotonic()
                    next_frame[i] = time.monotonic()+.15
                if time.monotonic()-last_good[i] > 3:
                    raise TimeoutError(d.drone_id+": камера не возвращает пригодные кадры")
                p = d.control.get_telemetry(frame_id="map")
                if not p.armed:
                    raise RuntimeError(d.drone_id+": моторы выключены во время поиска")
                reached = math.dist((p.x, p.y, p.z), (x, y, HEIGHT)) < .20
                if reached:
                    arrived[i] = arrived[i] if arrived[i] is not None else time.monotonic()
                    if time.monotonic()-arrived[i] >= .5 and valid[i] >= 3:
                        indices[i] += 1
                        sent[i] = None
                else:
                    arrived[i] = None
                if sent[i] is not None and time.monotonic()-sent[i] > 18:
                    raise TimeoutError(d.drone_id+": точка маршрута не достигнута")
            if all(i == len(p) for i, p in zip(indices, paths)):
                complete = True
                break
            time.sleep(.04)
    except Exception as e:
        print("Ошибка поиска:", repr(e), flush=True)
    finally:
        for d in drones:
            try:
                check(d.control.land())
            except Exception as e:
                print(d.drone_id, "ошибка посадки:", e, flush=True)
        pending = list(drones)
        while pending and time.monotonic() < END:
            for d in pending[:]:
                try:
                    if not d.control.get_telemetry(frame_id="map").armed:
                        pending.remove(d)
                except Exception:
                    pass
            if pending:
                time.sleep(.15)
        for color, name in enumerate(NAMES):
            found = [t for t in items if t["color"] == color and t["n"] >= 3]
            print(f"{name}: {len(found)} шт.", flush=True)
            for t in found:
                print(f"  x={t['x']:.2f}, y={t['y']:.2f}", flush=True)
        print("STATUS", "LANDING_UNCONFIRMED" if pending else "COMPLETE" if complete else "PARTIAL", flush=True)
        for d in drones:
            try:
                d.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()