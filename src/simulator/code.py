import time, math
import numpy as np
import sverk_interfaces
from aruco_det_loc.msg import MarkerArray

# Camera points down; X is forward and Y is left in the map at yaw=0.
HEIGHT = 1.2
SPEED = 0.8
FOV = 65.0
MERGE = 0.23
NAMES = ('Красный', 'Жёлтый', 'Зелёный')
last_frames = {}
calibration_errors = []

def components(mask):
    # Connected components on a half-resolution mask.
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
    a = frame.astype(np.int16)
    r,g,b = a[:,:,0],a[:,:,1],a[:,:,2]
    masks = ((r>100)&(r>g*1.45)&(r>b*1.45),
             (r>120)&(g>110)&(b<r*0.65)&(b<g*0.65)&(abs(r-g)<100),
             (g>90)&(g>r*1.45)&(g>b*1.3))
    for color,mask in enumerate(masks):
        for u,v in components(mask):
            yield color,u,v

def remember(items, color, x, y, drone_id, frame_no):
    best = None
    for item in items:
        if item['color'] == color and math.hypot(item['x']-x,item['y']-y) < MERGE:
            best = item; break
    if best is None:
        items.append(dict(color=color,x=x,y=y,n=1,frames={(drone_id,frame_no)},announced=False))
    elif (drone_id,frame_no) not in best['frames']:
        best['frames'].add((drone_id,frame_no))
        best['n'] += 1
        k = best['n']
        best['x'] += (x-best['x'])/k
        best['y'] += (y-best['y'])/k
        if k >= 3 and not best['announced']:
            print('%s: %s x=%.2f y=%.2f' % (drone_id,NAMES[color],best['x'],best['y']),flush=True)
            best['announced']=True

def sample(d, items, frame_no):
    frame=d.camera.read_numpy(encoding='rgb8',timeout=0.25)
    if not isinstance(frame,np.ndarray): return
    signature=hash(frame.tobytes())
    if last_frames.get(d.drone_id)==signature: return
    last_frames[d.drone_id]=signature
    p=d.control.get_telemetry(frame_id='map')
    if p.z < 0.5: return
    h,w=frame.shape[:2]
    markers=d.topic.wait_for_message(MarkerArray,'/aruco/det/markers',timeout=0.1)
    anchors=[]
    if markers:
        for mark in markers.markers:
            if 0 <= mark.id < 25 and len(mark.corners) >= 4:
                u0=sum(c.x for c in mark.corners)/len(mark.corners)
                v0=sum(c.y for c in mark.corners)/len(mark.corners)
                anchors.append((u0,v0,-1.6+0.8*(mark.id%5),-1.6+0.8*(mark.id//5)))
    if not anchors: return
    focal=w/(2*math.tan(math.radians(FOV)/2))
    transform=None
    if len(anchors)>=2:
        mat=[]; val=[]
        for u0,v0,x0,y0 in anchors:
            mat.extend([[u0,v0,1,0],[-v0,u0,0,1]])
            val.extend([x0,y0])
        transform=np.linalg.lstsq(np.array(mat),np.array(val),rcond=None)[0]
    c,s=math.cos(p.yaw),math.sin(p.yaw)
    for u0,v0,x0,y0 in anchors:
        dx=-(v0-h/2)*p.z/focal; dy=-(u0-w/2)*p.z/focal
        calibration_errors.append(math.hypot(p.x+c*dx-s*dy-x0,p.y+s*dx+c*dy-y0))
    for color,u,v in detect(frame):
        # Optical image up is body forward, image right is body right.
        bx=-(v-h/2)*p.z/focal
        by=-(u-w/2)*p.z/focal
        if transform is not None:
            aa,bb,tx,ty=transform
            x=aa*u+bb*v+tx; y=-aa*v+bb*u+ty
        else:
            u0,v0,x0,y0=anchors[0]
            dx=-(v-v0)*p.z/focal; dy=-(u-u0)*p.z/focal
            x=x0+c*dx-s*dy; y=y0+s*dx+c*dy
        if -2.15 <= x <= 2.15 and -2.15 <= y <= 2.15:
            remember(items,color,x,y,d.drone_id,frame_no)

def main():
    ds=[]; items=[]; start=time.monotonic()
    try:
        for name in ('drone_1','drone_2'):
            ds.append(sverk_interfaces.init(drone_id=name))
        for d in ds: d.camera.configure(width=320,height=240,fov=FOV,frame_rate=10)
        for d in ds:
            d.control.navigate_wait(x=0,y=0,z=HEIGHT,speed=0.6,frame_id='body',auto_arm=True,timeout=25)
        for d in ds:
            msg=d.topic.wait_for_message(MarkerArray,'/aruco/det/markers',timeout=1)
            print(d.drone_id, 'ArUco:', [m.id for m in msg.markers] if msg else [], flush=True)
        # Assign halves by actual position, keeping a separation at y=0.
        ds.sort(key=lambda d: d.control.get_telemetry(frame_id='map').y)
        paths=[]
        for sign in (-1,1):
            paths.append([(-1.5,sign*1.45),(1.5,sign*1.45),(1.5,sign*0.50),(-1.5,sign*0.50)])
        for leg in range(4):
            for i,d in enumerate(ds):
                x,y=paths[i][leg]
                d.control.navigate(x=x,y=y,z=HEIGHT,speed=SPEED,frame_id='map')
            deadline=min(start+94,time.monotonic()+13)
            count=0
            while time.monotonic()<deadline:
                count+=1
                for d in ds: sample(d,items,leg*1000+count)
                reached=True
                for i,d in enumerate(ds):
                    p=d.control.get_telemetry(frame_id='map'); x,y=paths[i][leg]
                    if math.hypot(p.x-x,p.y-y)>0.20: reached=False
                if reached and count>=3: break
            if time.monotonic()>=start+94: break
    except Exception as e:
        print('Ошибка поиска:',repr(e),flush=True)
    finally:
        for d in ds:
            try: d.control.land()
            except Exception as e: print('Ошибка посадки:',e,flush=True)
        for color,name in enumerate(NAMES):
            found=[p for p in items if p['color']==color and p['n']>=3]
            print(name+': '+str(len(found))+' шт. '+ ' '.join('x=%.2f, y=%.2f'%(p['x'],p['y']) for p in found),flush=True)
        if calibration_errors:
            print('Проверка ArUco/телеметрии: N=%d, медиана=%.2f м, максимум=%.2f м' % (len(calibration_errors),float(np.median(calibration_errors)),max(calibration_errors)),flush=True)
        else: print('Проверка ArUco: нет измерений',flush=True)
        limit=time.monotonic()+15
        while time.monotonic()<limit:
            try:
                if all(not d.control.get_telemetry(frame_id='map').armed for d in ds): break
            except Exception: break
            time.sleep(0.2)
        for d in ds:
            try: d.close()
            except Exception: pass

if __name__=='__main__': main()