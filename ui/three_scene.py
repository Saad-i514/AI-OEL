"""Interactive, procedural Three.js city and courier scene for the planner tab."""
import json

import streamlit.components.v1 as components


def show_delivery_scene(world, route, height=690, opponent_route=None, game_result=None, autoplay=False):
    """Render a 3D city with a planned courier route or a live two-agent match replay."""
    game_mode = opponent_route is not None
    grid = world.grid.tolist()
    size = len(grid)
    if not size or any(len(row) != size for row in grid):
        raise ValueError("The 3D scene requires a non-empty square terrain grid.")

    def pack_cells(cells, name):
        packed = []
        for index, cell in enumerate(cells):
            if len(cell) != 2:
                raise ValueError(f"Invalid {name} grid position at step {index}: expected (row, column).")
            row, col = int(cell[0]), int(cell[1])
            if not (0 <= row < size and 0 <= col < size):
                raise ValueError(f"Invalid {name} grid position at step {index}: ({row}, {col}) is outside the map.")
            packed.append([row, col])
        return packed

    route_cells = pack_cells(route, "courier route")
    opponent_cells = pack_cells(opponent_route, "rival route") if game_mode else []
    payload = {
        "grid": grid,
        "size": size,
        "terrainNames": list(world.inst["terrain_cost"]),
        "terrainCosts": world.inst["terrain_cost"],
        "start": pack_cells([world.start], "depot")[0],
        "stops": pack_cells(world.stops, "delivery stops"),
        "stations": pack_cells(world.stations, "charging stations"),
        "route": route_cells,
        "opponentRoute": opponent_cells,
        "gameMode": game_mode,
        "autoPlay": bool(autoplay),
        "gameResult": game_result or {},
    }
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html = r'''<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
*{box-sizing:border-box}html,body{margin:0;width:100%;height:100%;overflow:hidden;background:#eef3f8;color:#17263c;
font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}#stage{position:absolute;inset:0}
canvas{display:block;width:100%;height:100%}.glass{position:absolute;z-index:2;background:rgba(255,255,255,.9);
border:1px solid rgba(222,231,240,.94);box-shadow:0 10px 32px rgba(17,34,55,.12);backdrop-filter:blur(12px)}
#top{left:16px;right:16px;top:14px;display:flex;align-items:center;justify-content:space-between;padding:12px 15px;border-radius:14px}
.brand{display:flex;align-items:center;gap:10px}.brand-icon{width:34px;height:34px;border-radius:10px;background:#e8efff;
display:grid;place-items:center;color:#356df3;font-weight:800}.title{font-size:13px;font-weight:750;letter-spacing:-.1px}.subtitle{font-size:10px;color:#7d8a9c;margin-top:2px}
.live{display:flex;align-items:center;gap:7px;color:#18866d;background:#e9f7f1;border-radius:20px;padding:7px 10px;font-size:10px;font-weight:750;letter-spacing:.04em}
.pulse{width:7px;height:7px;background:#20b486;border-radius:50%;box-shadow:0 0 0 4px #20b48622}
#minimap{left:16px;top:82px;width:180px;padding:10px;border-radius:12px}
#minimap canvas{width:100%;height:auto;display:block;border-radius:7px;background:#d98938}
#legend{left:16px;top:288px;padding:12px 14px;border-radius:12px;display:grid;gap:8px;min-width:168px}
.legend-title{font-size:10px;letter-spacing:.12em;text-transform:uppercase;color:#8190a3;font-weight:750;margin-bottom:2px}
.legend-item{display:flex;align-items:center;gap:8px;color:#53647a;font-size:12px}.swatch{width:12px;height:12px;border-radius:3px;flex:none}
#controls{left:16px;right:16px;bottom:14px;display:flex;align-items:center;gap:10px;padding:11px 13px;border-radius:14px}
button{border:0;border-radius:9px;padding:9px 12px;background:#356df3;color:#fff;font-size:11px;font-weight:750;cursor:pointer}
button{transition:transform .16s ease,box-shadow .16s ease,background .16s ease;box-shadow:0 3px 8px rgba(28,55,92,.14)}
button:hover{background:#285bd4;transform:translateY(-2px) scale(1.035);box-shadow:0 7px 16px rgba(39,83,170,.28)}
button:active{transform:scale(.97)}button.secondary{background:#edf2f8;color:#45566d}button.secondary:hover{background:#e2e9f2;color:#233b5b}
#progress-wrap{height:5px;flex:1;background:#e7edf4;border-radius:6px;overflow:hidden;min-width:70px}
#progress{height:100%;width:0;background:linear-gradient(90deg,#356df3,#19b6a5);transition:width .12s linear}
#step{color:#53647a;font-size:10px;font-variant-numeric:tabular-nums;white-space:nowrap;min-width:72px;text-align:right}
#speed{width:78px;accent-color:#356df3}#speed-label{font-size:9px;color:#738198;white-space:nowrap}
#position{right:16px;top:82px;padding:12px 14px;border-radius:12px;min-width:145px;text-align:right}
#position-label{font-size:9px;color:#8290a2;letter-spacing:.1em;text-transform:uppercase;font-weight:750}
#coords{font-size:16px;font-weight:750;color:#1a2a42;margin-top:4px;font-variant-numeric:tabular-nums}
#rival-coords{display:none;margin-top:6px;padding-top:6px;border-top:1px solid #e4eaf0;color:#a23454;font-size:12px;font-weight:750;font-variant-numeric:tabular-nums}
#finish{display:none;left:50%;top:50%;transform:translate(-50%,-50%);width:min(420px,calc(100% - 32px));padding:20px 22px;border-radius:16px;text-align:center}
#finish-title{color:#17392f;font-size:22px;font-weight:850;letter-spacing:-.3px}
#finish-detail{color:#52647a;font-size:13px;margin-top:6px}
#finish-stats{display:flex;justify-content:center;gap:18px;flex-wrap:wrap;margin-top:12px;color:#233c58;font-size:12px;font-weight:700}
#finish-stats span{padding:7px 9px;border-radius:8px;background:#f0f4f8}
#settings-toggle{background:#edf2f8;color:#45566d;padding:8px 10px;min-height:34px;font-size:10px}
#settings-toggle:hover{background:#e2e9f2}
#settings-panel{display:none;position:absolute;z-index:5;right:16px;top:72px;width:230px;padding:14px;border-radius:13px;gap:12px}
#settings-panel.open{display:grid}.setting-row{display:flex;align-items:center;justify-content:space-between;gap:10px;color:#53647a;font-size:11px;font-weight:600}
.setting-row input[type=checkbox]{accent-color:#356df3;width:15px;height:15px}.setting-size{display:grid;gap:5px;color:#53647a;font-size:10px;font-weight:700}
.setting-size input{width:100%;accent-color:#356df3}.setting-size span{color:#8290a2;font-weight:500}
#hint{position:absolute;right:17px;bottom:70px;color:#fff;background:rgba(24,40,62,.62);padding:7px 10px;border-radius:8px;font-size:9px;pointer-events:none}
#error{display:none;position:absolute;inset:0;z-index:4;background:#f6f8fb;align-items:center;justify-content:center;text-align:center;padding:28px;color:#53647a}
@media(max-width:520px){#controls{flex-wrap:wrap;gap:6px;padding:8px}#controls button{padding:8px 9px;font-size:9px}
#progress-wrap{order:4;flex-basis:42%}#step{order:5;font-size:9px;min-width:60px}#speed-label{margin-left:auto}
#speed{width:56px}#position{right:10px;top:86px;min-width:120px;padding:9px 10px}#minimap{left:10px;top:86px;width:132px;padding:7px}#legend{left:10px;top:232px;min-width:148px;padding:10px 11px}
#settings-panel{top:65px;right:10px;width:205px;padding:11px}.legend-item{font-size:11px}.legend-title{font-size:9px}
#top{left:10px;right:10px;top:10px;padding:9px 10px}.subtitle{font-size:9px}.live{font-size:9px;padding:6px 8px}}
</style></head><body>
<div id="stage"></div>
<div id="error"><div><strong>3D scene could not start</strong><p id="error-text">Check that WebGL is enabled and the Three.js CDN is reachable.</p></div></div>
<div id="top" class="glass"><div class="brand"><div class="brand-icon">3D</div><div><div class="title">Live route simulation</div><div class="subtitle">Terrain-aware delivery · algorithm-generated path</div></div></div><div style="display:flex;align-items:center;gap:8px"><button id="settings-toggle" aria-expanded="false">⚙ Display</button><div class="live"><span class="pulse"></span><span id="run-state">READY</span></div></div></div>
<div id="settings-panel" class="glass" aria-label="Scene display settings">
<div class="legend-title">Visible overlays</div>
<label class="setting-row">District minimap<input id="show-minimap" type="checkbox" checked></label>
<label class="setting-row">Terrain guide<input id="show-legend" type="checkbox" checked></label>
<label class="setting-row">Stop & depot tags<input id="show-tags" type="checkbox" checked></label>
<label class="setting-size">Courier & marker size <span id="size-value">125%</span><input id="visual-size" type="range" min="1" max="1.75" step="0.25" value="1.25"></label>
</div>
<div id="minimap" class="glass"><div class="legend-title">District map · north ↑</div><canvas id="mini-map" width="176" height="176"></canvas></div>
<div id="legend" class="glass"><div class="legend-title">Terrain guide</div>
<div class="legend-item"><span class="swatch" style="background:#d98b3a"></span>Terracotta road</div>
<div class="legend-item"><span class="swatch" style="background:#648d49"></span>Garden</div>
<div class="legend-item"><span class="swatch" style="background:#94684d"></span>Earth / mud</div>
<div class="legend-item"><span class="swatch" style="background:#52b9c8"></span>Water crossing</div>
<div class="legend-item"><span class="swatch" style="background:#777a7d"></span>Stone structures</div>
<div class="legend-item"><span style="color:#356df3">●</span>Delivery courier</div>
<div class="legend-item game-only" style="display:none"><span style="color:#d94767">✕</span>Rival interceptor</div>
</div>
<div id="position" class="glass"><div id="position-label">Courier location</div><div id="coords">Preparing…</div><div id="rival-coords"></div></div>
<div id="finish" class="glass" role="status" aria-live="polite"><div id="finish-title"></div><div id="finish-detail"></div><div id="finish-stats"></div></div>
<div id="controls" class="glass"><button id="play">▶&nbsp; Start route</button><button id="reset" class="secondary">↺&nbsp; Reset</button><button id="follow" class="secondary">◎&nbsp; Follow courier</button>
<div id="progress-wrap"><div id="progress"></div></div><div id="step">Move 0 / 0</div><span id="speed-label">PACE</span><input id="speed" aria-label="Playback speed" type="range" min="0.5" max="5" step="0.25" value="2.5">
</div><div id="hint">Drag to orbit · scroll to zoom</div>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.180.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.180.0/examples/jsm/"}}</script>
<script>
const DATA = __DATA__;
</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

if(DATA.gameMode){
 document.querySelector('.title').textContent='Live game replay';
 document.querySelector('.subtitle').textContent='Autonomous courier vs rival interceptor · replay starts automatically';
 document.querySelector('#position-label').textContent='Live match positions';
 document.querySelector('#rival-coords').style.display='block';
 document.querySelectorAll('.game-only').forEach(e=>e.style.display='flex');
}
const stage=document.querySelector('#stage'), scene=new THREE.Scene();
scene.background=new THREE.Color('#e9c28c'); scene.fog=new THREE.Fog('#e9c28c',86,150);
const renderer=new THREE.WebGLRenderer({antialias:true,alpha:false});
renderer.setPixelRatio(Math.min(devicePixelRatio,2)); renderer.shadowMap.enabled=true;
renderer.shadowMap.type=THREE.PCFSoftShadowMap; renderer.outputColorSpace=THREE.SRGBColorSpace;
renderer.toneMapping=THREE.ACESFilmicToneMapping; renderer.toneMappingExposure=1.15; stage.appendChild(renderer.domElement);
const camera=new THREE.PerspectiveCamera(48,1,.1,280); const N=DATA.size,S=2.0,half=(N-1)*S/2;
camera.position.set(half+15,45,half+18);
const controls=new OrbitControls(camera,renderer.domElement); controls.target.set(half,0,half); controls.enableDamping=true;
controls.dampingFactor=.07; controls.minDistance=12; controls.maxDistance=140; controls.minPolarAngle=.12; controls.maxPolarAngle=1.42;
scene.add(new THREE.HemisphereLight(0xfff5df,0x795638,2.25));
const sun=new THREE.DirectionalLight(0xffedcc,3.1); sun.position.set(-24,58,28); sun.castShadow=true;
sun.shadow.mapSize.set(2048,2048); sun.shadow.camera.left=-28; sun.shadow.camera.right=28; sun.shadow.camera.top=28; sun.shadow.camera.bottom=-28; scene.add(sun);
const fill=new THREE.DirectionalLight(0xffd599,1.0);fill.position.set(30,30,-20);scene.add(fill);
function seeded(r,c){return ((r*92821+c*68917+71)%997)/997}
function tileTexture(base,kind){const canvas=document.createElement('canvas');canvas.width=canvas.height=192;const ctx=canvas.getContext('2d');ctx.fillStyle=base;ctx.fillRect(0,0,192,192);
 let seed=kind*131+29;const rand=()=>{seed=(seed*16807)%2147483647;return(seed-1)/2147483646};
 for(let i=0;i<720;i++){const x=rand()*192,y=rand()*192,r=rand()*2.1+.25;ctx.globalAlpha=.05+rand()*.18;ctx.fillStyle=rand()>.5?'#fff4d2':'#54351f';ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);ctx.fill()}
 ctx.globalAlpha=.2;ctx.strokeStyle=kind===0?'#f4c270':'#e8dcc4';ctx.lineWidth=2;ctx.strokeRect(2,2,188,188);ctx.globalAlpha=1;
 const t=new THREE.CanvasTexture(canvas);t.colorSpace=THREE.SRGBColorSpace;t.anisotropy=renderer.capabilities.getMaxAnisotropy();return t}
const mats={road:new THREE.MeshStandardMaterial({map:tileTexture('#d88b3c',0),roughness:1}),grass:new THREE.MeshStandardMaterial({map:tileTexture('#648b4c',1),roughness:1}),
mud:new THREE.MeshStandardMaterial({map:tileTexture('#95694e',2),roughness:1}),ford:new THREE.MeshStandardMaterial({map:tileTexture('#48adbf',3),roughness:.36,metalness:.02}),
block:new THREE.MeshStandardMaterial({map:tileTexture('#888988',4),roughness:1}),line:new THREE.MeshStandardMaterial({color:'#f7eacb',roughness:.8}),
building:new THREE.MeshStandardMaterial({map:tileTexture('#a0a0a0',5),roughness:.95}),building2:new THREE.MeshStandardMaterial({map:tileTexture('#8b8c8d',6),roughness:.95}),
window:new THREE.MeshStandardMaterial({color:'#f6c875',emissive:'#d68135',emissiveIntensity:.35}),
route:new THREE.MeshStandardMaterial({color:'#2771ff',emissive:'#1854df',emissiveIntensity:.45,roughness:.32}),
skin:new THREE.MeshStandardMaterial({color:'#c98c67',roughness:.83}),jacket:new THREE.MeshStandardMaterial({color:'#1979b9',roughness:.75}),
dark:new THREE.MeshStandardMaterial({color:'#26364b',roughness:.86}),pants:new THREE.MeshStandardMaterial({color:'#243b5c',roughness:.85}),
pack:new THREE.MeshStandardMaterial({color:'#e77536',roughness:.78}),white:new THREE.MeshStandardMaterial({color:'#f5f7fa',roughness:.72}),
gold:new THREE.MeshStandardMaterial({color:'#ffca63',emissive:'#7a4d12',emissiveIntensity:.18})};
const matCell=[mats.road,mats.grass,mats.mud,mats.ford];
let visualScale=1.25;
const scaleTargets=[],tagTargets=[],routeScaleTargets=[];
const ground=new THREE.Mesh(new THREE.PlaneGeometry(N*S+5,N*S+5),new THREE.MeshStandardMaterial({map:tileTexture('#c98748',9),roughness:1}));
ground.rotation.x=-Math.PI/2;ground.position.set(half,-.16,half);ground.receiveShadow=true;scene.add(ground);
const gridGroup=new THREE.Group();scene.add(gridGroup);
const boxGeo=new THREE.BoxGeometry(S-.055,.12,S-.055), buildingGeo=new THREE.BoxGeometry(1,1,1);
const routeCells=new Set(DATA.route.map(p=>p.join(',')));
const pointCells=new Set([DATA.start,...DATA.stops,...DATA.stations].map(p=>p.join(',')));
const heights=Array.from({length:N},()=>Array(N).fill(.02));
for(let r=0;r<N;r++)for(let c=0;c<N;c++){
 const kind=DATA.grid[r][c],x=c*S,z=r*S,blocked=kind<0;
 const tile=new THREE.Mesh(boxGeo,blocked?mats.block:matCell[kind]);tile.position.set(x,0,z);tile.receiveShadow=true;tile.castShadow=blocked;gridGroup.add(tile);
 if(blocked){
   const h=.48+seeded(r,c)*.38,w=1.72+seeded(c,r)*.12,d=1.72+seeded(r+c,c)*.12;
   const b=new THREE.Mesh(buildingGeo,((r+c)%3===0)?mats.building2:mats.building);
   b.scale.set(w,h,d);b.position.set(x,.08+h/2,z);b.castShadow=true;b.receiveShadow=true;gridGroup.add(b);heights[r][c]=.08+h;
   const roof=new THREE.Mesh(new THREE.BoxGeometry(w-.18,.035,d-.18),mats.building2);roof.position.set(x,.08+h+.015,z);roof.receiveShadow=true;gridGroup.add(roof);
   const seam=new THREE.Mesh(new THREE.BoxGeometry(w-.38,.012,.025),mats.line);seam.position.set(x,.11+h,z-.43);gridGroup.add(seam);
 } else {
   heights[r][c]=.08;
   if(kind===0){
     // occasional low timber benches decorate open plaza cells away from the route.
     if(seeded(r,c)>.91&&!routeCells.has(`${r},${c}`)&&!pointCells.has(`${r},${c}`)){
       const bench=new THREE.Group();bench.position.set(x,.12,z);bench.rotation.y=seeded(c,r)*Math.PI;
       for(let slat=0;slat<3;slat++){const seat=new THREE.Mesh(new THREE.BoxGeometry(1.08,.09,.13),new THREE.MeshStandardMaterial({color:slat===1?'#a16d3d':'#8e5c32',roughness:.9}));seat.position.set(0,.37,(slat-1)*.15);seat.castShadow=true;bench.add(seat)}
       for(const dx of [-.4,.4]){const leg=new THREE.Mesh(new THREE.BoxGeometry(.09,.36,.1),mats.dark);leg.position.set(dx,.18,0);bench.add(leg)}gridGroup.add(bench)
     }
   } else if(kind===1){
     if(seeded(r,c)>.68&&!routeCells.has(`${r},${c}`)&&!pointCells.has(`${r},${c}`)){
       const bed=new THREE.Mesh(new THREE.BoxGeometry(1.48,.18,1.48),new THREE.MeshStandardMaterial({color:'#70604b',roughness:1}));bed.position.set(x,.15,z);bed.castShadow=true;gridGroup.add(bed);
       const earth=new THREE.Mesh(new THREE.BoxGeometry(1.28,.035,1.28),new THREE.MeshStandardMaterial({color:'#513e2c',roughness:1}));earth.position.set(x,.25,z);gridGroup.add(earth);
       for(let leaf=0;leaf<5;leaf++){const dx=(seeded(r,leaf)-.5)*.72,dz=(seeded(c,leaf+4)-.5)*.72;const bush=new THREE.Mesh(new THREE.SphereGeometry(.23+seeded(leaf,c)*.08,12,9),new THREE.MeshStandardMaterial({color:['#4f853c','#679d45','#81ae51'][leaf%3],roughness:.94}));bush.position.set(x+dx,.45+seeded(c,leaf)*.12,z+dz);bush.castShadow=true;gridGroup.add(bush)}
     } else if(seeded(r,c)>.38){
       for(let g=0;g<3;g++){const blade=new THREE.Mesh(new THREE.ConeGeometry(.075,.28,5),new THREE.MeshStandardMaterial({color:g%2?'#4d8140':'#79a84d',roughness:1}));blade.position.set(x+(seeded(r,g)-.5)*1.25,.22,z+(seeded(c,g+3)-.5)*1.25);blade.castShadow=true;gridGroup.add(blade)}
     }
   } else if(kind===2 && seeded(r,c)>.45){
     const stain=new THREE.Mesh(new THREE.CircleGeometry(.22+seeded(c,r)*.18,16),new THREE.MeshStandardMaterial({color:'#73503b',roughness:.55}));stain.rotation.x=-Math.PI/2;stain.position.set(x,.067,z);gridGroup.add(stain)
   } else if(kind===3){
     const ripple=new THREE.Mesh(new THREE.TorusGeometry(.36,.018,5,20),new THREE.MeshStandardMaterial({color:'#d4f5fb',roughness:.25,transparent:true,opacity:.72}));ripple.rotation.x=Math.PI/2;ripple.position.set(x,.07,z);gridGroup.add(ripple)
   }
 }
}
// A restrained tile grid keeps the world legible like a top-down route simulation.
const grid=new THREE.GridHelper(N*S,N,0xf6d29a,0xf3d9b2);grid.position.set(half,.071,half);grid.material.transparent=true;grid.material.opacity=.28;scene.add(grid);
function world(cell,source){
 if(!Array.isArray(cell)||cell.length<2)throw new Error('Invalid scene position for '+source+': '+JSON.stringify(cell)+' (expected [row, column]).');
 const row=Number(cell[0]),col=Number(cell[1]);
 if(!Number.isInteger(row)||!Number.isInteger(col)||!heights[row]||heights[row][col]===undefined)
  throw new Error('Scene position outside the '+N+'×'+N+' terrain grid for '+source+': ['+cell[0]+', '+cell[1]+'].');
 return new THREE.Vector3(col*S,heights[row][col],row*S);
}
function labelSprite(text,color){const canvas=document.createElement('canvas'),ctx=canvas.getContext('2d');canvas.width=320;canvas.height=96;
 ctx.fillStyle='rgba(20,36,58,.96)';ctx.beginPath();ctx.roundRect(5,8,310,80,24);ctx.fill();ctx.font='900 54px system-ui';ctx.fillStyle=color;ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(text,160,49);
 const tex=new THREE.CanvasTexture(canvas);tex.colorSpace=THREE.SRGBColorSpace;const m=new THREE.Sprite(new THREE.SpriteMaterial({map:tex,transparent:true,depthTest:false}));m.scale.set(2.25,.58,1);return m}
function marker(cell,color,text,shape='stop'){
 const p=world(cell,'marker '+text),g=new THREE.Group();g.position.set(p.x,p.y,p.z);
 const ring=new THREE.Mesh(new THREE.TorusGeometry(.36,.045,8,28),new THREE.MeshStandardMaterial({color,emissive:color,emissiveIntensity:.42}));ring.rotation.x=Math.PI/2;ring.position.y=.12;g.add(ring);
 if(shape==='depot'){const box=new THREE.Mesh(new THREE.BoxGeometry(.5,.24,.5),new THREE.MeshStandardMaterial({color:'#e76550',roughness:.65}));box.position.y=.16;box.castShadow=true;g.add(box)}
 else {const starShape=new THREE.Shape();for(let i=0;i<10;i++){const a=Math.PI/2+i*Math.PI/5,rad=i%2?.19:.4;const x=Math.cos(a)*rad,y=Math.sin(a)*rad;i?starShape.lineTo(x,y):starShape.moveTo(x,y)}starShape.closePath();const star=new THREE.Mesh(new THREE.ShapeGeometry(starShape),new THREE.MeshStandardMaterial({color:'#ffd36a',emissive:'#9c671a',emissiveIntensity:.18,side:THREE.DoubleSide}));star.rotation.x=-Math.PI/2;star.position.y=.15;g.add(star)}
 const tag=labelSprite(text,color);tag.position.set(shape==='depot'?-1.35:0,1.35,shape==='depot'?-1.15:0);
 tag.scale.set(shape==='depot'?2.5:4.0,shape==='depot'?.65:1.0,1);g.add(tag);tagTargets.push(tag);g.scale.setScalar(visualScale);scaleTargets.push(g);scene.add(g);return g;
}
marker(DATA.start,'#ff796f','DEPOT','depot');DATA.stops.forEach((p,i)=>marker(p,'#f6bd52','STOP '+(i+1)));
DATA.stations.forEach((p,i)=>{const p3=world(p,'charging station'),pin=new THREE.Group();pin.position.set(p3.x,p3.y,p3.z);const pad=new THREE.Mesh(new THREE.BoxGeometry(1.35,.12,1.35),new THREE.MeshStandardMaterial({color:'#647277',roughness:.85}));pad.position.y=.13;pin.add(pad);const charge=new THREE.Mesh(new THREE.BoxGeometry(.92,.06,.92),new THREE.MeshStandardMaterial({color:'#47c9c0',emissive:'#087b78',emissiveIntensity:.6,roughness:.3}));charge.position.y=.22;pin.add(charge);const orb=new THREE.Mesh(new THREE.SphereGeometry(.18,16,12),new THREE.MeshStandardMaterial({color:'#d8fff5',emissive:'#27d5bb',emissiveIntensity:1.3,metalness:.18}));orb.position.y=.52;pin.add(orb);pin.scale.setScalar(visualScale);scaleTargets.push(pin);scene.add(pin)});
// Route ribbon is slightly lifted above the terrain so its direction stays visible.
const routePoints=DATA.route.map(p=>{const q=world(p,'courier route');return new THREE.Vector3(q.x,q.y+.25,q.z)});
if(!DATA.gameMode&&routePoints.length>1){for(let i=0;i<routePoints.length-1;i++){const a=routePoints[i],b=routePoints[i+1],delta=b.clone().sub(a),segment=new THREE.Mesh(new THREE.CylinderGeometry(.075,.075,delta.length()+.04,8),mats.route);segment.position.copy(a).add(b).multiplyScalar(.5);segment.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),delta.normalize());segment.scale.set(visualScale,1,visualScale);segment.castShadow=true;routeScaleTargets.push(segment);scene.add(segment)}}
function makeTrail(route,color){
 if(!route.length)return null;
 const points=route.map(cell=>{const p=world(cell,'animated trail');return new THREE.Vector3(p.x,p.y+.28,p.z)});
 const geometry=new THREE.BufferGeometry().setFromPoints(points);geometry.setDrawRange(0,1);
 const line=new THREE.Line(geometry,new THREE.LineBasicMaterial({color,transparent:true,opacity:.92}));
 line.frustumCulled=false;scene.add(line);return {line,geometry,points};
}
const agentTrail=DATA.gameMode?makeTrail(DATA.route,'#2771ff'):null;
const rivalTrail=DATA.gameMode?makeTrail(DATA.opponentRoute,'#dc4264'):null;
// Procedural courier: face, jacket, backpack, articulated limbs and visible delivery parcel.
const courier=new THREE.Group();scene.add(courier);
function part(parent,geo,mat,pos,scale){const m=new THREE.Mesh(geo,mat);m.position.set(...pos);if(scale)m.scale.set(...scale);m.castShadow=true;m.receiveShadow=true;parent.add(m);return m}
const sphere=new THREE.SphereGeometry(1,20,16),capsule=new THREE.CapsuleGeometry(.17,.42,5,10),limbGeo=new THREE.CapsuleGeometry(.075,.3,4,8);
const torso=part(courier,new THREE.CapsuleGeometry(.28,.42,6,12),mats.jacket,[0,.94,0],[1,1,.76]);
part(courier,sphere,mats.skin,[0,1.62,.015],[.205,.225,.2]);
part(courier,new THREE.SphereGeometry(.218,16,10),mats.dark,[0,1.77,0],[1,.38,1]); // cap
part(courier,new THREE.BoxGeometry(.26,.035,.13),mats.dark,[0,1.72,.17]);
// Face details point toward local +Z.
for(const x of [-.072,.072])part(courier,sphere,mats.white,[x,1.64,.19],[.033,.026,.018]);
part(courier,sphere,mats.dark,[0,1.585,.205],[.06,.025,.02]);
const pack=part(courier,new THREE.BoxGeometry(.37,.42,.19),mats.pack,[0,1.02,-.24]);pack.rotation.x=-.1;
part(courier,new THREE.BoxGeometry(.17,.09,.025),mats.white,[0,1.08,-.343]);
const leftArm=new THREE.Group();leftArm.position.set(-.3,1.26,0);courier.add(leftArm);part(leftArm,limbGeo,mats.jacket,[0,-.2,0]);part(leftArm,sphere,mats.skin,[0,-.43,.015],[.075,.075,.075]);
const rightArm=new THREE.Group();rightArm.position.set(.3,1.26,0);courier.add(rightArm);part(rightArm,limbGeo,mats.jacket,[0,-.2,0]);part(rightArm,sphere,mats.skin,[0,-.43,.015],[.075,.075,.075]);
const leftLeg=new THREE.Group();leftLeg.position.set(-.13,.68,0);courier.add(leftLeg);part(leftLeg,limbGeo,mats.pants,[0,-.22,0]);part(leftLeg,new THREE.BoxGeometry(.16,.085,.28),mats.dark,[0,-.46,.07]);
const rightLeg=new THREE.Group();rightLeg.position.set(.13,.68,0);courier.add(rightLeg);part(rightLeg,limbGeo,mats.pants,[0,-.22,0]);part(rightLeg,new THREE.BoxGeometry(.16,.085,.28),mats.dark,[0,-.46,.07]);
part(courier,new THREE.BoxGeometry(.28,.09,.1),mats.gold,[0,1.38,.01]); // reflective jacket stripe
courier.scale.setScalar(1.5*visualScale);
const trackingRing=new THREE.Mesh(new THREE.TorusGeometry(.43,.055,9,36),new THREE.MeshStandardMaterial({color:'#42d6c2',emissive:'#13a996',emissiveIntensity:.8}));
trackingRing.rotation.x=Math.PI/2;trackingRing.position.y=.18;courier.add(trackingRing);
let rivalCourier=null,rivalTrackingRing=null,rivalTag=null;
let rivalLeftArm=null,rivalRightArm=null,rivalLeftLeg=null,rivalRightLeg=null,rivalWalkPhase=0;
if(DATA.gameMode){
 const rivalCoat=new THREE.MeshStandardMaterial({color:'#bd3557',roughness:.74});
 const rivalPack=new THREE.MeshStandardMaterial({color:'#492c43',roughness:.82});
 rivalCourier=new THREE.Group();scene.add(rivalCourier);
 part(rivalCourier,new THREE.CapsuleGeometry(.29,.44,6,12),rivalCoat,[0,.94,0],[1,1,.76]);
 part(rivalCourier,sphere,mats.skin,[0,1.62,.015],[.205,.225,.2]);
 part(rivalCourier,new THREE.SphereGeometry(.218,16,10),rivalPack,[0,1.77,0],[1,.38,1]);
 part(rivalCourier,new THREE.BoxGeometry(.36,.4,.2),rivalPack,[0,1.02,-.24]);
 rivalLeftArm=new THREE.Group();rivalLeftArm.position.set(-.3,1.26,0);rivalCourier.add(rivalLeftArm);
 part(rivalLeftArm,limbGeo,rivalCoat,[0,-.2,0]);part(rivalLeftArm,sphere,mats.skin,[0,-.43,.015],[.075,.075,.075]);
 rivalRightArm=new THREE.Group();rivalRightArm.position.set(.3,1.26,0);rivalCourier.add(rivalRightArm);
 part(rivalRightArm,limbGeo,rivalCoat,[0,-.2,0]);part(rivalRightArm,sphere,mats.skin,[0,-.43,.015],[.075,.075,.075]);
 rivalLeftLeg=new THREE.Group();rivalLeftLeg.position.set(-.13,.68,0);rivalCourier.add(rivalLeftLeg);
 part(rivalLeftLeg,limbGeo,mats.pants,[0,-.22,0]);part(rivalLeftLeg,new THREE.BoxGeometry(.17,.09,.28),mats.dark,[0,-.46,.07]);
 rivalRightLeg=new THREE.Group();rivalRightLeg.position.set(.13,.68,0);rivalCourier.add(rivalRightLeg);
 part(rivalRightLeg,limbGeo,mats.pants,[0,-.22,0]);part(rivalRightLeg,new THREE.BoxGeometry(.17,.09,.28),mats.dark,[0,-.46,.07]);
 rivalCourier.scale.setScalar(1.38*visualScale);
 rivalTrackingRing=new THREE.Mesh(new THREE.TorusGeometry(.45,.06,9,36),new THREE.MeshStandardMaterial({color:'#f45578',emissive:'#c82750',emissiveIntensity:.75}));
 rivalTrackingRing.rotation.x=Math.PI/2;rivalTrackingRing.position.y=.18;rivalCourier.add(rivalTrackingRing);
 rivalTag=labelSprite('RIVAL','#ff9bb0');rivalTag.position.set(0,2.15,0);rivalCourier.add(rivalTag);
}

const nRoute=Math.max(0,DATA.route.length-1);let routeProgress=0,playing=DATA.autoPlay,last=null,pace=DATA.gameMode?6:2.5,walkPhase=0,follow=DATA.gameMode||DATA.autoPlay,previousFollowPoint=null;
const playBtn=document.querySelector('#play'),progressEl=document.querySelector('#progress'),stepEl=document.querySelector('#step'),coordsEl=document.querySelector('#coords'),rivalCoordsEl=document.querySelector('#rival-coords'),stateEl=document.querySelector('#run-state'),followBtn=document.querySelector('#follow'),speedInput=document.querySelector('#speed'),finish=document.querySelector('#finish');
if(DATA.gameMode){speedInput.min='1';speedInput.max='30';speedInput.step='1';speedInput.value='6';document.querySelector('#speed-label').textContent='TURNS / SEC';
 playBtn.innerHTML='❚❚&nbsp; Pause match';stateEl.textContent='LIVE';followBtn.classList.add('active');followBtn.textContent='◎  Following both';}
else if(DATA.autoPlay){playBtn.innerHTML='❚❚&nbsp; Pause route';stateEl.textContent='IN TRANSIT';followBtn.classList.add('active');followBtn.textContent='◎  Following courier';}
function focusCourier(){
 const target=DATA.gameMode&&rivalCourier?courier.position.clone().add(rivalCourier.position).multiplyScalar(.5).add(new THREE.Vector3(0,1.2,0)):courier.position.clone().add(new THREE.Vector3(0,1.45,0));
 controls.target.copy(target);
 if(DATA.gameMode&&rivalCourier){
  const separation=courier.position.distanceTo(rivalCourier.position),distance=THREE.MathUtils.clamp(separation*1.12+22,34,82);
  camera.position.copy(target).add(new THREE.Vector3(.38,.88,.46).normalize().multiplyScalar(distance));
 }else camera.position.copy(target).add(new THREE.Vector3(8,24,10));
 controls.update();previousFollowPoint=target.clone()
}
const settingsPanel=document.querySelector('#settings-panel'),settingsToggle=document.querySelector('#settings-toggle');
settingsToggle.addEventListener('click',()=>{const open=settingsPanel.classList.toggle('open');settingsToggle.setAttribute('aria-expanded',String(open))});
document.querySelector('#show-minimap').addEventListener('change',e=>document.querySelector('#minimap').style.display=e.target.checked?'block':'none');
document.querySelector('#show-legend').addEventListener('change',e=>document.querySelector('#legend').style.display=e.target.checked?'grid':'none');
document.querySelector('#show-tags').addEventListener('change',e=>tagTargets.forEach(tag=>tag.visible=e.target.checked));
document.querySelector('#visual-size').addEventListener('input',e=>{visualScale=Number(e.target.value);document.querySelector('#size-value').textContent=`${Math.round(visualScale*100)}%`;
 scaleTargets.forEach(target=>target.scale.setScalar(visualScale));routeScaleTargets.forEach(target=>target.scale.set(visualScale,1,visualScale));courier.scale.setScalar(1.5*visualScale)});
const mini=document.querySelector('#mini-map'),miniCtx=mini.getContext('2d');
function drawMiniMap(agentRow,agentCol,progress=routeProgress,rivalRow=null,rivalCol=null){const cell=mini.width/N,colors=['#d98b3a','#648d49','#94684d','#52b9c8','#777a7d'];miniCtx.clearRect(0,0,mini.width,mini.height);
 for(let r=0;r<N;r++)for(let c=0;c<N;c++){const type=DATA.grid[r][c];miniCtx.fillStyle=type<0?colors[4]:colors[type];miniCtx.fillRect(c*cell,r*cell,cell+.5,cell+.5)}
 miniCtx.strokeStyle='rgba(255,245,222,.32)';miniCtx.lineWidth=.5;for(let i=0;i<=N;i++){miniCtx.beginPath();miniCtx.moveTo(i*cell,0);miniCtx.lineTo(i*cell,mini.height);miniCtx.stroke();miniCtx.beginPath();miniCtx.moveTo(0,i*cell);miniCtx.lineTo(mini.width,i*cell);miniCtx.stroke()}
miniCtx.strokeStyle='#1763ed';miniCtx.lineWidth=Math.max(1.4,cell*.42);miniCtx.lineJoin='round';miniCtx.lineCap='round';miniCtx.beginPath();const step=Math.floor(progress),start=DATA.gameMode?Math.max(0,step-28):0,visibleRoute=DATA.gameMode?DATA.route.slice(start,step+1):DATA.route;visibleRoute.forEach(([r,c],i)=>{if(i)miniCtx.lineTo((c+.5)*cell,(r+.5)*cell);else miniCtx.moveTo((c+.5)*cell,(r+.5)*cell)});miniCtx.stroke();
 DATA.stations.forEach(([r,c])=>{miniCtx.fillStyle='#53e1ce';miniCtx.fillRect(c*cell+cell*.2,r*cell+cell*.2,cell*.6,cell*.6)});
 DATA.stops.forEach(([r,c])=>{miniCtx.fillStyle='#ffda68';miniCtx.beginPath();miniCtx.arc((c+.5)*cell,(r+.5)*cell,Math.max(1.6,cell*.36),0,Math.PI*2);miniCtx.fill()});
 const [sr,sc]=DATA.start;miniCtx.fillStyle='#ee6558';miniCtx.fillRect(sc*cell+cell*.12,sr*cell+cell*.12,cell*.76,cell*.76);
 miniCtx.fillStyle='#fff';miniCtx.beginPath();miniCtx.arc((agentCol+.5)*cell,(agentRow+.5)*cell,Math.max(2,cell*.46),0,Math.PI*2);miniCtx.fill();miniCtx.fillStyle='#1969ff';miniCtx.beginPath();miniCtx.arc((agentCol+.5)*cell,(agentRow+.5)*cell,Math.max(1.4,cell*.32),0,Math.PI*2);miniCtx.fill();
 if(rivalRow!==null){miniCtx.fillStyle='#fff';miniCtx.beginPath();miniCtx.arc((rivalCol+.5)*cell,(rivalRow+.5)*cell,Math.max(2,cell*.46),0,Math.PI*2);miniCtx.fill();miniCtx.fillStyle='#dc4264';miniCtx.beginPath();miniCtx.arc((rivalCol+.5)*cell,(rivalRow+.5)*cell,Math.max(1.4,cell*.32),0,Math.PI*2);miniCtx.fill()}
}
speedInput.addEventListener('input',e=>pace=Number(e.target.value));
function updateTrail(trail,progress){
 if(!trail)return;
 const segment=Math.max(0,Math.min(Math.floor(progress),trail.points.length-1)),t=Math.max(0,Math.min(progress-segment,1)),next=Math.min(segment+1,trail.points.length-1);
 const a=trail.points[segment],b=trail.points[next],attr=trail.geometry.getAttribute('position');
 attr.setXYZ(next,a.x+(b.x-a.x)*t,a.y+(b.y-a.y)*t,a.z+(b.z-a.z)*t);attr.needsUpdate=true;
 const trailStart=DATA.gameMode?Math.max(0,segment-28):0;
 trail.geometry.setDrawRange(trailStart,Math.min(trail.points.length-trailStart,segment-trailStart+2));
}
function updateCourier(){const safeProgress=Math.max(0,Math.min(routeProgress,nRoute)),segment=Math.max(0,Math.min(Math.floor(safeProgress),DATA.route.length-1));if(!DATA.route[segment])throw new Error('Missing courier route segment '+segment+'; progress='+routeProgress+', pace='+pace+', route length='+DATA.route.length);const t=Math.max(0,Math.min(safeProgress-segment,1)),a=world(DATA.route[segment],'courier animation current'),b=world(DATA.route[Math.min(segment+1,DATA.route.length-1)],'courier animation next');
 courier.position.set(THREE.MathUtils.lerp(a.x,b.x,t),THREE.MathUtils.lerp(a.y,b.y,t),THREE.MathUtils.lerp(a.z,b.z,t));
 if(segment<DATA.route.length-1){const dx=b.x-a.x,dz=b.z-a.z;courier.rotation.y=Math.atan2(dx,dz)}
 const moving=playing&&segment<nRoute&&(DATA.route[segment][0]!==DATA.route[segment+1][0]||DATA.route[segment][1]!==DATA.route[segment+1][1]);walkPhase+=moving?.28:0;const swing=moving?Math.sin(walkPhase)*.62:0;
 leftLeg.rotation.x=swing;rightLeg.rotation.x=-swing;leftArm.rotation.x=-swing*.8;rightArm.rotation.x=swing*.8;
 courier.position.y+=moving?Math.abs(Math.sin(walkPhase*2))*.035:0;
 const row=Math.round(THREE.MathUtils.lerp(DATA.route[segment][0],DATA.route[Math.min(segment+1,DATA.route.length-1)][0],t));
 const col=Math.round(THREE.MathUtils.lerp(DATA.route[segment][1],DATA.route[Math.min(segment+1,DATA.route.length-1)][1],t));
 let rivalRow=null,rivalCol=null;
 if(DATA.gameMode){
  const safeProgress=Math.max(0,Math.min(routeProgress,DATA.opponentRoute.length-1)),ri=Math.max(0,Math.min(Math.floor(safeProgress),DATA.opponentRoute.length-1)),rt=Math.max(0,Math.min(safeProgress-ri,1)),r0=world(DATA.opponentRoute[ri],'rival animation current'),r1=world(DATA.opponentRoute[Math.min(ri+1,DATA.opponentRoute.length-1)],'rival animation next');
  rivalCourier.position.set(THREE.MathUtils.lerp(r0.x,r1.x,rt),THREE.MathUtils.lerp(r0.y,r1.y,rt)+Math.abs(Math.sin(walkPhase*1.7))*.025,THREE.MathUtils.lerp(r0.z,r1.z,rt));
  if(ri<DATA.opponentRoute.length-1){const dx=r1.x-r0.x,dz=r1.z-r0.z;rivalCourier.rotation.y=Math.atan2(dx,dz)}
  rivalRow=Math.round(THREE.MathUtils.lerp(DATA.opponentRoute[ri][0],DATA.opponentRoute[Math.min(ri+1,DATA.opponentRoute.length-1)][0],rt));
  rivalCol=Math.round(THREE.MathUtils.lerp(DATA.opponentRoute[ri][1],DATA.opponentRoute[Math.min(ri+1,DATA.opponentRoute.length-1)][1],rt));
  const rivalMoving=playing&&ri<DATA.opponentRoute.length-1&&
   (DATA.opponentRoute[ri][0]!==DATA.opponentRoute[ri+1][0]||DATA.opponentRoute[ri][1]!==DATA.opponentRoute[ri+1][1]);
  rivalWalkPhase+=rivalMoving?.28:0;const rivalSwing=rivalMoving?Math.sin(rivalWalkPhase)*.62:0;
  rivalLeftLeg.rotation.x=rivalSwing;rivalRightLeg.rotation.x=-rivalSwing;
  rivalLeftArm.rotation.x=-rivalSwing*.8;rivalRightArm.rotation.x=rivalSwing*.8;
  rivalCoordsEl.textContent=`Rival · Row ${rivalRow+1}, Col ${rivalCol+1}`;
 }
 updateTrail(agentTrail,routeProgress);updateTrail(rivalTrail,routeProgress);
 drawMiniMap(row,col,routeProgress,rivalRow,rivalCol);
 coordsEl.textContent=DATA.gameMode?`Courier · Row ${row+1}, Col ${col+1}`:`Row ${row+1} · Col ${col+1}`;
 stepEl.textContent=DATA.gameMode?`Turn ${Math.floor(routeProgress)} / ${nRoute}`:`Move ${Math.floor(routeProgress)} / ${nRoute}`;
 progressEl.style.width=(nRoute?routeProgress/nRoute*100:100)+'%';
}
function showFinalResult(){
 const result=DATA.gameResult||{},label=String(result.result||'draw').toUpperCase();
 document.querySelector('#finish-title').textContent=label==='WIN'?'Delivery successful':label==='LOSS'?'Match ended in a loss':'Match ended in a draw';
 document.querySelector('#finish-detail').textContent=result.why?String(result.why).replaceAll('_',' '):'The game reached its terminal state.';
 document.querySelector('#finish-stats').innerHTML=`<span>${result.turns||nRoute} turns</span><span>${result.cost||0} travel cost</span><span>${result.slips||0} slips</span>`;
 finish.style.display='block';stateEl.textContent=label;playBtn.innerHTML='↻&nbsp; Replay match';
}
playBtn.addEventListener('click',()=>{if(routeProgress>=nRoute){routeProgress=0;finish.style.display='none'}playing=!playing;
 playBtn.innerHTML=playing?(DATA.gameMode?'❚❚&nbsp; Pause match':'❚❚&nbsp; Pause route'):(DATA.gameMode?'▶&nbsp; Resume match':'▶&nbsp; Resume route');
 stateEl.textContent=playing?(DATA.gameMode?'LIVE':'IN TRANSIT'):'PAUSED';
 if(playing){follow=true;followBtn.classList.add('active');followBtn.textContent=DATA.gameMode?'◎  Following both':'◎  Following courier';focusCourier()}});
followBtn.addEventListener('click',()=>{follow=!follow;followBtn.classList.toggle('active',follow);followBtn.textContent=follow?(DATA.gameMode?'◎  Following both':'◎  Following courier'):'◎  Follow courier';if(follow)focusCourier();else previousFollowPoint=null});
document.querySelector('#reset').addEventListener('click',()=>{playing=false;routeProgress=0;finish.style.display='none';playBtn.innerHTML=DATA.gameMode?'▶&nbsp; Replay match':'▶&nbsp; Start route';stateEl.textContent=DATA.gameMode?'READY':'READY';updateCourier();if(follow){controls.target.set(half,0,half);camera.position.set(half+15,45,half+18);previousFollowPoint=courier.position.clone()}});
function resize(){const w=stage.clientWidth,h=stage.clientHeight;if(!w||!h)return;renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix()}
new ResizeObserver(resize).observe(stage);resize();updateCourier();
if(follow)focusCourier();
renderer.setAnimationLoop(()=>{const now=performance.now(),dt=last===null?0:Math.max(0,Math.min((now-last)/1000,.05));last=now;
 if(playing){routeProgress=Math.max(0,Math.min(nRoute,routeProgress+dt*pace));if(routeProgress>=nRoute){routeProgress=nRoute;playing=false;if(DATA.gameMode)showFinalResult();else{playBtn.innerHTML='✓&nbsp; Route complete';stateEl.textContent='DELIVERED'}}}
 updateCourier();
 if(follow&&DATA.gameMode&&rivalCourier){
  const target=courier.position.clone().add(rivalCourier.position).multiplyScalar(.5).add(new THREE.Vector3(0,1.2,0));
  const separation=courier.position.distanceTo(rivalCourier.position),distance=THREE.MathUtils.clamp(separation*1.12+22,34,82);
  controls.target.lerp(target,.09);
  const cameraTarget=controls.target.clone().add(new THREE.Vector3(.38,.88,.46).normalize().multiplyScalar(distance));
  camera.position.lerp(cameraTarget,.055);
 }else if(follow&&previousFollowPoint){
  const target=courier.position.clone().add(new THREE.Vector3(0,1.45,0)),delta=target.clone().sub(previousFollowPoint);
  controls.target.add(delta);camera.position.add(delta);previousFollowPoint.copy(target)
 }
 controls.update();renderer.render(scene,camera)});
</script>
<script>
window.addEventListener('error',event=>{const box=document.querySelector('#error');if(box){box.style.display='flex';document.querySelector('#error-text').textContent=(event.message||'WebGL or Three.js failed to load.')+(event.lineno?' · scene line '+event.lineno:'')+(event.error&&event.error.stack?' · '+event.error.stack.split('\\n').slice(1,3).join(' '):'')}});
</script>
</body></html>'''.replace("__DATA__", data)
    components.html(html, height=height, scrolling=False)
