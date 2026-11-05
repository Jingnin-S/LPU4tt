#!/usr/bin/env python3
"""MIT. Draw the existing Tiny-LPU architecture; no model or RTL changes."""
from pathlib import Path
import json
import math
import html
from collections import Counter
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/diagrams'
W,H,SCALE=1680,1370,2
INK='#17324d';BLUE='#2563eb';TEAL='#0f766e';GRAY='#718096';ORANGE='#b96716'
elements=[];ops=[]
font_path='/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc'
bold_path='/usr/share/fonts/google-noto-cjk/NotoSansCJK-Bold.ttc'
font_index=next(i for i in range(10) if ImageFont.truetype(font_path,20,index=i).getname()[0]=='Noto Sans CJK SC')
fonts={}


def font(size,bold=False):
    key=(size,bold)
    if key not in fonts:fonts[key]=ImageFont.truetype(bold_path if bold else font_path,size,index=font_index)
    return fonts[key]


def element(kind,x,y,w,h,color=INK,fill='transparent',dash=False,group=None):
    n=len(elements)
    digits='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    order='a'+digits[n] if n<62 else 'b'+digits[(n-62)//62]+digits[(n-62)%62]
    e=dict(id=f'tinylpu_{n:03d}',type=kind,x=x,y=y,width=w,height=h,angle=0,
           strokeColor=color,backgroundColor=fill,fillStyle='solid',strokeWidth=2,
           strokeStyle='dashed' if dash else 'solid',roughness=0,opacity=100,
           groupIds=[group] if group else [],frameId=None,index=order,roundness=None,
           seed=10400+n,version=1,versionNonce=10600+n,isDeleted=False,
           boundElements=None,updated=0,link=None,locked=False)
    elements.append(e);return e


def rect(x,y,w,h,fill,stroke=BLUE,dash=False,group=None,radius=15):
    e=element('rectangle',x,y,w,h,stroke,fill,dash,group)
    e['roundness']={'type':3} if radius else None
    ops.append(('rect',x,y,w,h,fill,stroke,dash,radius));return e


def text(x,y,value,size=24,color=INK,center=None,bold=False,group=None):
    f=font(size,bold);width=f.getlength(value)
    if center is not None:x=center-width/2
    e=element('text',x,y,width,size*1.3,color,group=group)
    e.update(text=value,originalText=value,fontSize=size,fontFamily=5,
             textAlign='left',verticalAlign='top',containerId=None,
             autoResize=True,lineHeight=1.3)
    ops.append(('text',x,y,value,size,color,bold))
    assert 0<=x and x+width<=W+1,(value,x,width)
    assert 0<=y and y+size*1.3<=H


def box(x,y,w,h,title,lines,fill='#ffffff',stroke=BLUE,dash=False,size=22):
    group=f'group_{len(elements)}'
    rect(x,y,w,h,fill,stroke,dash,group)
    total=30+len(lines)*29
    top=y+(h-total)/2
    text(x,top,title,25,stroke,center=x+w/2,bold=True,group=group)
    for i,line in enumerate(lines):
        assert font(size).getlength(line)<=w-22,(title,line)
        text(x,top+36+i*29,line,size,center=x+w/2,group=group)


def arrow(points,color=BLUE,dash=False,both=False,width=3):
    x,y=points[0]
    e=element('arrow',x,y,max(p[0] for p in points)-min(p[0] for p in points),
              max(p[1] for p in points)-min(p[1] for p in points),color,dash=dash)
    e.update(points=[[px-x,py-y] for px,py in points],lastCommittedPoint=None,
             startBinding=None,endBinding=None,startArrowhead='arrow' if both else None,
             endArrowhead='arrow',elbowed=False)
    e['strokeWidth']=width
    ops.append(('arrow',points,color,dash,both,width))


# Background panels first, then edges, then nodes and labels.
rect(40,145,1600,185,'#edf1f7','#b8c6d8',radius=20)
rect(40,505,1600,620,'#edf5fc','#92b5d8',radius=20)
rect(1280,600,310,280,'#f7fbff','#a7c3df',radius=14)

arrow([(330,250),(405,250)])
arrow([(745,250),(820,250)])
arrow([(1140,250),(1215,250)])
arrow([(1410,310),(1410,365)],both=True)
arrow([(220,450),(220,620)])
arrow([(365,700),(420,700)])
arrow([(870,700),(930,700)])
arrow([(1210,690),(1300,690)])
arrow([(1210,715),(1240,715),(1240,810),(1300,810)],ORANGE,True)
arrow([(1570,685),(1620,685),(1620,995),(1590,995)])
arrow([(1435,855),(1435,945)],ORANGE,True)
arrow([(1280,995),(900,995),(900,815),(870,815)])
arrow([(420,805),(390,805),(390,985),(365,985)])
arrow([(75,985),(55,985),(55,405),(70,405)],TEAL)
arrow([(365,645),(390,645),(390,575),(1070,575),(1070,620)],GRAY,True,width=2)


text(65,35,'LPU4tt: Software Stack and Tiny-LPU Architecture',38,bold=True)
text(68,98,'W4A4: 4-bit weights and activations; ACC32: 32-bit integer accumulation',24,GRAY)
text(65,151,'Host software',21,GRAY,bold=True)
box(70,200,260,105,'SGLang / PyTorch',['Linear graph / tensors'],size=20)
box(405,185,340,125,'Compiler',['Numeric contract; M/K tiles','Bank layout; reuse; schedule'],size=20)
box(820,185,320,125,'ISA v0.1',['LOAD / STORE / MVM / END','4B/8B headers + LOAD data'],size=20)
box(1215,185,395,125,'Host Runtime',['Quant/dequant; bias; nonlinear','Pack byte stream / decode result'],size=20)
text(1120,334,'Commands, data and results',19,GRAY)
box(70,365,1540,85,'TT GPIO / ready-valid byte interface',
    ['8-bit RX/TX; valid/ready handshake; start / busy / done / error'],fill='#e6f3f0',stroke=TEAL,size=23)
text(80,550,'Commands /',19,BLUE)
text(80,580,'LOAD data',19,BLUE)
text(70,470,'Results / status',19,TEAL)
text(420,522,'Tiny-LPU digital core / one clock',28,bold=True)
text(1190,525,'Dashed: second-lane extension',19,ORANGE)
box(75,620,290,170,'RX / Check / Control',
    ['Assemble header; dispatch op','Range / alignment / alias checks','M/K groups; reset / ena / errors'],size=17)
rect(420,610,450,230,'#ffffff',BLUE,group='memory')
text(420,627,'4-bank Scratchpad + Valid',26,BLUE,center=645,bold=True,group='memory')
for i,x in enumerate((442,547,652,757)):
    box(x,678,95,70,f'Bank {i}',['1R'],fill='#dcecff',size=20)
text(440,770,'Sync read: 1 cycle; valid bit / byte',20,group='memory')
text(440,802,'LOAD/WB share one 1B/clock write port',19,group='memory')
text(365,668,'LOAD',18,BLUE)
text(883,668,'Read',17,BLUE)
text(690,548,'Control / valid',19,GRAY)
box(930,620,280,140,'Gather / Arbitration',
    ['Bank arbitration; broadcast','Collect A / W0 / W1','Read seed32 for ACCUM'],size=18)
text(1218,660,'A,W0',18,BLUE)
text(1160,785,'A,W1',18,ORANGE)
text(1300,607,'MAC pipeline',24,BLUE,bold=True)
box(1300,640,270,90,'Lane 0',['INT4 x INT4 -> ACC32'],fill='#e6f4eb',stroke='#24704b',size=20)
box(1300,765,270,90,'Lane 1 (extension)',['INT4 x INT4 -> ACC32'],fill='#fff2df',stroke=ORANGE,dash=True,size=20)
text(1500,895,'ACC32',20,BLUE)
box(1280,945,310,100,'Writeback',['raw32 or final SAT16'],size=23)
text(950,955,'Serialized writeback: 1B/clock',19,BLUE)
box(75,925,290,120,'STORE / TX',['Sync read -> response bytes','Hold data under backpressure'],stroke=TEAL,size=18)
text(280,850,'STORE read',18,TEAL)
text(440,925,'Datapath notes',23,GRAY,bold=True)
text(440,966,'Broadcast A; each lane reads its own weights.',20,GRAY)
text(440,1003,'Serialize conflicts; conflict-free core issue II=1.',20,GRAY)
text(440,1040,'Keep partial sums raw32; SAT16 only at the end.',20,GRAY)
box(70,1160,720,75,'Qualified macro profile',
    ['64B = 4 x 16B; Lane 0; SKY130A user macro'],fill='#e6f4eb',stroke='#24704b',size=22)
box(825,1160,785,75,'Modeling / expansion candidates',
    ['128B = 4 x 32B, two lanes; 192B = 4 x 48B, longer-K study'],fill='#fff2df',stroke=ORANGE,size=20)
text(70,1265,'Scope: CPU SGLang connects to the C++ device model; physical GPIO driver and silicon bring-up remain future work.',20,GRAY)
text(70,1305,'Clock: 100MHz is internal macro STA; nominal external input pad rating is ~66MHz. No PLL or CDC in this design.',20,GRAY)


def dashed_line(draw,points,fill,width,dash=10,gap=7):
    for a,b in zip(points,points[1:]):
        dx,dy=b[0]-a[0],b[1]-a[1];length=math.hypot(dx,dy)
        if not length:continue
        for t in range(0,math.ceil(length),dash+gap):
            t1=min(t+dash,length)
            draw.line([(a[0]+dx*t/length,a[1]+dy*t/length),
                       (a[0]+dx*t1/length,a[1]+dy*t1/length)],fill=fill,width=width)


OUT.mkdir(parents=True,exist_ok=True)
im=Image.new('RGB',(W*SCALE,H*SCALE),'#ffffff');d=ImageDraw.Draw(im)
svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
     '<rect width="100%" height="100%" fill="white"/>']
for op in ops:
    if op[0]=='rect':
        _,x,y,w,h,fill,stroke,dash,radius=op
        xy=(x*SCALE,y*SCALE,(x+w)*SCALE,(y+h)*SCALE)
        d.rounded_rectangle(xy,radius=radius*SCALE,fill=fill,
                            outline=stroke if not dash else None,width=2*SCALE)
        if dash:dashed_line(d,[(x*SCALE,y*SCALE),((x+w)*SCALE,y*SCALE),
                              ((x+w)*SCALE,(y+h)*SCALE),(x*SCALE,(y+h)*SCALE),(x*SCALE,y*SCALE)],stroke,2*SCALE,10*SCALE,7*SCALE)
        dash_attr='stroke-dasharray="10 7"' if dash else ''
        svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="2" {dash_attr}/>')
    elif op[0]=='text':
        _,x,y,value,size,color,bold=op
        d.text((x*SCALE,y*SCALE),value,font=font(size*SCALE,bold),fill=color,anchor='lt')
        weight='700' if bold else '400'
        svg.append(f'<text x="{x:.2f}" y="{y+size:.2f}" font-family="Noto Sans CJK SC,Microsoft YaHei,sans-serif" font-size="{size}" font-weight="{weight}" fill="{color}">{html.escape(value)}</text>')
    else:
        _,points,color,dash,both,width=op
        pts=[(x*SCALE,y*SCALE) for x,y in points]
        if dash:dashed_line(d,pts,color,width*SCALE,10*SCALE,7*SCALE)
        else:d.line(pts,fill=color,width=width*SCALE,joint='curve')
        path=' '.join(f'{x},{y}' for x,y in points)
        dash_attr='stroke-dasharray="10 7"' if dash else ''
        svg.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" {dash_attr}/>')
        ends=[(points[-2],points[-1])]
        if both:ends.append((points[1],points[0]))
        for a,b in ends:
            ang=math.atan2(b[1]-a[1],b[0]-a[0]);length=11;half=5
            q=[b,(b[0]-length*math.cos(ang)+half*math.sin(ang),b[1]-length*math.sin(ang)-half*math.cos(ang)),
                 (b[0]-length*math.cos(ang)-half*math.sin(ang),b[1]-length*math.sin(ang)+half*math.cos(ang))]
            d.polygon([(x*SCALE,y*SCALE) for x,y in q],fill=color)
            svg.append(f'<polygon points="'+ ' '.join(f'{x:.2f},{y:.2f}' for x,y in q)+f'" fill="{color}"/>')
svg.append('</svg>')
base=OUT/'lpu4tt_architecture'
im.save(base.with_suffix('.png'))
base.with_suffix('.svg').write_text('\n'.join(svg),encoding='utf-8')
file=dict(type='excalidraw',version=2,source='https://excalidraw.com',elements=elements,
          appState={'viewBackgroundColor':'#ffffff','gridSize':None},files={})
base.with_suffix('.excalidraw').write_text(json.dumps(file,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert len({e['id'] for e in elements})==len(elements)
assert all(e['fontFamily']==5 for e in elements if e['type']=='text')
summary=dict(canvas=[W,H],png=[W*SCALE,H*SCALE],elements=dict(Counter(e['type'] for e in elements)),
             all_text_font_family=5,scope='Logical architecture; solid baseline plus dashed dual-lane extension',
             files=[p.name for p in (base.with_suffix('.png'),base.with_suffix('.svg'),base.with_suffix('.excalidraw'))])
base.with_suffix('.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary),flush=True)
