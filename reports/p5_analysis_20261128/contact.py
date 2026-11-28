from pathlib import Path
from PIL import Image,ImageOps,ImageDraw
p=Path(__file__).resolve().parent
pages=sorted((p/'audit/pages').glob('page-*.png'))
for start in range(0,len(pages),4):
 subset=pages[start:start+4];canvas=Image.new('RGB',(1000,1450),'#d1d5db')
 for j,f in enumerate(subset):
  im=Image.open(f).convert('RGB');im.thumbnail((480,680))
  x=(j%2)*500+10;y=(j//2)*725+30
  canvas.paste(im,(x,y));ImageDraw.Draw(canvas).text((x,y-20),f.stem,fill='black')
 canvas.save(p/f'audit/contact_{start//4+1}.png')
print('pages',len(pages))
