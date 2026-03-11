from PIL import Image

img = Image.open("dataset/bee.png")

sprites = {
"bee1":(60,130,260,290),
"bee2":(60,310,260,470),
"bee3":(60,490,260,650),

"pipe_top":(370,110,590,310),
"pipe_bottom":(370,340,590,630),

"honey":(640,150,760,280),
"flower":(770,150,910,280),

"background":(980,110,1320,420),

"ground":(60,690,820,820),

"gameover":(510,660,840,770),
"start":(530,770,830,860),
}

for name,box in sprites.items():
    img.crop(box).save(f"{name}.png")