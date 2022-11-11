# cv2.putTrueTypeText works just like cv2.putText, but with TTF fonts!

<img src="https://github.com/hansalemaos/screenshots/raw/main/cv2_putTrueTypeText_000000.png"/>

### You can add putTrueTypeText to cv2, or ...

```python
$pip install a_cv2_putTrueTypeText

from a_cv_imwrite_imread_plus import add_imwrite_plus_imread_plus_to_cv2
from random import choice
from a_cv2_imshow_thread import add_imshow_thread_to_cv2
from a_cv2_putTrueTypeText import add_truetypetext_to_cv2, get_all_ttf_fonts
import cv2
add_imshow_thread_to_cv2()
add_truetypetext_to_cv2()
add_imwrite_plus_imread_plus_to_cv2()


url = r"https://raw.githubusercontent.com/hansalemaos/screenshots/main/templatematching1.png"
filepath = "c:\\temptemptemppic.png"
pic = cv2.imread_plus(url)
cv2.imwrite_plus(filepath, pic)
ttfonts = get_all_ttf_fonts()



font1 = choice(ttfonts)
test1 = cv2.putTrueTypeText(
    img=filepath,
    text=f"{font1}".lower(),
    org=(50, 120),
    fontFace=font1, #needs to be a file path!
    fontScale=46,
    color=(255, 255, 0),
)

font2 = choice(ttfonts)
test2 = cv2.putTrueTypeText(
    img=url,
    text=f"{font2}".lower(),
    org=(50, 120),
    fontFace=font2,
    fontScale=46,
    color=(255, 0, 255),
)

font3 = choice(ttfonts)
test3 = cv2.putTrueTypeText(
    img=cv2.imread(filepath),
    text=f"{font3}".lower(),
    org=(50, 120),
    fontFace=font3,
    fontScale=46,
    color=(123, 50, 110),
)

font4 = choice(ttfonts)

test4 = cv2.putTrueTypeText(
    img=cv2.imread(filepath, cv2.IMREAD_GRAYSCALE),
    text=f"{font4}".lower(),
    org=(50, 120),
    fontFace=font4,
    fontScale=46,
    color=(255, 0, 255),
)


test5 = cv2.putTrueTypeText(
    img=cv2.imread(filepath),
    text=f"cv2.FONT_HERSHEY_SIMPLEX",
    org=(50, 120),
    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    fontScale=2,
    color=(255, 0, 255),
)


cv2.imshow_thread([test1, test2, test3, test4, test5])
```

<img src="https://github.com/hansalemaos/screenshots/raw/main/cv2_putTrueTypeText_000001.png"/>
<img src="https://github.com/hansalemaos/screenshots/raw/main/cv2_putTrueTypeText_000002.png"/>
<img src="https://github.com/hansalemaos/screenshots/raw/main/cv2_putTrueTypeText_000003.png"/>
<img src="https://github.com/hansalemaos/screenshots/raw/main/cv2_putTrueTypeText_000004.png"/>
<img src="https://github.com/hansalemaos/screenshots/raw/main/cv2_putTrueTypeText_000005.png"/>
<img src="https://github.com/hansalemaos/screenshots/raw/main/cv2_putTrueTypeText_000006.png"/>
<img src="https://github.com/hansalemaos/screenshots/raw/main/cv2_putTrueTypeText_000007.png"/>
<img src="https://github.com/hansalemaos/screenshots/raw/main/cv2_putTrueTypeText_000008.png"/>
<img src="https://github.com/hansalemaos/screenshots/raw/main/cv2_putTrueTypeText_000009.png"/>
<img src="https://github.com/hansalemaos/screenshots/raw/main/cv2_putTrueTypeText_000010.png"/>
<img src="https://github.com/hansalemaos/screenshots/raw/main/cv2_putTrueTypeText_000011.png"/>
<img src="https://github.com/hansalemaos/screenshots/raw/main/cv2_putTrueTypeText_000012.png"/>
<img src="https://github.com/hansalemaos/screenshots/raw/main/cv2_putTrueTypeText_000013.png"/>
<img src="https://github.com/hansalemaos/screenshots/raw/main/cv2_putTrueTypeText_000014.png"/>
<img src="https://github.com/hansalemaos/screenshots/raw/main/cv2_putTrueTypeText_000015.png"/>
<img src="https://github.com/hansalemaos/screenshots/raw/main/cv2_putTrueTypeText_000017.png"/>

### ... import the function

```python
from a_cv_imwrite_imread_plus import add_imwrite_plus_imread_plus_to_cv2
from random import choice
from a_cv2_imshow_thread import add_imshow_thread_to_cv2
from a_cv2_putTrueTypeText import putTrueTypeText, get_all_ttf_fonts
import cv2
add_imwrite_plus_imread_plus_to_cv2()


url = r"https://raw.githubusercontent.com/hansalemaos/screenshots/main/templatematching1.png"
filepath = "c:\\temptemptemppic.png"
pic = cv2.imread_plus(url)
cv2.imwrite_plus(filepath, pic)
ttfonts = get_all_ttf_fonts()
add_imshow_thread_to_cv2()


font1 = choice(ttfonts)
test1 = putTrueTypeText(
    img=filepath,
    text=f"{font1}".lower(),
    org=(50, 120),
    fontFace=font1, #needs to be a file path!
    fontScale=46,
    color=(255, 255, 0),
)

font2 = choice(ttfonts)
test2 = putTrueTypeText(
    img=url,
    text=f"{font2}".lower(),
    org=(50, 120),
    fontFace=font2,
    fontScale=46,
    color=(255, 0, 255),
)

font3 = choice(ttfonts)
test3 = putTrueTypeText(
    img=cv2.imread(filepath),
    text=f"{font3}".lower(),
    org=(50, 120),
    fontFace=font3,
    fontScale=46,
    color=(123, 50, 110),
)

font4 = choice(ttfonts)

test4 = putTrueTypeText(
    img=cv2.imread(filepath, cv2.IMREAD_GRAYSCALE),
    text=f"{font4}".lower(),
    org=(50, 120),
    fontFace=font4,
    fontScale=46,
    color=(255, 0, 255),
)


test5 = putTrueTypeText(
    img=cv2.imread(filepath),
    text=f"cv2.FONT_HERSHEY_SIMPLEX",
    org=(50, 120),
    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    fontScale=2,
    color=(255, 0, 255),
)


cv2.imshow_thread([test1, test2, test3, test4, test5])
```