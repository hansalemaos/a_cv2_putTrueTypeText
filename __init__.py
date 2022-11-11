# Based on https://github.com/bunkahle/PILasOPENCV
# MIT License
#
# Copyright (c) 2019 Andreas Bunkahle
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import os
import sys
import freetype
import cv2
import numpy as np
from a_cv_imwrite_imread_plus import open_image_in_cv


def get_all_ttf_fonts() -> list:

    dirs = []
    if sys.platform == "win32":
        # check the windows font repository
        # NOTE: must use uppercase WINDIR, to work around bugs in
        # 1.5.2's os.environ.get()
        windir = os.environ.get("WINDIR")
        if windir:
            dirs.append(os.path.join(windir, "Fonts"))
    elif sys.platform in ("linux", "linux2"):
        lindirs = os.environ.get("XDG_DATA_DIRS", "")
        if not lindirs:
            # According to the freedesktop spec, XDG_DATA_DIRS should
            # default to /usr/share
            lindirs = "/usr/share"
        dirs += [os.path.join(lindir, "fonts") for lindir in lindirs.split(":")]
    elif sys.platform == "darwin":
        dirs += [
            "/Library/Fonts",
            "/System/Library/Fonts",
            os.path.expanduser("~/Library/Fonts"),
        ]
    allttffonts = []
    for directory in dirs:
        for walkroot, walkdir, walkfilenames in os.walk(directory):
            for walkfilename in walkfilenames:
                fontpath = os.path.join(walkroot, walkfilename)
                if os.path.splitext(fontpath)[1] == ".ttf":
                    allttffonts.append(fontpath)
    return allttffonts


def reverse_color(color):
    if len(color) == 3:
        return list(reversed(color))
    elif len(color) == 4:
        return list(reversed(color[:3])) + [color[-1]]
    return color


def image_to_original(im, oldshape):
    bala = im.copy()
    if len(oldshape) == 2:
        bala = cv2.cvtColor(bala, cv2.COLOR_BGRA2BGR)
    elif len(oldshape) == 3:
        if oldshape[-1] == 3:
            bala = cv2.cvtColor(bala, cv2.COLOR_BGRA2BGR)
    return bala


def convert_cv_image_to_4_channels(img):
    image = img.copy()
    if len(image.shape) == 3:
        if image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)

    elif len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
    return image


def get_right_font_fix(text, font=r"C:\Windows\Fonts\ANTQUAB.TTF", size=32):
    if isinstance(font, str):
        ttf_font = freetype.Face(font)
    else:
        ttf_font = font
    ttf_font.set_char_size(size * size)
    width, height, baseline = getsize(text, ttf_font)
    _, height, _ = getsize("|", ttf_font)
    return ttf_font, size, width, height, getmask(text, ttf_font)


def getmask(text_to_write, ttf_font):
    text_to_write = f"{text_to_write}"
    text = f"|{text_to_write}|"
    slot = ttf_font.glyph
    _, height, baseline = getsize(text, ttf_font)
    width, _, _ = getsize(text_to_write, ttf_font)
    Z = np.zeros((height, width), dtype=np.ubyte)
    x, y = 0, 0
    previous = 0
    maxlentext = len(text) - 1
    prevone = 0
    for ini, c in enumerate(text):
        ttf_font.load_char(c)
        bitmap = slot.bitmap
        top = slot.bitmap_top
        w, h = bitmap.width, bitmap.rows
        y = height - baseline - top
        if y <= 0:
            y = 0
        kerning = ttf_font.get_kerning(previous, c)
        x += kerning.x >> 6
        prevone = x

        character = np.array(bitmap.buffer, dtype="uint8").reshape(h, w)
        if ini != 0 and ini != maxlentext:
            try:
                Z[y : y + h, x : x + w] += character
            except ValueError:
                while x + w > Z.shape[1]:
                    x = x - 1
                if x > 0:
                    Z[: character.shape[0], x : x + w] += character
            x += slot.advance.x >> 6

            previous = c
    Z = Z[:, : prevone + 1]
    return Z


def getsize(text, ttf_font):
    slot = ttf_font.glyph
    width, height, baseline = 0, 0, 0
    previous = 0
    for i, c in enumerate(text):
        ttf_font.load_char(c)
        bitmap = slot.bitmap
        height = max(height, bitmap.rows + max(0, -(slot.bitmap_top - bitmap.rows)))
        baseline = max(baseline, max(0, -(slot.bitmap_top - bitmap.rows)))
        kerning = ttf_font.get_kerning(previous, c)
        width += (slot.advance.x >> 6) + (kerning.x >> 6)
        previous = c
    return width, height, baseline


def get_image_mask(image, color):
    ink = reverse_color(color)
    img = np.zeros(image.shape, dtype=image.dtype)
    if len(img.shape) > 2:
        if img.shape[2] >= 2:
            img[:, :, 0] = ink[0]
            img[:, :, 1] = ink[1]
        if img.shape[2] >= 3:
            img[:, :, 2] = ink[2]
        if img.shape[2] == 4:
            img[:, :, 3] = 0
    else:
        img[:] = ink
    return img


def _paste(mother, child, x, y):
    "Pastes the numpy image child into the numpy image mother at position (x, y)"
    size = mother.shape
    csize = child.shape
    if y + csize[0] < 0 or x + csize[1] < 0 or y > size[0] or x > size[1]:
        return mother
    sel = [int(y), int(x), csize[0], csize[1]]
    csel = [0, 0, csize[0], csize[1]]
    if y < 0:
        sel[0] = 0
        sel[2] = csel[2] + y
        csel[0] = -y
    elif y + sel[2] >= size[0]:
        sel[2] = int(size[0])
        csel[2] = size[0] - y
    else:
        sel[2] = sel[0] + sel[2]
    if x < 0:
        sel[1] = 0
        sel[3] = csel[3] + x
        csel[1] = -x
    elif x + sel[3] >= size[1]:
        sel[3] = int(size[1])
        csel[3] = size[1] - x
    else:
        sel[3] = sel[1] + sel[3]
    childpart = child[csel[0] : csel[2], csel[1] : csel[3]]
    mother[sel[0] : sel[2], sel[1] : sel[3]] = childpart
    return mother


def split(im, image=None):
    _instance = im.copy()
    "splits the image into its color bands"
    if image is None:
        if len(_instance.shape) == 3:
            if _instance.shape[2] == 1:
                return _instance.copy()
            elif _instance.shape[2] == 2:
                l, a = cv2.split(_instance)
                return l, a
            elif _instance.shape[2] == 3:
                b, g, r = cv2.split(_instance)
                return b, g, r
            else:
                b, g, r, a = cv2.split(_instance)
                return b, g, r, a
        else:
            return _instance
    else:
        if len(_instance.shape) == 3:
            if image.shape[2] == 1:
                return image.copy()
            elif image.shape[2] == 2:
                l, a = cv2.split(image)
                return l, a
            elif image.shape[2] == 3:
                b, g, r = cv2.split(image)
                return b, g, r
            else:
                b, g, r, a = cv2.split(image)
                return b, g, r, a
        else:
            return _instance


def paste(imgbase, img_color, box=None, mask=None, transparency=-1):
    "pastes either an image or a color to a region of interest defined in box with a mask"
    _instance = imgbase.copy()
    if box is None:
        raise ValueError("cannot determine region size; use 4-item box")
    img_dim = (box[3] + 1, box[2] + 1)
    channels, depth = (
        2 if len(img_color.shape) == 2 else img_color.shape[-1],
        img_color.dtype,
    )
    colorbox = np.zeros((img_dim[0], img_dim[1], channels), dtype=depth)
    if channels > 2:
        colorbox[:] = img_color[0][0]
    else:
        colorbox[:] = img_color[0]
    _img_color = colorbox.copy()
    if mask is None:
        _instance = _paste(_instance, _img_color, box[0], box[1])
    else:
        # enlarge the image _img_color without resizing to the new_canvas
        new_canvas = np.zeros(_instance.shape, dtype=_instance.dtype)
        new_canvas = _paste(new_canvas, _img_color, box[0], box[1])

        if len(_instance.shape) == 3:
            if _instance.shape[2] == 4:  # RGBA
                *_, _mask = split(mask)

            elif _instance.shape[2] == 1:
                _mask = _instance.copy()
        else:
            _mask = _instance.copy()

        if mask.shape[:2] != new_canvas.shape[:2]:
            _new_mask = np.zeros(_instance.shape[:2], dtype=_instance.dtype)
            _new_mask = ~(_paste(_new_mask, mask, box[0], box[1]))
        else:
            _new_mask = ~mask
        if transparency > 0:
            calcmaxadd = _new_mask.copy()
            calcmaxadd[np.where(calcmaxadd >= np.max(img_color[0][0]))] = 0
            maxcolor = np.max(img_color[0][0])
            maxadd = 254 - maxcolor
            if transparency < maxadd:
                maxadd = transparency
            _new_mask[np.where(_new_mask <= maxcolor)] = (
                maxadd + _new_mask[np.where(_new_mask <= maxcolor)]
            )
        _instance = composite(_instance, new_canvas, _new_mask)
    return _instance


def composite(background, foreground, mask, neg_mask=False):
    "pastes the foreground image into the background image using the mask"
    # Convert uint8 to float
    foreground = foreground.astype(float)
    old_type = background.dtype
    background = background.astype(float)
    # Normalize the alpha mask to keep intensity between 0 and 1
    if neg_mask:
        alphamask = mask.astype(float) / 255
    else:
        alphamask = (~mask).astype(float) / 255

    fslen = len(foreground.shape)
    if len(alphamask.shape) != fslen:
        img = np.zeros(foreground.shape, dtype=foreground.dtype)
        if fslen > 2:
            if foreground.shape[2] >= 2:
                img[:, :, 0] = alphamask
                img[:, :, 1] = alphamask
            if foreground.shape[2] >= 3:
                img[:, :, 2] = alphamask
            if foreground.shape[2] == 4:
                img[:, :, 3] = alphamask
            alphamask = img.copy()
    # Multiply the foreground with the alpha mask
    try:
        foreground = cv2.multiply(alphamask, foreground)
    except:
        if alphamask.shape[2] == 1 and foreground.shape[2] == 3:
            triplemask = cv2.merge((alphamask, alphamask, alphamask))
            foreground = cv2.multiply(triplemask, foreground)
        else:
            raise ValueError(
                "OpenCV Error: Sizes of input arguments do not match (The operation is neither 'array op array' (where arrays have the same size and the same number of channels), nor 'array op scalar', nor 'scalar op array') in cv::arithm_op, file ..\..\..\..\opencv\modules\core\src\arithm.cpp"
            )
    # Multiply the background with ( 1 - alpha )
    bslen = len(background.shape)
    if len(alphamask.shape) != bslen:
        img = np.zeros(background.shape, dtype=background.dtype)
        if bslen > 2:
            if background.shape[2] >= 2:
                img[:, :, 0] = alphamask
                img[:, :, 1] = alphamask
            if background.shape[2] >= 3:
                img[:, :, 2] = alphamask
            if background.shape[2] == 4:
                img[:, :, 3] = alphamask
            alphamask = img.copy()
    try:
        background = cv2.multiply(1.0 - alphamask, background)
    except:
        if alphamask.shape[2] == 1 and foreground.shape[2] == 3:
            background = cv2.multiply(1.0 - triplemask, background)
        else:
            raise ValueError(
                "OpenCV Error: Sizes of input arguments do not match (The operation is neither 'array op array' (where arrays have the same size and the same number of channels), nor 'array op scalar', nor 'scalar op array') in cv::arithm_op, file ..\..\..\..\opencv\modules\core\src\arithm.cpp"
            )
    # Add the masked foreground and background
    outImage = cv2.add(foreground, background)
    outImage = outImage / 255
    outImage = outImage * 255
    outImage = outImage.astype(old_type)
    return outImage


def putTrueTypeText(img, text, org, fontFace, fontScale, color, *args, **kwargs):
    image = open_image_in_cv(img)
    text = f"{text} "
    if isinstance(fontFace, int):
        image2 = cv2.putText(
            image.copy(), text[:-1], org, fontFace, fontScale, color, *args, **kwargs
        )
        return image2
    ttf_font, fontScale, width, height, fontmask = get_right_font_fix(
        text, font=fontFace, size=fontScale
    )
    oldshape = image.shape
    img2 = convert_cv_image_to_4_channels(image)
    imga = get_image_mask(image=img2, color=color)
    box = [int(org[0]), int(org[1] - height), int(org[0] + width), int(org[1])]
    bala = paste(imgbase=img2, img_color=imga, box=box, mask=fontmask, transparency=-1)
    bala = image_to_original(bala, oldshape)
    return bala


def add_truetypetext_to_cv2():
    cv2.putTrueTypeText = putTrueTypeText
