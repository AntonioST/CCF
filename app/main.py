import argparse
import base64
from io import BytesIO

import cv2
import numpy as np
from bg_atlasapi import BrainGlobeAtlas
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import Slider, ColumnDataSource, Button, RadioButtonGroup, PointDrawTool, FileInput
from bokeh.plotting import figure, Figure

import _feature
from _slice import *

# CLI
AP = argparse.ArgumentParser()
AP.add_argument(
    '--use-source',
    default='allen_mouse_25um',
    help='atlas source name. default: allen_mouse_25um'
)
AP.add_argument(
    '--check-latest',
    action='store_true',
    help='check atlas source version to latest'
)
OPT = AP.parse_args()

# Data
atlas = BrainGlobeAtlas(OPT.use_source, check_latest=OPT.check_latest)
slice_view = CoronalView(atlas.reference)
offset_map = slice_view.offset(0, 0)
image_data: np.ndarray

# View
slice_view_name = ['Coronal', 'Sagittal', 'Transverse']
slice_view_btn: RadioButtonGroup
image_file_btn: FileInput
frame_slider: Slider
rotate_h_slider: Slider
rotate_v_slider: Slider
image_ref: ColumnDataSource
image_tar: ColumnDataSource
figure_ref: Figure
figure_tar: Figure
annotation_ref: ColumnDataSource
annotation_tar: ColumnDataSource


def _change_view(e):
    global slice_view, offset_map
    v = slice_view_btn.active
    c = slice_view_name.index(slice_view.name)
    if v == c:
        return

    if v == 0:
        slice_view = CoronalView(slice_view.reference)
    elif v == 1:
        slice_view = SagittalView(slice_view.reference)
    elif v == 2:
        slice_view = TransverseView(slice_view.reference)
    else:
        return

    frame_slider.end = slice_view.n_frame
    h = rotate_h_slider.value
    v = rotate_v_slider.value
    offset_map = slice_view.offset(h, v)
    _update_image_ref(frame_slider.value)
    _feature_detect('clear')


def _update_frame(attr, old: int, new: int):
    _update_image_ref(new)


def _update_file(attr, old, new):
    global image_data
    filename = image_file_btn.filename

    if len(filename) == 0:
        image_data = None
    else:
        image = BytesIO(base64.b64decode(image_file_btn.value))
        # https://stackoverflow.com/a/25198846
        image = np.asarray(bytearray(image.getvalue()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image_data = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)

    _update_image_tar()


def _update_rotate(attr, old: int, new: int):
    global offset_map
    h = rotate_h_slider.value
    v = rotate_v_slider.value
    offset_map = slice_view.offset(h, v)
    _update_image_ref(frame_slider.value)


def _update_image_ref(frame: int):
    r_frame = slice_view.plane(frame + offset_map)
    image_ref.data = dict(
        image=[r_frame[::-1, :]],
        dw=[slice_view.width],
        dh=[slice_view.height]
    )


def _update_image_tar():
    if image_data is None:
        w = slice_view.width
        h = slice_view.height
        image = np.zeros((w, h), dtype=np.uint32)
        view = image.view(dtype=np.uint8).reshape((w, h, 4))
        for i in range(w):
            for j in range(h):
                view[i, j, 0] = int(255 * i / w)
                view[i, j, 1] = 158
                view[i, j, 2] = int(255 * j / h)
                view[i, j, 3] = 255

    else:
        w, h, _ = image_data.shape
        image = image_data.view(dtype=np.uint32).reshape((w, h))

    image_tar.data = dict(
        image=[image],
        dw=[w],
        dh=[h]
    )


def _reset_rotate(target: str):
    if target == 'h':
        rotate_h_slider.value = 0
    elif target == 'v':
        rotate_v_slider.value = 0


def _feature_detect(target: str):
    """

    https://www.analyticsvidhya.com/blog/2021/06/feature-detection-description-and-matching-of-images-using-opencv/

    :param target:
    :return:
    """
    if target == 'clear':
        annotation_ref.data = dict(x=[], y=[])
    else:
        r_frame = slice_view.plane(frame_slider.value + offset_map)
        annotation_ref.data = _feature.feature_detect_blob(r_frame[::-1, :])


# init slice_view_btn
slice_view_btn = RadioButtonGroup(
    labels=['Coronal', 'Sagittal', 'Transverse'],
    active=0
)
slice_view_btn.on_click(_change_view)

# init frame_slider
frame_slider = Slider(
    start=0,
    end=slice_view.n_frame,
    value=slice_view.n_frame // 2,
    step=1,
    title='Frame'
)
frame_slider.on_change('value', _update_frame)

# init rotate_h_slider
rotate_h_slider = Slider(
    start=-100,
    end=100,
    value=0,
    step=1,
    title='vertical rotate',
    sizing_mode='scale_width',
)
rotate_h_slider.on_change('value', _update_rotate)

# init rotate_v_slider
rotate_v_slider = Slider(
    start=-100,
    end=100,
    value=0,
    step=1,
    title='horizontal rotate',
    sizing_mode='scale_width',
)
rotate_v_slider.on_change('value', _update_rotate)

# init rotate_h_reset and rotate_v_reset
rotate_h_reset = Button(label='reset', width_policy='min')
rotate_h_reset.on_click(lambda it: _reset_rotate('h'))

rotate_v_reset = Button(label='reset', width_policy='min')
rotate_v_reset.on_click(lambda it: _reset_rotate('v'))

# init reference figure
figure_ref = figure(
    width=2 * slice_view.width,
    height=2 * slice_view.height,
    toolbar_location='right',
    tools='pan,wheel_zoom,box_zoom,save,reset'
)

# reference image
image_ref = ColumnDataSource(data=dict(
    image=[None],
    dw=[0],
    dh=[0],
))
figure_ref.image(
    'image',
    x=0, y=0,
    dw='dw', dh='dh',
    palette="Cividis256", level="image",
    source=image_ref
)

figure_tar = figure(
    width=2 * slice_view.width,
    height=2 * slice_view.height,
    toolbar_location='right',
    tools='pan,wheel_zoom,box_zoom,save,reset'
)
image_tar = ColumnDataSource(data=dict(
    image=[None],
    dw=[0],
    dh=[0],
))
figure_tar.image_rgba(
    'image',
    x=0, y=0,
    dw='dw', dh='dh',
    source=image_tar
)

image_file_btn = FileInput(accept='image/*')
image_file_btn.on_change('filename', _update_file)

# reference annotations
# http://docs.bokeh.org/en/1.0.0/docs/user_guide/examples/tools_point_draw.html
annotation_ref = ColumnDataSource(data=dict(x=[], y=[]))
annotation_ref_plot = figure_ref.scatter(
    'x', 'y', size=12, color='green',
    source=annotation_ref,
)
figure_ref.add_tools(PointDrawTool(
    renderers=[annotation_ref_plot],
    description='add reference points'
))

#
feature_detect_btn = Button(label='detect', width_policy='min')
feature_detect_btn.on_click(lambda it: _feature_detect('ref'))
feature_clear_btn = Button(label='clear', width_policy='min')
feature_clear_btn.on_click(lambda it: _feature_detect('clear'))

# update
_update_image_ref(frame_slider.value)
_update_image_tar()

# layout

model_ref = column(
    slice_view_btn,
    frame_slider,
    row(rotate_h_slider, rotate_h_reset),
    row(rotate_v_slider, rotate_v_reset),
    figure_ref,
    row(feature_detect_btn, feature_clear_btn)
)
model_tar = column(
    image_file_btn,
    figure_tar
)
model = row(model_ref, model_tar)

# Document
doc = curdoc()
doc.add_root(model)
doc.title = 'CCF'
