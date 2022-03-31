import argparse

from bg_atlasapi import BrainGlobeAtlas
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import Slider, ColumnDataSource, Button, RadioButtonGroup, PointDrawTool
from bokeh.plotting import figure, Figure

import _feature
from _slice import *

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

atlas = BrainGlobeAtlas(OPT.use_source, check_latest=OPT.check_latest)
slice_view = CoronalView(atlas.reference)
offset_map = slice_view.offset(0, 0)

slice_view_name = ['Coronal', 'Sagittal', 'Transverse']
frame_slider: Slider
rotate_h_slider: Slider
rotate_v_slider: Slider
image_data: ColumnDataSource
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
    _update_image(frame_slider.value)
    _feature_detect('clear')


def _update_frame(attr, old: int, new: int):
    _update_image(new)


def _update_rotate(attr, old: int, new: int):
    global offset_map
    h = rotate_h_slider.value
    v = rotate_v_slider.value
    offset_map = slice_view.offset(h, v)
    _update_image(frame_slider.value)


def _update_image(frame: int):
    r_frame = slice_view.plane(frame + offset_map)
    image_data.data = dict(
        image=[r_frame[::-1, :]],
        dw=[slice_view.width],
        dh=[slice_view.height]
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
image_data = ColumnDataSource(data=dict(
    image=[None],
    dw=[0],
    dh=[0],
))
figure_ref.image(
    'image',
    x=0, y=0,
    dw='dw', dh='dh',
    palette="Cividis256", level="image",
    source=image_data
)

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
_update_image(frame_slider.value)

# layout
model = column(
    slice_view_btn,
    frame_slider,
    row(rotate_h_slider, rotate_h_reset),
    row(rotate_v_slider, rotate_v_reset),
    figure_ref,
    row(feature_detect_btn, feature_clear_btn)
)

doc = curdoc()
doc.add_root(model)
doc.title = 'CCF'
