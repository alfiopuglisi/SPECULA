'''
Reproduce issue #397: macOS crash when a matplotlib window is opened
after drawing text with cairo (as orthogram does for the block diagram).

Usage: python repro.py control|cairo|orthogram|specula
'''
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
mode = sys.argv[1]


def show_window():
    import matplotlib
    import matplotlib.pyplot as plt
    print('matplotlib backend:', matplotlib.get_backend(), flush=True)
    fig = plt.figure()
    plt.plot([1, 2, 3])
    fig.show()
    plt.pause(1)
    plt.close(fig)


if mode == 'control':
    show_window()

elif mode == 'cairo':
    import cairo
    print('cairo', cairo.cairo_version_string(), 'pycairo', cairo.version, flush=True)
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 300, 100)
    ctx = cairo.Context(surface)
    ctx.select_font_face('sans-serif')
    ctx.set_font_size(30)
    ctx.move_to(10, 50)
    ctx.show_text('SPECULA')
    surface.write_to_png(os.path.join(HERE, 'cairo.png'))
    show_window()

elif mode == 'orthogram':
    from orthogram import DiagramDef, write_png
    d = DiagramDef(label='test')
    d.add_row(['a', 'b'])
    d.add_connection('a', 'b', label='a->b')
    write_png(d, os.path.join(HERE, 'orthogram.png'))
    show_window()

elif mode == 'specula':
    import specula
    specula.main_simul(yml_files=[os.path.join(HERE, 'p.yml')], cpu=True,
                       diagram_filename=os.path.join(HERE, 'diagram.png'))

print(f'REPRO {mode}: no crash', flush=True)
