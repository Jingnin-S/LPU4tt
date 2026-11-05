# Architecture figure

The PNG is the README preview; the SVG is a scalable vector export.
The Excalidraw file has 105 individually editable elements. Drag it into
https://excalidraw.com to edit; all text uses Excalidraw font family 5.

Solid paths show the one-lane base. Orange dashed paths show the second-lane
extension. The 64B qualified macro and 128B/192B candidates are labeled separately.

The optional generator is scripts/draw_architecture.py. It needs
requirements-diagram.txt and Noto Sans CJK Regular/Bold font collections at the
paths recorded in the script. Noto supplies Latin glyphs for this English figure.
Figure generation is separate from model/RTL tests and the default CI.
